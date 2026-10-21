from __future__ import annotations

from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.models.schemas import (
    ApplicationView,
    ChatResponse,
    Job,
    JobRecommendation,
    ToolCall,
)
from server.app.services.llm import LLMTurn, NativeToolCall
from server.app.services.sessions import SessionStore
from server.app.tools.applications import (
    APPLY_DEFINITION,
    LIST_DEFINITION,
)
from server.app.tools.base import ToolContext, ToolDefinition
from server.app.tools.jobs import (
    DETAILS_DEFINITION,
    RECOMMEND_DEFINITION,
)
from server.app.tools.resume import DEFINITION as PARSE_RESUME_DEFINITION

MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = """You are AIJob, a concise and supportive job-search assistant.
Use tools to inspect resumes, find jobs, retrieve job details, record applications, and list applications.
Never invent a job, employer, score, application, or resume fact. You may discuss jobs only when they are
returned by a tool. When a tool says clarification_required, ask the user its exact clarification question.
When a tool says resume_required, ask the user to upload a PDF or DOCX resume before continuing.
No job application is submitted externally; apply_to_job records the user's application in this assistant.
For filters, preserve earlier active filters unless the user asks to remove or replace them.
Be transparent about missing information, no matches, and failures. Keep replies brief."""

TOOL_DEFINITIONS = [
    PARSE_RESUME_DEFINITION,
    RECOMMEND_DEFINITION,
    DETAILS_DEFINITION,
    APPLY_DEFINITION,
    LIST_DEFINITION,
]


class ChatModel(Protocol):
    async def generate_chat(
        self,
        contents: list[Any],
        tools: list[ToolDefinition],
        system_instruction: str,
    ) -> LLMTurn: ...

    def format_tool_response(
        self, call: NativeToolCall, output: dict[str, Any]
    ) -> Any: ...


class Orchestrator:
    def __init__(
        self,
        model: ChatModel,
        store: SessionStore,
        jobs: list[Job],
    ) -> None:
        self.model = model
        self.store = store
        self.jobs = jobs
        self._tools = {tool.name: tool for tool in TOOL_DEFINITIONS}

    async def chat(
        self,
        session_id: str,
        message: str,
        db_session: AsyncSession,
    ) -> ChatResponse:
        state = self.store.get(session_id)
        context = ToolContext(
            session_id=session_id,
            state=state,
            session_store=self.store,
            db_session=db_session,
            jobs=self.jobs,
        )
        contents: list[Any] = [message]
        calls_for_response: list[ToolCall] = []
        displayed_jobs: list[JobRecommendation] | None = None
        applications: list[ApplicationView] | None = None
        reply = ""
        iterations = 0
        while iterations < MAX_TOOL_ITERATIONS:
            turn = await self.model.generate_chat(
                contents=contents,
                tools=TOOL_DEFINITIONS,
                system_instruction=SYSTEM_PROMPT,
            )
            iterations += 1
            if not turn.tool_calls:
                reply = turn.text
                break
            contents.append(turn.model_content)
            for call in turn.tool_calls:
                if len(calls_for_response) >= MAX_TOOL_ITERATIONS:
                    break
                definition = self._tools.get(call.name)
                if definition is None:
                    tool_output: dict[str, Any] = {
                        "status": "unknown_tool",
                        "message": "That tool isn't available. Use one of the provided tools.",
                    }
                else:
                    calls_for_response.append(
                        ToolCall(name=call.name, arguments=call.arguments)
                    )
                    try:
                        arguments = definition.input_model.model_validate(call.arguments)
                    except ValidationError as exc:
                        tool_output = {
                            "status": "invalid_arguments",
                            "message": "The tool arguments were invalid. Retry using the declared schema.",
                            "details": exc.errors(include_input=False),
                        }
                    else:
                        tool_output = await definition.execute(context, arguments)

                if call.name == "recommend_jobs" and "jobs" in tool_output:
                    displayed_jobs = [
                        JobRecommendation.model_validate(item)
                        for item in tool_output["jobs"]
                    ]
                if call.name == "list_applications" and "applications" in tool_output:
                    applications = [
                        ApplicationView.model_validate(item)
                        for item in tool_output["applications"]
                    ]
                contents.append(self.model.format_tool_response(call, tool_output))
            if len(calls_for_response) >= MAX_TOOL_ITERATIONS:
                break
        if not reply:
            reply = (
                "I completed the available steps, but this request reached the tool-call limit. "
                "Please check the results shown and tell me what you'd like to do next."
            )
        return ChatResponse(
            reply=reply,
            tool_calls=calls_for_response,
            jobs=displayed_jobs,
            applications=applications,
            profile=state.profile,
        )
