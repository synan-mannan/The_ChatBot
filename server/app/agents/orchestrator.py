from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.models.schemas import (
    ApplicationActionResponse,
    ApplicationConfirmation,
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
    ApplyToJobInput,
    apply_to_job_tool,
)
from server.app.tools.application_answers import SAVE_APPLICATION_ANSWERS_DEFINITION
from server.app.tools.application_documents import (
    APPROVE_COVER_LETTER_DEFINITION,
    GENERATE_COVER_LETTER_DEFINITION,
    ApproveCoverLetterInput,
    approve_cover_letter_tool,
)
from server.app.tools.base import ToolContext, ToolDefinition
from server.app.config import Settings, get_settings
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
For any request to find, search, recommend, or match jobs, call recommend_jobs directly. That tool uses
the resume already uploaded to this session; do not ask the user to upload or parse it again just to
recommend jobs. Do not call parse_resume as a prerequisite to recommend_jobs.
For requests to apply to a numbered job, call apply_to_job and use the matching position from the
most recent recommendations (for example, job 2 means displayed:2). Use an explicit id:N only when
the user clearly provides a dataset job ID. Do not ask for clarification when the numbered position
exists in the recent recommendations.
For applications, call apply_to_job first with confirm=false. Only call it again with confirm=true
after the user explicitly agrees in a later message containing a clear affirmative confirmation.
For generic employer career portals, this confirmation only prepares the form for review. If the
form is review_ready, it is NOT submitted until the user explicitly says to submit it in a later
message. If the report is needs_input, ask for that batch of missing answers; never invent them.
Report exactly the status, filled field list, missing questions, blockers, and screenshot returned
by the tool. Never claim submission unless the report status is submitted.
For filters, preserve earlier active filters unless the user asks to remove or replace them.
When a user provides an application answer such as work authorization, salary, relocation, notice
period, availability, or EEO information, save only the answer they explicitly stated with
save_application_answers. Never infer or invent application answers.
Generate cover letters only when asked; show the generated draft and require explicit approval
before including it in any application.
An application is submitted only when apply_to_job returns status submitted. For dry_run, say that
submission is disabled and nothing was sent. For review_ready, say that the form is filled but NOT
submitted. For needs_input, ask for the listed missing answers in one batch. For needs_manual,
provide the apply link and say it was not submitted. For failed, explain the failure and offer the
link. Never claim success for any other status.
Be transparent about missing information, no matches, and failures. Keep replies brief."""

TOOL_DEFINITIONS = [
    PARSE_RESUME_DEFINITION,
    RECOMMEND_DEFINITION,
    DETAILS_DEFINITION,
    APPLY_DEFINITION,
    LIST_DEFINITION,
    SAVE_APPLICATION_ANSWERS_DEFINITION,
    GENERATE_COVER_LETTER_DEFINITION,
    APPROVE_COVER_LETTER_DEFINITION,
]
CONFIRMATION_PHRASES = {
    "yes",
    "yes submit",
    "yes prepare",
    "yes prepare for review",
    "prepare it",
    "yes please",
    "yes please submit",
    "confirm",
    "confirm application",
    "go ahead",
    "proceed",
    "submit it",
    "submit the application",
}
REJECTION_PHRASES = {
    "no",
    "no thanks",
    "cancel",
    "cancel application",
    "do not submit",
    "dont submit",
}


def _normalized_short_reply(message: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", message.casefold()).strip()


def _is_explicit_confirmation(message: str) -> bool:
    return _normalized_short_reply(message) in CONFIRMATION_PHRASES


def _is_explicit_rejection(message: str) -> bool:
    return _normalized_short_reply(message) in REJECTION_PHRASES


def _is_explicit_cover_letter_approval(message: str) -> bool:
    return _normalized_short_reply(message) in {
        "approve",
        "approve it",
        "approve the cover letter",
        "approve this cover letter",
        "use this cover letter",
        "use it",
    }


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
        settings: Settings | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.jobs = jobs
        self.settings = settings or get_settings()
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
            model=self.model,
            settings=self.settings,
        )
        system_instruction = (
            f"{SYSTEM_PROMPT}\n\n"
            "Session state: an uploaded resume profile is "
            f"{'available' if state.profile is not None else 'not available'}."
        )
        if state.last_shown_jobs:
            recent_jobs = "\n".join(
                f"{index}. {recommendation.job.title} at {recommendation.job.company} "
                f"(dataset ID {recommendation.job.id})"
                for index, recommendation in enumerate(state.last_shown_jobs, start=1)
            )
            system_instruction += f"\nMost recent numbered job results:\n{recent_jobs}"
        if state.pending_application_job_id:
            pending_job = next(
                (
                    job for job in self.jobs
                    if job.id == state.pending_application_job_id
                ),
                None,
            )
            if pending_job is not None:
                system_instruction += (
                    "\nPending application confirmation: "
                    f"{pending_job.title} at {pending_job.company} "
                    f"(dataset ID {pending_job.id}). Only proceed if the user's current message "
                    "is an explicit affirmative confirmation."
                )
        if (
            _is_explicit_rejection(message)
            and (
                state.pending_application_job_id
                or state.pending_autofill_job_id
                or state.pending_submission_job_id
            )
        ):
            state.pending_application_job_id = None
            state.pending_autofill_job_id = None
            state.pending_submission_job_id = None
            pending_reply = "Understood. I cancelled the application; nothing was submitted."
            return ChatResponse(reply=pending_reply, profile=state.profile)
        if (
            state.pending_cover_letter_job_id
            and _is_explicit_cover_letter_approval(message)
        ):
            approved_job_id = state.pending_cover_letter_job_id
            approved = await approve_cover_letter_tool(
                context,
                ApproveCoverLetterInput(job_id=approved_job_id),
            )
            return ChatResponse(
                reply=str(approved["message"]),
                tool_calls=[
                    ToolCall(
                        name="approve_cover_letter",
                        arguments={"job_id": approved_job_id},
                    )
                ],
                profile=state.profile,
            )
        contents: list[Any] = [message]
        calls_for_response: list[ToolCall] = []
        displayed_jobs: list[JobRecommendation] | None = None
        applications: list[ApplicationView] | None = None
        application_confirmation: ApplicationConfirmation | None = None
        application_result: ApplicationActionResponse | None = None
        profile = state.profile
        reply = ""
        terminal_tool_result = False
        iterations = 0
        while iterations < MAX_TOOL_ITERATIONS:
            turn = await self.model.generate_chat(
                contents=contents,
                tools=TOOL_DEFINITIONS,
                system_instruction=system_instruction,
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
                call_arguments = dict(call.arguments)
                if call.name == "apply_to_job":
                    if (
                        state.pending_submission_job_id
                        and _is_explicit_confirmation(message)
                    ):
                        call_arguments["job_id"] = (
                            f"id:{state.pending_submission_job_id}"
                        )
                        call_arguments["confirm"] = False
                        call_arguments["submit"] = True
                    elif (
                        state.pending_application_job_id
                        and _is_explicit_confirmation(message)
                    ):
                        call_arguments["job_id"] = (
                            f"id:{state.pending_application_job_id}"
                        )
                        call_arguments["confirm"] = True
                    elif call_arguments.get("confirm") is True:
                        call_arguments["confirm"] = False
                if definition is None:
                    tool_output: dict[str, Any] = {
                        "status": "unknown_tool",
                        "message": "That tool isn't available. Use one of the provided tools.",
                    }
                else:
                    calls_for_response.append(
                        ToolCall(name=call.name, arguments=call_arguments)
                    )
                    try:
                        arguments = definition.input_model.model_validate(call_arguments)
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
                    reply = self._recommendations_reply(tool_output)
                    terminal_tool_result = True
                    break
                if call.name == "list_applications" and "applications" in tool_output:
                    applications = [
                        ApplicationView.model_validate(item)
                        for item in tool_output["applications"]
                    ]
                if call.name == "apply_to_job":
                    if tool_output.get("status") not in {
                        "invalid_arguments",
                        "unknown_tool",
                    }:
                        application_result = ApplicationActionResponse.model_validate(
                            tool_output
                        )
                        if tool_output.get("status") == "confirmation_required":
                            application_confirmation = (
                                ApplicationConfirmation.model_validate(tool_output)
                            )
                    reply = str(tool_output["message"])
                    terminal_tool_result = True
                    break
                if call.name == "list_applications":
                    reply = (
                        self._applications_reply(applications or [])
                        if tool_output.get("status") == "ok"
                        else str(tool_output["message"])
                    )
                    terminal_tool_result = True
                    break
                if call.name == "save_application_answers":
                    reply = str(tool_output.get("message", ""))
                    if (
                        tool_output.get("status") == "saved"
                        and state.pending_autofill_job_id
                    ):
                        rerun = await apply_to_job_tool(
                            context,
                            ApplyToJobInput(
                                job_id=f"id:{state.pending_autofill_job_id}",
                                confirm=True,
                            ),
                        )
                        application_result = ApplicationActionResponse.model_validate(
                            rerun
                        )
                        reply = str(rerun["message"])
                    terminal_tool_result = True
                    break
                if call.name == "generate_cover_letter":
                    if tool_output.get("status") == "draft_ready":
                        reply = (
                            f"Draft cover letter for {tool_output['title']} at "
                            f"{tool_output['company']} (not approved):\n\n"
                            f"{tool_output['cover_letter']}\n\n"
                            "Review it and say “approve the cover letter” if you want it "
                            "included for this job. It will not be submitted."
                        )
                    else:
                        reply = str(tool_output["message"])
                    terminal_tool_result = True
                    break
                if call.name == "approve_cover_letter":
                    reply = str(tool_output["message"])
                    terminal_tool_result = True
                    break
                contents.append(self.model.format_tool_response(call, tool_output))
            if terminal_tool_result:
                break
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
            application_confirmation=application_confirmation,
            application_result=application_result,
            profile=profile,
        )

    @staticmethod
    def _recommendations_reply(tool_output: dict[str, Any]) -> str:
        if tool_output.get("status") == "no_matches":
            return str(tool_output["message"])
        count = len(tool_output.get("jobs", []))
        return (
            f"I found {count} matching job{'s' if count != 1 else ''}. "
            "The ranked cards below include match reasons and application buttons. "
            "You can say, for example, “apply to job 2.”"
        )

    @staticmethod
    def _applications_reply(applications: list[ApplicationView]) -> str:
        if not applications:
            return "You don't have any recorded applications yet."
        lines = ["Here are your recorded applications:"]
        for application in applications:
            lines.append(
                f"- {application.title} at {application.company} "
                f"({application.status}, applied {application.applied_date:%Y-%m-%d})"
            )
        return "\n".join(lines)
