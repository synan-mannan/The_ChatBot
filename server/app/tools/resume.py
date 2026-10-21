from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from server.app.models.schemas import ResumeProfile
from server.app.tools.base import ToolContext, ToolDefinition


class ParseResumeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def parse_resume_tool(
    context: ToolContext, _: ParseResumeInput
) -> dict[str, object]:
    if context.state.profile is None:
        return {
            "status": "resume_required",
            "message": "Ask the user to upload a PDF or DOCX resume first.",
        }
    return {
        "status": "ok",
        "profile": context.state.profile.model_dump(mode="json"),
    }


DEFINITION = ToolDefinition(
    name="parse_resume",
    description=(
        "Return the structured details from the resume already uploaded in this chat. "
        "If none has been uploaded, tell the user to upload one."
    ),
    input_model=ParseResumeInput,
    execute=parse_resume_tool,
)
