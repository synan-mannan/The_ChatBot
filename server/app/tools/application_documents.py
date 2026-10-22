from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.app.models.db import ResumeProfileRow
from server.app.services.autofill import generate_cover_letter
from server.app.tools.base import (
    ToolContext,
    ToolDefinition,
    resolve_job_reference,
)


class GenerateCoverLetterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)


class ApproveCoverLetterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)


async def generate_cover_letter_tool(
    context: ToolContext, arguments: GenerateCoverLetterInput
) -> dict[str, Any]:
    if context.state.profile is None:
        return {
            "status": "resume_required",
            "message": "Upload a resume before generating a cover letter.",
        }
    job, clarification = resolve_job_reference(arguments.job_id, context)
    if clarification:
        return {"status": "clarification_required", "message": clarification}
    if job is None:
        return {"status": "not_found", "message": "The selected job was not found."}
    if context.model is None:
        raise RuntimeError("Cover letter generation requires a configured language model.")
    draft = await generate_cover_letter(context.state.profile, job, context.model)
    context.state.pending_cover_letter_job_id = job.id
    context.state.pending_cover_letter = draft
    return {
        "status": "draft_ready",
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "cover_letter": draft,
        "message": "This is an unapproved cover letter draft. Ask the user to review it.",
    }


async def approve_cover_letter_tool(
    context: ToolContext, arguments: ApproveCoverLetterInput
) -> dict[str, Any]:
    if (
        context.state.pending_cover_letter_job_id != arguments.job_id
        or not context.state.pending_cover_letter
    ):
        return {
            "status": "approval_required",
            "message": "Generate and review a cover letter before approving it.",
        }
    job = next((item for item in context.jobs if item.id == arguments.job_id), None)
    if job is None:
        return {
            "status": "not_found",
            "message": "The selected job was not found.",
        }
    row = await context.db_session.get(ResumeProfileRow, context.session_id)
    if row is None:
        raise RuntimeError("The candidate profile must be persisted before approving a letter.")
    approved = dict(row.approved_cover_letters_data or {})
    approved[job.id] = context.state.pending_cover_letter
    row.approved_cover_letters_data = approved
    await context.db_session.commit()
    context.state.approved_cover_letters[job.id] = context.state.pending_cover_letter
    context.state.pending_cover_letter_job_id = None
    context.state.pending_cover_letter = None
    return {
        "status": "approved",
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "message": (
            f"The cover letter was approved for {job.title} at {job.company} and may be "
            "included in a reviewed application. It has not been submitted."
        ),
    }


GENERATE_COVER_LETTER_DEFINITION = ToolDefinition(
    name="generate_cover_letter",
    description=(
        "Generate a draft cover letter for a selected job from the candidate's parsed profile. "
        "Present the draft to the user for review. Never treat a generated draft as approved."
    ),
    input_model=GenerateCoverLetterInput,
    execute=generate_cover_letter_tool,
)

APPROVE_COVER_LETTER_DEFINITION = ToolDefinition(
    name="approve_cover_letter",
    description=(
        "Record explicit user approval of the currently displayed cover letter draft for its job. "
        "Call only after the user clearly says to approve or use that specific draft."
    ),
    input_model=ApproveCoverLetterInput,
    execute=approve_cover_letter_tool,
)
