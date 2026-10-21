from __future__ import annotations

from datetime import timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from server.app.models.db import ApplicationRow
from server.app.tools.base import (
    ToolContext,
    ToolDefinition,
    get_job,
    resolve_job_reference,
)


class ApplyToJobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)


class ListApplicationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


async def apply_to_job_tool(
    context: ToolContext, arguments: ApplyToJobInput
) -> dict[str, Any]:
    if context.state.profile is None:
        return {
            "status": "resume_required",
            "message": "Ask the user to upload a resume before applying.",
        }
    job, clarification = resolve_job_reference(arguments.job_id, context)
    if clarification:
        return {"status": "clarification_required", "message": clarification}
    if job is None:
        return {"status": "not_found", "message": f"Job '{arguments.job_id}' was not found."}

    application = ApplicationRow(session_id=context.session_id, job_id=job.id)
    context.db_session.add(application)
    try:
        await context.db_session.commit()
    except IntegrityError:
        await context.db_session.rollback()
        existing = await context.db_session.scalar(
            select(ApplicationRow).where(
                ApplicationRow.session_id == context.session_id,
                ApplicationRow.job_id == job.id,
            )
        )
        if existing is None:
            raise
        applied_date = existing.applied_date
        if applied_date.tzinfo is None:
            applied_date = applied_date.replace(tzinfo=timezone.utc)
        return {
            "status": "duplicate",
            "job_id": job.id,
            "title": job.title,
            "company": job.company,
            "applied_date": applied_date.isoformat(),
            "message": (
                f"You already applied to {job.title} at {job.company} on "
                f"{applied_date:%Y-%m-%d %H:%M UTC}."
            ),
        }

    return {
        "status": "applied",
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "applied_date": application.applied_date.isoformat(),
        "message": f"Application recorded for {job.title} at {job.company}.",
    }


async def list_applications_tool(
    context: ToolContext, _: ListApplicationsInput
) -> dict[str, Any]:
    if context.state.profile is None:
        return {
            "status": "resume_required",
            "message": "Ask the user to upload a resume before showing applications.",
        }
    rows = await context.db_session.scalars(
        select(ApplicationRow)
        .where(ApplicationRow.session_id == context.session_id)
        .order_by(ApplicationRow.applied_date.desc(), ApplicationRow.id.desc())
    )
    applications: list[dict[str, Any]] = []
    for row in rows:
        job = get_job(row.job_id, context.jobs)
        if job is None:
            continue
        applied_date = row.applied_date
        if applied_date.tzinfo is None:
            applied_date = applied_date.replace(tzinfo=timezone.utc)
        applications.append(
            {
                "job_id": row.job_id,
                "title": job.title,
                "company": job.company,
                "applied_date": applied_date.isoformat(),
                "status": row.status,
            }
        )
    return {"status": "ok", "applications": applications}


APPLY_DEFINITION = ToolDefinition(
    name="apply_to_job",
    description=(
        "Record an application to a valid job from this dataset. Use a displayed result number "
        "when the user refers to a position such as 'job 3'. Bare numbers refer to the latest "
        "displayed result; use id:N only for an explicitly provided dataset ID. "
        "Duplicate applications are prevented."
    ),
    input_model=ApplyToJobInput,
    execute=apply_to_job_tool,
)
LIST_DEFINITION = ToolDefinition(
    name="list_applications",
    description="List applications already recorded for this anonymous chat session.",
    input_model=ListApplicationsInput,
    execute=list_applications_tool,
)
