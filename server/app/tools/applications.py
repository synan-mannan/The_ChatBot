from __future__ import annotations

from datetime import timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from server.app.models.db import ApplicationRow
from server.app.models.schemas import Job
from server.app.services.applicant import apply, describe_plan, pick_method
from server.app.tools.base import (
    ToolContext,
    ToolDefinition,
    get_job,
    resolve_job_reference,
)


class ApplyToJobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)
    confirm: bool = False


class ListApplicationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _applied_date(row: ApplicationRow):
    applied_date = row.applied_date
    if applied_date.tzinfo is None:
        applied_date = applied_date.replace(tzinfo=timezone.utc)
    return applied_date


def _duplicate_result(row: ApplicationRow, job: Job) -> dict[str, Any]:
    return {
        "status": "duplicate",
        "application_status": row.status,
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "method": row.method,
        "detail": row.detail,
        "apply_url": row.apply_url,
        "applied_date": _applied_date(row).isoformat(),
        "message": (
            f"This application is already recorded with status '{row.status}' "
            f"for {job.title} at {job.company} (since "
            f"{_applied_date(row):%Y-%m-%d %H:%M UTC})."
        ),
    }


async def _find_application(
    context: ToolContext, job_id: str
) -> ApplicationRow | None:
    return await context.db_session.scalar(
        select(ApplicationRow).where(
            ApplicationRow.session_id == context.session_id,
            ApplicationRow.job_id == job_id,
        )
    )


async def _claim_failed_application(
    context: ToolContext,
    row: ApplicationRow,
    job: Job,
) -> bool:
    result = await context.db_session.execute(
        update(ApplicationRow)
        .where(
            ApplicationRow.id == row.id,
            ApplicationRow.status == "failed",
        )
        .values(
            status="in_progress",
            method=pick_method(job),
            detail="Preparing the application after explicit confirmation.",
            apply_url=str(job.source_link),
        )
    )
    await context.db_session.commit()
    return result.rowcount == 1


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

    existing = await _find_application(context, job.id)
    if existing is not None and existing.status != "failed":
        return _duplicate_result(existing, job)

    if not arguments.confirm or context.state.pending_application_job_id != job.id:
        context.state.pending_application_job_id = job.id
        return {
            "status": "confirmation_required",
            "job_id": job.id,
            "title": job.title,
            "company": job.company,
            "plan": describe_plan(job),
            "message": (
                f"I can {describe_plan(job)} for {job.title} at {job.company}. "
                "Nothing has been sent yet. Reply “yes, submit” to continue."
            ),
        }

    method = pick_method(job)
    if existing is None:
        application = ApplicationRow(
            session_id=context.session_id,
            job_id=job.id,
            status="in_progress",
            method=method,
            detail="Preparing the application after explicit confirmation.",
            apply_url=str(job.source_link),
        )
        context.db_session.add(application)
        try:
            await context.db_session.commit()
        except IntegrityError:
            await context.db_session.rollback()
            existing = await _find_application(context, job.id)
            if existing is None:
                raise
            if existing.status != "failed":
                context.state.pending_application_job_id = None
                return _duplicate_result(existing, job)
            claimed = await _claim_failed_application(context, existing, job)
            if not claimed:
                await context.db_session.refresh(existing)
                context.state.pending_application_job_id = None
                return _duplicate_result(existing, job)
            application = existing
    else:
        claimed = await _claim_failed_application(context, existing, job)
        if not claimed:
            await context.db_session.refresh(existing)
            context.state.pending_application_job_id = None
            return _duplicate_result(existing, job)
        application = existing

    result = await apply(job, context.state.profile, context.state.resume_path)
    application.status = result.status
    application.method = result.method
    application.detail = result.detail
    application.apply_url = result.apply_url
    await context.db_session.commit()
    context.state.pending_application_job_id = None
    return {
        "status": result.status,
        "method": result.method,
        "detail": result.detail,
        "message": result.detail,
        "apply_url": result.apply_url,
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "applied_date": _applied_date(application).isoformat(),
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
        applications.append(
            {
                "job_id": row.job_id,
                "title": job.title,
                "company": job.company,
                "applied_date": _applied_date(row).isoformat(),
                "status": row.status,
                "method": row.method,
                "detail": row.detail,
                "apply_url": row.apply_url,
            }
        )
    return {"status": "ok", "applications": applications}


APPLY_DEFINITION = ToolDefinition(
    name="apply_to_job",
    description=(
        "Prepare an application for a valid job in this dataset. First call with confirm=false "
        "to show the plan; only call again with confirm=true after the user explicitly agrees "
        "in a later message. Never set confirm=true on the first call or without clear user "
        "consent. Bare numbers refer to the latest displayed result; use id:N only for an "
        "explicit dataset ID. One application is processed at a time."
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
