from __future__ import annotations

from datetime import timezone
from typing import Any, Literal

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
    submit: bool = False


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


async def _claim_retryable_application(
    context: ToolContext,
    row: ApplicationRow,
    job: Job,
    allowed_statuses: tuple[str, ...] = ("failed", "needs_input", "review_ready"),
) -> bool:
    result = await context.db_session.execute(
        update(ApplicationRow)
        .where(
            ApplicationRow.id == row.id,
            ApplicationRow.status.in_(allowed_statuses),
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


def _autofill_output(
    context: ToolContext,
    job: Job,
    application: ApplicationRow,
    report: Any,
) -> dict[str, Any]:
    status = report.status
    if status == "review_ready":
        detail = "The application form is filled and ready for review. It has NOT been submitted."
    elif status == "dry_run":
        detail = (
            "Dry run only: the application form was prepared, but nothing was submitted "
            "because REAL_APPLY_ENABLED=false."
        )
    elif status == "needs_input":
        detail = "The form is not ready to submit; please answer the listed questions."
    elif status == "needs_manual":
        detail = report.blocker_reason or "Continue this application manually."
    elif status == "submitted":
        detail = "The portal confirmed receipt of the application."
    else:
        detail = report.blocker_reason or "The form could not be completed; please verify manually."
    application.report_data = report.model_dump(mode="json")
    application.status = status
    application.method = "autofill"
    application.detail = detail
    application.apply_url = report.final_url or str(job.source_link)
    return {
        "status": status,
        "method": "autofill",
        "detail": detail,
        "message": detail,
        "apply_url": application.apply_url,
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "applied_date": _applied_date(application).isoformat(),
        "screenshot_url": (
            f"/api/applications/{job.id}/preview?session_id={context.session_id}"
            if report.screenshot_path
            else None
        ),
        "filled": [item.model_dump() for item in report.filled],
        "needs_input": [item.model_dump() for item in report.needs_input],
        "blocker_reason": report.blocker_reason,
    }


async def _run_generic_autofill(
    context: ToolContext,
    job: Job,
    application: ApplicationRow,
    mode: Literal["review", "dry_run", "submit"],
) -> dict[str, Any]:
    from server.app.services.autofill import fill_application_form

    profile = context.state.profile
    if profile is None:
        raise ValueError("A parsed resume profile is required for application autofill.")
    report = await fill_application_form(
        str(job.source_link),
        profile,
        context.state.answers,
        context.state.resume_path,
        mode,
        job,
        session_id=context.session_id,
        uploads_dir=context.settings.uploads_dir,
        model=context.model,
        real_apply_enabled=context.settings.real_apply_enabled,
        approved_cover_letter=context.state.approved_cover_letters.get(job.id),
    )
    output = _autofill_output(context, job, application, report)
    await context.db_session.commit()
    if report.status == "needs_input":
        context.state.pending_autofill_job_id = job.id
        context.state.pending_submission_job_id = None
    elif report.status == "review_ready":
        context.state.pending_autofill_job_id = None
        context.state.pending_submission_job_id = job.id
    elif report.status == "dry_run":
        context.state.pending_autofill_job_id = None
        context.state.pending_submission_job_id = job.id
    elif (
        report.status == "failed"
        and report.blocker_reason
        and "rate-limited" in report.blocker_reason.casefold()
    ):
        context.state.pending_autofill_job_id = job.id
        context.state.pending_submission_job_id = None
    else:
        context.state.pending_autofill_job_id = None
        context.state.pending_submission_job_id = None
    return output


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
    method = pick_method(job)

    if arguments.submit:
        if method != "autofill":
            return {
                "status": "needs_manual",
                "message": "This application cannot be submitted through the generic review flow.",
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "method": method,
            }
        if (
            existing is not None
            and existing.status == "dry_run"
            and context.state.pending_submission_job_id == job.id
            and not context.settings.real_apply_enabled
        ):
            return {
                **_duplicate_result(existing, job),
                "status": "dry_run",
                "message": (
                    "Dry run only: REAL_APPLY_ENABLED=false, so the form will not be submitted."
                ),
                "detail": (
                    "Dry run only: REAL_APPLY_ENABLED=false, so the form will not be submitted."
                ),
            }
        if (
            existing is None
            or existing.status != "review_ready"
            or context.state.pending_submission_job_id != job.id
        ):
            return {
                "status": "needs_manual",
                "message": "Review the application form before requesting submission.",
                "job_id": job.id,
                "title": job.title,
                "company": job.company,
                "method": "autofill",
            }
        if not context.settings.real_apply_enabled:
            existing.status = "dry_run"
            existing.detail = (
                "Dry run only: REAL_APPLY_ENABLED=false, so the form was not submitted."
            )
            await context.db_session.commit()
            return {
                **_duplicate_result(existing, job),
                "status": "dry_run",
                "message": existing.detail,
                "detail": existing.detail,
            }
        claimed = await _claim_retryable_application(
            context, existing, job, allowed_statuses=("review_ready",)
        )
        if not claimed:
            await context.db_session.refresh(existing)
            return _duplicate_result(existing, job)
        output = await _run_generic_autofill(context, job, existing, "submit")
        if existing.status == "submitted":
            context.state.pending_submission_job_id = None
        return output

    auto_rerun = (
        existing is not None
        and existing.status in {"needs_input", "failed"}
        and context.state.pending_autofill_job_id == job.id
    )
    retryable_statuses = {"failed", "needs_input"}
    if existing is not None and existing.status not in retryable_statuses and not auto_rerun:
        return _duplicate_result(existing, job)

    if not auto_rerun and (
        not arguments.confirm or context.state.pending_application_job_id != job.id
    ):
        context.state.pending_application_job_id = job.id
        return {
            "status": "confirmation_required",
            "job_id": job.id,
            "title": job.title,
            "company": job.company,
            "plan": describe_plan(job),
            "message": (
                f"I can {describe_plan(job)} for {job.title} at {job.company}. "
                + (
                    "Nothing will be submitted yet. Reply “yes, prepare for review” to continue."
                    if method == "autofill"
                    else "Nothing has been sent yet. Reply “yes, submit” to continue."
                )
            ),
        }

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
            claimed = await _claim_retryable_application(context, existing, job)
            if not claimed:
                await context.db_session.refresh(existing)
                context.state.pending_application_job_id = None
                return _duplicate_result(existing, job)
            application = existing
    else:
        claimed = await _claim_retryable_application(context, existing, job)
        if not claimed:
            await context.db_session.refresh(existing)
            context.state.pending_application_job_id = None
            return _duplicate_result(existing, job)
        application = existing

    if method == "autofill":
        mode = "review" if context.settings.real_apply_enabled else "dry_run"
        context.state.pending_application_job_id = None
        return await _run_generic_autofill(context, job, application, mode)

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
        "in a later message. For generic career portals, confirm prepares a review; use submit=true "
        "only after a separate explicit user instruction to submit a reviewed form. Never submit "
        "without that confirmation. Bare numbers refer to the latest displayed result; use id:N "
        "only for an explicit dataset ID. One application is processed at a time."
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
