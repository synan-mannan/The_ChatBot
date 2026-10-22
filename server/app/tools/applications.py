from __future__ import annotations

from datetime import datetime, timedelta, timezone
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


def _application_snapshot(
    row: ApplicationRow,
) -> tuple[str, str, str, str | None, dict[str, Any] | None]:
    return row.status, row.method, row.detail, row.apply_url, row.report_data


def _review_is_recent(report_data: dict[str, Any] | None) -> bool:
    if not report_data or not report_data.get("reviewed_at"):
        return False
    try:
        reviewed_at = datetime.fromisoformat(report_data["reviewed_at"])
    except (TypeError, ValueError):
        return False
    if reviewed_at.tzinfo is None:
        reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - reviewed_at
    return timedelta(0) <= age <= timedelta(minutes=15)


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
    *,
    persist: bool = True,
    answers_snapshot: dict[str, str | bool | float] | None = None,
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
    elif status == "retry_later":
        detail = report.blocker_reason or "Please retry the application shortly."
    else:
        detail = report.blocker_reason or "The form could not be completed; please verify manually."
    existing_report = application.report_data or {}
    apply_url = report.final_url or application.apply_url or str(job.source_link)
    if persist:
        report_data = report.model_dump(mode="json")
        report_data["answers_snapshot"] = (
            answers_snapshot if answers_snapshot is not None
            else context.state.answers
        ).copy()
        if status == "review_ready":
            report_data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        application.report_data = {
            **report_data,
        }
        application.status = status
        application.method = "autofill"
        application.detail = detail
        application.apply_url = apply_url
    return {
        "status": status,
        "method": "autofill",
        "detail": detail,
        "message": detail,
        "apply_url": apply_url,
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "applied_date": _applied_date(application).isoformat(),
        "screenshot_url": (
            f"/api/applications/{job.id}/preview?session_id={context.session_id}"
            if report.screenshot_path or existing_report.get("screenshot_path")
            else None
        ),
        "detected_ats": report.detected_ats or existing_report.get("detected_ats"),
        "discovered_form_labels": (
            report.discovered_form_labels
            or existing_report.get("discovered_form_labels", [])
        ),
        "debug_screenshot_url": (
            f"/api/applications/{job.id}/debug-preview?session_id={context.session_id}"
            if report.debug_screenshot_path
            or existing_report.get("debug_screenshot_path")
            else None
        ),
        "filled": (
            [item.model_dump() for item in report.filled]
            or existing_report.get("filled", [])
        ),
        "needs_input": (
            [item.model_dump() for item in report.needs_input]
            or existing_report.get("needs_input", [])
        ),
        "blocker_reason": report.blocker_reason,
    }


async def _run_generic_autofill(
    context: ToolContext,
    job: Job,
    application: ApplicationRow,
    mode: Literal["review", "dry_run", "submit"],
    *,
    answers: dict[str, str | bool | float] | None = None,
    previous_state: tuple[str, str, str, str | None, dict[str, Any] | None] | None = None,
    application_is_new: bool = False,
    review_submit_exemption: bool = False,
) -> dict[str, Any]:
    from server.app.services.autofill import fill_application_form

    profile = context.state.profile
    if profile is None:
        raise ValueError("A parsed resume profile is required for application autofill.")
    answers_used = answers if answers is not None else context.state.answers
    report = await fill_application_form(
        str(job.source_link),
        profile,
        answers_used,
        context.state.resume_path,
        mode,
        job,
        session_id=context.session_id,
        uploads_dir=context.settings.uploads_dir,
        model=context.model,
        real_apply_enabled=context.settings.real_apply_enabled,
        approved_cover_letter=context.state.approved_cover_letters.get(job.id),
        review_submit_exemption=review_submit_exemption,
    )
    if report.status == "retry_later":
        output = _autofill_output(
            context,
            job,
            application,
            report,
            persist=False,
            answers_snapshot=answers_used,
        )
        if application_is_new:
            await context.db_session.delete(application)
        elif previous_state is not None:
            (
                application.status,
                application.method,
                application.detail,
                application.apply_url,
                application.report_data,
            ) = previous_state
        await context.db_session.commit()
        return output

    output = _autofill_output(
        context,
        job,
        application,
        report,
        answers_snapshot=answers_used,
    )
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
    requested_job_id = arguments.job_id
    if arguments.submit and requested_job_id == "pending":
        statuses = (
            ("review_ready",)
            if context.settings.real_apply_enabled
            else ("review_ready", "dry_run")
        )
        reviewed_rows = list(
            await context.db_session.scalars(
                select(ApplicationRow).where(
                    ApplicationRow.session_id == context.session_id,
                    ApplicationRow.status.in_(statuses),
                    ApplicationRow.method == "autofill",
                )
            )
        )
        if len(reviewed_rows) != 1:
            if not reviewed_rows:
                return {
                    "status": "needs_manual",
                    "message": "There is no reviewed application ready to submit.",
                }
            reviewed_jobs = [
                get_job(row.job_id, context.jobs)
                for row in reviewed_rows
            ]
            choices = [
                f"{job.title} at {job.company} (id:{job.id})"
                for job in reviewed_jobs
                if job is not None
            ]
            return {
                "status": "clarification_required",
                "message": (
                    "Several reviewed applications are ready. Which one should I submit? "
                    + "; ".join(choices)
                ),
            }
        requested_job_id = f"id:{reviewed_rows[0].job_id}"

    job, clarification = resolve_job_reference(requested_job_id, context)
    if clarification:
        return {"status": "clarification_required", "message": clarification}
    if job is None:
        return {"status": "not_found", "message": f"Job '{arguments.job_id}' was not found."}

    existing = await _find_application(context, job.id)
    method = pick_method(job)
    previous_state = None

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
        previous_state = _application_snapshot(existing)
        claimed = await _claim_retryable_application(
            context, existing, job, allowed_statuses=("review_ready",)
        )
        if not claimed:
            await context.db_session.refresh(existing)
            return _duplicate_result(existing, job)
        final_url = previous_state[3] or str(job.source_link)
        submit_job = job.model_copy(
            update={"source_link": final_url}
        )
        saved_answers = (existing.report_data or {}).get("answers_snapshot")
        output = await _run_generic_autofill(
            context,
            submit_job,
            existing,
            "submit",
            answers=saved_answers if isinstance(saved_answers, dict) else None,
            previous_state=previous_state,
            review_submit_exemption=_review_is_recent(existing.report_data),
        )
        if existing.status == "submitted":
            context.state.pending_submission_job_id = None
        return output

    auto_rerun = (
        existing is None
        and context.state.pending_application_job_id == job.id
    ) or (
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
        application_is_new = True
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
            previous_state = _application_snapshot(existing)
            application_is_new = False
    else:
        previous_state = _application_snapshot(existing)
        application_is_new = False
        claimed = await _claim_retryable_application(context, existing, job)
        if not claimed:
            await context.db_session.refresh(existing)
            context.state.pending_application_job_id = None
            return _duplicate_result(existing, job)
        application = existing

    if method == "autofill":
        mode = "review" if context.settings.real_apply_enabled else "dry_run"
        output = await _run_generic_autofill(
            context,
            job,
            application,
            mode,
            previous_state=previous_state,
            application_is_new=application_is_new,
        )
        if output["status"] != "retry_later":
            context.state.pending_application_job_id = None
        return output

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
                "filled": (row.report_data or {}).get("filled", []),
                "needs_input": (row.report_data or {}).get("needs_input", []),
                "screenshot_url": (
                    f"/api/applications/{job.id}/preview?session_id={context.session_id}"
                    if (row.report_data or {}).get("screenshot_path")
                    else None
                ),
                "detected_ats": (row.report_data or {}).get("detected_ats"),
                "debug_screenshot_url": (
                    f"/api/applications/{job.id}/debug-preview?session_id={context.session_id}"
                    if (row.report_data or {}).get("debug_screenshot_path")
                    else None
                ),
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
