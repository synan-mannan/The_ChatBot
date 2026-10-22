from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Path as PathParam, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import StringConstraints
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from server.app.models.db import ApplicationRow, ResumeProfileRow
from server.app.models.schemas import (
    ApplicationActionRequest,
    ApplicationActionResponse,
    ApplicationsResponse,
    ApplicationView,
    Job,
    ResumeProfile,
)
from server.app.routes.dependencies import get_database_session
from server.app.services.sessions import SessionStore, restore_resume_path
from server.app.services.autofill import clear_session_rate_limits
from server.app.tools.applications import ApplyToJobInput, apply_to_job_tool
from server.app.tools.base import ToolContext

router = APIRouter(prefix="/api", tags=["applications"])
SessionQuery = Annotated[str, StringConstraints(min_length=8, max_length=128)]
JobPath = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]


@router.delete("/session", status_code=204)
async def clear_session(
    request: Request,
    session_id: SessionQuery = Query(...),
    session: AsyncSession = Depends(get_database_session),
) -> Response:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id):
        raise HTTPException(status_code=400, detail="The session ID contains unsupported characters.")
    store: SessionStore = request.app.state.session_store
    store.clear(session_id, request.app.state.settings.uploads_dir)
    clear_session_rate_limits(session_id)
    await session.execute(
        delete(ApplicationRow).where(ApplicationRow.session_id == session_id)
    )
    profile_row = await session.get(ResumeProfileRow, session_id)
    if profile_row is not None:
        await session.delete(profile_row)
    await session.commit()
    return Response(status_code=204)


@router.post("/applications/apply", response_model=ApplicationActionResponse)
async def apply_to_job(
    body: ApplicationActionRequest,
    request: Request,
    session: AsyncSession = Depends(get_database_session),
) -> ApplicationActionResponse:
    store: SessionStore = request.app.state.session_store
    state = store.get(body.session_id)
    if state.profile is None:
        profile_row = await session.get(ResumeProfileRow, body.session_id)
        if profile_row is not None:
            state.answers = profile_row.answers_data or {}
            state.approved_cover_letters = (
                profile_row.approved_cover_letters_data or {}
            )
            if profile_row.profile_data:
                state.profile = ResumeProfile.model_validate(profile_row.profile_data)
    if state.profile is not None:
        await restore_resume_path(
            state, body.session_id, request.app.state.settings.uploads_dir
        )

    context = ToolContext(
        session_id=body.session_id,
        state=state,
        session_store=store,
        db_session=session,
        jobs=request.app.state.jobs,
        model=request.app.state.model,
        settings=request.app.state.settings,
    )
    result = await apply_to_job_tool(
        context,
        ApplyToJobInput(
            job_id=body.job_id,
            confirm=body.confirm,
            submit=body.submit,
        ),
    )
    return ApplicationActionResponse.model_validate(result)


@router.get("/applications/{job_id}/preview", include_in_schema=True)
async def get_application_preview(
    request: Request,
    job_id: JobPath = PathParam(...),
    session_id: SessionQuery = Query(...),
    session: AsyncSession = Depends(get_database_session),
) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id):
        raise HTTPException(status_code=404, detail="Application preview not found.")
    uploads_root = Path(request.app.state.settings.uploads_dir).resolve()
    session_directory = (uploads_root / session_id).resolve()
    preview_path = (session_directory / "previews" / f"{job_id}.png").resolve()
    try:
        session_directory.relative_to(uploads_root)
        preview_path.relative_to(session_directory)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Application preview not found.") from exc
    row = await session.scalar(
        select(ApplicationRow).where(
            ApplicationRow.session_id == session_id,
            ApplicationRow.job_id == job_id,
        )
    )
    report_path = (
        Path(row.report_data["screenshot_path"]).resolve()
        if row is not None
        and row.report_data is not None
        and row.report_data.get("screenshot_path")
        else None
    )
    if (
        row is None
        or report_path != preview_path
    ):
        raise HTTPException(status_code=404, detail="Application preview not found.")
    if not preview_path.is_file():
        raise HTTPException(status_code=404, detail="Application preview not found.")
    return FileResponse(preview_path, media_type="image/png")


@router.get("/applications", response_model=ApplicationsResponse)
async def list_applications(
    request: Request,
    session_id: SessionQuery = Query(...),
    session: AsyncSession = Depends(get_database_session),
) -> ApplicationsResponse:
    store: SessionStore = request.app.state.session_store
    state = store.get(session_id)
    if state.profile is None:
        profile_row = await session.get(ResumeProfileRow, session_id)
        if profile_row is None or not profile_row.profile_data:
            return ApplicationsResponse(
                message="Please upload a resume before viewing applications."
            )
        state.profile = ResumeProfile.model_validate(profile_row.profile_data)
        state.answers = profile_row.answers_data or {}
        state.approved_cover_letters = profile_row.approved_cover_letters_data or {}
    await restore_resume_path(
        state, session_id, request.app.state.settings.uploads_dir
    )

    job_by_id: dict[str, Job] = {job.id: job for job in request.app.state.jobs}
    rows = await session.scalars(
        select(ApplicationRow)
        .where(ApplicationRow.session_id == session_id)
        .order_by(ApplicationRow.applied_date.desc(), ApplicationRow.id.desc())
    )
    applications = [
        ApplicationView(
            job_id=row.job_id,
            title=job_by_id[row.job_id].title,
            company=job_by_id[row.job_id].company,
            applied_date=row.applied_date,
            status=row.status,
            method=row.method,
            detail=row.detail,
            apply_url=row.apply_url,
        )
        for row in rows
        if row.job_id in job_by_id
    ]
    return ApplicationsResponse(applications=applications)
