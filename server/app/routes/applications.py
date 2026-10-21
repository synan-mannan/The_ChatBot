from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import StringConstraints
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated

from server.app.models.db import ApplicationRow, ResumeProfileRow
from server.app.models.schemas import (
    ApplicationsResponse,
    ApplicationView,
    Job,
    ResumeProfile,
)
from server.app.routes.dependencies import get_database_session
from server.app.services.sessions import SessionStore

router = APIRouter(prefix="/api", tags=["applications"])
SessionQuery = Annotated[str, StringConstraints(min_length=8, max_length=128)]


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
        if profile_row is None:
            return ApplicationsResponse(
                message="Please upload a resume before viewing applications."
            )
        state.profile = ResumeProfile.model_validate(profile_row.profile_data)

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
        )
        for row in rows
        if row.job_id in job_by_id
    ]
    return ApplicationsResponse(applications=applications)
