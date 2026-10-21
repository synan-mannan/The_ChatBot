from __future__ import annotations

import asyncio

import httpx

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from google.genai import errors
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.models.db import ResumeProfileRow
from server.app.models.schemas import ResumeProfile, UploadResponse
from server.app.routes.dependencies import get_database_session
from server.app.services.llm import (
    LLMUnavailableError,
    ProfileExtractionError,
)
from server.app.services.resume_parser import ResumeValidationError, extract_resume_text
from server.app.services.sessions import SessionStore

router = APIRouter(prefix="/api", tags=["resume"])


@router.post("/upload", response_model=UploadResponse)
async def upload_resume(
    request: Request,
    session_id: str = Form(min_length=8, max_length=128),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_database_session),
) -> UploadResponse:
    settings = request.app.state.settings
    content = await file.read(settings.max_resume_bytes + 1)
    try:
        resume_text = extract_resume_text(file.filename or "", content, settings.max_resume_bytes)
    except ResumeValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = request.app.state.model
    try:
        profile = await model.extract_profile(resume_text)
    except ProfileExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (
        errors.APIError,
        asyncio.TimeoutError,
        httpx.TimeoutException,
        httpx.TransportError,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail="The AI service is temporarily unavailable after retries. Please try again shortly.",
        ) from exc

    row = await session.get(ResumeProfileRow, session_id)
    if row is None:
        row = ResumeProfileRow(session_id=session_id, profile_data=profile.model_dump(mode="json"))
        session.add(row)
    else:
        row.profile_data = profile.model_dump(mode="json")
    await session.commit()
    store: SessionStore = request.app.state.session_store
    store.set_profile(session_id, profile)
    return UploadResponse(profile=profile)
