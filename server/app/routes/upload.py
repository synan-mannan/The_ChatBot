from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

import httpx

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from google.genai import errors
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.models.db import ResumeProfileRow
from server.app.models.schemas import UploadResponse
from server.app.routes.dependencies import get_database_session
from server.app.services.llm import (
    LLMUnavailableError,
    ProfileExtractionError,
)
from server.app.services.resume_parser import ResumeValidationError, extract_resume_text
from server.app.services.sessions import SessionStore

router = APIRouter(prefix="/api", tags=["resume"])


def _persist_resume(upload_root: Path, session_id: str, suffix: str, content: bytes) -> Path:
    directory = upload_root / session_id
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"resume{suffix}"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".resume-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
        os.replace(temporary_name, destination)
    except OSError:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


@router.post("/upload", response_model=UploadResponse)
async def upload_resume(
    request: Request,
    session_id: str = Form(min_length=8, max_length=128),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_database_session),
) -> UploadResponse:
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id):
        raise HTTPException(status_code=400, detail="The session ID contains unsupported characters.")
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

    suffix = Path(file.filename or "").suffix.lower()
    try:
        resume_path = await asyncio.to_thread(
            _persist_resume,
            Path(settings.uploads_dir),
            session_id,
            suffix,
            content,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="The resume was parsed, but could not be saved for applications.",
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
    state = store.get(session_id)
    state.resume_path = str(resume_path)
    state.pending_application_job_id = None
    return UploadResponse(profile=profile)
