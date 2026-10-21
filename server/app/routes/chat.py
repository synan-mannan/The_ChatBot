from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends, Request
from google.genai import errors
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.agents.orchestrator import Orchestrator
from server.app.models.db import ResumeProfileRow
from server.app.models.schemas import ChatRequest, ChatResponse, ResumeProfile
from server.app.services.llm import LLMUnavailableError
from server.app.services.sessions import SessionStore
from server.app.routes.dependencies import get_database_session

router = APIRouter(prefix="/api", tags=["chat"])


async def _restore_profile(
    session_id: str, session: AsyncSession, store: SessionStore
) -> None:
    state = store.get(session_id)
    if state.profile is not None:
        return
    row = await session.scalar(
        select(ResumeProfileRow).where(ResumeProfileRow.session_id == session_id)
    )
    if row is not None:
        state.profile = ResumeProfile.model_validate(row.profile_data)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_database_session),
) -> ChatResponse:
    store: SessionStore = request.app.state.session_store
    await _restore_profile(body.session_id, session, store)
    orchestrator: Orchestrator = request.app.state.orchestrator
    try:
        return await orchestrator.chat(body.session_id, body.message, session)
    except LLMUnavailableError as exc:
        return ChatResponse(reply=str(exc), profile=store.get(body.session_id).profile)
    except (errors.APIError, asyncio.TimeoutError, httpx.TimeoutException, httpx.TransportError):
        return ChatResponse(
            reply="The AI service is temporarily unavailable after retries. Please try again shortly.",
            profile=store.get(body.session_id).profile,
        )
    except ValidationError:
        return ChatResponse(
            reply="I couldn't process that request safely. Please try rephrasing it.",
            profile=store.get(body.session_id).profile,
        )
