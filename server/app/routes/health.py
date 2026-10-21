from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.routes.dependencies import get_database_session

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health(
    request: Request, session: AsyncSession = Depends(get_database_session)
) -> dict[str, object]:
    await session.execute(text("SELECT 1"))
    return {
        "status": "ok",
        "jobs_loaded": len(request.app.state.jobs),
        "llm_configured": bool(
            request.app.state.settings.groq_api_key
            or request.app.state.settings.gemini_api_key
        ),
        "llm_provider": (
            "groq"
            if request.app.state.settings.groq_api_key
            else "gemini"
            if request.app.state.settings.gemini_api_key
            else None
        ),
    }
