from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.app.agents.orchestrator import Orchestrator
from server.app.config import Settings, get_settings
from server.app.models.db import create_database, initialize_database
from server.app.routes.applications import router as applications_router
from server.app.routes.chat import router as chat_router
from server.app.routes.health import router as health_router
from server.app.routes.upload import router as upload_router
from server.app.services.job_catalog import load_jobs
from server.app.services.llm import GeminiService
from server.app.services.sessions import SessionStore

logger = logging.getLogger("aijob")


def create_app(settings: Settings | None = None, model: object | None = None) -> FastAPI:
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        jobs = load_jobs(active_settings.jobs_file)
        engine, session_factory = create_database(active_settings.database_url)
        await initialize_database(engine)
        session_store = SessionStore()
        llm = model or GeminiService(active_settings)
        app.state.settings = active_settings
        app.state.jobs = jobs
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.session_store = session_store
        app.state.model = llm
        app.state.orchestrator = Orchestrator(llm, session_store, jobs)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(
        title=active_settings.app_name,
        description="Conversational job assistant backed by real listing data.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(chat_router)
    app.include_router(upload_router)
    app.include_router(applications_router)
    app.include_router(health_router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request error for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Something went wrong. Please try again."},
        )

    client_dist = Path(active_settings.client_dist)
    if client_dist.is_dir() and (client_dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=client_dist, html=True), name="client")
    else:
        @app.get("/", include_in_schema=False)
        async def service_root() -> dict[str, str]:
            return {"name": active_settings.app_name, "docs": "/docs"}

    return app


app = create_app()
