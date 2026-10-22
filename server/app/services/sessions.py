from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from server.app.models.schemas import JobFilters, JobRecommendation, ResumeProfile


@dataclass
class SessionState:
    profile: ResumeProfile | None = None
    answers: dict[str, str | bool | float] = field(default_factory=dict)
    approved_cover_letters: dict[str, str] = field(default_factory=dict)
    pending_cover_letter_job_id: str | None = None
    pending_cover_letter: str | None = None
    resume_path: str | None = None
    filters: JobFilters = field(default_factory=JobFilters)
    last_shown_jobs: list[JobRecommendation] = field(default_factory=list)
    pending_application_job_id: str | None = None
    pending_autofill_job_id: str | None = None
    pending_submission_job_id: str | None = None


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        return self._sessions.setdefault(session_id, SessionState())

    def set_profile(self, session_id: str, profile: ResumeProfile) -> SessionState:
        state = self.get(session_id)
        state.profile = profile
        return state

    def set_answers(
        self, session_id: str, answers: dict[str, str | bool | float]
    ) -> SessionState:
        state = self.get(session_id)
        state.answers.update(answers)
        return state

    def set_filters(self, session_id: str, filters: JobFilters) -> SessionState:
        state = self.get(session_id)
        state.filters = filters
        return state

    def remember_jobs(
        self, session_id: str, jobs: list[JobRecommendation]
    ) -> SessionState:
        state = self.get(session_id)
        state.last_shown_jobs = jobs
        return state

    def clear(self, session_id: str, uploads_dir: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id):
            raise ValueError("The session ID contains unsupported characters.")
        uploads_root = Path(uploads_dir).resolve()
        session_directory = (uploads_root / session_id).resolve()
        session_directory.relative_to(uploads_root)
        self._sessions.pop(session_id, None)
        if session_directory.is_dir():
            shutil.rmtree(session_directory)


async def restore_resume_path(
    state: SessionState,
    session_id: str,
    uploads_dir: str,
) -> str | None:
    if state.resume_path and await asyncio.to_thread(Path(state.resume_path).is_file):
        return state.resume_path
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", session_id):
        return None
    session_dir = Path(uploads_dir) / session_id
    for suffix in (".pdf", ".docx"):
        candidate = session_dir / f"resume{suffix}"
        if await asyncio.to_thread(candidate.is_file):
            state.resume_path = str(candidate)
            return state.resume_path
    return None
