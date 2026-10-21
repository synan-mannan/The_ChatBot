from __future__ import annotations

from dataclasses import dataclass, field

from server.app.models.schemas import JobFilters, JobRecommendation, ResumeProfile


@dataclass
class SessionState:
    profile: ResumeProfile | None = None
    resume_path: str | None = None
    filters: JobFilters = field(default_factory=JobFilters)
    last_shown_jobs: list[JobRecommendation] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        return self._sessions.setdefault(session_id, SessionState())

    def set_profile(self, session_id: str, profile: ResumeProfile) -> SessionState:
        state = self.get(session_id)
        state.profile = profile
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
