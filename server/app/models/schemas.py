from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints
from typing_extensions import Annotated


SessionId = Annotated[str, StringConstraints(min_length=8, max_length=128, strip_whitespace=True)]


class ResumeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    designation: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience_years: float | None = Field(default=None, ge=0, le=70)
    experience_summary: str | None = None
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


class JobFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remote_only: bool | None = None
    location: str | None = None
    required_skill: str | None = None
    experience_min: float | None = Field(default=None, ge=0)
    experience_max: float | None = Field(default=None, ge=0)
    company: str | None = None


class Job(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    company: str
    location: str
    contact_email: str | None = None
    is_remote: bool
    experience_required: str
    skills: list[str] = Field(default_factory=list)
    description: str = ""
    source_link: HttpUrl | str
    source: str
    posted_date: str = ""


class JobRecommendation(BaseModel):
    job: Job
    score: float = Field(ge=0, le=1)
    match_reasons: list[str]


class ApplicationView(BaseModel):
    job_id: str
    title: str
    company: str
    applied_date: datetime
    status: str
    method: str = "manual"
    detail: str = ""
    apply_url: str | None = None


class ApplicationConfirmation(BaseModel):
    job_id: str
    title: str
    company: str
    plan: str


class ApplicationActionResponse(BaseModel):
    status: Literal[
        "confirmation_required",
        "in_progress",
        "submitted",
        "dry_run",
        "needs_manual",
        "failed",
        "duplicate",
        "not_found",
        "resume_required",
        "clarification_required",
    ]
    message: str
    method: str | None = None
    detail: str | None = None
    plan: str | None = None
    job_id: str | None = None
    title: str | None = None
    company: str | None = None
    applied_date: datetime | None = None
    apply_url: str | None = None
    application_status: str | None = None


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    session_id: SessionId
    message: Annotated[str, StringConstraints(min_length=1, max_length=4_000, strip_whitespace=True)]


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    jobs: list[JobRecommendation] | None = None
    applications: list[ApplicationView] | None = None
    application_confirmation: ApplicationConfirmation | None = None
    application_result: ApplicationActionResponse | None = None
    profile: ResumeProfile | None = None


class UploadResponse(BaseModel):
    profile: ResumeProfile
    message: str = "Resume parsed successfully."


class ApplicationsResponse(BaseModel):
    applications: list[ApplicationView] = Field(default_factory=list)
    message: str | None = None


class ApplicationActionRequest(BaseModel):
    session_id: SessionId
    job_id: Annotated[str, StringConstraints(min_length=1, max_length=128, strip_whitespace=True)]
    confirm: bool = False
