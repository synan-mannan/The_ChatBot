from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.models.schemas import Job, JobRecommendation
from server.app.services.sessions import SessionState, SessionStore


@dataclass
class ToolContext:
    session_id: str
    state: SessionState
    session_store: SessionStore
    db_session: AsyncSession
    jobs: list[Job]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    execute: Callable[[ToolContext, BaseModel], Awaitable[dict[str, Any]]]

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        definitions = schema.pop("$defs", {})

        def expand(value: Any) -> Any:
            if isinstance(value, dict):
                reference = value.get("$ref")
                if reference:
                    key = reference.rsplit("/", 1)[-1]
                    return expand(definitions[key])
                return {
                    name: expand(item)
                    for name, item in value.items()
                    if name not in {"$defs", "additionalProperties"}
                }
            if isinstance(value, list):
                return [expand(item) for item in value]
            return value

        schema = expand(schema)
        schema["description"] = self.description
        return schema


def get_job(job_id: str, jobs: list[Job]) -> Job | None:
    return next((job for job in jobs if job.id == job_id), None)


def resolve_job_reference(
    reference: str, context: ToolContext
) -> tuple[Job | None, str | None]:
    reference = reference.strip()
    lowered_reference = reference.casefold()
    if lowered_reference.startswith("displayed:"):
        position_text = reference.split(":", 1)[1]
        if position_text.isdecimal():
            position = int(position_text)
            if 1 <= position <= len(context.state.last_shown_jobs):
                return get_job(context.state.last_shown_jobs[position - 1].job.id, context.jobs), None
        return None, f"There is no displayed result {position_text}."
    if lowered_reference.startswith("id:"):
        reference = reference.split(":", 1)[1]
    exact = get_job(reference, context.jobs)
    positional: Job | None = None
    if reference.isdecimal():
        position = int(reference)
        if 1 <= position <= len(context.state.last_shown_jobs):
            positional_id = context.state.last_shown_jobs[position - 1].job.id
            positional = get_job(positional_id, context.jobs)
    if exact and positional and exact.id != positional.id:
        return None, (
            f"'{reference}' could mean displayed result {reference} ({positional.title} at "
            f"{positional.company}) or job ID {exact.id} ({exact.title} at {exact.company}). "
            f"Reply 'displayed:{reference}' or 'id:{exact.id}'."
        )
    if exact:
        return exact, None
    if positional:
        return positional, None
    maximum = len(context.jobs)
    return None, f"I couldn't find job '{reference}'. Valid job IDs are 1-{maximum}."


def serialize_recommendations(
    recommendations: list[JobRecommendation],
) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in recommendations]
