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
        maximum = len(context.state.last_shown_jobs)
        return (
            None,
            f"There is no displayed result {position_text}. Valid displayed result numbers are 1-{maximum}.",
        )
    explicit_id = lowered_reference.startswith("id:")
    if explicit_id:
        reference = reference.split(":", 1)[1]
    exact = get_job(reference, context.jobs)
    if not explicit_id and reference.isdecimal():
        position = int(reference)
        if 1 <= position <= len(context.state.last_shown_jobs):
            positional_id = context.state.last_shown_jobs[position - 1].job.id
            return get_job(positional_id, context.jobs), None
    if exact:
        return exact, None
    maximum = len(context.jobs)
    return None, f"I couldn't find job '{reference}'. Valid job IDs are 1-{maximum}."


def serialize_recommendations(
    recommendations: list[JobRecommendation],
) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in recommendations]
