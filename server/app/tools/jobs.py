from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.app.models.schemas import JobFilters
from server.app.services.matcher import recommend_jobs
from server.app.tools.base import (
    ToolContext,
    ToolDefinition,
    get_job,
    resolve_job_reference,
    serialize_recommendations,
)


class RecommendJobsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: JobFilters = Field(default_factory=JobFilters)
    top_k: int = Field(default=5, ge=1, le=20)


class GetJobDetailsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)


def _merge_filters(current: JobFilters, incoming: JobFilters) -> JobFilters:
    changes = {
        field: getattr(incoming, field) for field in incoming.model_fields_set
    }
    return current.model_copy(update=changes)


async def recommend_jobs_tool(
    context: ToolContext, arguments: RecommendJobsInput
) -> dict[str, Any]:
    filters = _merge_filters(context.state.filters, arguments.filters)
    context.session_store.set_filters(context.session_id, filters)
    if context.state.profile is None:
        return {
            "status": "resume_required",
            "message": "Ask the user to upload a resume before recommending jobs.",
        }
    results = recommend_jobs(
        context.jobs,
        context.state.profile,
        filters,
        top_k=arguments.top_k,
        profile_location=filters.location,
    )
    context.session_store.remember_jobs(context.session_id, results)
    return {
        "status": "ok" if results else "no_matches",
        "filters": filters.model_dump(mode="json"),
        "jobs": serialize_recommendations(results),
        "message": (
            f"Found {len(results)} matching jobs."
            if results
            else "No jobs match all active filters. Suggest relaxing one or more filters."
        ),
    }


async def get_job_details_tool(
    context: ToolContext, arguments: GetJobDetailsInput
) -> dict[str, Any]:
    job, clarification = resolve_job_reference(arguments.job_id, context)
    if clarification:
        return {"status": "clarification_required", "message": clarification}
    if job is None:
        return {"status": "not_found", "message": f"Job '{arguments.job_id}' was not found."}
    return {"status": "ok", "job": job.model_dump(mode="json")}


RECOMMEND_DEFINITION = ToolDefinition(
    name="recommend_jobs",
    description=(
        "Find and rank real jobs from the loaded dataset for the uploaded resume. "
        "Preserve existing filters when refining; set a filter to null only when the user "
        "explicitly clears it. Only describe jobs returned by this tool."
    ),
    input_model=RecommendJobsInput,
    execute=recommend_jobs_tool,
)
DETAILS_DEFINITION = ToolDefinition(
    name="get_job_details",
    description="Return details for one real dataset job ID or a numbered displayed result.",
    input_model=GetJobDetailsInput,
    execute=get_job_details_tool,
)
