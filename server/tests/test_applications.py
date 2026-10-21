import pytest
import pytest_asyncio

from server.app.models.db import create_database, initialize_database
from server.app.models.schemas import Job, JobRecommendation, ResumeProfile
from server.app.services.sessions import SessionStore
from server.app.tools.applications import (
    ApplyToJobInput,
    ListApplicationsInput,
    apply_to_job_tool,
    list_applications_tool,
)
from server.app.tools.base import ToolContext


@pytest_asyncio.fixture
async def database_session():
    engine, factory = create_database("sqlite+aiosqlite:///:memory:")
    await initialize_database(engine)
    async with factory() as session:
        yield session
    await engine.dispose()


def make_job(job_id: str) -> Job:
    return Job(
        id=job_id,
        title="Backend Developer",
        company="Example Co",
        location="Remote",
        is_remote=True,
        experience_required="2+ years",
        skills=["Python"],
        description="Build APIs.",
        source_link="https://example.com/jobs/backend",
        source="test",
    )


def make_context(session, store: SessionStore, jobs: list[Job]) -> ToolContext:
    session_id = "anonymous-session-123"
    state = store.set_profile(session_id, ResumeProfile(name="Jordan"))
    return ToolContext(session_id, state, store, session, jobs)


@pytest.mark.asyncio
async def test_duplicate_application_is_idempotent(database_session) -> None:
    job = make_job("1")
    store = SessionStore()
    context = make_context(database_session, store, [job])

    first = await apply_to_job_tool(context, ApplyToJobInput(job_id="1"))
    second = await apply_to_job_tool(context, ApplyToJobInput(job_id="1"))
    listed = await list_applications_tool(context, ListApplicationsInput())

    assert first["status"] == "applied"
    assert second["status"] == "duplicate"
    assert "already applied" in str(second["message"])
    assert second["applied_date"][:10] in str(second["message"])
    assert listed["applications"] == [
        {
            "job_id": "1",
            "title": "Backend Developer",
            "company": "Example Co",
            "applied_date": first["applied_date"],
            "status": "applied",
        }
    ]
    assert database_session.get_bind().dialect.name == "sqlite"


@pytest.mark.asyncio
async def test_numbered_reference_uses_recent_results_and_id_prefix_selects_dataset_id(
    database_session,
) -> None:
    jobs = [make_job("1"), make_job("3"), make_job("7")]
    store = SessionStore()
    context = make_context(database_session, store, jobs)
    store.remember_jobs(
        context.session_id,
        [JobRecommendation(job=jobs[2], score=0.8, match_reasons=["Skills match"])],
    )

    displayed_result = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="1")
    )
    explicit_id_result = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="id:1")
    )

    assert displayed_result["status"] == "applied"
    assert displayed_result["job_id"] == "7"
    assert explicit_id_result["status"] == "applied"
    assert explicit_id_result["job_id"] == "1"


@pytest.mark.asyncio
async def test_resume_required_before_application(database_session) -> None:
    job = make_job("1")
    store = SessionStore()
    state = store.get("anonymous-session-123")
    context = ToolContext("anonymous-session-123", state, store, database_session, [job])

    result = await apply_to_job_tool(context, ApplyToJobInput(job_id="1"))

    assert result["status"] == "resume_required"
