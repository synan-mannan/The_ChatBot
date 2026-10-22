import pytest
import pytest_asyncio
from sqlalchemy import select

from server.app.config import Settings
from server.app.services.applicant import ApplyResult
from server.app.tools import applications as application_tools
from server.app.models.db import ApplicationRow, create_database, initialize_database
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


def make_context(
    session,
    store: SessionStore,
    jobs: list[Job],
    *,
    real_apply_enabled: bool = False,
    require_review: bool = False,
) -> ToolContext:
    session_id = "anonymous-session-123"
    state = store.set_profile(session_id, ResumeProfile(name="Jordan"))
    return ToolContext(
        session_id,
        state,
        store,
        session,
        jobs,
        settings=Settings(
            real_apply_enabled=real_apply_enabled,
            require_review=require_review,
        ),
    )


@pytest.mark.asyncio
async def test_submitted_application_is_duplicate(database_session, monkeypatch) -> None:
    job = make_job("1")
    store = SessionStore()
    context = make_context(
        database_session, store, [job], real_apply_enabled=True
    )
    calls = 0

    async def fake_apply(*_: object, **__: object) -> ApplyResult:
        nonlocal calls
        calls += 1
        return ApplyResult("manual", "submitted", "Portal confirmed receipt.", job.source_link)

    monkeypatch.setattr(application_tools, "pick_method", lambda _: "manual")
    monkeypatch.setattr(application_tools, "apply", fake_apply)

    first = await apply_to_job_tool(context, ApplyToJobInput(job_id="1"))
    second = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="1")
    )
    listed = await list_applications_tool(context, ListApplicationsInput())

    assert first["status"] == "submitted"
    assert second["status"] == "duplicate"
    assert second["application_status"] == "submitted"
    assert "already recorded" in str(second["message"])
    assert second["applied_date"][:10] in str(second["message"])
    assert calls == 1
    assert listed["applications"] == [
        {
            "job_id": "1",
            "title": "Backend Developer",
            "company": "Example Co",
            "applied_date": first["applied_date"],
            "status": "submitted",
            "method": "manual",
            "detail": first["detail"],
            "apply_url": "https://example.com/jobs/backend",
            "filled": [],
            "needs_input": [],
            "screenshot_url": None,
            "detected_ats": None,
            "debug_screenshot_url": None,
        }
    ]
    assert database_session.get_bind().dialect.name == "sqlite"


@pytest.mark.asyncio
async def test_submit_pending_requires_a_single_reviewed_application(
    database_session,
) -> None:
    jobs = [make_job("1"), make_job("2")]
    store = SessionStore()
    context = make_context(
        database_session,
        store,
        jobs,
        real_apply_enabled=True,
        require_review=True,
    )
    database_session.add_all(
        [
            ApplicationRow(
                session_id=context.session_id,
                job_id=job.id,
                status="review_ready",
                method="autofill",
                detail="Ready for review.",
                apply_url=job.source_link,
                report_data={"filled": []},
            )
            for job in jobs
        ]
    )
    await database_session.commit()

    result = await apply_to_job_tool(
        context,
        ApplyToJobInput(job_id="pending", submit=True),
    )

    assert result["status"] == "clarification_required"
    assert "Backend Developer at Example Co (id:1)" in result["message"]
    assert "Backend Developer at Example Co (id:2)" in result["message"]


@pytest.mark.asyncio
async def test_numbered_reference_uses_recent_results_and_id_prefix_selects_dataset_id(
    database_session, monkeypatch
) -> None:
    jobs = [make_job("1"), make_job("3"), make_job("7")]
    store = SessionStore()
    context = make_context(database_session, store, jobs)
    async def fake_apply(*_: object, **__: object) -> ApplyResult:
        return ApplyResult("manual", "needs_manual", "Manual.", "https://example.test/apply")

    monkeypatch.setattr(application_tools, "pick_method", lambda _: "manual")
    monkeypatch.setattr(application_tools, "apply", fake_apply)
    store.remember_jobs(
        context.session_id,
        [JobRecommendation(job=jobs[2], score=0.8, match_reasons=["Skills match"])],
    )

    displayed_confirmation = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="1")
    )
    assert displayed_confirmation["job_id"] == "7"
    displayed_result = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="id:7", confirm=True)
    )
    explicit_store = SessionStore()
    explicit_context = make_context(database_session, explicit_store, jobs)
    explicit_confirmation = await apply_to_job_tool(
        explicit_context, ApplyToJobInput(job_id="id:1")
    )
    explicit_id_result = await apply_to_job_tool(
        explicit_context, ApplyToJobInput(job_id="id:1", confirm=True)
    )

    assert displayed_confirmation["status"] == "needs_manual"
    assert displayed_result["status"] == "needs_manual"
    assert explicit_confirmation["status"] == "needs_manual"
    assert explicit_id_result["status"] == "needs_manual"
    assert explicit_id_result["job_id"] == "1"


@pytest.mark.asyncio
async def test_resume_required_before_application(database_session) -> None:
    job = make_job("1")
    store = SessionStore()
    state = store.get("anonymous-session-123")
    context = ToolContext("anonymous-session-123", state, store, database_session, [job])

    result = await apply_to_job_tool(context, ApplyToJobInput(job_id="1"))

    assert result["status"] == "resume_required"


@pytest.mark.asyncio
async def test_failed_application_retry_reuses_the_same_database_row(
    database_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = make_job("1")
    store = SessionStore()
    context = make_context(database_session, store, [job])
    results = iter(
        [
            ApplyResult("manual", "failed", "Temporary failure.", job.source_link),
            ApplyResult("manual", "needs_manual", "Complete it manually.", job.source_link),
        ]
    )
    calls = 0

    async def fake_apply(*_: object, **__: object) -> ApplyResult:
        nonlocal calls
        calls += 1
        return next(results)

    monkeypatch.setattr(application_tools, "apply", fake_apply)
    monkeypatch.setattr(application_tools, "pick_method", lambda job: "manual")

    first_attempt = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="1")
    )
    original_row = await database_session.scalar(
        select(ApplicationRow).where(ApplicationRow.job_id == "1")
    )
    assert first_attempt["status"] == "failed"
    assert original_row is not None
    original_id = original_row.id

    retry = await apply_to_job_tool(
        context, ApplyToJobInput(job_id="1")
    )
    rows = list(await database_session.scalars(select(ApplicationRow)))

    assert retry["status"] == "needs_manual"
    assert calls == 2
    assert len(rows) == 1
    assert rows[0].id == original_id
    assert rows[0].status == "needs_manual"
