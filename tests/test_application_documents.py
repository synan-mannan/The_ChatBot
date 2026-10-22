from __future__ import annotations

import pytest

from server.app.models.db import ResumeProfileRow, create_database, initialize_database
from server.app.models.schemas import Job, ResumeProfile
from server.app.services.llm import LLMTurn
from server.app.services.sessions import SessionState, SessionStore
from server.app.tools.application_documents import (
    ApproveCoverLetterInput,
    GenerateCoverLetterInput,
    approve_cover_letter_tool,
    generate_cover_letter_tool,
)
from server.app.tools.base import ToolContext


class CoverLetterModel:
    async def generate_chat(self, contents, tools, system_instruction):
        return LLMTurn(
            text="Dear hiring team,\n\nI am applying based on my stated experience.",
            tool_calls=[],
            model_content=None,
        )


@pytest.mark.asyncio
async def test_cover_letter_requires_approval_and_persists_only_after_approval() -> None:
    engine, factory = create_database("sqlite+aiosqlite:///:memory:")
    await initialize_database(engine)
    job = Job(
        id="job-docs",
        title="Engineer",
        company="Example",
        location="Remote",
        is_remote=True,
        experience_required="Not specified",
        source_link="https://careers.example.test/apply",
        source="Fixture",
        description="Build reliable software.",
    )
    store = SessionStore()
    state = store.set_profile("session-docs-123", ResumeProfile(name="Ada Lovelace"))
    async with factory() as db_session:
        db_session.add(
            ResumeProfileRow(
                session_id="session-docs-123",
                profile_data=state.profile.model_dump(mode="json"),
            )
        )
        await db_session.commit()
        context = ToolContext(
            session_id="session-docs-123",
            state=state,
            session_store=store,
            db_session=db_session,
            jobs=[job],
            model=CoverLetterModel(),
        )
        draft = await generate_cover_letter_tool(
            context, GenerateCoverLetterInput(job_id=job.id)
        )
        row = await db_session.get(ResumeProfileRow, "session-docs-123")
        assert row is not None
        assert row.approved_cover_letters_data == {}
        assert state.approved_cover_letters == {}

        approval = await approve_cover_letter_tool(
            context, ApproveCoverLetterInput(job_id=job.id)
        )
        await db_session.refresh(row)

    assert draft["status"] == "draft_ready"
    assert state.pending_cover_letter is None
    assert approval["status"] == "approved"
    assert row.approved_cover_letters_data[job.id] == draft["cover_letter"]
    assert state.approved_cover_letters[job.id] == draft["cover_letter"]
    await engine.dispose()
