from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from server.app.models.db import ResumeProfileRow, create_database, initialize_database
from server.app.models.schemas import ResumeProfile
from server.app.services.llm import _parse_profile_output, _profile_prompt
from server.app.services.sessions import SessionState, SessionStore
from server.app.tools.application_answers import (
    SaveApplicationAnswersInput,
    save_application_answers_tool,
)
from server.app.tools.base import ToolContext


def test_richer_resume_fields_remain_optional_and_backward_compatible() -> None:
    profile = ResumeProfile.model_validate(
        {"name": "Ada Lovelace", "email": "ada@example.test"}
    )

    assert profile.first_name is None
    assert profile.total_experience_years is None
    assert profile.skills == []


def test_resume_extraction_prompt_forbids_invented_details() -> None:
    prompt = _profile_prompt("Ada Lovelace")

    assert "Never infer or invent personal details" in prompt
    assert "linkedin_url" in prompt
    parsed = _parse_profile_output('{"name":"Ada Lovelace"}')
    assert parsed.linkedin_url is None


def test_application_answer_tool_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        SaveApplicationAnswersInput.model_validate(
            {"answers": {"made_up_answer": "invented"}}
        )


@pytest.mark.asyncio
async def test_application_answers_are_saved_to_session_and_database() -> None:
    engine, factory = create_database("sqlite+aiosqlite:///:memory:")
    await initialize_database(engine)
    store = SessionStore()
    state = store.get("session-123456")
    async with factory() as db_session:
        context = ToolContext(
            session_id="session-123456",
            state=state,
            session_store=store,
            db_session=db_session,
            jobs=[],
        )
        result = await save_application_answers_tool(
            context,
            SaveApplicationAnswersInput(
                answers={"notice_period": "30 days", "requires_sponsorship": False}
            ),
        )
        row = await db_session.get(ResumeProfileRow, "session-123456")

    assert result["status"] == "saved"
    assert "notice_period" in result["saved_keys"]
    assert state.answers == {
        "notice_period": "30 days",
        "requires_sponsorship": False,
    }
    assert row is not None
    assert row.answers_data == state.answers
    await engine.dispose()


@pytest.mark.asyncio
async def test_initialize_database_migrates_existing_profile_rows() -> None:
    engine, _ = create_database("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE resume_profiles (session_id VARCHAR(128) PRIMARY KEY, "
                "profile_data JSON NOT NULL)"
            )
        )
        await connection.execute(
            text(
                "INSERT INTO resume_profiles (session_id, profile_data) "
                "VALUES ('legacy-session', '{}')"
            )
        )

    await initialize_database(engine)
    async with engine.connect() as connection:
        row = await connection.execute(
            text(
                "SELECT answers_data FROM resume_profiles "
                "WHERE session_id = 'legacy-session'"
            )
        )
        assert row.scalar_one() == "{}"
    await engine.dispose()
