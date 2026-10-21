from typing import Any
import json

import pytest
import pytest_asyncio
from sqlalchemy import select

from server.app.agents.orchestrator import Orchestrator
from server.app.models.db import ApplicationRow, create_database, initialize_database
from server.app.models.schemas import Job, JobRecommendation, ResumeProfile
from server.app.services.llm import LLMTurn, NativeToolCall
from server.app.services.sessions import SessionStore
from server.app.tools.base import ToolDefinition


class MockLLM:
    def __init__(self, turns: list[LLMTurn]) -> None:
        self.turns = turns
        self.calls = 0
        self.system_instructions: list[str] = []

    async def generate_chat(
        self,
        contents: list[Any],
        tools: list[ToolDefinition],
        system_instruction: str,
    ) -> LLMTurn:
        assert contents
        assert "Never invent a job" in system_instruction
        assert "call recommend_jobs directly" in system_instruction
        assert "profile is available" in system_instruction
        assert {tool.name for tool in tools} >= {"recommend_jobs", "apply_to_job"}
        self.system_instructions.append(system_instruction)
        turn = self.turns[self.calls]
        self.calls += 1
        return turn

    def format_tool_response(
        self, call: NativeToolCall, output: dict[str, Any]
    ) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": call.call_id or "mock-call",
            "content": json.dumps(output),
        }


@pytest_asyncio.fixture
async def database_session():
    engine, factory = create_database("sqlite+aiosqlite:///:memory:")
    await initialize_database(engine)
    async with factory() as session:
        yield session
    await engine.dispose()


def make_job() -> Job:
    return Job(
        id="1",
        title="Backend Developer",
        company="Example Co",
        location="London",
        is_remote=False,
        experience_required="1-3 years",
        skills=["Python"],
        description="Build APIs.",
        source_link="https://example.com/backend",
        source="test",
    )


@pytest.mark.asyncio
async def test_orchestrator_returns_structured_recommendations_without_second_call(
    database_session,
) -> None:
    store = SessionStore()
    store.set_profile(
        "anonymous-session-123",
        ResumeProfile(designation="Backend Developer", skills=["Python"]),
    )
    model = MockLLM(
        [
            LLMTurn(
                text="",
                tool_calls=[
                    NativeToolCall(
                        "recommend_jobs",
                        {"filters": {}, "top_k": 1},
                        "mock-call-1",
                    )
                ],
                model_content="assistant-tool-call",
            )
        ]
    )
    orchestrator = Orchestrator(model, store, [make_job()])

    result = await orchestrator.chat(
        "anonymous-session-123",
        "Find me a backend job.",
        database_session,
    )

    assert result.reply.startswith("I found 1 matching job.")
    assert [call.name for call in result.tool_calls] == ["recommend_jobs"]
    assert result.jobs is not None
    assert [recommendation.job.id for recommendation in result.jobs] == ["1"]
    assert result.profile is not None
    assert result.profile.designation == "Backend Developer"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_orchestrator_caps_model_iterations(database_session) -> None:
    store = SessionStore()
    store.set_profile("anonymous-session-123", ResumeProfile(name="Jordan"))
    repeated_tool_turns = [
        LLMTurn(
            text="",
            tool_calls=[NativeToolCall("parse_resume", {}, f"mock-call-{index}")],
            model_content=f"tool-call-{index}",
        )
        for index in range(5)
    ]
    model = MockLLM(repeated_tool_turns)
    orchestrator = Orchestrator(model, store, [make_job()])

    result = await orchestrator.chat(
        "anonymous-session-123",
        "Show profile",
        database_session,
    )

    assert model.calls == 5
    assert len(result.tool_calls) == 5
    assert "tool-call limit" in result.reply


@pytest.mark.asyncio
async def test_orchestrator_supplies_recent_numbered_jobs_to_model(database_session) -> None:
    store = SessionStore()
    store.set_profile("anonymous-session-123", ResumeProfile(name="Jordan"))
    store.remember_jobs(
        "anonymous-session-123",
        [
            # The displayed number is the stable reference the user sees.
            JobRecommendation(job=make_job(), score=0.8, match_reasons=["Python"]),
        ],
    )
    model = MockLLM(
        [
            LLMTurn(
                text="",
                tool_calls=[
                    NativeToolCall(
                        "apply_to_job",
                        {"job_id": "displayed:1"},
                        "apply-call",
                    )
                ],
                model_content="assistant-apply-call",
            )
        ]
    )
    orchestrator = Orchestrator(model, store, [make_job()])

    await orchestrator.chat("anonymous-session-123", "Apply to job 1", database_session)

    assert "1. Backend Developer at Example Co (dataset ID 1)" in model.system_instructions[0]
    assert "job 2 means displayed:2" in model.system_instructions[0]


@pytest.mark.asyncio
async def test_apply_result_is_returned_without_a_second_model_call(database_session) -> None:
    store = SessionStore()
    store.set_profile("anonymous-session-123", ResumeProfile(name="Jordan"))
    model = MockLLM(
        [
            LLMTurn(
                text="",
                tool_calls=[
                    NativeToolCall(
                        "apply_to_job",
                        {"job_id": "1"},
                        "apply-call",
                    )
                ],
                model_content="assistant-apply-call",
            )
        ]
    )
    orchestrator = Orchestrator(model, store, [make_job()])

    result = await orchestrator.chat(
        "anonymous-session-123",
        "Apply to job 1",
        database_session,
    )
    applications = await database_session.scalars(
        select(ApplicationRow).where(
            ApplicationRow.session_id == "anonymous-session-123"
        )
    )

    assert result.reply == "Application recorded for Backend Developer at Example Co."
    assert len(result.tool_calls) == 1
    assert model.calls == 1
    assert len(list(applications)) == 1
