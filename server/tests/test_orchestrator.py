from typing import Any

import pytest
import pytest_asyncio

from server.app.agents.orchestrator import Orchestrator
from server.app.models.db import create_database, initialize_database
from server.app.models.schemas import Job, ResumeProfile
from server.app.services.llm import LLMTurn, NativeToolCall
from server.app.services.sessions import SessionStore
from server.app.tools.base import ToolDefinition


class MockLLM:
    def __init__(self, turns: list[LLMTurn]) -> None:
        self.turns = turns
        self.calls = 0

    async def generate_chat(
        self,
        contents: list[Any],
        tools: list[ToolDefinition],
        system_instruction: str,
    ) -> LLMTurn:
        assert "Never invent a job" in system_instruction
        assert {tool.name for tool in tools} >= {"recommend_jobs", "apply_to_job"}
        turn = self.turns[self.calls]
        self.calls += 1
        return turn


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
async def test_orchestrator_runs_native_tool_then_returns_model_reply(database_session) -> None:
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
                    )
                ],
                model_content="assistant-tool-call",
            ),
            LLMTurn("Found one role for you.", [], "assistant-final"),
        ]
    )
    orchestrator = Orchestrator(model, store, [make_job()])

    result = await orchestrator.chat(
        "anonymous-session-123",
        "Find me a backend job.",
        database_session,
    )

    assert result.reply == "Found one role for you."
    assert [call.name for call in result.tool_calls] == ["recommend_jobs"]
    assert result.jobs is not None
    assert [recommendation.job.id for recommendation in result.jobs] == ["1"]
    assert model.calls == 2


@pytest.mark.asyncio
async def test_orchestrator_caps_model_iterations(database_session) -> None:
    store = SessionStore()
    store.set_profile("anonymous-session-123", ResumeProfile(name="Jordan"))
    repeated_tool_turns = [
        LLMTurn(
            text="",
            tool_calls=[NativeToolCall("parse_resume", {})],
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
