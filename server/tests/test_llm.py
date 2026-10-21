from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from server.app.config import Settings
from server.app.services.llm import (
    GeminiService,
    GroqService,
    LLMUnavailableError,
    NativeToolCall,
    ProfileExtractionError,
    _parse_profile_output,
    create_llm_service,
)
from server.app.tools.jobs import RECOMMEND_DEFINITION


class FakeGroqMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in self.tool_calls or []
                ],
            }.items()
            if not exclude_none or value is not None
        }


class FakeGroqClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    async def create_completion(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.responses.pop(0)


def groq_response(message: FakeGroqMessage) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "status", "message"),
    [
        (
            401,
            "UNAUTHENTICATED",
            "Gemini rejected the API credentials. Check GEMINI_API_KEY.",
        ),
        (
            429,
            "RESOURCE_EXHAUSTED",
            "Gemini rate limit or quota reached.",
        ),
        (
            404,
            "NOT_FOUND",
            "Gemini model 'gemini-2.5-flash' was not found",
        ),
        (
            503,
            "UNAVAILABLE",
            "Gemini is temporarily unavailable after retries.",
        ),
    ],
)
async def test_gemini_api_errors_return_actionable_safe_message(
    status_code: int, status: str, message: str
) -> None:
    async def fail_request(**_: Any) -> Any:
        raise errors.APIError(
            status_code,
            {"message": "Provider detail", "status": status},
        )

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=fail_request)
        )
    )
    service = GeminiService(
        Settings(gemini_api_key="test", llm_max_retries=1),
        client=client,
    )

    with pytest.raises(LLMUnavailableError, match=message):
        await service._generate(contents="test")


@pytest.mark.asyncio
async def test_profile_extraction_repairs_invalid_json_once(monkeypatch) -> None:
    service = GeminiService(Settings(gemini_api_key="test"), client=object())
    prompts: list[str] = []
    outputs = [
        '{"name":',
        '{"name":"Jordan Lee","skills":["Python"]}',
    ]

    async def fake_generate(**kwargs: Any):
        prompts.append(kwargs["contents"])
        return SimpleNamespace(text=outputs.pop(0))

    monkeypatch.setattr(service, "_generate", fake_generate)

    profile = await service.extract_profile("Jordan Lee Python developer")

    assert profile.name == "Jordan Lee"
    assert profile.skills == ["Python"]
    assert len(prompts) == 2
    assert "previous response did not match" in prompts[1]


@pytest.mark.asyncio
async def test_profile_extraction_fails_politely_after_repair(monkeypatch) -> None:
    service = GeminiService(Settings(gemini_api_key="test"), client=object())
    calls = 0

    async def fake_generate(**_: Any):
        nonlocal calls
        calls += 1
        return SimpleNamespace(text="not JSON")

    monkeypatch.setattr(service, "_generate", fake_generate)

    with pytest.raises(ProfileExtractionError, match="couldn't be structured"):
        await service.extract_profile("Some readable resume text")

    assert calls == 2


@pytest.mark.asyncio
async def test_groq_profile_extraction_uses_json_mode_and_validates() -> None:
    client = FakeGroqClient(
        [
            groq_response(
                FakeGroqMessage(
                    '{"name":"Jordan Lee","skills":["Python"],"experience_years":4}'
                )
            )
        ]
    )
    service = GroqService(
        Settings(_env_file=None, groq_api_key="test"),
        client=client,
    )

    profile = await service.extract_profile("Jordan Lee is a Python developer.")

    assert profile.name == "Jordan Lee"
    assert profile.skills == ["Python"]
    assert client.requests[0]["response_format"] == {"type": "json_object"}
    assert client.requests[0]["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_groq_repairs_invalid_profile_json_once() -> None:
    client = FakeGroqClient(
        [
            groq_response(FakeGroqMessage("{invalid")),
            groq_response(FakeGroqMessage('{"name":"Jordan"}')),
        ]
    )
    service = GroqService(Settings(groq_api_key="test"), client=client)

    profile = await service.extract_profile("Jordan's resume")

    assert profile.name == "Jordan"
    assert len(client.requests) == 2
    assert "previous response did not match" in client.requests[1]["messages"][0]["content"]


@pytest.mark.parametrize(
    ("raw_output", "experience_years", "skills"),
    [
        (
            '{"name":"Jordan","skills":"Python, React","experience_years":"5 years"}',
            5,
            ["Python", "React"],
        ),
        (
            '```json\n{"name":"Jordan","skills":["Python"],"experience_years":"5+"}\n```',
            5,
            ["Python"],
        ),
        (
            '{"name":"Jordan","skills":["Python"],"experience_years":"5-7 years","location":"London"}',
            None,
            ["Python"],
        ),
        (
            '{"name":"Jordan","skills":null,"education":null,"experience_years":null}',
            None,
            [],
        ),
    ],
)
def test_resume_profile_extraction_normalizes_common_model_output(
    raw_output: str, experience_years: int | None, skills: list[str]
) -> None:
    profile = _parse_profile_output(raw_output)

    assert profile.name == "Jordan"
    assert profile.experience_years == experience_years
    assert profile.skills == skills


@pytest.mark.asyncio
async def test_groq_native_tool_call_and_tool_response_round_trip() -> None:
    client = FakeGroqClient(
        [
            groq_response(
                FakeGroqMessage(
                    tool_calls=[
                        SimpleNamespace(
                            id="call-123",
                            function=SimpleNamespace(
                                name="recommend_jobs",
                                arguments='{"filters":{"remote_only":true},"top_k":3}',
                            ),
                        )
                    ]
                )
            ),
            groq_response(FakeGroqMessage("Found three remote jobs.")),
        ]
    )
    service = GroqService(Settings(groq_api_key="test"), client=client)

    turn = await service.generate_chat(
        ["Find remote jobs."],
        [RECOMMEND_DEFINITION],
        "You are AIJob.",
    )

    call = NativeToolCall(
        "recommend_jobs",
        {"filters": {"remote_only": True}, "top_k": 3},
        "call-123",
    )
    assert turn.tool_calls == [call]
    assert turn.model_content["role"] == "assistant"
    assert client.requests[0]["messages"] == [
        {"role": "system", "content": "You are AIJob."},
        {"role": "user", "content": "Find remote jobs."},
    ]
    assert client.requests[0]["tools"][0]["function"]["name"] == "recommend_jobs"
    tool_response = service.format_tool_response(call, {"status": "ok", "jobs": []})
    assert tool_response == {
        "role": "tool",
        "tool_call_id": "call-123",
        "content": '{"status": "ok", "jobs": []}',
    }
    final_turn = await service.generate_chat(
        ["Find remote jobs.", turn.model_content, tool_response],
        [RECOMMEND_DEFINITION],
        "You are AIJob.",
    )

    assert final_turn.text == "Found three remote jobs."
    assert client.requests[1]["messages"][2] == turn.model_content
    assert client.requests[1]["messages"][3] == tool_response


@pytest.mark.asyncio
async def test_groq_missing_key_is_reported_without_network_call() -> None:
    service = GroqService(Settings(_env_file=None))

    with pytest.raises(LLMUnavailableError, match="GROQ_API_KEY"):
        await service.generate_chat([], [], "You are AIJob.")


def test_groq_provider_is_preferred_and_gemini_remains_available() -> None:
    groq_service = create_llm_service(
        Settings(
            _env_file=None,
            groq_api_key="groq-test",
            gemini_api_key="gemini-test",
        )
    )
    gemini_service = create_llm_service(
        Settings(_env_file=None, gemini_api_key="gemini-test")
    )
    unconfigured_service = create_llm_service(Settings(_env_file=None))

    assert isinstance(groq_service, GroqService)
    assert isinstance(gemini_service, GeminiService)
    assert isinstance(unconfigured_service, GroqService)


def test_settings_load_groq_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    settings = Settings(_env_file=None)

    assert settings.groq_api_key == "test-groq-key"
