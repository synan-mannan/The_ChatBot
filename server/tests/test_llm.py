from types import SimpleNamespace

import pytest

from server.app.config import Settings
from server.app.services.llm import GeminiService, ProfileExtractionError


@pytest.mark.asyncio
async def test_profile_extraction_repairs_invalid_json_once(monkeypatch) -> None:
    service = GeminiService(Settings(gemini_api_key="test"), client=object())
    prompts: list[str] = []
    outputs = [
        '{"name":',
        '{"name":"Jordan Lee","skills":["Python"]}',
    ]

    async def fake_generate(**kwargs):
        prompts.append(kwargs["contents"])
        return SimpleNamespace(text=outputs.pop(0))

    monkeypatch.setattr(service, "_generate", fake_generate)

    profile = await service.extract_profile("Jordan Lee Python developer")

    assert profile.name == "Jordan Lee"
    assert profile.skills == ["Python"]
    assert len(prompts) == 2
    assert "Repair the previous invalid output" in prompts[1]


@pytest.mark.asyncio
async def test_profile_extraction_fails_politely_after_repair(monkeypatch) -> None:
    service = GeminiService(Settings(gemini_api_key="test"), client=object())
    calls = 0

    async def fake_generate(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(text="not JSON")

    monkeypatch.setattr(service, "_generate", fake_generate)

    with pytest.raises(ProfileExtractionError, match="couldn't be structured"):
        await service.extract_profile("Some readable resume text")

    assert calls == 2
