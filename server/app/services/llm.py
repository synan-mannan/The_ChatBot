from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from server.app.config import Settings
from server.app.models.schemas import ResumeProfile
from server.app.tools.base import ToolDefinition


class LLMUnavailableError(RuntimeError):
    """The configured language model cannot be used."""


class ProfileExtractionError(ValueError):
    """The model could not return a valid structured resume profile."""


@dataclass
class NativeToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMTurn:
    text: str
    tool_calls: list[NativeToolCall]
    model_content: Any


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or (exc.code is not None and exc.code >= 500)
    return False


class GeminiService:
    def __init__(self, settings: Settings, client: genai.Client | None = None) -> None:
        if client is None and not settings.gemini_api_key:
            self.client: genai.Client | None = None
        else:
            self.client = client or genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=30_000),
            )
        self.model = settings.gemini_model
        self.retries = settings.llm_max_retries

    def _require_client(self) -> genai.Client:
        if self.client is None:
            raise LLMUnavailableError(
                "The AI service isn't configured yet. Set GEMINI_API_KEY to enable chat."
            )
        return self.client

    async def _generate(self, **kwargs: Any) -> Any:
        client = self._require_client()
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_retryable),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            stop=stop_after_attempt(self.retries),
            reraise=True,
        ):
            with attempt:
                return await client.aio.models.generate_content(
                    model=self.model,
                    **kwargs,
                )
        raise LLMUnavailableError("The AI service did not return a response.")

    async def extract_profile(self, resume_text: str) -> ResumeProfile:
        prompt = (
            "Extract a resume profile from the text below. Return only a JSON object with these "
            "keys: name, email, phone, designation, skills, experience_years, "
            "experience_summary, education, projects. Use null for unknown scalar values and "
            "empty arrays for unknown list values. Do not infer or invent facts.\n\n"
            f"RESUME TEXT:\n{resume_text}"
        )
        for attempt_number in range(2):
            if attempt_number:
                prompt = (
                    "Repair the previous invalid output. Return only one valid JSON object "
                    "matching the required resume fields. Do not include markdown or commentary. "
                    "Use null or empty arrays for unknown values. Do not invent facts.\n\n"
                    f"RESUME TEXT:\n{resume_text}\n\nINVALID OUTPUT:\n{invalid_output}"
                )
            try:
                response = await self._generate(
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                invalid_output = response.text or ""
                profile = ResumeProfile.model_validate_json(invalid_output)
                return profile
            except (ValueError, TypeError) as exc:
                invalid_output = locals().get("invalid_output", "")
                if attempt_number == 1:
                    raise ProfileExtractionError(
                        "The resume details couldn't be structured. Please try another resume."
                    ) from exc
        raise ProfileExtractionError(
            "The resume details couldn't be structured. Please try another resume."
        )

    async def generate_chat(
        self,
        contents: list[Any],
        tools: list[ToolDefinition],
        system_instruction: str,
    ) -> LLMTurn:
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=types.Schema.from_json_schema(
                    json_schema=types.JSONSchema.model_validate(tool.json_schema())
                ),
            )
            for tool in tools
        ]
        response = await self._generate(
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[types.Tool(function_declarations=declarations)],
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if not response.candidates or response.candidates[0].content is None:
            raise LLMUnavailableError("The AI service returned an empty response.")
        content = response.candidates[0].content
        text_parts: list[str] = []
        calls: list[NativeToolCall] = []
        for part in content.parts or []:
            if part.text:
                text_parts.append(part.text)
            if part.function_call:
                calls.append(
                    NativeToolCall(
                        name=part.function_call.name or "",
                        arguments=dict(part.function_call.args or {}),
                    )
                )
        return LLMTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            model_content=content,
        )
