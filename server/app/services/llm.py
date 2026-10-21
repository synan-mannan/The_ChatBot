from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
)
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from server.app.config import Settings
from server.app.models.schemas import ResumeProfile
from server.app.tools.base import ToolDefinition

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """The configured language model cannot be used."""


class ProfileExtractionError(ValueError):
    """The model could not return a valid structured resume profile."""


@dataclass
class NativeToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclass
class LLMTurn:
    text: str
    tool_calls: list[NativeToolCall]
    model_content: Any


def _retryable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            asyncio.TimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
            APIConnectionError,
            APITimeoutError,
        ),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or (exc.code is not None and exc.code >= 500)
    return False


def _profile_prompt(resume_text: str) -> str:
    schema = json.dumps(ResumeProfile.model_json_schema(), ensure_ascii=False)
    return (
        "Extract only facts explicitly supported by the resume text. Return exactly one JSON "
        "object matching this JSON Schema. Do not use markdown or add fields. "
        "For unknown values, use null for nullable scalar fields and [] for arrays. "
        "experience_years must be a JSON number (not text such as '5 years') or null. "
        "skills, education, and projects must be arrays of strings.\n\n"
        f"JSON SCHEMA:\n{schema}\n\nRESUME TEXT:\n{resume_text}"
    )


def _profile_validation_summary(exc: ValueError | TypeError) -> str:
    if isinstance(exc, ValidationError):
        return json.dumps(exc.errors(include_input=False), ensure_ascii=False)
    if isinstance(exc, json.JSONDecodeError):
        return f"Invalid JSON: {exc.msg} at character {exc.pos}."
    return str(exc)


def _parse_profile_output(raw: str) -> ResumeProfile:
    candidate = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.I | re.S)
    if fenced:
        candidate = fenced.group(1)
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("The response must be a JSON object.")

    fields = ResumeProfile.model_fields
    normalized = {key: value for key, value in data.items() if key in fields}
    for field_name in ("skills", "education", "projects"):
        value = normalized.get(field_name)
        if value is None:
            normalized[field_name] = []
        elif isinstance(value, str):
            normalized[field_name] = [
                item.strip()
                for item in re.split(r"[,;\n]+", value)
                if item.strip()
            ]
    years = normalized.get("experience_years")
    if isinstance(years, bool):
        normalized["experience_years"] = None
    elif isinstance(years, str):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*(?:\+|years?|yrs?)?\s*",
            years,
            re.I,
        )
        normalized["experience_years"] = float(match.group(1)) if match else None

    return ResumeProfile.model_validate(normalized)


def _profile_repair_prompt(
    resume_text: str, invalid_output: str, validation_issue: str
) -> str:
    return (
        f"{_profile_prompt(resume_text)}\n\n"
        "Your previous response did not match the required schema. Correct these validation "
        f"issues and return one corrected JSON object:\n{validation_issue}\n\n"
        f"PREVIOUS RESPONSE:\n{invalid_output[:12_000]}"
    )


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
        try:
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
        except errors.APIError as exc:
            logger.warning(
                "Gemini request failed after retries: code=%s status=%s",
                exc.code,
                exc.status,
            )
            if exc.code in {401, 403}:
                raise LLMUnavailableError(
                    "Gemini rejected the API credentials. Check GEMINI_API_KEY."
                ) from exc
            if exc.code == 429:
                raise LLMUnavailableError(
                    "Gemini rate limit or quota reached. Check your Google AI billing/quota, "
                    "or configure GROQ_API_KEY to use Groq."
                ) from exc
            if exc.code == 404:
                raise LLMUnavailableError(
                    f"Gemini model '{self.model}' was not found or is unavailable for this API key. "
                    "Check GEMINI_MODEL."
                ) from exc
            if exc.code is not None and exc.code >= 500:
                raise LLMUnavailableError(
                    "Gemini is temporarily unavailable after retries. Please try again shortly."
                ) from exc
            raise LLMUnavailableError(
                f"Gemini rejected the request (HTTP {exc.code or 'unknown'}). "
                "Check the configured model and request settings."
            ) from exc
        except (asyncio.TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning(
                "Gemini request timed out or could not connect after retries: %s",
                type(exc).__name__,
            )
            raise LLMUnavailableError(
                "Gemini could not be reached after retries. Check your network connection "
                "and try again."
            ) from exc
        raise LLMUnavailableError("The AI service did not return a response.")

    async def extract_profile(self, resume_text: str) -> ResumeProfile:
        prompt = _profile_prompt(resume_text)
        for attempt_number in range(2):
            if attempt_number:
                prompt = _profile_repair_prompt(
                    resume_text, invalid_output, validation_issue
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
                return _parse_profile_output(invalid_output)
            except (ValueError, TypeError) as exc:
                validation_issue = _profile_validation_summary(exc)
                logger.warning(
                    "Gemini resume extraction validation failed on attempt %s/2: %s",
                    attempt_number + 1,
                    validation_issue,
                )
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
                        call_id=part.function_call.id,
                    )
                )
        return LLMTurn(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            model_content=content,
        )

    def format_tool_response(
        self, call: NativeToolCall, output: dict[str, Any]
    ) -> types.Content:
        return types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=call.name,
                    response=output,
                )
            ],
        )


class GroqService:
    def __init__(
        self,
        settings: Settings,
        client: AsyncGroq | None = None,
    ) -> None:
        if client is None and not settings.groq_api_key:
            self.client: AsyncGroq | None = None
        else:
            self.client = client or AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=30.0,
                max_retries=0,
            )
        self.model = settings.groq_model
        self.retries = settings.llm_max_retries

    def _require_client(self) -> AsyncGroq:
        if self.client is None:
            raise LLMUnavailableError(
                "The AI service isn't configured yet. Set GROQ_API_KEY (or GEMINI_API_KEY) to enable chat."
            )
        return self.client

    async def _generate(self, **kwargs: Any) -> Any:
        client = self._require_client()
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_retryable),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                stop=stop_after_attempt(self.retries),
                reraise=True,
            ):
                with attempt:
                    return await client.chat.completions.create(
                        model=self.model,
                        **kwargs,
                    )
        except APIStatusError as exc:
            if exc.status_code in {401, 403}:
                raise LLMUnavailableError(
                    "Groq rejected the API credentials. Check GROQ_API_KEY."
                ) from exc
            if _retryable(exc):
                raise LLMUnavailableError(
                    "Groq is temporarily unavailable after retries. Please try again shortly."
                ) from exc
            raise LLMUnavailableError(
                "Groq could not process the request. Check the configured model and try again."
            ) from exc
        except (APIConnectionError, APITimeoutError, httpx.HTTPError, asyncio.TimeoutError) as exc:
            raise LLMUnavailableError(
                "Groq is temporarily unavailable after retries. Please try again shortly."
            ) from exc
        raise LLMUnavailableError("Groq did not return a response.")

    async def extract_profile(self, resume_text: str) -> ResumeProfile:
        prompt = _profile_prompt(resume_text)
        for attempt_number in range(2):
            if attempt_number:
                prompt = _profile_repair_prompt(
                    resume_text, invalid_output, validation_issue
                )
            response = await self._generate(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            invalid_output = response.choices[0].message.content or ""
            try:
                return _parse_profile_output(invalid_output)
            except (ValueError, TypeError) as exc:
                validation_issue = _profile_validation_summary(exc)
                logger.warning(
                    "Groq resume extraction validation failed on attempt %s/2: %s",
                    attempt_number + 1,
                    validation_issue,
                )
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
        messages = [{"role": "system", "content": system_instruction}]
        for item in contents:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                messages.append(item)
            else:
                raise LLMUnavailableError(
                    "The chat history contains a message unsupported by the Groq provider."
                )

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema(),
                },
            }
            for tool in tools
        ]
        response = await self._generate(
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=0.2,
        )
        if not response.choices:
            raise LLMUnavailableError("Groq returned an empty response.")
        message = response.choices[0].message
        calls: list[NativeToolCall] = []
        for tool_call in message.tool_calls or []:
            raw_arguments = tool_call.function.arguments or "{}"
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_arguments
            calls.append(
                NativeToolCall(
                    name=tool_call.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                    call_id=tool_call.id,
                )
            )
        return LLMTurn(
            text=message.content or "",
            tool_calls=calls,
            model_content=message.model_dump(exclude_none=True),
        )

    def format_tool_response(
        self, call: NativeToolCall, output: dict[str, Any]
    ) -> dict[str, str]:
        if call.call_id is None:
            raise LLMUnavailableError("Groq returned a tool call without an identifier.")
        return {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": json.dumps(output, ensure_ascii=False),
        }


def create_llm_service(settings: Settings) -> GeminiService | GroqService:
    if settings.groq_api_key:
        return GroqService(settings)
    if settings.gemini_api_key:
        return GeminiService(settings)
    return GroqService(settings)
