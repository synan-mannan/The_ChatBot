from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, RootModel, ValidationError

from server.app.config import get_settings
from server.app.models.schemas import Job, ResumeProfile

AutofillMode = Literal["dry_run", "review", "submit"]
BLOCKED_DOMAINS = {"linkedin.com", "indeed.com", "naukri.com", "glassdoor.com"}
AGGREGATOR_DOMAINS = {
    "remotive.com",
    "remoteok.com",
    "jobicy.com",
    "himalayas.app",
}
ALLOWED_PROFILE_KEYS = {
    "first_name",
    "last_name",
    "name",
    "email",
    "phone",
    "city",
    "country",
    "linkedin_url",
    "github_url",
    "portfolio_url",
    "current_company",
    "current_title",
    "total_experience_years",
    "experience_years",
    "highest_education",
    "college",
    "graduation_year",
    "resume_path",
    "cover_letter",
}
ALLOWED_ANSWER_KEYS = {
    "work_authorization",
    "requires_sponsorship",
    "notice_period",
    "expected_salary",
    "current_salary",
    "willing_to_relocate",
    "earliest_start_date",
    "gender",
    "ethnicity",
    "disability_status",
    "veteran_status",
    "criminal_history",
    "terms_consent",
    "privacy_consent",
    "legal_attestation",
}
ALLOWED_MAPPING_KEYS = ALLOWED_PROFILE_KEYS | ALLOWED_ANSWER_KEYS
FORM_TIMEOUT_SECONDS = 30
ACTION_TIMEOUT_MS = 8_000
DOMAIN_INTERVAL = timedelta(seconds=30)
SESSION_HOUR = timedelta(hours=1)
_RATE_LIMIT_LOCK = asyncio.Lock()
_DOMAIN_RUNS: dict[str, datetime] = {}
_SESSION_SUBMISSIONS: dict[str, list[datetime]] = {}
_REVIEW_RUNS: dict[tuple[str, str, str], datetime] = {}
logger = logging.getLogger(__name__)


def clear_session_rate_limits(session_id: str) -> None:
    _SESSION_SUBMISSIONS.pop(session_id, None)
    for key in [key for key in _REVIEW_RUNS if key[0] == session_id]:
        _REVIEW_RUNS.pop(key, None)


class FilledField(BaseModel):
    label: str
    key: str


class NeedsInputField(BaseModel):
    label: str
    type: str
    options: list[str] = []
    required: bool


class AutofillReport(BaseModel):
    status: Literal[
        "submitted",
        "review_ready",
        "dry_run",
        "needs_input",
        "needs_manual",
        "failed",
        "retry_later",
    ]
    filled: list[FilledField] = []
    needs_input: list[NeedsInputField] = []
    blocker_reason: str | None = None
    screenshot_path: str | None = None
    final_url: str | None = None
    detected_ats: Literal[
        "greenhouse", "lever", "ashby", "workable", "smartrecruiters", "workday", "other"
    ] = "other"
    debug_screenshot_path: str | None = None
    discovered_form_labels: list[str] = []
    debug_labels_path: str | None = None


class MappingOutput(RootModel[dict[str, str | None]]):
    pass


@dataclass
class FormField:
    frame: Any
    controls: list[Any]
    label: str
    name: str
    field_id: str
    field_type: str
    required: bool
    options: list[tuple[str, str]] = field(default_factory=list)
    metadata: str = ""
    form_key: str = "page"
    form_metadata: str = ""
    visible: bool = True


class FieldMappingError(RuntimeError):
    """The optional AI field-mapping response could not be validated."""


class CoverLetterGenerationError(RuntimeError):
    """The configured model could not draft a cover letter."""


async def generate_cover_letter(
    profile: ResumeProfile, job: Job, model: Any
) -> str:
    experience_years = (
        profile.total_experience_years
        if profile.total_experience_years is not None
        else profile.experience_years
    )
    candidate_facts = {
        "name": profile.name,
        "current_title": profile.current_title or profile.designation,
        "current_company": profile.current_company,
        "experience_years": experience_years,
        "experience_summary": profile.experience_summary,
        "skills": profile.skills,
        "education": profile.education,
        "projects": profile.projects,
    }
    prompt = (
        "Draft a concise cover letter for the job using only explicit facts from the "
        "candidate profile and the supplied job description. Do not invent experience, "
        "achievements, qualifications, or contact details. Return only the letter text.\n"
        f"JOB TITLE: {job.title}\nCOMPANY: {job.company}\n"
        f"JOB DESCRIPTION:\n{job.description}\n"
        f"CANDIDATE PROFILE FACTS:\n{json.dumps(candidate_facts, ensure_ascii=False)}"
    )
    turn = await model.generate_chat(
        contents=[prompt],
        tools=[],
        system_instruction=(
            "Write a professional cover letter using only the provided job details and "
            "candidate facts. Do not claim it has been approved or submitted."
        ),
    )
    draft = turn.text.strip()
    if not draft:
        raise CoverLetterGenerationError("The AI did not return a cover letter draft.")
    return draft


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _blocked_domain(hostname: str) -> bool:
    host = hostname.casefold().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in BLOCKED_DOMAINS)


def _safe_file_component(value: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def _parse_mapping_output(raw: str) -> dict[str, str]:
    candidate = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.I | re.S)
    if fenced:
        candidate = fenced.group(1)
    try:
        output = MappingOutput.model_validate_json(candidate).root
    except (ValidationError, ValueError) as exc:
        raise FieldMappingError("The AI returned an invalid field mapping.") from exc
    return {
        label: key
        for label, key in output.items()
        if key in ALLOWED_MAPPING_KEYS
    }


def _deterministic_key(field: FormField) -> str | None:
    if field.field_type == "file" and re.search(
        r"\b(resume|cv|pdf|docx?)\b", field.metadata, re.I
    ):
        return "resume_path"
    label = _normalize(" ".join((field.label, field.name, field.field_id, field.metadata)))
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("resume_path", (r"\b(resume|cv|curriculum vitae)\b",)),
        ("cover_letter", (r"\bcover letter\b",)),
        ("first_name", (r"\b(first|given|forename)\s+name\b",)),
        ("last_name", (r"\b(last|family|surname)\s+name\b",)),
        ("linkedin_url", (r"\blinked?in\b",)),
        ("github_url", (r"\bgithub\b",)),
        ("portfolio_url", (r"\b(portfolio|personal website|website|web site)\b",)),
        ("email", (r"\b(e mail|email address|email)\b",)),
        ("phone", (r"\b(phone|mobile|cell|telephone)\b",)),
        ("total_experience_years", (r"\b(years? of experience|experience in years)\b",)),
        ("graduation_year", (r"\b(graduation|graduating|degree completion) year\b",)),
        ("highest_education", (r"\b(highest education|education level|degree)\b",)),
        ("college", (r"\b(college|university|school)\b",)),
        ("current_company", (r"\b(current company|current employer|employer name)\b",)),
        ("current_title", (r"\b(current title|current position|current role|job title)\b",)),
        ("city", (r"\b(city|location|town)\b",)),
        ("country", (r"\b(country|country of residence)\b",)),
        ("work_authorization", (r"\b(work authorization|authorized to work|right to work)\b",)),
        ("requires_sponsorship", (r"\b(sponsorship|visa sponsorship)\b",)),
        ("notice_period", (r"\bnotice period\b",)),
        ("expected_salary", (r"\b(expected|desired|salary expectation|salary requirement)\b",)),
        ("current_salary", (r"\b(current salary|current compensation)\b",)),
        ("willing_to_relocate", (r"\b(relocat|willing to move)\b",)),
        ("earliest_start_date", (r"\b(start date|available to start|earliest start)\b",)),
        ("gender", (r"\bgender\b",)),
        ("ethnicity", (r"\b(ethnicity|race)\b",)),
        ("disability_status", (r"\b(disability|disabled)\b",)),
        ("veteran_status", (r"\bveteran\b",)),
        ("criminal_history", (r"\b(convicted|criminal history|felony)\b",)),
        ("terms_consent", (r"\bterms of (?:service|use)\b",)),
        ("privacy_consent", (r"\bprivacy (?:policy|notice|terms)\b",)),
        ("legal_attestation", (r"\b(attest|legal declaration)\b",)),
        ("name", (r"\b(full|preferred) name\b", r"^name$")),
    )
    for key, patterns in rules:
        if any(re.search(pattern, label) for pattern in patterns):
            return key
    return None


def _detected_ats(url: str) -> Literal[
    "greenhouse", "lever", "ashby", "workable", "smartrecruiters", "workday", "other"
]:
    host = (urlsplit(url).hostname or "").casefold()
    for vendor, domains in (
        ("greenhouse", ("greenhouse.io",)),
        ("lever", ("lever.co",)),
        ("ashby", ("ashbyhq.com",)),
        ("workable", ("workable.com",)),
        ("smartrecruiters", ("smartrecruiters.com",)),
        ("workday", ("myworkdayjobs.com", "workday.com")),
    ):
        if any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return vendor  # type: ignore[return-value]
    return "other"


def _registrable_domain(host: str) -> str:
    labels = host.casefold().rstrip(".").split(".")
    if len(labels) < 2:
        return host.casefold()
    common_compound_suffixes = {
        "co.uk", "com.au", "co.nz", "co.in", "com.br", "com.mx", "co.jp"
    }
    suffix = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix in common_compound_suffixes and len(labels) > 2 else suffix


def _is_aggregator(host: str) -> bool:
    domain = _registrable_domain(host)
    return domain.startswith("arbeitnow.") or domain in AGGREGATOR_DOMAINS


_NON_APPLICATION_FORM = re.compile(
    r"newsletter|subscribe|job alert|get notified|search|login|log in|sign in|"
    r"cookie|comment",
    re.I,
)
_APPLY_CONTROL = re.compile(
    r"^\s*apply(?:\s+now|\s+for this job|\s+on (?:the )?company(?: site)?|\s+here)?\s*$",
    re.I,
)
_PROFILE_FORM_KEYS = ALLOWED_PROFILE_KEYS - {
    "cover_letter", "current_company", "current_title", "total_experience_years",
    "experience_years", "highest_education", "college", "graduation_year",
}


def is_application_form(fields: list[FormField]) -> bool:
    if not fields or not any(field.visible for field in fields):
        return False
    description = " ".join(
        " ".join(
            (
                field.form_metadata,
                field.label,
                field.name,
                field.field_id,
                field.metadata,
            )
        )
        for field in fields
    )
    if _NON_APPLICATION_FORM.search(description):
        return False
    file_upload = any(
        field.field_type == "file"
        and _deterministic_key(field) == "resume_path"
        for field in fields
    )
    mapped_profile_fields = [
        field
        for field in fields
        if _deterministic_key(field) in _PROFILE_FORM_KEYS
    ]
    return file_upload or len(mapped_profile_fields) >= 3


async def _application_form_fields(frames: list[Any]) -> tuple[list[FormField], list[FormField]]:
    discovered = await _discover_fields(frames)
    grouped: dict[tuple[int, str], list[FormField]] = {}
    for field_info in discovered:
        grouped.setdefault(
            (id(field_info.frame), field_info.form_key), []
        ).append(field_info)
    candidates = [group for group in grouped.values() if is_application_form(group)]
    labels = [field_info.label for field_info in discovered]
    if not candidates:
        return [], discovered
    candidates.sort(
        key=lambda group: (
            any(
                field_info.field_type == "file"
                and _deterministic_key(field_info) == "resume_path"
                for field_info in group
            ),
            len(
                [
                    field_info
                    for field_info in group
                    if _deterministic_key(field_info) in _PROFILE_FORM_KEYS
                ]
            ),
        ),
        reverse=True,
    )
    selected = candidates[0]
    logger.info(
        "autofill form selected: fields=%d resume_upload=%s",
        len(selected),
        any(field_info.field_type == "file" for field_info in selected),
    )
    return selected, discovered


def _sensitive_question(field: FormField) -> bool:
    options = " ".join(label for label, _ in field.options)
    description = _normalize(f"{field.label} {field.metadata} {options}")
    return bool(
        re.search(
            r"\b(gender|ethnicity|race|disability|disabled|veteran|diversity|"
            r"equal employment|consent|privacy|terms|agree|attest|convicted|"
            r"criminal|felony)\b",
            description,
        )
    )


def _profile_value(profile: ResumeProfile, key: str) -> str | None:
    if key == "first_name":
        value = profile.first_name
        if value:
            return value
        return (profile.name or "").strip().split(maxsplit=1)[0] or None
    if key == "last_name":
        value = profile.last_name
        if value:
            return value
        parts = (profile.name or "").strip().split(maxsplit=1)
        return parts[1] if len(parts) == 2 else None
    if key == "name":
        if profile.name:
            return profile.name
        if profile.first_name or profile.last_name:
            return " ".join(part for part in (profile.first_name, profile.last_name) if part)
        return None
    if key == "total_experience_years":
        value = profile.total_experience_years
        if value is None:
            value = profile.experience_years
    elif key == "current_title":
        value = profile.current_title or profile.designation
    elif key == "highest_education":
        value = profile.highest_education or (profile.education[0] if profile.education else None)
    elif key in ResumeProfile.model_fields:
        value = getattr(profile, key)
    else:
        return None
    return str(value) if value is not None else None


def _value_for_key(
    key: str,
    profile: ResumeProfile,
    answers: dict[str, str | bool | float],
    resume_path: str | None,
) -> str | None:
    if key == "resume_path":
        return resume_path
    if key in ALLOWED_ANSWER_KEYS:
        value = answers.get(key)
        return str(value) if value is not None else None
    return _profile_value(profile, key)


def _option_match(value: str, options: list[tuple[str, str]]) -> tuple[str, str] | None:
    wanted = _normalize(value)
    if not wanted:
        return None
    for option in options:
        if wanted in {_normalize(option[0]), _normalize(option[1])}:
            return option
        if len(wanted) > 3 and wanted in _normalize(option[0]):
            return option
    scored = [
        (
            max(
                SequenceMatcher(None, wanted, _normalize(label)).ratio(),
                SequenceMatcher(None, wanted, _normalize(option_value)).ratio(),
            ),
            (label, option_value),
        )
        for label, option_value in options
        if _normalize(label) not in {"", "other"}
    ]
    if scored:
        score, option = max(scored, key=lambda item: item[0])
        if score >= 0.68:
            return option
    return None


def _option_value(key: str | None, value: str | bool | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        if key in {"requires_sponsorship"}:
            return "yes" if value else "no"
        if key in {"willing_to_relocate"}:
            return "yes" if value else "no"
        return "yes" if value else "no"
    return str(value)


async def _field_mapping(
    fields: list[FormField], model: Any | None
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    unmapped: list[FormField] = []
    for field_info in fields:
        key = _deterministic_key(field_info)
        if key is None:
            unmapped.append(field_info)
        else:
            mapping[field_info.label] = key
    if not unmapped or model is None:
        return mapping
    descriptions = [
        {"label": item.label, "options": [label for label, _ in item.options]}
        for item in unmapped
    ]
    prompt = (
        "Map each form label to one allowed key or null. Return only a JSON object whose "
        "keys are the exact labels and whose values are allowed key names or null. "
        "Never provide a value or answer. Do not infer information from the label.\n"
        f"ALLOWED KEYS: {json.dumps(sorted(ALLOWED_MAPPING_KEYS))}\n"
        f"FIELDS: {json.dumps(descriptions, ensure_ascii=False)}"
    )
    turn = await model.generate_chat(
        contents=[prompt],
        tools=[],
        system_instruction=(
            "You classify application form labels only. Output a JSON label-to-key mapping; "
            "do not generate personal answers or values."
        ),
    )
    response_mapping = _parse_mapping_output(turn.text)
    mapping.update(response_mapping)
    return mapping


async def _discover_fields(frames: list[Any]) -> list[FormField]:
    fields: list[FormField] = []
    for frame in frames:
        controls = frame.locator("input, textarea, select")
        count = await controls.count()
        radio_groups: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
        for index in range(count):
            control = controls.nth(index)
            visible = await control.is_visible()
            info = await control.evaluate(
                """element => {
                  const text = node => (node?.innerText || node?.textContent || '').trim();
                  const labels = element.labels ? Array.from(element.labels).map(text).filter(Boolean) : [];
                  const fieldset = element.closest('fieldset');
                  let preceding = '';
                  let node = element;
                  for (let depth = 0; depth < 4 && node && !preceding; depth++, node = node.parentElement) {
                    let sibling = node.previousElementSibling;
                    while (sibling && !preceding) {
                      preceding = text(sibling);
                      sibling = sibling.previousElementSibling;
                    }
                  }
                  return {
                    tag: element.tagName.toLowerCase(),
                    type: (element.getAttribute('type') || element.tagName).toLowerCase(),
                    value: element.value || '',
                    name: element.getAttribute('name') || '',
                    id: element.id || '',
                    accept: element.getAttribute('accept') || '',
                    label: labels[0] || element.getAttribute('aria-label') ||
                      element.getAttribute('placeholder') || text(fieldset?.querySelector('legend')) ||
                      preceding || '',
                    group: text(fieldset?.querySelector('legend')) || '',
                    formKey: element.closest('form')
                      ? `form-${Array.from(document.forms).indexOf(element.closest('form'))}`
                      : 'page',
                    formMetadata: element.closest('form')
                      ? [
                          text(element.closest('form')),
                          element.closest('form').id,
                          element.closest('form').className,
                          element.closest('form').getAttribute('action') || '',
                          Array.from(element.closest('form').querySelectorAll(
                            'button, input[type="submit"]'
                          )).map(button => text(button) || button.value || '').join(' ')
                        ].join(' ')
                      : '',
                    visible: true,
                    required: element.required || element.getAttribute('aria-required') === 'true',
                    options: element.tagName.toLowerCase() === 'select'
                      ? Array.from(element.options).map(option => ({
                          label: text(option), value: option.value
                        })) : []
                  };
                }"""
            )
            info["visible"] = visible
            field_type = str(info["type"])
            if field_type in {"submit", "button", "reset", "hidden", "image"}:
                continue
            if field_type == "radio":
                group = str(info["name"] or f"radio-{index}")
                radio_groups.setdefault(group, []).append((control, info))
                continue
            options = [
                (str(option["label"]), str(option["value"]))
                for option in info["options"]
                if str(option["label"]).strip()
            ]
            fields.append(
                FormField(
                    frame=frame,
                    controls=[control],
                    label=str(info["label"] or info["name"] or info["id"] or "Unlabelled field"),
                    name=str(info["name"]),
                    field_id=str(info["id"]),
                    field_type=field_type,
                    required=bool(info["required"]),
                    options=options,
                    metadata=" ".join(
                        value for value in (str(info["group"]), str(info["accept"])) if value
                    ),
                    form_key=str(info["formKey"]),
                    form_metadata=str(info["formMetadata"]),
                    visible=bool(info["visible"]),
                )
            )
        for name, group_items in radio_groups.items():
            first = group_items[0][1]
            options = [
                (
                    str(item["label"] or item.get("value") or ""),
                    str(item.get("value") or ""),
                )
                for _, item in group_items
            ]
            fields.append(
                FormField(
                    frame=frame,
                    controls=[control for control, _ in group_items],
                    label=str(first["group"] or first["label"] or name),
                    name=name,
                    field_id=str(first["id"]),
                    field_type="radio",
                    required=any(bool(item["required"]) for _, item in group_items),
                    options=options,
                    metadata=" ".join(label for label, _ in options),
                    form_key=str(first["formKey"]),
                    form_metadata=str(first["formMetadata"]),
                    visible=any(
                        bool(item["visible"]) for _, item in group_items
                    ),
                )
            )
    return fields


async def _blocker_reason(frames: list[Any], url: str) -> str | None:
    for frame in frames:
        captcha = frame.locator(
            'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="turnstile"], '
            '.h-captcha, .g-recaptcha, .cf-turnstile, '
            '[name="g-recaptcha-response"], [name="h-captcha-response"]'
        )
        if await captcha.count():
            return "A CAPTCHA is present. I stopped without attempting to solve it."
    page_text = ""
    for frame in frames:
        try:
            page_text += "\n" + (await frame.locator("body").inner_text(timeout=1_000))
        except Exception:
            continue
    if re.search(
        r"captcha|verify you are human|checking your browser|security challenge",
        page_text,
        re.I,
    ):
        return "A CAPTCHA or anti-bot challenge is present. I stopped without attempting to bypass it."
    if re.search(
        r"sign in|log in|create an account|create account|single sign.on|sso|"
        r"account required|register to apply",
        page_text,
        re.I,
    ):
        return "This page requires login, account creation, or SSO. Continue manually."
    parsed = urlsplit(url)
    if re.search(r"/(login|signin|sign-in|register|signup)(?:/|$)", parsed.path, re.I):
        return "This URL is a login or account-creation page. Continue manually."
    return None


async def _wait_for_destination(page: Any) -> Any:
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    try:
        await page.wait_for_load_state("domcontentloaded", timeout=10_000)
    except PlaywrightTimeoutError:
        pass
    try:
        await page.wait_for_load_state("networkidle", timeout=3_000)
    except PlaywrightTimeoutError:
        pass
    return page


async def _activate_apply_control(page: Any, frames: list[Any]) -> Any | None:
    candidates: list[tuple[bool, Any, str | None]] = []
    for frame in frames:
        for role in ("link", "button"):
            controls = frame.get_by_role(role, name=_APPLY_CONTROL)
            for index in range(await controls.count()):
                control = controls.nth(index)
                if not await control.is_visible():
                    continue
                href = await control.get_attribute("href") if role == "link" else None
                external = False
                if href:
                    target_host = urlsplit(href).hostname
                    current_host = urlsplit(page.url).hostname
                    external = bool(
                        target_host
                        and current_host
                        and _registrable_domain(target_host)
                        != _registrable_domain(current_host)
                    )
                candidates.append((external, control, href))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, control, href = candidates[0]
    if href and urlsplit(href).scheme in {"http", "https"}:
        await page.goto(href, wait_until="domcontentloaded", timeout=30_000)
        logger.info("autofill hop via external apply link: host=%s", urlsplit(page.url).hostname)
        return await _wait_for_destination(page)

    from playwright.async_api import TimeoutError as PlaywrightTimeoutError

    context = page.context
    try:
        async with context.expect_page(timeout=1_500) as page_info:
            await control.click(timeout=ACTION_TIMEOUT_MS)
        destination = await page_info.value
        logger.info("autofill hop via new page: host=%s", urlsplit(destination.url).hostname)
        return await _wait_for_destination(destination)
    except PlaywrightTimeoutError:
        await _wait_for_destination(page)
        logger.info("autofill hop via same page: host=%s", urlsplit(page.url).hostname)
        return page


async def _resolve_application_entry(
    page: Any,
) -> tuple[Any, list[FormField], list[FormField], str | None]:
    for hop in range(5):
        if hop == 0 and _is_aggregator(urlsplit(page.url).hostname or ""):
            logger.info("autofill aggregator listing detected: host=%s", urlsplit(page.url).hostname)
        frames = list(page.frames)
        fields, discovered = await _application_form_fields(frames)
        reason = await _blocker_reason(frames, page.url)
        if reason:
            logger.info(
                "autofill stopped at hop=%d reason=%s host=%s",
                hop,
                reason,
                urlsplit(page.url).hostname,
            )
            return page, [], discovered, reason
        if fields:
            logger.info(
                "autofill application form found at hop=%d host=%s",
                hop,
                urlsplit(page.url).hostname,
            )
            return page, fields, discovered, None
        if hop == 4:
            break
        destination = await _activate_apply_control(page, frames)
        if destination is None:
            break
        page = destination
    reason = f"No application form found on {page.url}"
    logger.info(
        "autofill stopped: reason=no_application_form host=%s",
        urlsplit(page.url).hostname,
    )
    return page, [], await _discover_fields(list(page.frames)), reason


def _needs_input(field_info: FormField) -> NeedsInputField:
    return NeedsInputField(
        label=field_info.label,
        type=field_info.field_type,
        options=[label for label, _ in field_info.options],
        required=field_info.required,
    )


async def _fill_fields(
    fields: list[FormField],
    mapping: dict[str, str],
    profile: ResumeProfile,
    answers: dict[str, str | bool | float],
    resume_path: str | None,
    approved_cover_letter: str | None,
) -> tuple[list[FilledField], list[NeedsInputField]]:
    filled: list[FilledField] = []
    needs_input: list[NeedsInputField] = []
    for field_info in fields:
        if not field_info.visible:
            continue
        key = mapping.get(field_info.label) or _deterministic_key(field_info)
        sensitive = _sensitive_question(field_info)
        if field_info.field_type == "checkbox":
            if key and key in answers:
                answer = answers[key]
                if answer is True:
                    await field_info.controls[0].check(timeout=ACTION_TIMEOUT_MS)
                    filled.append(FilledField(label=field_info.label, key=key))
                elif answer is False:
                    await field_info.controls[0].uncheck(timeout=ACTION_TIMEOUT_MS)
                    if field_info.required:
                        needs_input.append(_needs_input(field_info))
                    else:
                        filled.append(FilledField(label=field_info.label, key=key))
                elif isinstance(answer, str):
                    checked = _checkbox_answer(field_info, answer)
                    if checked is None:
                        needs_input.append(_needs_input(field_info))
                    elif checked:
                        await field_info.controls[0].check(timeout=ACTION_TIMEOUT_MS)
                        filled.append(FilledField(label=field_info.label, key=key))
                    else:
                        await field_info.controls[0].uncheck(timeout=ACTION_TIMEOUT_MS)
                        if field_info.required:
                            needs_input.append(_needs_input(field_info))
                        else:
                            filled.append(FilledField(label=field_info.label, key=key))
                else:
                    needs_input.append(_needs_input(field_info))
            else:
                needs_input.append(_needs_input(field_info))
            continue
        if key == "cover_letter" and not approved_cover_letter:
            needs_input.append(_needs_input(field_info))
            continue
        value = (
            approved_cover_letter
            if key == "cover_letter"
            else _value_for_key(key, profile, answers, resume_path) if key else None
        )
        if field_info.field_type == "file":
            if value and Path(value).is_file():
                await field_info.controls[0].set_input_files(value, timeout=ACTION_TIMEOUT_MS)
                filled.append(FilledField(label=field_info.label, key=key or "resume_path"))
            elif field_info.required or re.search(r"\b(resume|cv)\b", field_info.label, re.I):
                needs_input.append(_needs_input(field_info))
            continue
        if field_info.field_type in {"radio", "select-one", "select"}:
            if value is None and field_info.field_type == "select":
                country = profile.country
                if re.search(r"country code|calling code|dial code", field_info.label, re.I):
                    value = country
            selected_value = _option_value(key, answers.get(key)) if key in answers else value
            selected = (
                _option_match(selected_value, field_info.options)
                if selected_value
                else None
            )
            if selected is not None:
                if field_info.field_type == "radio":
                    index = field_info.options.index(selected)
                    await field_info.controls[index].check(timeout=ACTION_TIMEOUT_MS)
                else:
                    await field_info.controls[0].select_option(
                        label=selected[0], timeout=ACTION_TIMEOUT_MS
                    )
                filled.append(FilledField(label=field_info.label, key=key or "country"))
            elif field_info.required or sensitive:
                needs_input.append(_needs_input(field_info))
            continue
        if value:
            if field_info.field_type == "tel":
                # Preserve the phone exactly as stored by the candidate.
                value = profile.phone if key == "phone" and profile.phone else value
            await field_info.controls[0].press_sequentially(
                value, delay=30, timeout=ACTION_TIMEOUT_MS
            )
            filled.append(FilledField(label=field_info.label, key=key or ""))
        elif field_info.required or field_info.field_type == "textarea" or sensitive:
            needs_input.append(_needs_input(field_info))
    return filled, needs_input


def _checkbox_answer(field_info: FormField, answer: str) -> bool | None:
    normalized_answer = _normalize(answer)
    affirmative = {
        "yes",
        "true",
        "agree",
        "i agree",
        "authorized",
        "i am",
        "opt in",
    }
    negative = {
        "no",
        "false",
        "disagree",
        "i do not agree",
        "not applicable",
        "prefer not to say",
        "opt out",
    }
    if normalized_answer in affirmative:
        return True
    if normalized_answer in negative:
        return False
    return normalized_answer == _normalize(field_info.label)


async def _required_fields_missing(fields: list[FormField]) -> list[NeedsInputField]:
    missing: list[NeedsInputField] = []
    for field_info in fields:
        if not field_info.visible or not field_info.required:
            continue
        if field_info.field_type == "radio":
            empty = not any(
                [await control.is_checked() for control in field_info.controls]
            )
        elif field_info.field_type == "checkbox":
            empty = not await field_info.controls[0].is_checked()
        elif field_info.field_type == "file":
            empty = not await field_info.controls[0].evaluate(
                "element => Boolean(element.files && element.files.length)"
            )
        else:
            empty = not (await field_info.controls[0].input_value()).strip()
        if empty:
            missing.append(_needs_input(field_info))
    return missing


def _form_scope(field_info: FormField) -> Any:
    if not field_info.form_key.startswith("form-"):
        return field_info.frame
    try:
        form_index = int(field_info.form_key.removeprefix("form-"))
    except ValueError:
        return field_info.frame
    return field_info.frame.locator("form").nth(form_index)


async def _next_button(fields: list[FormField]) -> Any | None:
    pattern = re.compile(r"^\s*(next|continue)(?:\s+to\s+\w+)?\s*$", re.I)
    if not fields:
        return None
    scope = _form_scope(fields[0])
    for role in ("button", "link"):
        candidates = scope.get_by_role(role, name=pattern)
        for index in range(await candidates.count()):
            candidate = candidates.nth(index)
            if await candidate.is_visible():
                return candidate
    return None


async def _submit_button(fields: list[FormField]) -> Any | None:
    pattern = re.compile(r"^\s*(submit|submit application|send application|finish application)\s*$", re.I)
    if not fields:
        return None
    scope = _form_scope(fields[0])
    candidates = scope.get_by_role("button", name=pattern)
    for index in range(await candidates.count()):
        candidate = candidates.nth(index)
        if await candidate.is_visible():
            return candidate
    candidates = scope.locator('input[type="submit"]')
    if await candidates.count():
        return candidates.first
    return None


async def _confirmation_detected(page: Any) -> bool:
    text = "\n".join(
        [
            await frame.locator("body").inner_text(timeout=ACTION_TIMEOUT_MS)
            for frame in page.frames
        ]
    )
    return bool(
        re.search(
            r"thank you|application (?:has been )?(?:received|submitted)|"
            r"received your application|successfully submitted",
            text,
            re.I,
        )
    )


def _review_readiness_reason(
    fields: list[FormField], filled: list[FilledField]
) -> str | None:
    if len(filled) < 3:
        return "Fewer than three application fields were filled."
    has_resume_input = any(
        field_info.field_type == "file"
        and _deterministic_key(field_info) == "resume_path"
        for field_info in fields
    )
    filled_keys = {field_info.key for field_info in filled}
    if has_resume_input:
        if "resume_path" not in filled_keys:
            return "The resume upload was not completed."
        return None
    has_name = "name" in filled_keys or {
        "first_name", "last_name"
    }.issubset(filled_keys)
    if has_name and {"email", "phone"}.issubset(filled_keys):
        return None
    return "Without a resume upload, name, email, and phone must all be filled."


async def _save_debug_artifacts(
    page: Any | None,
    report: AutofillReport,
    *,
    uploads_dir: str,
    session_id: str,
    job_id: str,
) -> None:
    directory = Path(uploads_dir) / session_id / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    report.screenshot_path = str(directory / f"{job_id}.png")
    report.debug_screenshot_path = str(directory / f"{job_id}-debug.png")
    report.debug_labels_path = str(directory / f"{job_id}-labels.json")
    try:
        Path(report.debug_labels_path).write_text(
            json.dumps(report.discovered_form_labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if page is not None:
            await page.screenshot(path=report.screenshot_path, full_page=True)
            await page.screenshot(path=report.debug_screenshot_path, full_page=True)
    except Exception as exc:
        logger.warning("autofill debug artifact save failed: %s", type(exc).__name__)
        report.blocker_reason = (
            f"{report.blocker_reason or 'Application stopped.'} "
            f"Debug artifacts could not be saved ({type(exc).__name__})."
        )


async def _save_blocked_debug_artifacts(
    report: AutofillReport,
    *,
    uploads_dir: str,
    session_id: str,
    job_id: str,
) -> None:
    directory = Path(uploads_dir) / session_id / "previews"
    directory.mkdir(parents=True, exist_ok=True)
    report.screenshot_path = str(directory / f"{job_id}.png")
    report.debug_screenshot_path = str(directory / f"{job_id}-debug.png")
    report.debug_labels_path = str(directory / f"{job_id}-labels.json")
    try:
        Path(report.debug_labels_path).write_text("[]\n", encoding="utf-8")
        from playwright.async_api import async_playwright

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_content(
                "<!doctype html><html><body><h1>Manual application required</h1>"
                f"<p>{report.blocker_reason or 'Automation stopped.'}</p></body></html>"
            )
            await page.screenshot(path=report.screenshot_path, full_page=True)
            await page.screenshot(path=report.debug_screenshot_path, full_page=True)
            await browser.close()
    except Exception as exc:
        logger.warning("blocked autofill debug artifact save failed: %s", type(exc).__name__)
        report.blocker_reason = (
            f"{report.blocker_reason or 'Application stopped.'} "
            f"Debug artifacts could not be saved ({type(exc).__name__})."
        )


async def _reserve_run(
    domain: str,
    session_id: str,
    mode: AutofillMode,
    job_id: str = "",
    *,
    count_submission: bool = True,
    review_submit_exemption: bool = False,
) -> str | None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + DOMAIN_INTERVAL.total_seconds()
    while True:
        now = datetime.now(timezone.utc)
        wait_seconds = 0.0
        async with _RATE_LIMIT_LOCK:
            review_at = _REVIEW_RUNS.get((session_id, job_id, domain))
            review_exemption = bool(
                mode == "submit"
                and (
                    review_submit_exemption
                    or (
                        review_at
                        and now - review_at <= timedelta(minutes=15)
                    )
                )
            )
            previous = _DOMAIN_RUNS.get(domain)
            if (
                previous
                and now - previous < DOMAIN_INTERVAL
                and not review_exemption
            ):
                wait_seconds = (DOMAIN_INTERVAL - (now - previous)).total_seconds()
            else:
                if mode == "submit" and count_submission:
                    recent = [
                        value
                        for value in _SESSION_SUBMISSIONS.get(session_id, [])
                        if now - value < SESSION_HOUR
                    ]
                    if len(recent) >= 5:
                        return (
                            "The maximum of five real application attempts per session "
                            "per hour was reached."
                        )
                    recent.append(now)
                    _SESSION_SUBMISSIONS[session_id] = recent
                _DOMAIN_RUNS[domain] = now
                return None
        remaining = deadline - loop.time()
        if remaining <= 0:
            return "The domain remained rate-limited after waiting up to 30 seconds."
        await asyncio.sleep(min(wait_seconds, remaining))


def _mark_review_run(session_id: str, job_id: str, domains: set[str]) -> None:
    now = datetime.now(timezone.utc)
    for domain in domains:
        _REVIEW_RUNS[(session_id, job_id, domain)] = now


async def fill_application_form(
    url: str,
    profile: ResumeProfile,
    answers: dict[str, str | bool | float],
    resume_path: str | None,
    mode: AutofillMode,
    job: Job,
    *,
    session_id: str = "local-session",
    uploads_dir: str = "uploads",
    model: Any | None = None,
    real_apply_enabled: bool | None = None,
    headed: bool = False,
    approved_cover_letter: str | None = None,
    review_submit_exemption: bool = False,
    keep_open: bool = False,
    on_report: Callable[[AutofillReport], None] | None = None,
) -> AutofillReport:
    parsed_url = urlsplit(url)
    hostname = parsed_url.hostname or ""
    safe_identifiers = _safe_file_component(session_id) and _safe_file_component(job.id)
    if parsed_url.scheme not in {"http", "https"} or not hostname:
        report = AutofillReport(
            status="failed",
            blocker_reason="A valid HTTP(S) application URL is required.",
            final_url=url,
        )
        if safe_identifiers:
            await _save_blocked_debug_artifacts(
                report,
                uploads_dir=uploads_dir,
                session_id=session_id,
                job_id=job.id,
            )
        return report
    if _blocked_domain(hostname):
        report = AutofillReport(
            status="needs_manual",
            blocker_reason="This career portal does not permit automated applications. Continue manually.",
            final_url=url,
            detected_ats=_detected_ats(url),
        )
        if safe_identifiers:
            await _save_blocked_debug_artifacts(
                report,
                uploads_dir=uploads_dir,
                session_id=session_id,
                job_id=job.id,
            )
        return report
    if real_apply_enabled is None:
        real_apply_enabled = get_settings().real_apply_enabled
    if mode == "submit" and not real_apply_enabled:
        mode = "dry_run"
    if not safe_identifiers:
        return AutofillReport(
            status="failed",
            blocker_reason="The session or job identifier is not a safe file name.",
            final_url=url,
        )
    rate_reason = await _reserve_run(
        hostname.casefold(),
        session_id,
        mode,
        job.id,
        review_submit_exemption=review_submit_exemption,
    )
    if rate_reason:
        return AutofillReport(
            status="retry_later",
            blocker_reason=rate_reason,
            final_url=url,
            detected_ats=_detected_ats(url),
        )

    playwright = None
    browser = None
    page = None
    submit_clicked = False
    reserved_domains = {hostname.casefold()}
    report = AutofillReport(status="failed", final_url=url)
    try:
        from playwright.async_api import async_playwright

        async with asyncio.timeout(FORM_TIMEOUT_SECONDS):
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=not headed)
            page = await browser.new_page()
            page.set_default_timeout(ACTION_TIMEOUT_MS)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page, fields, discovered, reason = await _resolve_application_entry(page)
            report.final_url = page.url
            report.detected_ats = _detected_ats(page.url)
            report.discovered_form_labels = [field_info.label for field_info in discovered]
            if reason:
                report.status = "needs_manual"
                report.blocker_reason = reason
                return report
            final_host = (urlsplit(page.url).hostname or "").casefold()
            if final_host and final_host not in reserved_domains:
                rate_reason = await _reserve_run(
                    final_host,
                    session_id,
                    mode,
                    job.id,
                    count_submission=False,
                    review_submit_exemption=review_submit_exemption,
                )
                if rate_reason:
                    report.status = "retry_later"
                    report.blocker_reason = rate_reason
                    return report
                reserved_domains.add(final_host)
            all_filled: list[FilledField] = []
            all_needs_input: list[NeedsInputField] = []
            form_identity = (fields[0].frame.url, fields[0].form_key)
            form_fields_seen = list(fields)
            for step in range(8):
                mapping = await _field_mapping(fields, model)
                filled, missing = await _fill_fields(
                    fields,
                    mapping,
                    profile,
                    answers,
                    resume_path,
                    approved_cover_letter,
                )
                all_filled.extend(filled)
                all_needs_input.extend(missing)
                all_needs_input.extend(await _required_fields_missing(fields))
                unique_missing: dict[tuple[str, str], NeedsInputField] = {}
                for item in all_needs_input:
                    unique_missing[(item.label, item.type)] = item
                all_needs_input = list(unique_missing.values())
                next_control = await _next_button(fields)
                if next_control is None:
                    break
                await next_control.click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(350)
                reason = await _blocker_reason(list(page.frames), page.url)
                if reason:
                    report.status = "needs_manual"
                    report.blocker_reason = reason
                    report.final_url = page.url
                    report.filled = all_filled
                    report.needs_input = all_needs_input
                    return report
                step_fields = await _discover_fields(list(page.frames))
                fields = [
                    field_info
                    for field_info in step_fields
                    if (field_info.frame.url, field_info.form_key) == form_identity
                ]
                form_fields_seen.extend(fields)
                report.discovered_form_labels.extend(
                    field_info.label
                    for field_info in step_fields
                    if field_info.label not in report.discovered_form_labels
                )
                if not fields:
                    break
            else:
                report.status = "needs_manual"
                report.blocker_reason = "The form exceeded the eight-step safety limit."
                report.final_url = page.url
                return report

            screenshot_directory = Path(uploads_dir) / session_id / "previews"
            screenshot_directory.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_directory / f"{job.id}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            report.filled = all_filled
            report.needs_input = all_needs_input
            report.screenshot_path = str(screenshot_path)
            report.final_url = page.url
            report.detected_ats = _detected_ats(page.url)
            if all_needs_input:
                report.status = "needs_input"
                report.blocker_reason = "Some answers need your review before submission."
                return report
            readiness_reason = _review_readiness_reason(form_fields_seen, all_filled)
            if readiness_reason:
                report.status = "needs_manual"
                report.blocker_reason = readiness_reason
                return report
            if mode == "dry_run":
                report.status = "dry_run"
                if not real_apply_enabled:
                    report.blocker_reason = (
                        "REAL_APPLY_ENABLED=false; the form was not submitted."
                    )
                return report
            if mode == "review":
                report.status = "review_ready"
                return report

            submit = await _submit_button(fields)
            if submit is None:
                report.status = "needs_manual"
                report.blocker_reason = "No final submit button was found. Verify the form manually."
                return report
            await submit.click(timeout=ACTION_TIMEOUT_MS)
            submit_clicked = True
            await page.wait_for_timeout(1_500)
            report.final_url = page.url
            if await _confirmation_detected(page):
                report.status = "submitted"
                return report
            report.status = "failed"
            report.blocker_reason = "Submission could not be confirmed; please verify manually."
            return report
    except TimeoutError:
        report.status = "retry_later"
        report.blocker_reason = "The application page timed out; please verify manually."
        if page is not None:
            report.final_url = page.url
        return report
    except Exception as exc:
        exception_name = type(exc).__name__
        exception_message = str(exc).casefold()
        transient_browser_error = exception_name in {
            "BrowserClosedError",
            "BrowserDisconnectedError",
            "TargetClosedError",
        } or any(
            marker in exception_message
            for marker in (
                "browser disconnected",
                "browser has been closed",
                "target page, context or browser has been closed",
                "connection closed",
                "target closed",
            )
        )
        report.status = (
            "retry_later"
            if transient_browser_error
            else "failed" if submit_clicked
            else "needs_manual"
        )
        report.blocker_reason = f"The application form could not be completed ({exception_name})."
        if page is not None:
            report.final_url = page.url
        logger.warning("autofill run failed: %s", type(exc).__name__)
        return report
    finally:
        try:
            report.detected_ats = _detected_ats(report.final_url or url)
            if report.status == "review_ready":
                _mark_review_run(
                    session_id,
                    job.id,
                    reserved_domains | {
                        (urlsplit(report.final_url or "").hostname or "").casefold()
                    },
                )
            if report.status in {"needs_manual", "failed"} and _safe_file_component(session_id) and _safe_file_component(job.id):
                await _save_debug_artifacts(
                    page,
                    report,
                    uploads_dir=uploads_dir,
                    session_id=session_id,
                    job_id=job.id,
                )
            if keep_open:
                if on_report is not None:
                    on_report(report)
                if browser is not None:
                    try:
                        await asyncio.to_thread(input, "Press Enter to close the browser...")
                    except EOFError:
                        pass
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception as exc:
                    logger.warning("autofill browser close failed: %s", type(exc).__name__)
                    if report.status not in {"submitted", "review_ready", "needs_input"}:
                        report.status = "retry_later"
                        report.blocker_reason = "The browser could not close cleanly; try again later."
            if playwright is not None:
                try:
                    await playwright.stop()
                except Exception as exc:
                    logger.warning("autofill driver stop failed: %s", type(exc).__name__)
                    if report.status not in {"submitted", "review_ready", "needs_input"}:
                        report.status = "retry_later"
                        report.blocker_reason = "The browser driver could not stop cleanly; try again later."
