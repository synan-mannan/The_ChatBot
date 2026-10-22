from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, RootModel, ValidationError

from server.app.models.schemas import Job, ResumeProfile

AutofillMode = Literal["dry_run", "review", "submit"]
BLOCKED_DOMAINS = {"linkedin.com", "indeed.com", "naukri.com", "glassdoor.com"}
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
}
ALLOWED_MAPPING_KEYS = ALLOWED_PROFILE_KEYS | ALLOWED_ANSWER_KEYS
FORM_TIMEOUT_SECONDS = 30
ACTION_TIMEOUT_MS = 8_000
DOMAIN_INTERVAL = timedelta(seconds=30)
SESSION_HOUR = timedelta(hours=1)
_RATE_LIMIT_LOCK = asyncio.Lock()
_DOMAIN_RUNS: dict[str, datetime] = {}
_SESSION_SUBMISSIONS: dict[str, list[datetime]] = {}


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
    ]
    filled: list[FilledField] = []
    needs_input: list[NeedsInputField] = []
    blocker_reason: str | None = None
    screenshot_path: str | None = None
    final_url: str | None = None


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


class FieldMappingError(RuntimeError):
    """The optional AI field-mapping response could not be validated."""


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
        ("name", (r"\b(full|preferred) name\b", r"^name$")),
    )
    for key, patterns in rules:
        if any(re.search(pattern, label) for pattern in patterns):
            return key
    return None


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
    if key == "cover_letter":
        return None
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
            if not await control.is_visible():
                continue
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
                    label: labels[0] || element.getAttribute('aria-label') ||
                      element.getAttribute('placeholder') || text(fieldset?.querySelector('legend')) ||
                      preceding || '',
                    group: text(fieldset?.querySelector('legend')) || '',
                    required: element.required || element.getAttribute('aria-required') === 'true',
                    options: element.tagName.toLowerCase() === 'select'
                      ? Array.from(element.options).map(option => ({
                          label: text(option), value: option.value
                        })) : []
                  };
                }"""
            )
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
                    metadata=str(info["group"]),
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


async def _click_apply_if_needed(page: Any, frames: list[Any]) -> list[Any]:
    if await _discover_fields(frames):
        return frames
    pattern = re.compile(r"^\s*apply(?:\s+now)?(?:\s+for\s+this\s+job)?\s*$", re.I)
    for frame in frames:
        for role in ("button", "link"):
            candidates = frame.get_by_role(role, name=pattern)
            for index in range(await candidates.count()):
                candidate = candidates.nth(index)
                if await candidate.is_visible():
                    await candidate.click(timeout=ACTION_TIMEOUT_MS)
                    await page.wait_for_timeout(500)
                    return list(page.frames)
    return frames


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
) -> tuple[list[FilledField], list[NeedsInputField]]:
    filled: list[FilledField] = []
    needs_input: list[NeedsInputField] = []
    for field_info in fields:
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
                else:
                    needs_input.append(_needs_input(field_info))
            else:
                needs_input.append(_needs_input(field_info))
            continue
        if key == "cover_letter":
            needs_input.append(_needs_input(field_info))
            continue
        value = _value_for_key(key, profile, answers, resume_path) if key else None
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


async def _required_fields_missing(fields: list[FormField]) -> list[NeedsInputField]:
    missing: list[NeedsInputField] = []
    for field_info in fields:
        if not field_info.required:
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


async def _next_button(frames: list[Any]) -> Any | None:
    pattern = re.compile(r"^\s*(next|continue)(?:\s+to\s+\w+)?\s*$", re.I)
    for frame in frames:
        for role in ("button", "link"):
            candidates = frame.get_by_role(role, name=pattern)
            for index in range(await candidates.count()):
                candidate = candidates.nth(index)
                if await candidate.is_visible():
                    return candidate
    return None


async def _submit_button(frames: list[Any]) -> Any | None:
    pattern = re.compile(r"^\s*(submit|submit application|send application|finish application)\s*$", re.I)
    for frame in frames:
        for role in ("button",):
            candidates = frame.get_by_role(role, name=pattern)
            for index in range(await candidates.count()):
                candidate = candidates.nth(index)
                if await candidate.is_visible():
                    return candidate
        candidates = frame.locator('input[type="submit"]')
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


async def _reserve_run(domain: str, session_id: str, mode: AutofillMode) -> str | None:
    now = datetime.now(timezone.utc)
    async with _RATE_LIMIT_LOCK:
        previous = _DOMAIN_RUNS.get(domain)
        if previous and now - previous < DOMAIN_INTERVAL:
            wait_seconds = int((DOMAIN_INTERVAL - (now - previous)).total_seconds()) + 1
            return f"This domain is rate-limited. Try again in about {wait_seconds} seconds."
        if mode == "submit":
            recent = [
                value
                for value in _SESSION_SUBMISSIONS.get(session_id, [])
                if now - value < SESSION_HOUR
            ]
            if len(recent) >= 5:
                return "The maximum of five real application attempts per session per hour was reached."
            recent.append(now)
            _SESSION_SUBMISSIONS[session_id] = recent
        _DOMAIN_RUNS[domain] = now
    return None


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
    headed: bool = False,
) -> AutofillReport:
    parsed_url = urlsplit(url)
    hostname = parsed_url.hostname or ""
    if parsed_url.scheme not in {"http", "https"} or not hostname:
        return AutofillReport(status="failed", blocker_reason="A valid HTTP(S) application URL is required.")
    if _blocked_domain(hostname):
        return AutofillReport(
            status="needs_manual",
            blocker_reason="This career portal does not permit automated applications. Continue manually.",
            final_url=url,
        )
    if not _safe_file_component(session_id) or not _safe_file_component(job.id):
        return AutofillReport(
            status="failed",
            blocker_reason="The session or job identifier is not a safe file name.",
            final_url=url,
        )
    rate_reason = await _reserve_run(hostname.casefold(), session_id, mode)
    if rate_reason:
        return AutofillReport(status="needs_manual", blocker_reason=rate_reason, final_url=url)

    playwright = None
    browser = None
    report = AutofillReport(status="failed", final_url=url)
    try:
        from playwright.async_api import async_playwright

        async with asyncio.timeout(FORM_TIMEOUT_SECONDS):
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=not headed)
            page = await browser.new_page()
            page.set_default_timeout(ACTION_TIMEOUT_MS)
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            frames = list(page.frames)
            reason = await _blocker_reason(frames, page.url)
            if reason:
                report.status = "needs_manual"
                report.blocker_reason = reason
                report.final_url = page.url
                return report

            frames = await _click_apply_if_needed(page, frames)
            all_filled: list[FilledField] = []
            all_needs_input: list[NeedsInputField] = []
            for _ in range(8):
                frames = list(page.frames)
                reason = await _blocker_reason(frames, page.url)
                if reason:
                    report.status = "needs_manual"
                    report.blocker_reason = reason
                    report.final_url = page.url
                    return report
                fields = await _discover_fields(frames)
                if not fields:
                    if not all_filled:
                        report.status = "needs_manual"
                        report.blocker_reason = "No visible application form fields were found."
                        report.final_url = page.url
                        return report
                    break
                mapping = await _field_mapping(fields, model)
                filled, missing = await _fill_fields(
                    fields, mapping, profile, answers, resume_path
                )
                all_filled.extend(filled)
                all_needs_input.extend(missing)
                all_needs_input.extend(await _required_fields_missing(fields))
                unique_missing: dict[tuple[str, str], NeedsInputField] = {}
                for item in all_needs_input:
                    unique_missing[(item.label, item.type)] = item
                all_needs_input = list(unique_missing.values())
                next_control = await _next_button(frames)
                if next_control is None:
                    break
                await next_control.click(timeout=ACTION_TIMEOUT_MS)
                await page.wait_for_timeout(350)
            else:
                report.status = "needs_manual"
                report.blocker_reason = "The form exceeded the eight-step safety limit."
                report.final_url = page.url
                return report

            screenshot_directory = (
                Path(uploads_dir) / session_id / "previews"
            )
            screenshot_directory.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_directory / f"{job.id}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            report.filled = all_filled
            report.needs_input = all_needs_input
            report.screenshot_path = str(screenshot_path)
            report.final_url = page.url
            if all_needs_input:
                report.status = "needs_input"
                report.blocker_reason = "Some answers need your review before submission."
                return report
            if mode == "dry_run":
                report.status = "dry_run"
                return report
            if mode == "review":
                report.status = "review_ready"
                return report

            submit = await _submit_button(list(page.frames))
            if submit is None:
                report.status = "failed"
                report.blocker_reason = "No final submit button was found. Verify the form manually."
                return report
            await submit.click(timeout=ACTION_TIMEOUT_MS)
            await page.wait_for_timeout(1_500)
            report.final_url = page.url
            if await _confirmation_detected(page):
                report.status = "submitted"
                return report
            report.status = "failed"
            report.blocker_reason = "Submission could not be confirmed; please verify manually."
            return report
    except TimeoutError:
        report.status = "failed"
        report.blocker_reason = "The application page timed out; please verify manually."
        return report
    except Exception as exc:
        report.status = "failed"
        report.blocker_reason = (
            f"The application form could not be completed ({type(exc).__name__}); "
            "please verify manually."
        )
        return report
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                report.status = "failed"
                report.blocker_reason = "The browser could not close cleanly; verify manually."
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                report.status = "failed"
                report.blocker_reason = "The browser driver could not stop cleanly; verify manually."
