from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from server.app.config import get_settings
from server.app.models.schemas import Job, ResumeProfile

ApplyMethod = Literal["email", "greenhouse", "lever", "manual"]
ApplyStatus = Literal["submitted", "dry_run", "needs_manual", "failed"]
FORM_TIMEOUT_SECONDS = 45
ACTION_TIMEOUT_MS = 8_000


@dataclass(frozen=True)
class ApplyResult:
    method: ApplyMethod
    status: ApplyStatus
    detail: str
    apply_url: str | None


def _real_apply_enabled() -> bool:
    return get_settings().real_apply_enabled


def pick_method(job: Job) -> ApplyMethod:
    if job.contact_email:
        return "email"
    host = urlsplit(str(job.source_link)).hostname or ""
    host = host.casefold()
    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    return "manual"


def describe_plan(job: Job) -> str:
    method = pick_method(job)
    if method == "email":
        return f"email your resume to {job.contact_email}"
    if method == "greenhouse":
        return "fill and submit the Greenhouse application form"
    if method == "lever":
        return "fill and submit the Lever application form"
    return "open the employer's page (I can't submit this one automatically)"


def _resume_attachment(resume_path: str) -> tuple[bytes, str, str] | None:
    path = Path(resume_path)
    suffix = path.suffix.casefold()
    mime_types = {
        ".pdf": ("application", "pdf"),
        ".docx": (
            "application",
            "vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
    }
    mime_type = mime_types.get(suffix)
    if mime_type is None:
        return None
    return path.read_bytes(), mime_type[0], mime_type[1]


def _candidate_name(profile: ResumeProfile) -> tuple[str, str]:
    name = (profile.name or "").strip()
    if not name:
        return "", ""
    parts = name.split()
    return parts[0], " ".join(parts[1:])


async def apply_by_email(
    job: Job,
    profile: ResumeProfile,
    resume_path: str,
) -> ApplyResult:
    try:
        attachment = await asyncio.to_thread(_resume_attachment, resume_path)
        if attachment is None:
            return ApplyResult(
                "email",
                "failed",
                "The saved resume must be a PDF or DOCX file. Please upload it again.",
                str(job.source_link),
            )
        content, maintype, subtype = attachment
        name = profile.name or "Job applicant"
        message = EmailMessage()
        settings = get_settings()
        message["From"] = settings.smtp_from or "AIJob <dry-run@example.invalid>"
        message["To"] = job.contact_email or ""
        message["Subject"] = f"Application: {job.title} - {name}"
        if profile.email:
            message["Reply-To"] = profile.email
        message.set_content(
            f"Dear Hiring Team,\n\n"
            f"I am writing to apply for the {job.title} position. "
            f"My resume is attached for your consideration.\n\n"
            f"Name: {name}\n"
            f"Email: {profile.email or 'Not provided'}\n"
            f"Phone: {profile.phone or 'Not provided'}\n\n"
            "Thank you for your time.\n"
        )
        filename = f"resume{Path(resume_path).suffix.casefold()}"
        message.add_attachment(
            content,
            maintype=maintype,
            subtype=subtype,
            filename=filename,
        )
        if not _real_apply_enabled():
            return ApplyResult(
                "email",
                "dry_run",
                f"Dry run only: prepared an application email to {job.contact_email}; nothing was sent.",
                str(job.source_link),
            )

        smtp_host = settings.smtp_host.strip()
        smtp_user = settings.smtp_user
        smtp_password = settings.smtp_pass
        smtp_from = settings.smtp_from.strip()
        smtp_port = settings.smtp_port
        if not smtp_host or not smtp_from:
            return ApplyResult(
                "email",
                "failed",
                "SMTP_HOST and SMTP_FROM are required for real email applications. Nothing was sent.",
                str(job.source_link),
            )

        import aiosmtplib

        await asyncio.wait_for(
            aiosmtplib.send(
                message,
                hostname=smtp_host,
                port=smtp_port,
                username=smtp_user or None,
                password=smtp_password or None,
                start_tls=True,
                timeout=30,
            ),
            timeout=FORM_TIMEOUT_SECONDS,
        )
        return ApplyResult(
            "email",
            "submitted",
            f"Your application email to {job.contact_email} was sent.",
            str(job.source_link),
        )
    except Exception as exc:
        return ApplyResult(
            "email",
            "failed",
            f"Email delivery failed ({type(exc).__name__}). Please verify manually at {job.source_link}.",
            str(job.source_link),
        )


def _lever_apply_url(url: str) -> str:
    parts = urlsplit(url)
    if not parts.path.rstrip("/").endswith("/apply"):
        path = f"{parts.path.rstrip('/')}/apply"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return url


async def _has_captcha(page: Any) -> bool:
    captcha = page.locator(
        'iframe[src*="recaptcha"], iframe[src*="hcaptcha"], '
        '.h-captcha, .g-recaptcha, iframe[title*="captcha" i], '
        '[name="g-recaptcha-response"], [name="h-captcha-response"]'
    )
    return await captcha.count() > 0


async def _fill_first_present(
    page: Any, selectors: tuple[str, ...], value: str
) -> bool:
    if not value:
        return False
    for selector in selectors:
        locator = page.locator(selector)
        if await locator.count() > 0:
            await locator.first.fill(value, timeout=ACTION_TIMEOUT_MS)
            return True
    return False


async def _required_fields_left(page: Any) -> int:
    required = page.locator("input[required], textarea[required], select[required]")
    remaining = 0
    for index in range(await required.count()):
        field = required.nth(index)
        input_type = (await field.get_attribute("type") or "").casefold()
        if input_type in {"file", "checkbox"}:
            continue
        if input_type == "radio":
            if not await field.is_checked():
                remaining += 1
            continue
        if not (await field.input_value(timeout=ACTION_TIMEOUT_MS)).strip():
            remaining += 1
    return remaining


async def apply_via_form(
    job: Job,
    profile: ResumeProfile,
    resume_path: str,
    ats: Literal["greenhouse", "lever"],
) -> ApplyResult:
    browser = None
    playwright = None
    result: ApplyResult | None = None
    apply_url = str(job.source_link)
    if ats == "lever":
        apply_url = _lever_apply_url(apply_url)
    try:
        from playwright.async_api import async_playwright

        async with asyncio.timeout(FORM_TIMEOUT_SECONDS):
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(ACTION_TIMEOUT_MS)
            await page.goto(apply_url, wait_until="domcontentloaded")

            # Field IDs were verified against a live Anthropic Greenhouse posting; ATS selectors can drift.
            if await _has_captcha(page):
                result = ApplyResult(
                    ats,
                    "needs_manual",
                    f"A CAPTCHA is present. I stopped without attempting to solve it. Apply manually at {apply_url}.",
                    apply_url,
                )
            else:
                first_name, last_name = _candidate_name(profile)
                await _fill_first_present(
                    page,
                    ('input[name="first_name"]', "#first_name", 'input[id*="first_name"]'),
                    first_name,
                )
                await _fill_first_present(
                    page,
                    ('input[name="last_name"]', "#last_name", 'input[id*="last_name"]'),
                    last_name,
                )
                await _fill_first_present(
                    page,
                    ('input[name="name"]', "#name"),
                    profile.name or "",
                )
                await _fill_first_present(
                    page,
                    ('input[name="email"]', "#email", 'input[type="email"]'),
                    profile.email or "",
                )
                await _fill_first_present(
                    page,
                    ('input[name="phone"]', "#phone", 'input[type="tel"]'),
                    profile.phone or "",
                )
                file_inputs = page.locator('input[type="file"]')
                if await file_inputs.count() == 0:
                    result = ApplyResult(
                        ats,
                        "needs_manual",
                        f"No resume upload field was found. Complete the application manually at {apply_url}.",
                        apply_url,
                    )
                else:
                    await file_inputs.first.set_input_files(resume_path, timeout=ACTION_TIMEOUT_MS)
                    unanswered = await _required_fields_left(page)
                    if unanswered:
                        result = ApplyResult(
                            ats,
                            "needs_manual",
                            f"{unanswered} required custom question(s) need your own answer. "
                            f"Nothing was submitted. Continue manually at {apply_url}.",
                            apply_url,
                        )
                    elif not _real_apply_enabled():
                        result = ApplyResult(
                            ats,
                            "dry_run",
                            f"Dry run only: the {ats.title()} form was prepared; nothing was submitted. "
                            f"Review it manually at {apply_url}.",
                            apply_url,
                        )
                    else:
                        submit = page.locator(
                            'button[type="submit"], input[type="submit"]'
                        )
                        if await submit.count() == 0:
                            result = ApplyResult(
                                ats,
                                "needs_manual",
                                f"No submit control was found. Review the form manually at {apply_url}.",
                                apply_url,
                            )
                        else:
                            await submit.first.click(timeout=ACTION_TIMEOUT_MS)
                            await page.wait_for_timeout(4_000)
                            body_text = await page.locator("body").inner_text(
                                timeout=ACTION_TIMEOUT_MS
                            )
                            if re.search(
                                r"thank you|application has been submitted|received your application",
                                body_text,
                                re.IGNORECASE,
                            ):
                                result = ApplyResult(
                                    ats,
                                    "submitted",
                                    f"The {ats.title()} page confirmed receipt of your application.",
                                    apply_url,
                                )
                            else:
                                result = ApplyResult(
                                    ats,
                                    "failed",
                                    f"I couldn't confirm submission. Please verify manually at {apply_url}.",
                                    apply_url,
                                )
    except TimeoutError:
        result = ApplyResult(
            ats,
            "failed",
            f"The {ats.title()} application timed out. Please verify manually at {apply_url}.",
            apply_url,
        )
    except Exception as exc:
        result = ApplyResult(
            ats,
            "failed",
            f"The {ats.title()} form could not be completed ({type(exc).__name__}). "
            f"Please verify manually at {apply_url}.",
            apply_url,
        )
    finally:
        if browser is not None:
            try:
                await browser.close()
            except Exception:
                if result is None or result.status != "submitted":
                    result = ApplyResult(
                        ats,
                        "failed",
                        f"The {ats.title()} browser session could not close cleanly. "
                        f"Please verify manually at {apply_url}.",
                        apply_url,
                    )
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                if result is None or result.status != "submitted":
                    result = ApplyResult(
                        ats,
                        "failed",
                        f"The {ats.title()} browser driver could not stop cleanly. "
                        f"Please verify manually at {apply_url}.",
                        apply_url,
                    )
    if result is None:
        return ApplyResult(
            ats,
            "failed",
            f"The {ats.title()} application did not complete. Please verify manually at {apply_url}.",
            apply_url,
        )
    return result


async def apply(
    job: Job,
    profile: ResumeProfile,
    resume_path: str | None,
) -> ApplyResult:
    method = pick_method(job)
    if method == "manual":
        return ApplyResult(
            method,
            "needs_manual",
            f"This application was not submitted because this listing requires a manual application. "
            f"Continue at {job.source_link}.",
            str(job.source_link),
        )
    if not resume_path:
        return ApplyResult(
            method,
            "failed",
            "No saved resume file is available. Upload your resume again before applying.",
            str(job.source_link),
        )
    if method == "email":
        return await apply_by_email(job, profile, resume_path)
    return await apply_via_form(job, profile, resume_path, method)
