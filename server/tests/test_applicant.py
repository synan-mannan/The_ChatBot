from __future__ import annotations

import pytest
from unittest.mock import AsyncMock
from email.message import EmailMessage
from types import ModuleType
import sys

from server.app.services import applicant as applicant_service
from server.app.config import Settings
from server.app.models.schemas import Job, ResumeProfile
from server.app.services.applicant import (
    apply,
    apply_by_email,
    apply_via_form,
    describe_plan,
    pick_method,
)


def make_job(
    *,
    contact_email: str | None = None,
    source_link: str = "https://example.com/careers/role",
) -> Job:
    return Job(
        id="17",
        title="Backend Engineer",
        company="Example",
        location="Remote",
        contact_email=contact_email,
        is_remote=True,
        experience_required="Not specified",
        source_link=source_link,
        source="test",
    )


def configure_apply(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = False,
    **smtp: str | int,
) -> None:
    settings = Settings(
        _env_file=None,
        real_apply_enabled=enabled,
        **smtp,
    )
    monkeypatch.setattr(applicant_service, "get_settings", lambda: settings)


class FakeElement:
    def __init__(
        self,
        *,
        input_type: str | None = "text",
        value: str = "",
        attributes: dict[str, str] | None = None,
        checked: bool = False,
    ) -> None:
        self.input_type = input_type
        self.value = value
        self.attributes = attributes or {}
        self.checked = checked


class FakeLocator:
    def __init__(self, page: "FakePage", elements: list[FakeElement]) -> None:
        self.page = page
        self.elements = elements

    @property
    def first(self) -> "FakeLocator":
        return self.nth(0)

    async def count(self) -> int:
        return len(self.elements)

    def nth(self, index: int) -> "FakeLocator":
        return FakeLocator(self.page, self.elements[index:index + 1])

    async def fill(self, value: str, **_: object) -> None:
        self.elements[0].value = value

    async def set_input_files(self, path: str, **_: object) -> None:
        self.page.uploaded_file = path

    async def get_attribute(self, name: str) -> str | None:
        if name == "type":
            return self.elements[0].input_type
        return self.elements[0].attributes.get(name)

    async def input_value(self, **_: object) -> str:
        return self.elements[0].value

    async def is_checked(self) -> bool:
        return self.elements[0].checked

    async def click(self, **_: object) -> None:
        self.page.submit_clicked = True

    async def inner_text(self, **_: object) -> str:
        return self.page.confirmation_text


class FakePage:
    def __init__(
        self,
        *,
        captcha: bool = False,
        has_file: bool = True,
        required_fields: list[FakeElement] | None = None,
        confirmation_text: str = "",
    ) -> None:
        self.captcha = captcha
        self.has_file = has_file
        self.required_fields = required_fields or []
        self.confirmation_text = confirmation_text
        self.fields = {
            "first_name": FakeElement(),
            "last_name": FakeElement(),
            "name": FakeElement(),
            "email": FakeElement(input_type="email"),
            "phone": FakeElement(input_type="tel"),
        }
        self.uploaded_file: str | None = None
        self.submit_clicked = False
        self.url: str | None = None

    def set_default_timeout(self, _: int) -> None:
        return None

    async def goto(self, url: str, **_: object) -> None:
        self.url = url

    async def wait_for_timeout(self, _: int) -> None:
        return None

    def locator(self, selector: str) -> FakeLocator:
        if selector.startswith("input[required"):
            return FakeLocator(self, self.required_fields)
        if 'input[type="file"]' in selector:
            files = [FakeElement(input_type="file")] if self.has_file else []
            return FakeLocator(self, files)
        if selector == 'button[type="submit"], input[type="submit"]':
            return FakeLocator(self, [FakeElement(input_type="submit")])
        if selector == "body":
            return FakeLocator(self, [FakeElement()])
        if "captcha" in selector:
            markers = [FakeElement()] if self.captcha else []
            return FakeLocator(self, markers)
        for name, element in self.fields.items():
            if (
                f'name="{name}"' in selector
                or f"#{name}" in selector
                or (name == "email" and 'type="email"' in selector)
                or (name == "phone" and 'type="tel"' in selector)
            ):
                return FakeLocator(self, [element])
        return FakeLocator(self, [])


class FakeBrowser:
    def __init__(self, page: FakePage) -> None:
        self.page = page
        self.closed = False

    async def new_page(self) -> FakePage:
        return self.page

    async def close(self) -> None:
        self.closed = True


class FakePlaywright:
    def __init__(self, browser: FakeBrowser) -> None:
        self.chromium = self
        self.browser = browser
        self.stopped = False

    async def launch(self, **kwargs: object) -> FakeBrowser:
        assert kwargs == {"headless": True}
        return self.browser

    async def stop(self) -> None:
        self.stopped = True


class FakePlaywrightManager:
    def __init__(self, playwright: FakePlaywright) -> None:
        self.playwright = playwright

    async def start(self) -> FakePlaywright:
        return self.playwright


def install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch, page: FakePage
) -> tuple[FakeBrowser, FakePlaywright]:
    browser = FakeBrowser(page)
    playwright = FakePlaywright(browser)
    module = ModuleType("playwright.async_api")
    module.async_playwright = lambda: FakePlaywrightManager(playwright)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.async_api", module)
    return browser, playwright


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (make_job(contact_email="jobs@example.com"), "email"),
        (
            make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
            "greenhouse",
        ),
        (make_job(source_link="https://jobs.lever.co/example/17"), "lever"),
        (make_job(), "autofill"),
    ],
)
def test_pick_method(job: Job, expected: str) -> None:
    assert pick_method(job) == expected


@pytest.mark.asyncio
async def test_email_application_dry_run_prepares_email_without_sending(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF test resume")
    configure_apply(monkeypatch)
    job = make_job(contact_email="jobs@example.com")
    profile = ResumeProfile(
        name="Jordan Lee",
        email="jordan@example.net",
        phone="+1 555 0100",
    )

    result = await apply_by_email(job, profile, str(resume_path))

    assert result.method == "email"
    assert result.status == "dry_run"
    assert "nothing was sent" in result.detail
    assert result.apply_url == str(job.source_link)


@pytest.mark.asyncio
async def test_email_application_real_send_uses_smtp_mock_and_correct_attachment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.docx"
    resume_path.write_bytes(b"docx bytes")
    configure_apply(
        monkeypatch,
        enabled=True,
        smtp_host="smtp.example.test",
        smtp_port=587,
        smtp_user="aijob-user",
        smtp_pass="aijob-password",
        smtp_from="AIJob <apply@example.test>",
    )
    send = AsyncMock(return_value="sent")
    smtp_mock = ModuleType("aiosmtplib")
    smtp_mock.send = send  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiosmtplib", smtp_mock)
    job = make_job(contact_email="jobs@example.com")
    profile = ResumeProfile(
        name="Jordan Lee",
        email="jordan@example.net",
        phone="+1 555 0100",
    )

    result = await apply_by_email(job, profile, str(resume_path))

    assert result.status == "submitted"
    email_message = send.call_args.args[0]
    assert isinstance(email_message, EmailMessage)
    assert email_message["From"] == "AIJob <apply@example.test>"
    assert email_message["To"] == "jobs@example.com"
    assert email_message["Reply-To"] == "jordan@example.net"
    assert email_message["Subject"] == "Application: Backend Engineer - Jordan Lee"
    attachment = list(email_message.iter_attachments())[0]
    assert attachment.get_content_type() == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert send.call_args.kwargs == {
        "hostname": "smtp.example.test",
        "port": 587,
        "username": "aijob-user",
        "password": "aijob-password",
        "start_tls": True,
        "timeout": 30,
    }


@pytest.mark.asyncio
async def test_greenhouse_form_dry_run_fills_profile_and_resume_without_submit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch)
    page = FakePage()
    browser, playwright = install_fake_playwright(monkeypatch, page)
    job = make_job(source_link="https://boards.greenhouse.io/example/jobs/17")

    result = await apply_via_form(
        job,
        ResumeProfile(
            name="Jordan Lee",
            email="jordan@example.net",
            phone="+1 555 0100",
        ),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "dry_run"
    assert page.fields["first_name"].value == "Jordan"
    assert page.fields["last_name"].value == "Lee"
    assert page.fields["email"].value == "jordan@example.net"
    assert page.fields["phone"].value == "+1 555 0100"
    assert page.uploaded_file == str(resume_path)
    assert not page.submit_clicked
    assert browser.closed and playwright.stopped


@pytest.mark.asyncio
async def test_captcha_stops_form_application_without_filling(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch, enabled=True)
    page = FakePage(captcha=True)
    browser, playwright = install_fake_playwright(monkeypatch, page)

    result = await apply_via_form(
        make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
        ResumeProfile(name="Jordan Lee"),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "needs_manual"
    assert "CAPTCHA" in result.detail
    assert not page.fields["first_name"].value
    assert not page.submit_clicked
    assert browser.closed and playwright.stopped


@pytest.mark.asyncio
async def test_form_without_resume_input_requires_manual_action(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch)
    page = FakePage(has_file=False)
    install_fake_playwright(monkeypatch, page)

    result = await apply_via_form(
        make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
        ResumeProfile(name="Jordan Lee"),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "needs_manual"
    assert "No resume upload field" in result.detail


@pytest.mark.asyncio
async def test_unanswered_required_fields_stop_form_application(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch, enabled=True)
    page = FakePage(required_fields=[FakeElement(value="")])
    install_fake_playwright(monkeypatch, page)

    result = await apply_via_form(
        make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
        ResumeProfile(name="Jordan Lee"),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "needs_manual"
    assert "1 required custom question" in result.detail
    assert "Nothing was submitted" in result.detail
    assert not page.submit_clicked


@pytest.mark.asyncio
async def test_form_confirmation_page_marks_application_submitted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch, enabled=True)
    page = FakePage(confirmation_text="Thank you for applying.")
    install_fake_playwright(monkeypatch, page)

    result = await apply_via_form(
        make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
        ResumeProfile(name="Jordan Lee"),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "submitted"
    assert page.submit_clicked


@pytest.mark.asyncio
async def test_form_without_confirmation_fails_honestly(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume_path = tmp_path / "resume.pdf"
    resume_path.write_bytes(b"%PDF")
    configure_apply(monkeypatch, enabled=True)
    page = FakePage(confirmation_text="Please retry.")
    install_fake_playwright(monkeypatch, page)

    result = await apply_via_form(
        make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
        ResumeProfile(name="Jordan Lee"),
        str(resume_path),
        "greenhouse",
    )

    assert result.status == "failed"
    assert "couldn't confirm submission" in result.detail
    assert page.submit_clicked


def test_describe_plan_describes_one_step_application_and_review_mode() -> None:
    assert "fill and submit" in describe_plan(make_job())
    assert "prepare and review" in describe_plan(make_job(), require_review=True)


@pytest.mark.asyncio
async def test_application_without_resume_returns_failed() -> None:
    job = make_job()

    result = await apply(job, ResumeProfile(name="Jordan Lee"), None)

    assert result.method == "autofill"
    assert result.status == "failed"
    assert "No saved resume file" in result.detail
