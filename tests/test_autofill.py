from __future__ import annotations

from functools import partial
from datetime import datetime, timedelta, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from time import monotonic

import pytest

from server.app.models.schemas import Job, ResumeProfile
from server.app.services import autofill
from server.app.services.autofill import (
    _parse_mapping_output,
    fill_application_form,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "autofill"


@pytest.fixture
def fixture_server():
    handler = partial(SimpleHTTPRequestHandler, directory=str(FIXTURE_DIR))
    server = ThreadingHTTPServer(("0.0.0.0", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


@pytest.fixture(autouse=True)
def reset_rate_limits():
    autofill._DOMAIN_RUNS.clear()
    autofill._SESSION_SUBMISSIONS.clear()


def _job(url: str, job_id: str = "job-123") -> Job:
    return Job(
        id=job_id,
        title="Software Engineer",
        company="Example",
        location="Remote",
        is_remote=True,
        experience_required="Not specified",
        source_link=url,
        source="Fixture",
    )


@pytest.mark.asyncio
async def test_varied_labels_fill_resume_and_screenshot(fixture_server, tmp_path) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"test resume")
    url = f"{fixture_server}/basic.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
            linkedin_url="https://linkedin.com/in/ada",
        ),
        {},
        str(resume),
        "review",
        _job(url),
        session_id="session-123456",
        uploads_dir=str(tmp_path / "uploads"),
    )

    assert report.status == "review_ready", report.model_dump()
    assert {item.key for item in report.filled} >= {
        "first_name",
        "last_name",
        "email",
        "phone",
        "linkedin_url",
        "resume_path",
    }
    assert report.screenshot_path is not None
    assert Path(report.screenshot_path).is_file()
    assert report.detected_ats == "other"


@pytest.mark.asyncio
async def test_aggregator_follow_external_apply_skips_job_alert_form(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/aggregator.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "aggregator-job"),
        session_id="session-aggregator",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert report.final_url is not None
    assert report.final_url.endswith("/application.html")
    assert {field.label for field in report.filled} == {
        "Full name",
        "Email address",
        "Phone number",
    }
    assert all(field.label != "Enter your email" for field in report.filled)


@pytest.mark.asyncio
async def test_arbeitnow_apply_again_link_is_followed(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/apply_again.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "arbeitnow-apply-again"),
        session_id="session-arbeitnow",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert report.final_url is not None
    assert report.final_url.endswith("/application.html")
    assert {field.label for field in report.filled} == {
        "Full name",
        "Email address",
        "Phone number",
    }


@pytest.mark.asyncio
async def test_popup_tracking_redirects_and_employer_apply_click(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/redirect_listing.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "popup-job"),
        session_id="session-popup",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert report.final_url is not None
    assert report.final_url.endswith("/employer-apply.html")
    assert {field.key for field in report.filled} == {"name", "email", "phone"}


@pytest.mark.asyncio
async def test_single_email_listing_is_manual_and_saves_debug_artifacts(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/single_email.html"
    report = await fill_application_form(
        url,
        ResumeProfile(email="ada@example.test"),
        {},
        None,
        "review",
        _job(url, "single-email-job"),
        session_id="session-single-email",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_manual"
    assert report.blocker_reason == f"No application form found on {url}"
    assert report.filled == []
    assert report.final_url == url
    assert report.screenshot_path and Path(report.screenshot_path).is_file()
    assert report.debug_screenshot_path and Path(report.debug_screenshot_path).is_file()
    assert report.debug_labels_path and Path(report.debug_labels_path).is_file()
    assert report.discovered_form_labels == ["Enter your email"]


@pytest.mark.asyncio
async def test_fewer_than_three_profile_fields_never_become_review_ready(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/two_fields.html"
    report = await fill_application_form(
        url,
        ResumeProfile(name="Ada Lovelace", email="ada@example.test"),
        {},
        None,
        "review",
        _job(url, "two-fields-job"),
        session_id="session-two-fields",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_manual"
    assert report.status != "review_ready"
    assert report.filled == []
    assert report.screenshot_path and Path(report.screenshot_path).is_file()


@pytest.mark.asyncio
async def test_terms_checkbox_needs_explicit_consent(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/terms.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "terms-no-consent"),
        session_id="session-terms-no",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_input"
    assert [field.label for field in report.needs_input] == [
        "I agree to the terms and conditions"
    ]
    assert all(
        field.label != "accepted terms (user consented)"
        for field in report.filled
    )


@pytest.mark.asyncio
async def test_terms_checkbox_is_checked_only_with_saved_consent(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/terms.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {"accept_portal_terms": True},
        None,
        "review",
        _job(url, "terms-consent"),
        session_id="session-terms-yes",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert any(
        field.label == "accepted terms (user consented)"
        and field.key == "accept_portal_terms"
        for field in report.filled
    )


@pytest.mark.asyncio
async def test_eeo_questions_need_input_and_marketing_is_not_accepted(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/eeo_marketing.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "eeo-marketing"),
        session_id="session-eeo-marketing",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_input"
    assert any(field.label == "Gender" for field in report.needs_input)
    assert all("newsletter" not in field.label.casefold() for field in report.filled)


@pytest.mark.asyncio
async def test_select_and_radio_answers_match_options(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/select_radio.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {"notice_period": "30 days", "requires_sponsorship": False},
        None,
        "review",
        _job(url),
        session_id="session-select",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert {item.key for item in report.filled} == {
        "name",
        "email",
        "phone",
        "notice_period",
        "requires_sponsorship",
    }


@pytest.mark.asyncio
async def test_form_inside_iframe_is_discovered(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/iframe.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url),
        session_id="session-iframe",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready", report.model_dump()
    assert {item.key for item in report.filled} == {
        "first_name",
        "last_name",
        "email",
        "phone",
    }


@pytest.mark.asyncio
async def test_multistep_review_does_not_submit_and_submit_verifies(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/multistep.html"
    profile = ResumeProfile(
        name="Ada Lovelace",
        email="ada@example.test",
        phone="+44 20 1234 5678",
    )
    review = await fill_application_form(
        url,
        profile,
        {},
        None,
        "dry_run",
        _job(url, "dry-job"),
        session_id="session-multistep",
        uploads_dir=str(tmp_path),
    )
    assert review.status == "dry_run", review.model_dump()
    assert "submitted" not in (review.final_url or "")
    assert {item.key for item in review.filled} == {
        "first_name",
        "last_name",
        "email",
        "phone",
    }

    autofill._DOMAIN_RUNS.clear()
    submitted = await fill_application_form(
        url,
        profile,
        {},
        None,
        "submit",
        _job(url, "submit-job"),
        session_id="session-submit",
        uploads_dir=str(tmp_path),
        real_apply_enabled=True,
    )
    assert submitted.status == "submitted", submitted.blocker_reason
    assert "submitted" in (submitted.final_url or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        ("captcha.html", "CAPTCHA"),
        ("login.html", "login"),
    ],
)
async def test_captcha_and_login_are_manual_blockers(
    fixture_server, tmp_path, fixture, reason
) -> None:
    url = f"{fixture_server}/{fixture}"
    report = await fill_application_form(
        url,
        ResumeProfile(),
        {},
        None,
        "review",
        _job(url),
        session_id=f"session-{fixture.replace('.', '-')}",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_manual"
    assert report.blocker_reason is not None
    assert reason.casefold() in report.blocker_reason.casefold()
    assert report.final_url == url
    assert report.debug_screenshot_path and Path(report.debug_screenshot_path).is_file()
    assert report.debug_labels_path and Path(report.debug_labels_path).is_file()


@pytest.mark.asyncio
async def test_required_unknown_question_needs_answer_then_fills(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/unknown.html"
    profile = ResumeProfile(
        name="Ada Lovelace",
        email="ada@example.test",
        phone="+44 20 1234 5678",
    )
    missing = await fill_application_form(
        url,
        profile,
        {},
        None,
        "review",
        _job(url, "missing-job"),
        session_id="session-unknown",
        uploads_dir=str(tmp_path),
    )
    assert missing.status == "needs_input"
    assert missing.needs_input[0].label == "Notice period?"

    autofill._DOMAIN_RUNS.clear()
    answered = await fill_application_form(
        url,
        profile,
        {"notice_period": "30 days"},
        None,
        "review",
        _job(url, "answered-job"),
        session_id="session-unknown",
        uploads_dir=str(tmp_path),
    )
    assert answered.status == "review_ready"
    assert "notice_period" in {item.key for item in answered.filled}


@pytest.mark.asyncio
async def test_eeo_and_consent_stay_untouched_and_are_reported(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/eeo.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url),
        session_id="session-eeo",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "needs_input"
    assert {item.label for item in report.needs_input} >= {
        "Gender",
        "I agree to the privacy terms",
        "Have you been convicted of a felony?",
    }


@pytest.mark.asyncio
async def test_sensitive_answers_are_filled_only_after_explicit_user_input(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/eeo.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {
            "gender": "female",
            "privacy_consent": True,
            "criminal_history": "No",
        },
        None,
        "review",
        _job(url, "sensitive-answers"),
        session_id="session-sensitive-123",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert {item.key for item in report.filled} >= {
        "gender",
        "privacy_consent",
        "criminal_history",
    }


@pytest.mark.asyncio
async def test_cover_letter_is_never_filled_without_explicit_approval(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/cover_letter.html"
    missing_approval = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "cover-unapproved"),
        session_id="session-cover-letter",
        uploads_dir=str(tmp_path),
    )
    assert missing_approval.status == "needs_input"
    assert missing_approval.needs_input[0].label == "Cover letter"

    autofill._DOMAIN_RUNS.clear()
    approved = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "review",
        _job(url, "cover-approved"),
        session_id="session-cover-letter",
        uploads_dir=str(tmp_path),
        approved_cover_letter="Approved candidate cover letter.",
    )
    assert approved.status == "review_ready"
    assert "cover_letter" in {item.key for item in approved.filled}


@pytest.mark.asyncio
async def test_submit_refuses_missing_required_and_blocked_domain(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/unknown.html"
    missing = await fill_application_form(
        url,
        ResumeProfile(),
        {},
        None,
        "submit",
        _job(url),
        session_id="session-required",
        uploads_dir=str(tmp_path),
        real_apply_enabled=True,
    )
    assert missing.status == "needs_input"
    assert missing.needs_input[0].required

    blocked = await fill_application_form(
        "https://careers.linkedin.com/jobs/1",
        ResumeProfile(),
        {},
        None,
        "review",
        _job("https://careers.linkedin.com/jobs/1"),
        session_id="session-blocked-domain",
        uploads_dir=str(tmp_path),
    )
    assert blocked.status == "needs_manual"
    assert blocked.final_url == "https://careers.linkedin.com/jobs/1"
    assert blocked.debug_screenshot_path and Path(blocked.debug_screenshot_path).is_file()


@pytest.mark.asyncio
async def test_submit_mode_is_downgraded_when_real_apply_is_disabled(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/multistep.html"
    report = await fill_application_form(
        url,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
        {},
        None,
        "submit",
        _job(url),
        session_id="session-disabled-123",
        uploads_dir=str(tmp_path),
        real_apply_enabled=False,
    )

    assert report.status == "dry_run", report.model_dump()
    assert "REAL_APPLY_ENABLED=false" in (report.blocker_reason or "")


def test_llm_mapping_discards_invalid_keys_and_generated_values() -> None:
    mapped = _parse_mapping_output(
        '{"Notice period?":"notice_period","Email":"an invented answer",'
        '"Company":"not_a_profile_key"}'
    )

    assert mapped == {"Notice period?": "notice_period"}


@pytest.mark.asyncio
async def test_domain_and_session_real_application_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr(autofill, "DOMAIN_INTERVAL", timedelta(milliseconds=20))
    first = await autofill._reserve_run("example.test", "session-rate-123", "review")
    started = monotonic()
    immediate_retry = await autofill._reserve_run(
        "example.test", "another-session-123", "review"
    )

    assert first is None
    assert immediate_retry is None
    assert monotonic() - started >= 0.015

    autofill._DOMAIN_RUNS.clear()
    autofill._DOMAIN_RUNS["timeout.test"] = datetime.now(timezone.utc)

    class ExpiredLoop:
        calls = 0

        def time(self) -> float:
            self.calls += 1
            return float(self.calls)

    get_running_loop = autofill.asyncio.get_running_loop
    monkeypatch.setattr(autofill.asyncio, "get_running_loop", ExpiredLoop)
    timed_out = await autofill._reserve_run(
        "timeout.test", "session-timeout", "review", "job-timeout"
    )
    assert timed_out is not None
    assert "after waiting" in timed_out

    autofill._DOMAIN_RUNS.clear()
    monkeypatch.setattr(autofill.asyncio, "get_running_loop", get_running_loop)
    autofill._REVIEW_RUNS.clear()
    await autofill._reserve_run(
        "review.test", "session-review", "review", "job-review"
    )
    autofill._mark_review_run(
        "session-review", "job-review", {"review.test"}
    )
    started = monotonic()
    exempt = await autofill._reserve_run(
        "review.test", "session-review", "submit", "job-review"
    )
    assert exempt is None
    assert monotonic() - started < 0.015

    autofill._DOMAIN_RUNS.clear()
    for index in range(5):
        result = await autofill._reserve_run(
            f"domain-{index}.test", "session-rate-123", "submit"
        )
        assert result is None
    capped = await autofill._reserve_run(
        "sixth-domain.test", "session-rate-123", "submit"
    )
    assert capped is not None
    assert "five real application attempts" in capped


@pytest.mark.asyncio
async def test_closed_browser_returns_retry_later(monkeypatch, tmp_path) -> None:
    class ClosedBrowser:
        async def start(self):
            raise RuntimeError("Target page, context or browser has been closed")

    monkeypatch.setattr(
        "playwright.async_api.async_playwright",
        lambda: ClosedBrowser(),
    )
    url = "https://example.test/jobs/1"
    report = await fill_application_form(
        url,
        ResumeProfile(name="Ada Lovelace", email="ada@example.test"),
        {},
        None,
        "review",
        _job(url),
        session_id="session-browser-crash",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "retry_later"
    assert report.final_url == url
