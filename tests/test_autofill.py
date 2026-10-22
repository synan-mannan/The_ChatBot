from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

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
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
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

    assert report.status == "review_ready"
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


@pytest.mark.asyncio
async def test_select_and_radio_answers_match_options(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/select_radio.html"
    report = await fill_application_form(
        url,
        ResumeProfile(),
        {"notice_period": "30 days", "requires_sponsorship": False},
        None,
        "review",
        _job(url),
        session_id="session-select",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert {item.key for item in report.filled} == {
        "notice_period",
        "requires_sponsorship",
    }


@pytest.mark.asyncio
async def test_form_inside_iframe_is_discovered(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/iframe.html"
    report = await fill_application_form(
        url,
        ResumeProfile(name="Ada Lovelace", email="ada@example.test"),
        {},
        None,
        "review",
        _job(url),
        session_id="session-iframe",
        uploads_dir=str(tmp_path),
    )

    assert report.status == "review_ready"
    assert {item.key for item in report.filled} == {"first_name", "email"}


@pytest.mark.asyncio
async def test_multistep_review_does_not_submit_and_submit_verifies(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/multistep.html"
    profile = ResumeProfile(name="Ada Lovelace", email="ada@example.test")
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
    assert review.status == "dry_run"
    assert "submitted" not in (review.final_url or "")
    assert {item.key for item in review.filled} == {"first_name", "email"}

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


@pytest.mark.asyncio
async def test_required_unknown_question_needs_answer_then_fills(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/unknown.html"
    profile = ResumeProfile()
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
    assert answered.filled[0].key == "notice_period"


@pytest.mark.asyncio
async def test_eeo_and_consent_stay_untouched_and_are_reported(fixture_server, tmp_path) -> None:
    url = f"{fixture_server}/eeo.html"
    report = await fill_application_form(
        url,
        ResumeProfile(email="ada@example.test"),
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
        ResumeProfile(email="ada@example.test"),
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
        ResumeProfile(),
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
        ResumeProfile(),
        {},
        None,
        "review",
        _job(url, "cover-approved"),
        session_id="session-cover-letter",
        uploads_dir=str(tmp_path),
        approved_cover_letter="Approved candidate cover letter.",
    )
    assert approved.status == "review_ready"
    assert approved.filled[0].key == "cover_letter"


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
    )
    assert blocked.status == "needs_manual"


@pytest.mark.asyncio
async def test_submit_mode_is_downgraded_when_real_apply_is_disabled(
    fixture_server, tmp_path
) -> None:
    url = f"{fixture_server}/multistep.html"
    report = await fill_application_form(
        url,
        ResumeProfile(name="Ada Lovelace", email="ada@example.test"),
        {},
        None,
        "submit",
        _job(url),
        session_id="session-disabled-123",
        uploads_dir=str(tmp_path),
        real_apply_enabled=False,
    )

    assert report.status == "dry_run"
    assert "REAL_APPLY_ENABLED=false" in (report.blocker_reason or "")


def test_llm_mapping_discards_invalid_keys_and_generated_values() -> None:
    mapped = _parse_mapping_output(
        '{"Notice period?":"notice_period","Email":"an invented answer",'
        '"Company":"not_a_profile_key"}'
    )

    assert mapped == {"Notice period?": "notice_period"}


@pytest.mark.asyncio
async def test_domain_and_session_real_application_rate_limits() -> None:
    first = await autofill._reserve_run("example.test", "session-rate-123", "review")
    immediate_retry = await autofill._reserve_run(
        "example.test", "another-session-123", "review"
    )

    assert first is None
    assert immediate_retry is not None
    assert "rate-limited" in immediate_retry

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
