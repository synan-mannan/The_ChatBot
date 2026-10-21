from __future__ import annotations

import pytest

from server.app.models.schemas import Job, ResumeProfile
from server.app.services.applicant import apply, apply_by_email, describe_plan, pick_method


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


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (make_job(contact_email="jobs@example.com"), "email"),
        (
            make_job(source_link="https://boards.greenhouse.io/example/jobs/17"),
            "greenhouse",
        ),
        (make_job(source_link="https://jobs.lever.co/example/17"), "lever"),
        (make_job(), "manual"),
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
    monkeypatch.setenv("REAL_APPLY_ENABLED", "false")
    monkeypatch.delenv("SMTP_FROM", raising=False)
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


def test_describe_plan_explains_manual_fallback() -> None:
    assert "can't submit" in describe_plan(make_job())


@pytest.mark.asyncio
async def test_manual_application_returns_needs_manual_and_listing_url() -> None:
    job = make_job()

    result = await apply(job, ResumeProfile(name="Jordan Lee"), None)

    assert result.method == "manual"
    assert result.status == "needs_manual"
    assert str(job.source_link) in result.detail
