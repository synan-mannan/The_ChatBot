from __future__ import annotations

import pytest

from server.app.models.schemas import Job, ResumeProfile
from server.app.services import autofill
from server.app.services.applicant import apply, pick_method


def _job(url: str, contact_email: str | None = None) -> Job:
    return Job(
        id="job-applicant",
        title="Engineer",
        company="Example",
        location="Remote",
        contact_email=contact_email,
        is_remote=True,
        experience_required="Not specified",
        source_link=url,
        source="Fixture",
    )


def test_other_http_portals_and_blocked_hosts_use_autofill_method() -> None:
    assert pick_method(_job("https://careers.example.test/apply")) == "autofill"
    assert (
        pick_method(_job("https://jobs.linkedin.com/apply", "jobs@example.test"))
        == "autofill"
    )


@pytest.mark.asyncio
async def test_applicant_service_routes_generic_portal_to_autofill(
    tmp_path, monkeypatch
) -> None:
    resume = tmp_path / "resume.pdf"
    resume.write_bytes(b"local fixture resume")

    async def report(*args, **kwargs):
        return autofill.AutofillReport(
            status="review_ready",
            final_url="https://careers.example.test/apply",
        )

    monkeypatch.setattr(autofill, "fill_application_form", report)
    result = await apply(
        _job("https://careers.example.test/apply"),
        ResumeProfile(email="ada@example.test"),
        str(resume),
        answers={"notice_period": "30 days"},
        session_id="session-applicant",
        uploads_dir=str(tmp_path / "uploads"),
        real_apply_enabled=True,
    )

    assert result.method == "autofill"
    assert result.status == "review_ready"
    assert "NOT been submitted" in result.detail
