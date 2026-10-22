from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from server.app.config import Settings
from server.app.main import create_app
from server.app.models.schemas import Job, ResumeProfile
from server.app.services import autofill
from server.app.services.autofill import AutofillReport, FilledField
from server.app.services.llm import LLMTurn, NativeToolCall

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
    autofill._REVIEW_RUNS.clear()


def _settings(
    tmp_path: Path,
    real_apply_enabled: bool,
    *,
    require_review: bool = False,
) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(Path("data/jobs.json").resolve()),
        real_apply_enabled=real_apply_enabled,
        require_review=require_review,
        uploads_dir=str(tmp_path / "uploads"),
        client_dist=str(tmp_path / "client"),
    )


def _job(url: str) -> Job:
    return Job(
        id="job-flow",
        title="Software Engineer",
        company="Example",
        location="Remote",
        is_remote=True,
        experience_required="Not specified",
        source_link=url,
        source="Fixture",
    )


class SaveAnswerModel:
    async def generate_chat(self, contents, tools, system_instruction):
        return LLMTurn(
            text="",
            tool_calls=[
                NativeToolCall(
                    name="save_application_answers",
                    arguments={"answers": {"notice_period": "30 days"}},
                )
            ],
            model_content=None,
        )

    def format_tool_response(self, call, output):
        return output


class SubmitApplicationModel:
    async def generate_chat(self, contents, tools, system_instruction):
        return LLMTurn(
            text="",
            tool_calls=[
                NativeToolCall(
                    name="apply_to_job",
                    arguments={"job_id": "job-flow", "submit": False},
                )
            ],
            model_content=None,
        )

    def format_tool_response(self, call, output):
        return output


def _application_report(
    status: Literal["review_ready", "submitted", "retry_later"],
    final_url: str,
    screenshot_path: str | None = None,
) -> AutofillReport:
    return AutofillReport(
        status=status,
        final_url=final_url,
        screenshot_path=screenshot_path,
        filled=[
            FilledField(label="Full name", key="name"),
            FilledField(label="Email address", key="email"),
            FilledField(label="Phone number", key="phone"),
        ],
    )


def _prepare_session(app, session_id: str, url: str) -> None:
    app.state.jobs = [_job(url)]
    resume = Path(app.state.settings.uploads_dir) / session_id / "resume.pdf"
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_bytes(b"local fixture resume")
    state = app.state.session_store.set_profile(
        session_id,
        ResumeProfile(
            name="Ada Lovelace",
            email="ada@example.test",
            phone="+44 20 1234 5678",
        ),
    )
    state.resume_path = str(resume)


def test_confirmation_review_preview_and_submit_are_session_scoped(
    fixture_server, tmp_path, monkeypatch
) -> None:
    async def no_rate_limit(
        domain: str, session_id: str, mode: str, job_id: str = "", **kwargs
    ) -> None:
        return None

    monkeypatch.setattr(autofill, "_reserve_run", no_rate_limit)
    session_id = "session-flow-123"
    url = f"{fixture_server}/multistep.html"
    app = create_app(
        settings=_settings(tmp_path, True, require_review=True),
        model=object(),
    )
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        first = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        assert first.status_code == 200
        assert first.json()["status"] == "confirmation_required"
        assert "127.0.0.1" in first.json()["plan"]

        review = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow", "confirm": True},
        )
        assert review.status_code == 200
        assert review.json()["status"] == "review_ready"
        assert review.json()["screenshot_url"]
        assert {field["key"] for field in review.json()["filled"]} == {
            "first_name",
            "last_name",
            "email",
            "phone",
        }

        preview_url = review.json()["screenshot_url"]
        own_preview = client.get(preview_url)
        assert own_preview.status_code == 200
        assert own_preview.headers["content-type"] == "image/png"
        other_session = client.get(
            "/api/applications/job-flow/preview",
            params={"session_id": "session-other-123"},
        )
        assert other_session.status_code == 404
        traversal = client.get(
            "/api/applications/..%2F..%2Fsecrets/preview",
            params={"session_id": session_id},
        )
        assert traversal.status_code in {404, 422}

        submitted = client.post(
            "/api/applications/apply",
            json={
                "session_id": session_id,
                "job_id": "job-flow",
                "submit": True,
            },
        )
        assert submitted.status_code == 200
        assert submitted.json()["status"] == "submitted"
        cleared = client.delete(
            "/api/session",
            params={"session_id": session_id},
        )
        assert cleared.status_code == 204
        assert client.get(preview_url).status_code == 404


def test_rate_limited_application_can_be_retried(
    fixture_server, tmp_path, monkeypatch
) -> None:
    calls = 0

    async def rate_limit_once(
        domain: str, session_id: str, mode: str, job_id: str = "", **kwargs
    ) -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "Domain rate-limited: try again in 30 seconds."
        return None

    monkeypatch.setattr(autofill, "_reserve_run", rate_limit_once)
    session_id = "session-rate-limit"
    url = f"{fixture_server}/multistep.html"
    app = create_app(settings=_settings(tmp_path, True), model=object())
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        limited = client.post(
            "/api/applications/apply",
            json={
                "session_id": session_id,
                "job_id": "job-flow",
                "confirm": False,
            },
        )
        assert limited.json()["status"] == "retry_later"
        assert "rate-limited" in limited.json()["detail"]

        retried = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        assert retried.json()["status"] == "submitted"
        assert calls == 2


def test_one_step_apply_submits_with_real_apply_enabled(
    fixture_server, tmp_path, monkeypatch
) -> None:
    async def no_rate_limit(
        domain: str, session_id: str, mode: str, job_id: str = "", **kwargs
    ) -> None:
        return None

    monkeypatch.setattr(autofill, "_reserve_run", no_rate_limit)
    session_id = "session-one-step-123"
    url = f"{fixture_server}/multistep.html"
    app = create_app(settings=_settings(tmp_path, True), model=object())
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        result = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )

        assert result.status_code == 200
        assert result.json()["status"] == "submitted"
        assert "Software Engineer at Example" in result.json()["message"]
        assert result.json()["screenshot_url"]


def test_terms_answer_is_saved_and_application_retries_automatically(
    fixture_server, tmp_path, monkeypatch
) -> None:
    async def no_rate_limit(
        domain: str, session_id: str, mode: str, job_id: str = "", **kwargs
    ) -> None:
        return None

    monkeypatch.setattr(autofill, "_reserve_run", no_rate_limit)
    session_id = "session-terms-flow"
    url = f"{fixture_server}/terms.html"
    app = create_app(settings=_settings(tmp_path, True), model=None)
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        app.state.orchestrator.jobs = app.state.jobs

        first = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        assert first.json()["status"] == "needs_input"
        assert first.json()["needs_input"][0]["label"] == (
            "I agree to the terms and conditions"
        )

        consent = client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "message": "I agree to the terms and privacy policy",
            },
        )

        assert consent.status_code == 200
        assert consent.json()["application_result"]["status"] == "submitted"
        assert consent.json()["application_result"]["filled"][-1]["label"] == (
            "accepted terms (user consented)"
        )
        assert [call["name"] for call in consent.json()["tool_calls"]] == [
            "save_application_answers",
            "apply_to_job",
        ]
        assert app.state.session_store.get(session_id).answers[
            "accept_portal_terms"
        ] is True


def test_real_apply_disabled_stays_dry_run(fixture_server, tmp_path) -> None:
    session_id = "session-dryrun-123"
    url = f"{fixture_server}/multistep.html"
    app = create_app(
        settings=_settings(tmp_path, False),
        model=SubmitApplicationModel(),
    )
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        result = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        assert result.status_code == 200
        assert result.json()["status"] == "dry_run"
        assert result.json()["detail"] == "Dry run: nothing was sent."
        assert "Dry run: nothing was sent." in result.json()["message"]


def test_chat_saves_answers_and_reruns_needs_input(fixture_server, tmp_path, monkeypatch) -> None:
    async def no_rate_limit(
        domain: str, session_id: str, mode: str, job_id: str = "", **kwargs
    ) -> None:
        return None

    monkeypatch.setattr(autofill, "_reserve_run", no_rate_limit)
    session_id = "session-answer-flow"
    url = f"{fixture_server}/unknown.html"
    app = create_app(
        settings=_settings(tmp_path, True),
        model=SaveAnswerModel(),
    )
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        app.state.orchestrator.jobs = app.state.jobs
        client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        needs_input = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow", "confirm": True},
        )
        assert needs_input.json()["status"] == "needs_input"

        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "My notice period is 30 days."},
        )
        assert response.status_code == 200
        assert response.json()["application_result"]["status"] == "needs_manual"
        assert "notice_period" in {
            field["key"]
            for field in response.json()["application_result"]["filled"]
        }


def test_submit_it_reopens_stored_final_url(fixture_server, tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    final_url = f"{fixture_server}/application.html"

    async def fill(url, profile, answers, resume_path, mode, job, **kwargs):
        calls.append((url, mode))
        if mode == "review":
            preview = Path(kwargs["uploads_dir"]) / kwargs["session_id"] / "previews" / f"{job.id}.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(b"local preview")
            return _application_report("review_ready", final_url, str(preview))
        return _application_report("submitted", url)

    monkeypatch.setattr(autofill, "fill_application_form", fill)
    session_id = "session-submit-it"
    listing_url = f"{fixture_server}/aggregator.html"
    app = create_app(
        settings=_settings(tmp_path, True, require_review=True),
        model=SubmitApplicationModel(),
    )
    with TestClient(app) as client:
        _prepare_session(app, session_id, listing_url)
        app.state.orchestrator.jobs = app.state.jobs
        client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        review = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow", "confirm": True},
        )
        assert review.json()["status"] == "review_ready"

        submitted = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "submit it"},
        )

        assert submitted.status_code == 200
        assert submitted.json()["application_result"]["status"] == "submitted"
        assert calls == [
            (listing_url, "review"),
            (final_url, "submit"),
        ]


def test_retry_later_preserves_review_state_preview_and_answer_snapshot(
    fixture_server, tmp_path, monkeypatch
) -> None:
    calls: list[tuple[str, str, dict]] = []
    final_url = f"{fixture_server}/application.html"
    real_fill = autofill.fill_application_form

    async def fill(url, profile, answers, resume_path, mode, job, **kwargs):
        calls.append((url, mode, answers.copy()))
        if mode == "review":
            preview = (
                Path(kwargs["uploads_dir"])
                / kwargs["session_id"]
                / "previews"
                / f"{job.id}.png"
            )
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(b"local preview")
            return _application_report("review_ready", final_url, str(preview))
        return await real_fill(
            url,
            profile,
            answers,
            resume_path,
            mode,
            job,
            **kwargs,
        )

    monkeypatch.setattr(autofill, "fill_application_form", fill)
    session_id = "session-retry-state"
    url = f"{fixture_server}/aggregator.html"
    app = create_app(
        settings=_settings(tmp_path, True, require_review=True),
        model=object(),
    )
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        app.state.session_store.get(session_id).answers = {"notice_period": "30 days"}
        client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        review = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow", "confirm": True},
        )
        preview_url = review.json()["screenshot_url"]
        app.state.session_store.get(session_id).answers = {"notice_period": "60 days"}

        async def timed_out_rate_limit(
            domain, current_session, mode, job_id="", **kwargs
        ):
            return "The domain remained rate-limited after waiting up to 30 seconds."

        monkeypatch.setattr(autofill, "_reserve_run", timed_out_rate_limit)
        retry = client.post(
            "/api/applications/apply",
            json={
                "session_id": session_id,
                "job_id": "job-flow",
                "submit": True,
            },
        )
        applications = client.get(
            "/api/applications", params={"session_id": session_id}
        )

        assert retry.json()["status"] == "retry_later"
        assert applications.json()["applications"][0]["status"] == "review_ready"
        assert applications.json()["applications"][0]["apply_url"] == final_url
        assert client.get(preview_url).status_code == 200
        assert calls[1] == (final_url, "submit", {"notice_period": "30 days"})
