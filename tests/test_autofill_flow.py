from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from fastapi.testclient import TestClient

from server.app.config import Settings
from server.app.main import create_app
from server.app.models.schemas import Job, ResumeProfile
from server.app.services import autofill
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


def _settings(tmp_path: Path, real_apply_enabled: bool) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(Path("data/jobs.json").resolve()),
        real_apply_enabled=real_apply_enabled,
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


def _prepare_session(app, session_id: str, url: str) -> None:
    app.state.jobs = [_job(url)]
    resume = Path(app.state.settings.uploads_dir) / session_id / "resume.pdf"
    resume.parent.mkdir(parents=True, exist_ok=True)
    resume.write_bytes(b"local fixture resume")
    state = app.state.session_store.set_profile(
        session_id,
        ResumeProfile(name="Ada Lovelace", email="ada@example.test"),
    )
    state.resume_path = str(resume)


def test_confirmation_review_preview_and_submit_are_session_scoped(
    fixture_server, tmp_path, monkeypatch
) -> None:
    async def no_rate_limit(domain: str, session_id: str, mode: str) -> None:
        return None

    monkeypatch.setattr(autofill, "_reserve_run", no_rate_limit)
    session_id = "session-flow-123"
    url = f"{fixture_server}/multistep.html"
    app = create_app(settings=_settings(tmp_path, True), model=object())
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
            "email",
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


def test_real_apply_disabled_stays_dry_run(fixture_server, tmp_path) -> None:
    session_id = "session-dryrun-123"
    url = f"{fixture_server}/multistep.html"
    app = create_app(settings=_settings(tmp_path, False), model=object())
    with TestClient(app) as client:
        _prepare_session(app, session_id, url)
        client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow"},
        )
        result = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": "job-flow", "confirm": True},
        )
        assert result.status_code == 200
        assert result.json()["status"] == "dry_run"
        assert "nothing was submitted" in result.json()["detail"].casefold()
        submit_attempt = client.post(
            "/api/applications/apply",
            json={
                "session_id": session_id,
                "job_id": "job-flow",
                "submit": True,
            },
        )
        assert submit_attempt.json()["status"] == "dry_run"


def test_chat_saves_answers_and_reruns_needs_input(fixture_server, tmp_path, monkeypatch) -> None:
    async def no_rate_limit(domain: str, session_id: str, mode: str) -> None:
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
        assert response.json()["application_result"]["status"] == "review_ready"
        assert response.json()["application_result"]["filled"][0]["key"] == "notice_period"
