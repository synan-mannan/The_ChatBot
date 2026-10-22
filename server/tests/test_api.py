from __future__ import annotations

from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any
import json

from docx import Document
from fastapi.testclient import TestClient

from server.app.config import Settings
from server.app.main import create_app
from server.app.models.schemas import ResumeProfile
from server.app.services import autofill
from server.app.services.autofill import AutofillReport
from server.app.services.llm import LLMTurn, NativeToolCall
from server.app.tools.base import ToolDefinition

ROOT = Path(__file__).resolve().parents[2]


class ScriptedModel:
    def __init__(self, turns: list[LLMTurn]) -> None:
        self.turns = deque(turns)
        self.system_instructions: list[str] = []

    async def extract_profile(self, _: str) -> ResumeProfile:
        return ResumeProfile(
            name="Jordan Lee",
            email="jordan@example.com",
            designation="Software Engineer",
            skills=["Python", "React"],
            experience_years=3,
        )

    async def generate_chat(
        self,
        contents: list[Any],
        tools: list[ToolDefinition],
        system_instruction: str,
    ) -> LLMTurn:
        assert "Never invent a job" in system_instruction
        assert contents
        assert {tool.name for tool in tools} >= {"recommend_jobs", "apply_to_job"}
        self.system_instructions.append(system_instruction)
        return self.turns.popleft()

    def format_tool_response(
        self, call: NativeToolCall, output: dict[str, Any]
    ) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": call.call_id or "test-call",
            "content": json.dumps(output),
        }


def tool_turn(name: str, arguments: dict[str, Any]) -> LLMTurn:
    return LLMTurn(
        text="",
        tool_calls=[NativeToolCall(name, arguments, f"test-{name}")],
        model_content=f"call-{name}",
    )


def make_docx() -> bytes:
    document = Document()
    document.add_paragraph(
        "Jordan Lee jordan@example.com Software Engineer with three years of experience. "
        "Built Python services and React interfaces for customer projects. "
        "BSc Computer Science and open-source contributions."
    )
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_upload_recommend_apply_and_list_flow(tmp_path: Path, monkeypatch) -> None:
    session_id = "anonymous-browser-session-123"
    model = ScriptedModel(
        [
            tool_turn("recommend_jobs", {"filters": {}, "top_k": 2}),
            tool_turn("apply_to_job", {"job_id": "displayed:2"}),
            tool_turn("list_applications", {}),
            tool_turn("apply_to_job", {"job_id": "displayed:2"}),
        ]
    )

    async def fake_fill_application_form(url, *args, **kwargs):
        return AutofillReport(status="dry_run", final_url=url)

    monkeypatch.setattr(autofill, "fill_application_form", fake_fill_application_form)
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
        uploads_dir=str(tmp_path / "uploads"),
    )
    app = create_app(settings, model)

    with TestClient(app) as client:
        no_resume = client.get("/api/applications", params={"session_id": session_id})
        assert "upload a resume" in no_resume.json()["message"].lower()

        resume_bytes = make_docx()
        uploaded = client.post(
            "/api/upload",
            data={"session_id": session_id},
            files={
                "file": (
                    "jordan.docx",
                    resume_bytes,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["profile"]["name"] == "Jordan Lee"
        resume_path = tmp_path / "uploads" / session_id / "resume.docx"
        assert resume_path.read_bytes() == resume_bytes
        assert app.state.session_store.get(session_id).resume_path == str(resume_path)

        recommended = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Find jobs for me"},
        )
        assert recommended.status_code == 200
        assert recommended.json()["profile"]["name"] == "Jordan Lee"
        second_job_id = recommended.json()["jobs"][1]["job"]["id"]
        assert len(recommended.json()["jobs"]) == 2
        assert recommended.json()["reply"].startswith("I found 2 matching jobs.")
        assert "call recommend_jobs directly" in model.system_instructions[0]
        assert "uploaded resume profile is available" in model.system_instructions[0]
        applied = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Apply to job 2"},
        )
        assert applied.status_code == 200
        assert applied.json()["application_result"]["status"] == "dry_run"
        assert "Dry run: nothing was sent." in applied.json()["reply"]
        assert "2. " in model.system_instructions[1]

        listed = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Show my applied jobs"},
        )
        assert listed.status_code == 200
        assert len(listed.json()["applications"]) == 1

        duplicate = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Apply to job 2 again"},
        )
        assert duplicate.status_code == 200
        assert "already recorded" in duplicate.json()["reply"].lower()
        applications = client.get(
            "/api/applications", params={"session_id": session_id}
        ).json()["applications"]
        assert len(applications) == 1
        assert applications[0]["job_id"] == second_job_id

        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["jobs_loaded"] == 1800


def test_application_card_endpoint_does_not_call_llm(
    tmp_path: Path, monkeypatch
) -> None:
    session_id = "anonymous-browser-session-apply"
    model = ScriptedModel([])
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'apply-direct.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
        uploads_dir=str(tmp_path / "uploads"),
    )

    app = create_app(settings, model)
    async def fake_fill_application_form(url, *args, **kwargs):
        return AutofillReport(status="dry_run", final_url=url)

    monkeypatch.setattr(autofill, "fill_application_form", fake_fill_application_form)
    with TestClient(app) as client:
        job_id = "1"
        first = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": job_id},
        )
        assert first.status_code == 200
        assert first.json()["status"] == "resume_required"

        uploaded = client.post(
            "/api/upload",
            data={"session_id": session_id},
            files={
                "file": (
                    "jordan.docx",
                    make_docx(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert uploaded.status_code == 200

        first = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": job_id},
        )
        second = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": job_id},
        )
        duplicate = client.post(
            "/api/applications/apply",
            json={"session_id": session_id, "job_id": job_id},
        )

        assert first.status_code == 200
        assert first.json()["status"] == "dry_run"
        assert second.status_code == 200
        assert second.json()["status"] == "duplicate"
        assert duplicate.status_code == 200
        assert duplicate.json()["status"] == "duplicate"
    assert model.turns == deque()


def test_upload_rejects_unsupported_file_with_friendly_message(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
        uploads_dir=str(tmp_path / "uploads"),
    )
    app = create_app(settings, ScriptedModel([]))

    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            data={"session_id": "anonymous-browser-session-123"},
            files={"file": ("resume.txt", b"not a resume", "text/plain")},
        )

    assert response.status_code == 400
    assert "PDF or DOCX" in response.json()["detail"]


def test_resume_profile_is_restored_from_database_after_restart(tmp_path: Path) -> None:
    session_id = "anonymous-browser-session-456"
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'persistent.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
        uploads_dir=str(tmp_path / "uploads"),
    )
    upload_model = ScriptedModel([])
    with TestClient(create_app(settings, upload_model)) as client:
        response = client.post(
            "/api/upload",
            data={"session_id": session_id},
            files={
                "file": (
                    "jordan.docx",
                    make_docx(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        assert response.status_code == 200

    restart_model = ScriptedModel(
        [LLMTurn("Welcome back. Your resume is ready.", [], "restart-response")]
    )
    restarted_app = create_app(settings, restart_model)
    with TestClient(restarted_app) as client:
        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Hello again"},
        )

    assert response.status_code == 200
    assert response.json()["profile"]["name"] == "Jordan Lee"
    assert "uploaded resume profile is available" in restart_model.system_instructions[0]
    assert restarted_app.state.session_store.get(session_id).resume_path == str(
        tmp_path / "uploads" / session_id / "resume.docx"
    )


def test_health_reports_groq_when_groq_key_is_configured(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key="test-groq-key",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'health.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
    )

    with TestClient(create_app(settings, ScriptedModel([]))) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is True
    assert response.json()["llm_provider"] == "groq"
