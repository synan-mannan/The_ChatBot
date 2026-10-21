from __future__ import annotations

from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document
from fastapi.testclient import TestClient

from server.app.config import Settings
from server.app.main import create_app
from server.app.models.schemas import ResumeProfile
from server.app.services.llm import LLMTurn, NativeToolCall
from server.app.tools.base import ToolDefinition

ROOT = Path(__file__).resolve().parents[2]


class ScriptedModel:
    def __init__(self, turns: list[LLMTurn]) -> None:
        self.turns = deque(turns)

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
        return self.turns.popleft()


def tool_turn(name: str, arguments: dict[str, Any]) -> LLMTurn:
    return LLMTurn(
        text="",
        tool_calls=[NativeToolCall(name, arguments)],
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


def test_upload_recommend_apply_and_list_flow(tmp_path: Path) -> None:
    session_id = "anonymous-browser-session-123"
    model = ScriptedModel(
        [
            tool_turn("recommend_jobs", {"filters": {}, "top_k": 2}),
            LLMTurn("Here are two roles matched to your profile.", [], "final-recommendations"),
            tool_turn("apply_to_job", {"job_id": "id:1"}),
            LLMTurn("Your application has been recorded.", [], "final-apply"),
            tool_turn("list_applications", {}),
            LLMTurn("Here is your application history.", [], "final-list"),
            tool_turn("apply_to_job", {"job_id": "id:1"}),
            LLMTurn("You had already applied to that role.", [], "final-duplicate"),
        ]
    )
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
    )
    app = create_app(settings, model)

    with TestClient(app) as client:
        no_resume = client.get("/api/applications", params={"session_id": session_id})
        assert "upload a resume" in no_resume.json()["message"].lower()

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
        assert uploaded.json()["profile"]["name"] == "Jordan Lee"

        recommended = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Find jobs for me"},
        )
        assert recommended.status_code == 200
        first_job_id = recommended.json()["jobs"][0]["job"]["id"]
        assert len(recommended.json()["jobs"]) == 2
        model.turns[0] = tool_turn("apply_to_job", {"job_id": f"id:{first_job_id}"})

        applied = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": f"Apply to job ID id:{first_job_id}"},
        )
        assert applied.status_code == 200
        assert applied.json()["reply"] == "Your application has been recorded."

        listed = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Show my applied jobs"},
        )
        assert listed.status_code == 200
        assert len(listed.json()["applications"]) == 1

        model.turns[0] = tool_turn("apply_to_job", {"job_id": f"id:{first_job_id}"})
        duplicate = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Apply again"},
        )
        assert duplicate.status_code == 200
        applications = client.get(
            "/api/applications", params={"session_id": session_id}
        ).json()["applications"]
        assert len(applications) == 1

        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["jobs_loaded"] == 1800


def test_upload_rejects_unsupported_file_with_friendly_message(tmp_path: Path) -> None:
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'aijob.db'}",
        jobs_file=str(ROOT / "data" / "jobs.json"),
        client_dist=str(ROOT / "client" / "dist"),
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
    with TestClient(create_app(settings, restart_model)) as client:
        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "Hello again"},
        )

    assert response.status_code == 200
    assert response.json()["profile"]["name"] == "Jordan Lee"
