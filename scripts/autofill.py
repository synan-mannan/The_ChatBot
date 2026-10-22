from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app.config import get_settings
from server.app.models.db import ResumeProfileRow, create_database
from server.app.models.schemas import Job, ResumeProfile
from server.app.services.autofill import AutofillReport, fill_application_form
from server.app.services.llm import create_llm_service
from server.app.tools.application_answers import SaveApplicationAnswersInput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill a career-site application for human review; this CLI never submits."
    )
    parser.add_argument("--url", help="Employer application URL")
    parser.add_argument("--resume", help="Local PDF or DOCX resume file")
    parser.add_argument("--profile", help="JSON file with the ResumeProfile shape")
    parser.add_argument("--answers", help="Optional JSON object of saved application answers")
    parser.add_argument(
        "--headed",
        action="store_true",
        default=True,
        help="Show the browser window (the default and only supported mode)",
    )
    parser.add_argument(
        "--export-profile",
        metavar="SESSION_ID",
        help="Export a stored session profile to profile.json and exit",
    )
    return parser


async def _export_profile(session_id: str) -> None:
    settings = get_settings()
    engine, factory = create_database(settings.database_url)
    try:
        async with factory() as session:
            row = await session.get(ResumeProfileRow, session_id)
            if row is None or not row.profile_data:
                raise ValueError("No saved resume profile was found for that session.")
            profile = ResumeProfile.model_validate(row.profile_data)
            Path("profile.json").write_text(
                json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
    finally:
        await engine.dispose()
    print("Exported the session profile to profile.json.")


def _load_answers(path: str | None) -> dict[str, str | bool | float]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The answers file must contain a JSON object.")
    if not data:
        return {}
    return SaveApplicationAnswersInput.model_validate({"answers": data}).answers


def _job_for_url(url: str) -> Job:
    hostname = urlsplit(url).hostname
    if not hostname:
        raise ValueError("The application URL must include a hostname.")
    job_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return Job(
        id=f"cli-{job_id}",
        title="Application",
        company=hostname,
        location="Not specified",
        is_remote=False,
        experience_required="Not specified",
        source_link=url,
        source="Local CLI",
    )


async def _run_cli(args: argparse.Namespace) -> None:
    if args.export_profile:
        await _export_profile(args.export_profile)
        return
    if not args.url or not args.resume or not args.profile:
        raise ValueError("--url, --resume, and --profile are required unless --export-profile is used.")

    resume_path = Path(args.resume).expanduser().resolve()
    if not resume_path.is_file():
        raise ValueError("The resume file was not found.")
    profile_data = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    profile = ResumeProfile.model_validate(profile_data)
    answers = _load_answers(args.answers)
    settings = get_settings()
    model = create_llm_service(settings)
    session_id = f"cli-{uuid.uuid4().hex}"

    def print_report(report: AutofillReport) -> None:
        print(report.model_dump_json(indent=2))
        print(
            "\nThe browser is open for manual review. This CLI will never click "
            "the final submit button."
        )

    await fill_application_form(
        args.url,
        profile,
        answers,
        str(resume_path),
        "review",
        _job_for_url(args.url),
        session_id=session_id,
        uploads_dir=settings.uploads_dir,
        model=model,
        headed=args.headed,
        keep_open=True,
        on_report=print_report,
    )


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        asyncio.run(_run_cli(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
