from __future__ import annotations

import json
from pathlib import Path

from server.app.models.schemas import Job


def load_jobs(path: str | Path) -> list[Job]:
    job_file = Path(path)
    if not job_file.is_file():
        raise FileNotFoundError(f"Job dataset was not found: {job_file}")
    try:
        payload = json.loads(job_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Job dataset could not be loaded: {job_file}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("Job dataset must be a non-empty JSON array.")
    jobs = [Job.model_validate(record) for record in payload]
    ids = [job.id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Job dataset contains duplicate IDs.")
    return jobs
