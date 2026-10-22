from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.autofill import _job_for_url


ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_is_available_without_launching_a_browser() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "autofill.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--export-profile" in result.stdout
    assert "--headed" in result.stdout


def test_cli_job_identifiers_are_safe_and_stable() -> None:
    first = _job_for_url("https://careers.example.test/apply")
    second = _job_for_url("https://careers.example.test/apply")

    assert first.id == second.id
    assert "/" not in first.id
