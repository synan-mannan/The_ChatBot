from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from server.app.models.db import (
    ApplicationRow,
    create_database,
    initialize_database,
)


@pytest.mark.asyncio
async def test_existing_application_table_receives_new_columns_and_safe_status(
    tmp_path,
) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id VARCHAR(128) NOT NULL,
                job_id VARCHAR(128) NOT NULL,
                applied_date DATETIME NOT NULL,
                status VARCHAR(32) NOT NULL,
                CONSTRAINT uq_application_session_job UNIQUE (session_id, job_id)
            )
            """
        )
        connection.execute(
            "INSERT INTO applications (session_id, job_id, applied_date, status) "
            "VALUES (?, ?, ?, ?)",
            ("session-legacy", "42", datetime.now(timezone.utc).isoformat(), "applied"),
        )

    engine, factory = create_database(f"sqlite+aiosqlite:///{database_path}")
    await initialize_database(engine)
    async with factory() as session:
        row = await session.scalar(
            select(ApplicationRow).where(ApplicationRow.session_id == "session-legacy")
        )
    await engine.dispose()

    assert row is not None
    assert row.method == "manual"
    assert row.detail == "Legacy application record; submission was not verified."
    assert row.apply_url is None
    assert row.status == "dry_run"
