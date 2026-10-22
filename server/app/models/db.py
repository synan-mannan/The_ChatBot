from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, AsyncIterator

from sqlalchemy import (
    DateTime,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    inspect,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from server.app.config import get_settings


class Base(DeclarativeBase):
    pass


class ResumeProfileRow(Base):
    __tablename__ = "resume_profiles"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    profile_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    answers_data: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class ApplicationRow(Base):
    __tablename__ = "applications"
    __table_args__ = (UniqueConstraint("session_id", "job_id", name="uq_application_session_job"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    job_id: Mapped[str] = mapped_column(String(128), nullable=False)
    applied_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), default="in_progress", server_default="in_progress", nullable=False
    )
    method: Mapped[str] = mapped_column(
        String(32), default="manual", server_default="manual", nullable=False
    )
    detail: Mapped[str] = mapped_column(
        Text, default="", server_default="", nullable=False
    )
    apply_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    report_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)


def _async_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def create_database(url: str | None = None) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    database_url = _async_database_url(url or get_settings().database_url)
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def initialize_database(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_migrate_application_columns)
        await connection.run_sync(_migrate_resume_profile_columns)


def _migrate_application_columns(connection: Any) -> None:
    inspector = inspect(connection)
    if "applications" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("applications")}
    additions = {
        "method": "VARCHAR(32) NOT NULL DEFAULT 'manual'",
        "detail": "TEXT NOT NULL DEFAULT ''",
        "apply_url": "VARCHAR(2048)",
        "report_data": "JSON",
    }
    for name, definition in additions.items():
        if name not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE applications ADD COLUMN {name} {definition}"
            )
    connection.execute(
        text(
            "UPDATE applications SET status = 'dry_run', detail = "
            "CASE WHEN detail = '' THEN 'Legacy application record; submission was not verified.' "
            "ELSE detail END WHERE status = 'applied'"
        )
    )


def _migrate_resume_profile_columns(connection: Any) -> None:
    inspector = inspect(connection)
    if "resume_profiles" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("resume_profiles")}
    if "answers_data" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE resume_profiles ADD COLUMN answers_data JSON NOT NULL DEFAULT '{}'"
        )


async def get_db_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        yield session
