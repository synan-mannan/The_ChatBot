from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import DateTime, Integer, JSON, String, UniqueConstraint, func
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
    status: Mapped[str] = mapped_column(String(32), default="applied", nullable=False)


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


async def get_db_session(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        yield session
