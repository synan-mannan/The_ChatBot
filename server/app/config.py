from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "AIJob"
    database_url: str = "sqlite+aiosqlite:///./data/aijob.db"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    real_apply_enabled: bool = False
    require_review: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    smtp_from: str = ""
    jobs_file: str = "data/jobs.json"
    max_resume_bytes: int = 5 * 1024 * 1024
    uploads_dir: str = "uploads"
    client_dist: str = "client/dist"
    llm_max_retries: int = Field(default=3, ge=1, le=8)


@lru_cache
def get_settings() -> Settings:
    return Settings()
