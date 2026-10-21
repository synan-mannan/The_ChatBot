from functools import lru_cache

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
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    jobs_file: str = "data/jobs.json"
    max_resume_bytes: int = 5 * 1024 * 1024
    client_dist: str = "client/dist"
    llm_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
