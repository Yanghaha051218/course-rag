from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from COURSE_RAG_* environment variables."""

    app_name: str = "CourseRAG API"
    environment: Literal["development", "test", "production"] = "development"
    openai_api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_prefix="COURSE_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
