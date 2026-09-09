from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from COURSE_RAG_* environment variables."""

    app_name: str = "CourseRAG API"
    environment: Literal["development", "test", "production"] = "development"
    openai_api_key: SecretStr | None = None
    database_path: Path = Path("runtime/db/course-rag.sqlite3")
    chunk_target_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    max_document_bytes: int = Field(default=50 * 1024 * 1024, gt=0)

    model_config = SettingsConfigDict(
        env_prefix="COURSE_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_target_size:
            raise ValueError("chunk overlap must be smaller than target size")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
