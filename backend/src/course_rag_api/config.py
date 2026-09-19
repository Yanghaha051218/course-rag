from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from COURSE_RAG_* environment variables."""

    app_name: str = "CourseRAG API"
    environment: Literal["development", "test", "production"] = "development"
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_KEY", "COURSE_RAG_OPENAI_API_KEY"
        ),
    )
    database_path: Path = Path("runtime/db/course-rag.sqlite3")
    qdrant_path: Path = Path("runtime/qdrant")
    upload_path: Path = Path("runtime/uploads")
    qdrant_collection_prefix: str = Field(default="course_rag", min_length=1)
    embedding_provider: Literal["deterministic", "fastembed", "openai"] = "deterministic"
    embedding_model: str | None = Field(default=None, min_length=1)
    embedding_dimension: int | None = Field(default=None, gt=0)
    fastembed_model: str = Field(default="BAAI/bge-small-en-v1.5", min_length=1)
    fastembed_cache_dir: Path = Path("runtime/models/fastembed")
    embedding_batch_size: int = Field(default=64, gt=0)
    verifier_provider: Literal["openai"] = "openai"
    verifier_model: str = Field(default="gpt-4o-mini", min_length=1)
    generator_provider: Literal["openai"] = "openai"
    generator_model: str = Field(default="gpt-4o-mini", min_length=1)
    chunk_target_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    max_document_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_course_bytes: int = Field(default=500 * 1024 * 1024, gt=0)
    session_ttl_seconds: int = Field(default=7 * 24 * 60 * 60, gt=0)
    session_cookie_name: str = Field(default="course_rag_session", min_length=1)
    session_cookie_secure: bool = False
    auth_rate_limit_attempts: int = Field(default=10, gt=0)
    auth_rate_limit_window_seconds: int = Field(default=15 * 60, gt=0)

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
        if self.environment == "production" and not self.session_cookie_secure:
            raise ValueError("session_cookie_secure must be enabled in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
