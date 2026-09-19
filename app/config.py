"""Application configuration loaded from environment variables or a .env file.

Configuration is centralized here so that no other module reads `os.environ`
directly. Validation fails fast with a non-secret error message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    documents_path: Path = Field(default=Path("./data/documents"), alias="DOCUMENTS_PATH")
    index_path: Path = Field(default=Path("./data/index"), alias="INDEX_PATH")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE", gt=0)
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP", ge=0)
    top_k: int = Field(default=5, alias="TOP_K", gt=0)
    similarity_threshold: float = Field(
        default=0.45, alias="SIMILARITY_THRESHOLD", ge=0.0, le=1.0
    )
    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES", gt=0)
    max_input_characters: int = Field(default=8000, alias="MAX_INPUT_CHARACTERS", gt=0)
    model_timeout_seconds: float = Field(default=30.0, alias="MODEL_TIMEOUT_SECONDS", gt=0)
    model_max_tokens: int = Field(default=800, alias="MODEL_MAX_TOKENS", gt=0)
    model_max_retries: int = Field(default=3, alias="MODEL_MAX_RETRIES", ge=0)
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE", gt=0)

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    def require_groq_api_key(self) -> str:
        if not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Add it to your .env file or environment."
            )
        return self.groq_api_key

    def index_version_key(self) -> str:
        return f"{self.embedding_model}:{self.chunk_size}:{self.chunk_overlap}"


def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
