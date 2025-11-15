"""Centralized configuration powered by Pydantic settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application level configuration resolved from env and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AGENT_SMITH_",
        extra="ignore",
    )

    env: str = Field(
        default="development",
        validation_alias="AGENT_SMITH_ENV",
        description="Controls general runtime mode (development, staging, prod).",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="Optional key for enabling OpenAI hosted models.",
    )
    sqlite_path: Path = Field(
        default=Path("./var/agent_smith.db"),
        description="On-disk path for the SQLite database backing SQLModel.",
    )
    chroma_path: Path = Field(
        default=Path("./var/chroma"),
        description="Directory used by ChromaDB for persistent embeddings.",
    )

    @property
    def database_url(self) -> str:
        """Return a SQLAlchemy compatible SQLite connection string."""

        return f"sqlite:///{self.sqlite_path}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for app-wide reuse."""

    settings = Settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["Settings", "get_settings"]
