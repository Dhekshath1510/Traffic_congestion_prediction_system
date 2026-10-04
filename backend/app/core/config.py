from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(
        default="Traffic Congestion Prediction API",
        description="Application name.",
    )

    app_version: str = Field(
        default="0.1.0",
        description="Application version.",
    )

    environment: str = Field(
        default="development",
        description="Application environment.",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode.",
    )

    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5433/traffic_db",
        description="PostgreSQL database connection URL.",
    )

    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    log_level: str = Field(
        default="INFO",
        description="Application logging level.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Returns:
        Application settings instance.
    """
    return Settings()
