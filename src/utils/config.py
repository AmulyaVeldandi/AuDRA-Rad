from __future__ import annotations

"""Configuration helpers backed by environment variables and .env files."""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class RateLimitSettings(BaseModel):
    """API rate limit configuration."""

    requests_per_minute: int = Field(default=60, ge=1)
    burst_size: int = Field(default=15, ge=1)


class Settings(BaseSettings):
    """Global application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    ENVIRONMENT: Literal["dev", "staging", "prod"] = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")

    # LLM Backend selection
    LLM_BACKEND: Literal["nim", "ollama"] = Field(default="nim")

    # NIM configuration
    NIM_LLM_ENDPOINT: AnyHttpUrl = Field(default="http://nim-llm:8000/v1")
    NIM_EMBEDDING_ENDPOINT: AnyHttpUrl = Field(default="http://nim-embedding:8000/v1")
    NIM_LLM_API_KEY: SecretStr | None = Field(default=None)
    NIM_EMBEDDING_API_KEY: SecretStr | None = Field(default=None)

    # Ollama configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL_NAME: str = Field(default="llama3.2:latest")

    AWS_REGION: str | None = Field(default=None)
    OPENSEARCH_ENDPOINT: AnyHttpUrl | None = Field(default=None)

    API_RATE_LIMITS: RateLimitSettings = Field(default_factory=RateLimitSettings)

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
            raise ValueError(f"Unsupported log level '{value}'.")
        return level

    @model_validator(mode="after")
    def _validate_environment(self) -> "Settings":
        required_in_env = {"staging", "prod"}
        if self.ENVIRONMENT in required_in_env:
            missing: list[str] = []
            if self.NIM_LLM_API_KEY is None:
                missing.append("NIM_LLM_API_KEY")
            if self.NIM_EMBEDDING_API_KEY is None:
                missing.append("NIM_EMBEDDING_API_KEY")
            if self.AWS_REGION is None:
                missing.append("AWS_REGION")
            if self.OPENSEARCH_ENDPOINT is None:
                missing.append("OPENSEARCH_ENDPOINT")
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(
                    f"Missing required configuration for {self.ENVIRONMENT}: {missing_str}"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
