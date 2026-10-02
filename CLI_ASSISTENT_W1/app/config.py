"""Application configuration.

All tunable values come from environment variables (optionally loaded from a
``.env`` file) and are validated once at startup, so the rest of the code can
trust them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

# Maps each config field to the environment variable that sets it.
_ENV_VARS: dict[str, str] = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_model": "LLM_MODEL",
    "llm_api_key": "LLM_API_KEY",
    "llm_temperature": "LLM_TEMPERATURE",
    "llm_top_p": "LLM_TOP_P",
    "llm_request_timeout_seconds": "LLM_REQUEST_TIMEOUT_SECONDS",
    "max_history_messages": "MAX_HISTORY_MESSAGES",
    "log_level": "LOG_LEVEL",
    "log_file": "LOG_FILE",
    "show_token_usage": "SHOW_TOKEN_USAGE",
}


class ConfigError(Exception):
    """Raised when the environment does not describe a valid configuration."""


class AppConfig(BaseModel):
    """Validated, immutable application settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = Field(min_length=1)
    # SecretStr keeps the key out of repr() and therefore out of logs.
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0.0, le=600.0)
    max_history_messages: int = Field(default=20, ge=2, le=200)
    log_level: LogLevel = "INFO"
    log_file: Path = Path("logs/app.log")
    show_token_usage: bool = True

    @field_validator("llm_base_url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Build the configuration from environment variables.

    Args:
        env: Variables to read from. Defaults to the process environment after
            loading ``.env``. Passing a mapping keeps tests independent of the
            developer's machine.

    Raises:
        ConfigError: If a value is missing or invalid. The message names the
            environment variable so the user knows what to fix.
    """
    if env is None:
        load_dotenv()
        env = os.environ

    raw = {field: env[var] for field, var in _ENV_VARS.items() if var in env}
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_describe_validation_error(exc)) from exc


def _describe_validation_error(exc: ValidationError) -> str:
    problems = []
    for error in exc.errors():
        field = str(error["loc"][0]) if error["loc"] else "configuration"
        name = _ENV_VARS.get(field, field)
        problems.append(f"{name}: {error['msg']}")
    return "Invalid configuration. " + "; ".join(problems)
