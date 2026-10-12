"""Application configuration.

Every deployment-dependent value is read from the environment (or a local
`.env` file). Secrets are held as `SecretStr` so they never appear in reprs or
logs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["none", "low", "medium", "high", "xhigh", "max"]

ProviderName = Literal[
    "anthropic",
    "openai",
    "groq",
    "gemini",
    "xai",
    "qwen",
    "deepseek",
    "openrouter",
    "together",
    "ollama",
    "openai_compatible",
    "fake",
]


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Blank values (e.g. "GENERATION_API_KEY=") fall back to the default.
        env_ignore_empty=True,
    )

    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"

    # --- Database -----------------------------------------------------------
    database_url: str = "postgresql+asyncpg://companion:companion@localhost:5432/companion"
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_statement_timeout_ms: int = Field(default=10_000, ge=100)

    # --- Authentication -----------------------------------------------------
    # Only a development token scheme exists. It is NOT production
    # authentication; production startup is refused until one is implemented.
    auth_mode: Literal["dev"] = "dev"
    # Comma-separated "user_handle:token" pairs, e.g. "alice:dev-token-alice".
    dev_auth_tokens: SecretStr | None = None

    # --- Generative LLM -----------------------------------------------------
    # Provider presets (default model, base URL, key variable) live in
    # app/llm/providers/registry.py; unset values below fall back to them.
    generation_provider: ProviderName = "anthropic"
    generation_model: str | None = None
    # Overrides the preset's endpoint; required for "openai_compatible".
    generation_base_url: str | None = None
    # Applies to whichever provider is selected. When unset, the provider's own
    # key setting below is used (e.g. GROQ_API_KEY for "groq").
    generation_api_key: SecretStr | None = None
    generation_timeout_seconds: float = Field(default=45.0, gt=0)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
    # Reasoning/thinking depth. Unset = the provider preset's default (only
    # Anthropic sets one). Each adapter maps it to its provider's parameter;
    # models without reasoning controls reject it, so leave it unset for them.
    generation_effort: Effort | None = None
    # Leave unset for models that reject sampling parameters (e.g. claude-opus-5-5).
    generation_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # Anthropic only. "default": server-side refusal fallback chosen by the
    # provider per refusal category.
    generation_refusal_fallback: Literal["default", "off"] = "default"
    # Unset = the provider preset's default (sized per provider/tier).
    max_output_tokens: int | None = Field(default=None, ge=256, le=128_000)

    # --- Provider API keys (read from the environment or .env) ----------------
    # Only the selected provider's key is used. The Anthropic SDK can also
    # resolve ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile by itself.
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("gemini_api_key", "google_api_key")
    )
    xai_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    together_api_key: SecretStr | None = None

    # --- Turn limits --------------------------------------------------------
    turn_deadline_seconds: float = Field(default=90.0, gt=0)
    # A pending turn older than this is considered abandoned (process crash,
    # cancelled request) and may be taken over. Must exceed the turn deadline.
    stale_turn_seconds: float = Field(default=150.0, gt=0)
    max_input_characters: int = Field(default=8_000, ge=1, le=100_000)
    max_request_bytes: int = Field(default=64 * 1024, ge=1024)
    # Unset = the provider preset's default.
    max_context_tokens: int | None = Field(default=None, ge=1_000)
    history_max_messages: int = Field(default=50, ge=1)
    max_response_characters: int = Field(default=8_000, ge=100)

    # --- Prompts ------------------------------------------------------------
    companion_prompt_version: str = "1.0.0"

    # --- Diagnostics --------------------------------------------------------
    expose_timing_metadata: bool = True

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.stale_turn_seconds <= self.turn_deadline_seconds:
            raise ValueError("STALE_TURN_SECONDS must be greater than TURN_DEADLINE_SECONDS")
        # The JSON body carries escaping overhead (up to 6 bytes per character
        # for \\uXXXX escapes); keep the byte limit above the character limit so
        # valid messages are not rejected as too large.
        if self.max_request_bytes < self.max_input_characters * 6 + 1024:
            raise ValueError("MAX_REQUEST_BYTES is too small for MAX_INPUT_CHARACTERS")
        if self.generation_provider == "openai_compatible" and not (
            self.generation_base_url and self.generation_model
        ):
            raise ValueError(
                "GENERATION_PROVIDER=openai_compatible requires GENERATION_BASE_URL "
                "and GENERATION_MODEL"
            )
        if self.app_env is AppEnv.PRODUCTION:
            if self.auth_mode == "dev":
                raise ValueError(
                    "Development authentication cannot run with APP_ENV=production; "
                    "production authentication is not implemented yet"
                )
            if self.generation_provider == "fake":
                raise ValueError("GENERATION_PROVIDER=fake is a development stub")
            if self.expose_timing_metadata:
                raise ValueError("EXPOSE_TIMING_METADATA must be false in production")
        return self

    def dev_token_map(self) -> dict[str, str]:
        """Return a token -> user-handle mapping parsed from DEV_AUTH_TOKENS."""
        if self.dev_auth_tokens is None:
            return {}
        mapping: dict[str, str] = {}
        for pair in self.dev_auth_tokens.get_secret_value().split(","):
            pair = pair.strip()
            if not pair:
                continue
            handle, sep, token = pair.partition(":")
            if not sep or not handle.strip() or not token.strip():
                raise ValueError("DEV_AUTH_TOKENS entries must look like 'handle:token'")
            mapping[token.strip()] = handle.strip()
        return mapping


@lru_cache
def get_settings() -> Settings:
    return Settings()
