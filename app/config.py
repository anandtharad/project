"""Application configuration.

Every deployment-dependent value is read from the environment (or a local
`.env` file). Secrets are held as `SecretStr` so they never appear in reprs or
logs.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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
    "mistral",
    "ollama",
    "openai_compatible",
    "fake",
]


def detect_env_file_encoding(env_file: object) -> str | None:
    """Encoding of a .env file, from its byte-order mark.

    Windows tools save text in several encodings: PowerShell 5.1's
    `echo ... > .env` writes UTF-16, and some editors add a UTF-8 BOM.
    Reading those as plain UTF-8 fails or corrupts the first key.
    """
    if not isinstance(env_file, (str, Path)):
        return None
    try:
        with open(env_file, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return None
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return "utf-8-sig"  # plain UTF-8, with or without a BOM


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
    # Reasoning/thinking depth, mapped by each adapter to its provider's own
    # control. Unset = the preset default, which the anthropic, openai, groq
    # and gemini presets set to "low" for models whose controls are known.
    # "off" sends no effort parameter at all (for models without reasoning
    # controls).
    generation_effort: Effort | Literal["off"] | None = None
    # Leave unset for models that reject sampling parameters (e.g. claude-opus-5-5).
    # Anthropic accepts 0-1; OpenAI reasoning models only with effort "none".
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
    mistral_api_key: SecretStr | None = None

    # --- Conversation state (spec §8) ---------------------------------------
    # One structured LLM call before each reply updates the conversation state.
    state_extraction_enabled: bool = True
    # Unset = the generation model, except on Groq, where openai/gpt-oss-20b
    # is used so extraction draws on a separate free-tier rate limit.
    state_extraction_model: str | None = None
    state_extraction_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    state_extraction_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    state_prompt_version: str = "1.0.0"

    # --- Decisions (spec §7) -------------------------------------------------
    # "jev": TypeSafe's hosted API. "laya": a self-hosted Laya server
    # (laya-serve) speaking the same protocol. "off": documented fallbacks only.
    decision_provider: Literal["off", "jev", "laya"] = "off"
    jev_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("jev_api_key", "typesafe_api_key")
    )
    jev_base_url: str = "https://api.typesafe.ai"
    # A pinned version, not an alias, so evaluations stay reproducible.
    jev_model: str = "jev-1.13.0"
    laya_base_url: str = "http://localhost:8100"
    laya_api_key: SecretStr | None = None
    # A laya-serve checkpoint name; "english" suits this English-first app.
    laya_model: str = "english"
    decision_timeout_seconds: float = Field(
        default=3.0,
        gt=0,
        le=30,
        validation_alias=AliasChoices("decision_timeout_seconds", "jev_timeout_seconds"),
    )
    decision_max_retries: int = Field(default=1, ge=0, le=3)

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
    companion_prompt_version: str = "1.1.0"

    # --- Diagnostics --------------------------------------------------------
    expose_timing_metadata: bool = True

    def __init__(self, **values: Any) -> None:
        if values.get("_env_file_encoding") is None:
            env_file = values.get("_env_file", type(self).model_config.get("env_file"))
            values["_env_file_encoding"] = detect_env_file_encoding(env_file)
        super().__init__(**values)

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
        if self.decision_provider == "jev" and self.jev_api_key is None:
            raise ValueError("DECISION_PROVIDER=jev requires JEV_API_KEY (or TYPESAFE_API_KEY)")
        if self.decision_provider == "laya" and self.laya_api_key is None:
            # laya-serve accepts any caller when it has no key, so require one.
            raise ValueError(
                "DECISION_PROVIDER=laya requires LAYA_API_KEY (set the same on laya-serve)"
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
