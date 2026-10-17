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
    # Unset = the generation model. On Groq's free tier, qwen/qwen3.8-27b
    # (preview) is an option with its own rate limit; openai/gpt-oss-20b is
    # not (it fails strict schema validation for this schema).
    state_extraction_model: str | None = None
    state_extraction_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    state_extraction_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    state_prompt_version: str = "1.0.0"

    # --- Memory (spec §10) ---------------------------------------------------
    memory_enabled: bool = True
    # Post-turn candidate extraction (one structured LLM call per turn whose
    # message is long enough). Unset model = the reply model.
    memory_extraction_enabled: bool = True
    memory_extraction_model: str | None = None
    memory_extraction_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    memory_extraction_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    memory_prompt_version: str = "1.1.0"
    memory_max_retrieved: int = Field(default=5, ge=1, le=20)
    # Minimum cosine similarity for a semantic match. 0.49 is the best F1 on
    # the memory evaluation set's dev split with the local bge-small-en-v1.5
    # model (docs/memory.md); other embedding models need their own value.
    memory_min_semantic: float = Field(default=0.49, ge=0.0, le=1.0)
    memory_max_per_turn: int = Field(default=3, ge=1, le=10)
    # Every memory expires at most this many days after it is stored.
    memory_retention_days: int = Field(default=365, ge=1, le=3650)
    # Maintenance: delete expired memories and embed memories stored without a
    # vector (the local model was still loading, or EMBEDDING_MODEL changed).
    # The same pass deletes conversations past their retention period.
    memory_maintenance_interval_seconds: int = Field(default=600, ge=60)
    # A conversation with no new message for this many days is deleted with
    # its messages, state and per-turn records (memories have their own
    # retention and are kept).
    conversation_retention_days: int = Field(default=365, ge=1, le=3650)
    # Embeddings for semantic retrieval: "local" (fastembed on CPU, no key),
    # "openai", "gemini", or "none" (full-text matching only).
    embedding_provider: Literal["none", "local", "openai", "gemini", "fake"] = "local"
    # Unset = the provider default (see app/embeddings/factory.py).
    embedding_model: str | None = None
    embedding_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    # openai/gemini only: request shorter vectors (unset = the model's default).
    embedding_dimensions: int | None = Field(default=None, ge=1, le=4096)
    # local only: where the model is downloaded (about 65 MB). Unset = fastembed's
    # default, FASTEMBED_CACHE_PATH or the OS temp folder, which cleanup tools may empty.
    embedding_cache_dir: str | None = None

    # --- Knowledge retrieval (spec §11, docs/rag.md) ---------------------------
    # Runs only for explicit information requests (app/rag/routing.py), never
    # on safety pathways. Needs an ingested corpus (scripts/ingest_knowledge.py);
    # with none, lookups find nothing and replies go ahead without evidence.
    knowledge_enabled: bool = True
    knowledge_max_results: int = Field(default=3, ge=1, le=8)
    # Minimum cosine similarity, calibrated for the default local embedding
    # model on the retrieval evaluation's dev split (docs/rag.md); another
    # embedding model needs its own value.
    knowledge_min_semantic: float = Field(default=0.70, ge=0.0, le=1.0)
    # Minimum full-text rank, used only when no query embedding is available.
    knowledge_min_lexical: float = Field(default=0.46, ge=0.0, le=1.0)
    knowledge_max_per_document: int = Field(default=2, ge=1, le=8)
    knowledge_timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    # Chunking for ingestion; a change re-chunks every source on the next run.
    knowledge_chunk_max_words: int = Field(default=180, ge=20, le=400)
    knowledge_chunk_overlap_words: int = Field(default=30, ge=0, le=150)
    knowledge_chunk_min_words: int = Field(default=25, ge=0, le=200)

    # --- Decisions (spec §7) -------------------------------------------------
    # "jev": TypeSafe's hosted API. "laya": a self-hosted Laya server
    # (laya-serve) speaking the same protocol. "llm": a general LLM answers the
    # same typed questions (app/decisions/llm_backend.py). "off": documented
    # fallbacks only.
    decision_provider: Literal["off", "jev", "laya", "llm"] = "off"
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
    # DECISION_PROVIDER=llm: any provider preset, with that provider's own key
    # (e.g. GROQ_API_KEY). The default is deliberately not the reply model
    # (Groq's default is openai/gpt-oss-120b), so decisions do not share its
    # per-model rate limit. Changing the provider, model or prompt version
    # selects other thresholds (app/decisions/thresholds.py).
    decision_llm_provider: ProviderName = "groq"
    decision_llm_model: str = Field(default="openai/gpt-oss-20b", min_length=1)
    # As GENERATION_EFFORT: mapped per provider; "off" sends no effort parameter.
    decision_llm_effort: Effort | Literal["off"] = "low"
    # Includes hidden reasoning tokens on reasoning models.
    decision_llm_max_output_tokens: int = Field(default=1024, ge=256, le=8192)
    decision_prompt_version: str = "1.0.0"

    # --- Safety (spec §12, Phase 5) -----------------------------------------
    # The safety layer runs on every turn: a deterministic pattern screen, plus
    # an LLM risk classifier when one is configured. "auto": the classifier
    # runs on Groq when GROQ_API_KEY is set, otherwise it is off and every turn
    # runs in degraded mode (patterns only, recorded as such). The default
    # model is not the reply model, so it has its own free-tier limits.
    safety_enabled: bool = True
    safety_classifier_provider: ProviderName | Literal["auto", "off"] = "auto"
    safety_classifier_model: str = Field(default="openai/gpt-oss-safeguard-20b", min_length=1)
    safety_classifier_effort: Effort | Literal["off"] = "low"
    safety_classifier_max_output_tokens: int = Field(default=300, ge=64, le=4096)
    safety_classifier_timeout_seconds: float = Field(default=4.0, gt=0, le=30)
    safety_policy_version: str = "1.0.0"
    # Crisis resources: "" = generic (no numbers), or IN, US, GB
    # (app/safety/resources_v1.toml, verified official sources).
    safety_region: str = ""
    # A semantic check of replies on safety pathways only, with the
    # classifier's model; its failure never blocks a reply.
    safety_output_llm_check: bool = True
    safety_output_policy_version: str = "1.0.0"

    # --- Turn limits --------------------------------------------------------
    turn_deadline_seconds: float = Field(default=90.0, gt=0)
    # A pending turn older than this is considered abandoned (process crash,
    # cancelled request) and may be taken over. Must exceed the turn deadline.
    stale_turn_seconds: float = Field(default=150.0, gt=0)
    # Per-user message limits (spec §18, §22.3), counted from stored user
    # messages across all of the user's conversations, so they hold across
    # workers and restarts. 0 disables a limit.
    user_messages_per_minute: int = Field(default=20, ge=0)
    user_messages_per_day: int = Field(default=1000, ge=0)
    max_input_characters: int = Field(default=8_000, ge=1, le=100_000)
    max_request_bytes: int = Field(default=64 * 1024, ge=1024)
    # Unset = the provider preset's default.
    max_context_tokens: int | None = Field(default=None, ge=1_000)
    history_max_messages: int = Field(default=50, ge=1)
    max_response_characters: int = Field(default=8_000, ge=100)

    # --- Prompts ------------------------------------------------------------
    companion_prompt_version: str = "1.8.0"

    # --- Voice (docs/voice-interface.md) --------------------------------------
    # Speech in and out over a WebSocket, beside the text API. Off by default.
    voice_enabled: bool = False
    stt_provider: Literal["none", "deepgram", "sarvam", "gnani", "fake"] = "none"
    tts_provider: Literal["none", "deepgram", "sarvam", "gnani", "fake"] = "none"
    # Unset = the provider's default model / voice (app/voice/factory.py).
    stt_model: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    # BCP-47 language of the conversation, e.g. en-IN, en-US, hi-IN.
    voice_language: str = "en-IN"
    # The client sends 16-bit mono PCM at this rate.
    voice_input_sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    # One side speaks at a time. "full" (barge-in) is not available yet.
    voice_duplex: Literal["half"] = "half"
    # "client": push-to-talk, the turn ends when the person lets go.
    # "provider": hands-free, the speech-to-text provider's end of utterance.
    voice_end_of_turn: Literal["client", "provider"] = "client"
    voice_max_utterance_seconds: float = Field(default=60.0, ge=5.0, le=300.0)
    # A WebSocket connects with a single-use ticket from
    # POST /v1/conversations/{id}/voice/tickets, valid this long.
    voice_ticket_ttl_seconds: int = Field(default=60, ge=10, le=600)
    # Per person (docs/voice-interface.md, "Limits"): open sessions, seconds
    # of audio sent to speech-to-text per UTC day, and how long a session may
    # go without any message from the client.
    voice_max_sessions_per_user: int = Field(default=2, ge=1, le=20)
    voice_audio_seconds_per_day: int = Field(default=3600, ge=60, le=86_400)
    voice_idle_timeout_seconds: int = Field(default=600, ge=30, le=7200)
    deepgram_api_key: SecretStr | None = None
    sarvam_api_key: SecretStr | None = None
    gnani_api_key: SecretStr | None = None

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
        if self.decision_provider == "llm":
            # Imported here because the registry imports this module. Resolves the
            # decision model exactly as the reply model is resolved, so a missing
            # key (e.g. GROQ_API_KEY) fails at startup, not on the first turn.
            from app.llm.providers.registry import resolve_decision_llm

            resolve_decision_llm(self)
        if self.voice_enabled:
            for kind, provider in (("STT", self.stt_provider), ("TTS", self.tts_provider)):
                if provider in ("none", "fake"):
                    continue
                if getattr(self, f"{provider}_api_key") is None:
                    raise ValueError(
                        f"{kind}_PROVIDER={provider} requires {provider.upper()}_API_KEY"
                    )
        if self.safety_enabled:
            from app.safety.resources import normalize_region

            normalize_region(self.safety_region)  # unknown codes fail at startup
        if self.app_env is AppEnv.PRODUCTION:
            if not self.safety_enabled:
                raise ValueError("SAFETY_ENABLED=false is not allowed in production")
            if self.auth_mode == "dev":
                raise ValueError(
                    "Development authentication cannot run with APP_ENV=production; "
                    "production authentication is not implemented yet"
                )
            if self.generation_provider == "fake":
                raise ValueError("GENERATION_PROVIDER=fake is a development stub")
            if self.decision_provider == "llm" and self.decision_llm_provider == "fake":
                raise ValueError("DECISION_LLM_PROVIDER=fake is a development stub")
            if self.embedding_provider == "fake":
                raise ValueError("EMBEDDING_PROVIDER=fake is a development stub")
            if self.expose_timing_metadata:
                raise ValueError("EXPOSE_TIMING_METADATA must be false in production")
            if "fake" in (self.stt_provider, self.tts_provider):
                raise ValueError("STT_PROVIDER / TTS_PROVIDER=fake are development stubs")
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
