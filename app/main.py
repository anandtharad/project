"""Application factory. Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from fastapi import FastAPI

from app.api import conversations, health, messages
from app.api.errors import register_error_handlers
from app.api.middleware import BodySizeLimitMiddleware, TraceMiddleware
from app.config import AppEnv, Settings, get_settings
from app.context.builder import ContextBuilder
from app.db.session import create_engine, create_session_factory
from app.decisions.factory import build_decision_service
from app.decisions.questions import QUESTIONS, verify_locked
from app.decisions.service import DecisionBackend
from app.llm.base import LLMClient
from app.llm.factory import build_llm_client
from app.llm.providers.registry import ResolvedProvider, resolve_provider
from app.llm.schemas import GenerationParams
from app.observability import configure_logging, log_event
from app.orchestration.engine import TurnOrchestrator
from app.orchestration.persistence import TurnStore
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptRegistry, PromptTemplate
from app.state.extractor import StateExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPANION_PROMPT_ID = "companion_system"
STATE_PROMPT_ID = "state_extraction"

logger = logging.getLogger("app")


def migration_head() -> str:
    config = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise RuntimeError("No migrations found")
    return head


def state_extraction_model(settings: Settings, resolved: ResolvedProvider) -> str:
    # The reply model by default. A smaller model is not a safe default: on
    # Groq, openai/gpt-oss-20b failed Groq's own schema validation
    # (json_validate_failed) in 5 of 5 live extraction runs on 2026-10-13.
    return settings.state_extraction_model or resolved.model


def build_state_extractor(
    settings: Settings, resolved: ResolvedProvider, llm: LLMClient, prompt: PromptTemplate
) -> StateExtractor | None:
    if not settings.state_extraction_enabled:
        return None
    return StateExtractor(
        llm=llm,
        model=state_extraction_model(settings, resolved),
        prompt=prompt,
        params=GenerationParams(effort=resolved.effort),
        max_output_tokens=settings.state_extraction_max_output_tokens,
        timeout_seconds=settings.state_extraction_timeout_seconds,
    )


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    decision_backend: DecisionBackend | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # Fail fast on configuration/prompt problems before accepting traffic.
    registry = PromptRegistry()
    prompt = registry.get(COMPANION_PROMPT_ID, settings.companion_prompt_version)
    state_prompt = registry.get(STATE_PROMPT_ID, settings.state_prompt_version)
    verify_locked(QUESTIONS)
    dev_tokens = settings.dev_token_map()
    head = migration_head()
    resolved = resolve_provider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        session_factory = create_session_factory(engine)
        llm = llm_client or build_llm_client(settings, resolved)
        decisions = build_decision_service(settings, backend=decision_backend)
        state_extractor = build_state_extractor(settings, resolved, llm, state_prompt)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.llm = llm
        app.state.prompt = prompt
        app.state.dev_tokens = dev_tokens
        app.state.migration_head = head
        app.state.resolved_provider = resolved
        app.state.orchestrator = TurnOrchestrator(
            settings=settings,
            model=resolved.model,
            effort=resolved.effort,
            max_output_tokens=resolved.max_output_tokens,
            store=TurnStore(
                session_factory,
                stale_turn_seconds=settings.stale_turn_seconds,
                history_limit=settings.history_max_messages,
            ),
            llm=llm,
            prompt=prompt,
            context_builder=ContextBuilder(
                max_context_tokens=resolved.max_context_tokens,
                max_history_messages=settings.history_max_messages,
            ),
            validator=ResponseValidator(max_characters=settings.max_response_characters),
            decisions=decisions,
            state_extractor=state_extractor,
        )
        log_event(
            logger,
            "app.started",
            env=settings.app_env.value,
            llm_provider=llm.provider,
            model=resolved.model,
            prompt=prompt.key,
            migration_head=head,
            decision_provider=settings.decision_provider,
            state_model=state_extractor.model if state_extractor else None,
        )
        for warning in resolved.warnings:
            log_event(logger, "config.warning", level=logging.WARNING, detail=warning)
        try:
            yield
        finally:
            await decisions.aclose()
            await llm.aclose()
            await engine.dispose()

    is_production = settings.app_env is AppEnv.PRODUCTION
    app = FastAPI(
        title="Companion AI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None,
        openapi_url=None if is_production else "/openapi.json",
    )
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(conversations.router)
    app.include_router(messages.router)
    # Added last = outermost: tracing wraps the size limit.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(TraceMiddleware)
    return app
