"""Application factory. Run with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from fastapi import FastAPI

from app.api import conversations, health, memories, messages
from app.api.errors import register_error_handlers
from app.api.middleware import BodySizeLimitMiddleware, TraceMiddleware
from app.background import BackgroundTasks
from app.config import AppEnv, Effort, Settings, get_settings
from app.context.builder import ContextBuilder
from app.db.session import create_engine, create_session_factory
from app.decisions.factory import build_decision_service
from app.decisions.llm_backend import DECISION_PROMPT_ID
from app.decisions.questions import QUESTIONS, verify_locked
from app.decisions.service import DecisionBackend
from app.embeddings.base import EmbeddingClient
from app.embeddings.factory import build_embedding_client
from app.llm.base import LLMClient
from app.llm.factory import build_llm_client
from app.llm.providers.registry import ResolvedProvider, resolve_provider
from app.llm.schemas import GenerationParams
from app.memory.service import MemoryExtractor, MemoryService
from app.memory.store import MemoryStore, RetrievalParams
from app.observability import configure_logging, log_event
from app.orchestration.engine import TurnOrchestrator
from app.orchestration.persistence import TurnStore
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptRegistry, PromptTemplate
from app.rag.factory import build_knowledge_service
from app.safety.factory import build_safety_layer
from app.safety.policy import PATHWAY_RULES
from app.state.extractor import StateExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPANION_PROMPT_ID = "companion_system"
STATE_PROMPT_ID = "state_extraction"
MEMORY_PROMPT_ID = "memory_extraction"

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


def state_extraction_effort(
    settings: Settings, resolved: ResolvedProvider, model: str
) -> Effort | None:
    """The effort setting resolved for the extraction model itself: the reply
    model's effort can be invalid for another model (e.g. reasoning_effort sent
    to a model without reasoning controls)."""
    if model == resolved.model:
        return resolved.effort
    return resolve_provider(settings.model_copy(update={"generation_model": model})).effort


def build_state_extractor(
    settings: Settings, resolved: ResolvedProvider, llm: LLMClient, prompt: PromptTemplate
) -> StateExtractor | None:
    if not settings.state_extraction_enabled:
        return None
    model = state_extraction_model(settings, resolved)
    return StateExtractor(
        llm=llm,
        model=model,
        prompt=prompt,
        params=GenerationParams(effort=state_extraction_effort(settings, resolved, model)),
        max_output_tokens=settings.state_extraction_max_output_tokens,
        timeout_seconds=settings.state_extraction_timeout_seconds,
    )


def build_memory_service(
    settings: Settings,
    resolved: ResolvedProvider,
    llm: LLMClient,
    prompt: PromptTemplate,
    store: MemoryStore,
    embeddings: EmbeddingClient | None,
) -> MemoryService | None:
    if not settings.memory_enabled:
        return None
    return MemoryService(
        store=store,
        embeddings=embeddings,
        extractor=build_memory_extractor(settings, resolved, llm, prompt),
        retrieval=RetrievalParams(
            limit=settings.memory_max_retrieved, min_semantic=settings.memory_min_semantic
        ),
        retention_days=settings.memory_retention_days,
        max_per_turn=settings.memory_max_per_turn,
        embedding_timeout_seconds=settings.embedding_timeout_seconds,
    )


def build_memory_extractor(
    settings: Settings, resolved: ResolvedProvider, llm: LLMClient, prompt: PromptTemplate
) -> MemoryExtractor | None:
    if not settings.memory_extraction_enabled:
        return None
    model = settings.memory_extraction_model or resolved.model
    return MemoryExtractor(
        llm=llm,
        model=model,
        prompt=prompt,
        params=GenerationParams(effort=state_extraction_effort(settings, resolved, model)),
        max_output_tokens=settings.memory_extraction_max_output_tokens,
        timeout_seconds=settings.memory_extraction_timeout_seconds,
    )


async def maintain_memories(store: MemoryStore, memory: MemoryService | None) -> None:
    """One maintenance pass: retention purge, then embeddings for memories
    stored without a current-model vector. Each part fails on its own."""
    try:
        purged = await store.purge_expired(datetime.now(UTC))
        if purged:
            log_event(logger, "memory.purged", count=purged)
    except Exception as exc:  # retention must keep running
        log_event(
            logger, "memory.purge_failed", level=logging.WARNING, error_type=type(exc).__name__
        )
    if memory is None:
        return
    try:
        embedded = await memory.backfill_embeddings()
        if embedded:
            log_event(logger, "memory.embeddings_backfilled", count=embedded)
    except Exception as exc:
        log_event(
            logger, "memory.backfill_failed", level=logging.WARNING, error_type=type(exc).__name__
        )


async def maintain_memories_periodically(
    store: MemoryStore, memory: MemoryService | None, interval_seconds: int
) -> None:
    while True:
        await maintain_memories(store, memory)
        await asyncio.sleep(interval_seconds)


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    decision_backend: DecisionBackend | None = None,
    safety_llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    # Fail fast on configuration/prompt problems before accepting traffic.
    registry = PromptRegistry()
    prompt = registry.get(COMPANION_PROMPT_ID, settings.companion_prompt_version)
    state_prompt = registry.get(STATE_PROMPT_ID, settings.state_prompt_version)
    memory_prompt = registry.get(MEMORY_PROMPT_ID, settings.memory_prompt_version)
    if settings.decision_provider == "llm":
        registry.get(DECISION_PROMPT_ID, settings.decision_prompt_version)
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
        safety = build_safety_layer(settings, llm=safety_llm)
        state_extractor = build_state_extractor(settings, resolved, llm, state_prompt)
        memory_store = MemoryStore(session_factory, retention_days=settings.memory_retention_days)
        # One embedding client (one local model in memory) for memories and knowledge.
        embeddings = (
            build_embedding_client(settings)
            if settings.memory_enabled or settings.knowledge_enabled
            else None
        )
        memory = build_memory_service(
            settings, resolved, llm, memory_prompt, memory_store, embeddings
        )
        knowledge = build_knowledge_service(settings, session_factory, embeddings)
        background = BackgroundTasks()
        app.state.memory_store = memory_store
        app.state.memory = memory
        app.state.background = background
        if embeddings is not None:
            embeddings.start()  # loads a local embedding model in the background
        maintenance_task = asyncio.create_task(
            maintain_memories_periodically(
                memory_store, memory, settings.memory_maintenance_interval_seconds
            )
        )
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.llm = llm
        app.state.prompt = prompt
        app.state.dev_tokens = dev_tokens
        app.state.migration_head = head
        app.state.resolved_provider = resolved
        app.state.turn_store = turn_store = TurnStore(
            session_factory,
            stale_turn_seconds=settings.stale_turn_seconds,
            history_limit=settings.history_max_messages,
            messages_per_minute=settings.user_messages_per_minute,
            messages_per_day=settings.user_messages_per_day,
        )
        app.state.orchestrator = TurnOrchestrator(
            settings=settings,
            model=resolved.model,
            effort=resolved.effort,
            max_output_tokens=resolved.max_output_tokens,
            store=turn_store,
            llm=llm,
            prompt=prompt,
            context_builder=ContextBuilder(
                max_context_tokens=resolved.max_context_tokens,
                max_history_messages=settings.history_max_messages,
            ),
            validator=ResponseValidator(max_characters=settings.max_response_characters),
            decisions=decisions,
            state_extractor=state_extractor,
            memory=memory,
            background=background,
            safety=safety,
            knowledge=knowledge,
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
            decision_calibration=decisions.calibration_key,
            state_model=state_extractor.model if state_extractor else None,
            memory_enabled=memory is not None,
            embedding_model=memory.embedding_model if memory else None,
            safety_enabled=safety is not None,
            safety_classifier=(
                safety.analyzer.classifier.model
                if safety is not None and safety.analyzer.classifier is not None
                else None
            ),
            safety_region=safety.region if safety is not None else None,
            knowledge_enabled=knowledge is not None,
        )
        if safety is not None and not {
            rule.fragment for rule in PATHWAY_RULES.values() if rule.fragment
        } <= set(prompt.fragments):
            log_event(
                logger,
                "config.warning",
                level=logging.WARNING,
                detail=f"{prompt.key} has no safety pathway fragments; safety pathways use "
                "its danger guidance only (use companion_system 1.6.0 or later)",
            )
        for warning in resolved.warnings:
            log_event(logger, "config.warning", level=logging.WARNING, detail=warning)
        try:
            yield
        finally:
            maintenance_task.cancel()
            await asyncio.gather(maintenance_task, return_exceptions=True)
            await background.aclose()
            if memory is not None:
                await memory.aclose()  # closes the shared embedding client
            elif embeddings is not None:
                await embeddings.aclose()
            await decisions.aclose()
            if safety is not None:
                await safety.aclose()
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
    app.include_router(memories.router)
    # Added last = outermost: tracing wraps the size limit.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(TraceMiddleware)
    return app
