"""Memory retrieval for a turn and post-turn memory extraction (spec §9.2, §10).

`retrieve_for_turn` runs inside the turn and never raises: a failing store
means no memories (and the reply must not claim any, spec §23); a failing
embedding call falls back to full-text matching.

`process_turn` runs after the reply is persisted (post-turn processing). It
asks the LLM for candidates, applies the eligibility policy in code, embeds
what is kept and stores it. A failure there is logged and never affects the
completed turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from app.embeddings.base import EmbeddingClient, EmbeddingError
from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    ResponseSchema,
    TraceMetadata,
)
from app.memory.policy import decide
from app.memory.schemas import (
    MEMORY_EXTRACTION_JSON_SCHEMA,
    ExtractionOutput,
    Memory,
    MemoryStatus,
    RetrievedMemory,
)
from app.memory.store import MemoryStore, NewMemory, RetrievalParams
from app.observability import log_event
from app.prompting import PromptTemplate

logger = logging.getLogger("app.memory")

RetrievalMode = Literal["semantic", "lexical", "disabled", "unavailable"]
# Messages this short ("ok", "thanks!") never carry a memory: skip the LLM call.
MIN_EXTRACTION_CHARS = 20
_RECENT_MESSAGES = 4
_MESSAGE_CHARS = 1_500
_SHOWN_MEMORIES = 10


def _now() -> datetime:
    return datetime.now(UTC)


def _escape(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


@dataclass(frozen=True)
class RetrievalResult:
    memories: list[RetrievedMemory]
    mode: RetrievalMode
    failure: str | None = None


@dataclass(frozen=True)
class ProcessingResult:
    status: Literal["stored", "nothing_stored", "skipped", "failed"]
    reason: str | None = None
    stored: list[uuid.UUID] = field(default_factory=list)
    decisions: dict[str, int] = field(default_factory=dict)


def build_extraction_input(shown: Sequence[Memory], history: Sequence[ChatMessage]) -> str:
    memories = [
        {"number": i, "type": m.memory_type.value, "content": m.content, "status": m.status.value}
        for i, m in enumerate(shown)
    ]
    recent = []
    for message in history[-1 - _RECENT_MESSAGES : -1]:
        speaker = "Person" if message.role is ChatRole.USER else "Companion"
        recent.append(f"{speaker}: {_escape(message.content[:_MESSAGE_CHARS])}")
    return (
        "<existing_memories>\n"
        + _escape(json.dumps(memories, ensure_ascii=False))
        + "\n</existing_memories>\n\n<recent_messages>\n"
        + "\n\n".join(recent)
        + "\n</recent_messages>\n\n<latest_message>\n"
        + _escape(history[-1].content[:_MESSAGE_CHARS])
        + "\n</latest_message>\n\nPropose memory candidates from the latest message."
    )


class MemoryService:
    def __init__(
        self,
        *,
        store: MemoryStore,
        embeddings: EmbeddingClient | None,
        llm: LLMClient,
        model: str,
        prompt: PromptTemplate,
        params: GenerationParams,
        max_output_tokens: int,
        timeout_seconds: float,
        retrieval: RetrievalParams,
        retention_days: int,
        max_per_turn: int = 3,
        extraction_enabled: bool = True,
    ) -> None:
        self.store = store
        self._embeddings = embeddings
        self._llm = llm
        self._model = model
        self._prompt = prompt
        self._system = prompt.render()
        self._params = params
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout_seconds
        self._retrieval = retrieval
        self._retention_days = retention_days
        self._max_per_turn = max_per_turn
        self._extraction_enabled = extraction_enabled

    @property
    def embedding_model(self) -> str | None:
        return self._embeddings.model_id if self._embeddings is not None else None

    async def _embed(
        self, texts: Sequence[str], kind: Literal["query", "document"]
    ) -> list[list[float]] | None:
        if self._embeddings is None or not texts:
            return None
        try:
            return await self._embeddings.embed(texts, kind=kind)
        except EmbeddingError as exc:
            log_event(logger, "memory.embedding_failed", level=logging.WARNING, detail=exc.message)
            return None

    async def retrieve_for_turn(self, user_id: uuid.UUID, message: str) -> RetrievalResult:
        try:
            settings = await self.store.get_settings(user_id)
            if not settings.memory_enabled:
                return RetrievalResult([], "disabled")
            vectors = await self._embed([message], "query")
            memories = await self.store.retrieve(
                user_id,
                query=message,
                query_embedding=vectors[0] if vectors else None,
                embedding_model=self.embedding_model if vectors else None,
                now=_now(),
                params=self._retrieval,
                # Unconfirmed memories are shown flagged, so the companion can
                # ask which is current rather than guess (spec §10.6).
                statuses=(MemoryStatus.ACTIVE, MemoryStatus.PENDING_CONFIRMATION),
            )
        except (SQLAlchemyError, OSError):
            log_event(logger, "memory.store_unavailable", level=logging.WARNING, stage="retrieve")
            return RetrievalResult([], "unavailable", "store_unavailable")
        mode: RetrievalMode = "semantic" if vectors else "lexical"
        failure = "embedding_failed" if self._embeddings is not None and not vectors else None
        log_event(
            logger, "memory.retrieved", mode=mode, count=len(memories), embedding_failure=failure
        )
        return RetrievalResult(memories, mode, failure)

    async def process_turn(
        self,
        *,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        history: Sequence[ChatMessage],
        trace_id: str,
    ) -> ProcessingResult:
        latest = history[-1].content if history else ""
        if not self._extraction_enabled:
            return ProcessingResult("skipped", "extraction_disabled")
        if len(latest.strip()) < MIN_EXTRACTION_CHARS:
            return ProcessingResult("skipped", "too_short")
        settings = await self.store.get_settings(user_id)
        if not settings.memory_enabled:
            return ProcessingResult("skipped", "memory_disabled")
        shown = await self._related(user_id, latest)
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._llm.generate(
                    GenerationRequest(
                        model=self._model,
                        system=self._system,
                        messages=[
                            ChatMessage(
                                role=ChatRole.USER, content=build_extraction_input(shown, history)
                            )
                        ],
                        max_output_tokens=self._max_output_tokens,
                        timeout_seconds=self._timeout,
                        params=self._params,
                        trace=TraceMetadata(
                            trace_id=trace_id,
                            purpose=GenerationPurpose.MEMORY_EXTRACTION,
                            conversation_id=str(conversation_id),
                        ),
                        response_schema=ResponseSchema(
                            name="memory_candidates", json_schema=MEMORY_EXTRACTION_JSON_SCHEMA
                        ),
                    )
                )
        except TimeoutError:
            return self._failed("timeout")
        except LLMError as exc:
            return self._failed(f"llm:{exc.category.value}")
        try:
            output = ExtractionOutput.model_validate_json(response.content)
        except ValidationError:
            return self._failed("invalid_output")
        decisions = decide(
            output.candidates,
            latest_message=latest,
            shown=shown,
            settings=settings,
            now=_now(),
            retention_days=self._retention_days,
            max_per_turn=self._max_per_turn,
        )
        counts: dict[str, int] = {}
        for d in decisions:
            key = d.action if d.action != "reject" else f"reject:{d.reason}"
            counts[key] = counts.get(key, 0) + 1
        kept = [d for d in decisions if d.action != "reject"]
        vectors = await self._embed([d.content for d in kept], "document")
        new = [
            NewMemory(
                decision=d,
                embedding=vectors[i] if vectors else None,
                embedding_model=self.embedding_model if vectors else None,
            )
            for i, d in enumerate(kept)
        ]
        stored = (
            await self.store.apply(
                user_id,
                conversation_id=conversation_id,
                source_message_id=user_message_id,
                new=new,
                provenance={
                    "extraction_model": response.model_id,
                    "prompt": self._prompt.key,
                    "structured_mode": response.structured_mode.value
                    if response.structured_mode
                    else None,
                },
            )
            if new
            else []
        )
        log_event(
            logger,
            "memory.processed",
            candidates=len(output.candidates),
            stored=len(stored),
            decisions=counts,
        )
        return ProcessingResult("stored" if stored else "nothing_stored", None, stored, counts)

    async def _related(self, user_id: uuid.UUID, latest: str) -> list[Memory]:
        """Existing memories shown to the extractor, so it can spot duplicates,
        updates and conflicts: the most related active or pending ones."""
        vectors = await self._embed([latest], "query")
        retrieved = await self.store.retrieve(
            user_id,
            query=latest,
            query_embedding=vectors[0] if vectors else None,
            embedding_model=self.embedding_model if vectors else None,
            now=_now(),
            params=RetrievalParams(limit=_SHOWN_MEMORIES, min_semantic=0.3, lexical_only_ok=True),
            statuses=(MemoryStatus.ACTIVE, MemoryStatus.PENDING_CONFIRMATION),
        )
        return [r.memory for r in retrieved]

    def _failed(self, reason: str) -> ProcessingResult:
        log_event(logger, "memory.extraction_failed", level=logging.WARNING, failure=reason)
        return ProcessingResult("failed", reason)

    def start(self) -> None:
        if self._embeddings is not None:
            self._embeddings.start()

    async def backfill_embeddings(self, batch_size: int = 50, max_batches: int = 20) -> int:
        """Embed memories that have no vector from the current model: stored
        while the model was loading or a call failed, or embedded by a model
        configured earlier. Returns how many were embedded."""
        model = self.embedding_model
        if model is None:
            return 0
        done = 0
        for _ in range(max_batches):
            pending = await self.store.needing_embedding(model, limit=batch_size)
            if not pending:
                break
            vectors = await self._embed([content for _, content in pending], "document")
            if vectors is None:
                break
            done += await self.store.set_embeddings(
                [
                    (memory_id, vector)
                    for (memory_id, _), vector in zip(pending, vectors, strict=True)
                ],
                model,
            )
            if len(pending) < batch_size:
                break
        return done

    async def aclose(self) -> None:
        if self._embeddings is not None:
            await self._embeddings.aclose()
