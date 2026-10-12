"""Turn orchestrator: sequences one conversational turn end to end.

Phase 1 lifecycle:
RECEIVED -> VALIDATED -> AUTHORIZED -> USER_MESSAGE_PERSISTED -> CONTEXT_LOADED
-> CONTEXT_BUILT -> RESPONSE_GENERATED -> RESPONSE_VALIDATED
-> ASSISTANT_MESSAGE_PERSISTED -> COMPLETED

Failure behavior (the user message is always preserved once persisted):
- unknown / foreign conversation -> 404, nothing persisted, no model call
- turn already in flight          -> 409 turn_in_progress
- provider error                  -> turn marked failed, 503 dependency_unavailable
- turn deadline exceeded          -> turn marked failed, 504 timeout
- response fails validation       -> turn marked failed, 503 (no partial reply stored)
- assistant persistence fails     -> turn not reported complete, 503
Retrying with the same idempotency key re-attempts a failed turn without
creating a second user message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError

from app.config import Effort, Settings
from app.context.builder import ContextBuilder
from app.errors import AppError, ErrorCategory
from app.llm.base import LLMClient
from app.llm.schemas import (
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    GenerationResponse,
    LLMError,
    TraceMetadata,
)
from app.observability import log_event
from app.orchestration.lifecycle import TurnStage, TurnTrace
from app.orchestration.persistence import (
    AssistantProvenance,
    CompletedTurn,
    StartedTurn,
    TurnAttemptLost,
    TurnStore,
    turn_superseded,
)
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptTemplate

logger = logging.getLogger("app.turn")


@dataclass(frozen=True)
class TurnResult:
    turn: CompletedTurn
    timings_ms: dict[str, float]


class TurnOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        model: str,
        effort: Effort | None,
        max_output_tokens: int,
        store: TurnStore,
        llm: LLMClient,
        prompt: PromptTemplate,
        context_builder: ContextBuilder,
        validator: ResponseValidator,
    ) -> None:
        self._settings = settings
        self._model = model
        self._max_output_tokens = max_output_tokens
        # Built once so invalid generation settings fail at startup, never
        # mid-turn after the user message is already persisted.
        self._params = GenerationParams(effort=effort, temperature=settings.generation_temperature)
        self._store = store
        self._llm = llm
        self._prompt = prompt
        self._system_prompt = prompt.render()
        self._context_builder = context_builder
        self._validator = validator

    async def submit_message(
        self,
        *,
        trace_id: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
        idempotency_key: str | None,
    ) -> TurnResult:
        trace = TurnTrace(trace_id)
        trace.mark(TurnStage.VALIDATED)

        begun = await self._store.begin(
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
            idempotency_key=idempotency_key,
        )
        if isinstance(begun, CompletedTurn):
            log_event(logger, "turn.replayed", conversation_id=str(conversation_id))
            return TurnResult(turn=begun, timings_ms=trace.timings_ms())
        trace.mark(TurnStage.AUTHORIZED)
        trace.mark(TurnStage.USER_MESSAGE_PERSISTED)
        trace.mark(TurnStage.CONTEXT_LOADED)

        context = self._context_builder.build(system=self._system_prompt, history=begun.history)
        trace.mark(TurnStage.CONTEXT_BUILT)
        request = GenerationRequest(
            model=self._model,
            system=context.system,
            messages=context.messages,
            max_output_tokens=self._max_output_tokens,
            timeout_seconds=self._settings.generation_timeout_seconds,
            params=self._params,
            trace=TraceMetadata(
                trace_id=trace_id,
                purpose=GenerationPurpose.CONVERSATION,
                conversation_id=str(conversation_id),
            ),
        )

        response = await self._generate(request, begun, trace)
        trace.mark(TurnStage.RESPONSE_GENERATED)

        validation = self._validator.validate(response)
        log_event(
            logger,
            "response.validated",
            accepted=validation.accepted,
            reason=validation.reason.value if validation.reason else None,
            finish_reason=response.finish_reason.value,
        )
        if not validation.accepted:
            await self._fail(begun, f"invalid_response:{validation.reason}")
            raise AppError(
                ErrorCategory.DEPENDENCY_UNAVAILABLE,
                "invalid_generation",
                "The response could not be generated. Your message was saved; please retry.",
                retry_after_seconds=1,
            )
        trace.mark(TurnStage.RESPONSE_VALIDATED)

        provenance = AssistantProvenance(
            provider=response.provider,
            model_id=response.model_id,
            prompt_id=self._prompt.id,
            prompt_version=self._prompt.version,
            metadata={
                "requested_model": response.requested_model,
                "prompt_checksum": self._prompt.checksum,
                "finish_reason": response.finish_reason.value,
                "raw_finish_reason": response.raw_finish_reason,
                "fallback_used": response.fallback_used,
                "reasoning_removed": response.reasoning_removed,
                "provider_request_id": response.provider_request_id,
                "usage": response.usage.model_dump(),
                "latency_ms": round(response.latency_ms, 2),
                "generation_params": {
                    **request.params.model_dump(),
                    "max_output_tokens": request.max_output_tokens,
                },
                "context": {
                    "included_messages": context.included_messages,
                    "omitted_messages": context.omitted_messages,
                    "estimated_tokens": context.estimated_tokens,
                },
                "trace_id": trace_id,
            },
        )
        try:
            completed = await self._store.complete(
                conversation_id=begun.conversation_id,
                user_message_id=begun.user_message_id,
                attempt_id=begun.attempt_id,
                content=response.content,
                provenance=provenance,
            )
        except TurnAttemptLost as exc:
            log_event(logger, "turn.attempt_lost", level=logging.WARNING)
            raise turn_superseded() from exc
        except (SQLAlchemyError, OSError) as exc:
            log_event(logger, "persistence.failed", level=logging.ERROR, stage="assistant_message")
            await self._fail(begun, "persistence_error")
            raise AppError(
                ErrorCategory.DEPENDENCY_UNAVAILABLE,
                "persistence_failed",
                "The response could not be saved. Your message was kept; please retry.",
            ) from exc
        trace.mark(TurnStage.ASSISTANT_MESSAGE_PERSISTED)
        trace.mark(TurnStage.COMPLETED)
        log_event(
            logger,
            "turn.completed",
            conversation_id=str(conversation_id),
            provider=response.provider,
            model_id=response.model_id,
            prompt=self._prompt.key,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            generation_ms=round(response.latency_ms, 2),
            total_ms=round(trace.elapsed_seconds() * 1000, 2),
        )
        return TurnResult(turn=completed, timings_ms=trace.timings_ms())

    async def _generate(
        self, request: GenerationRequest, begun: StartedTurn, trace: TurnTrace
    ) -> GenerationResponse:
        remaining = self._settings.turn_deadline_seconds - trace.elapsed_seconds()
        try:
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await self._llm.generate(request)
        except TimeoutError as exc:
            log_event(logger, "generation.failed", level=logging.WARNING, category="turn_deadline")
            await self._fail(begun, "timeout")
            raise AppError(
                ErrorCategory.TIMEOUT,
                "generation_timeout",
                "The response took too long. Your message was saved; please retry.",
            ) from exc
        except LLMError as exc:
            log_event(
                logger,
                "generation.failed",
                level=logging.WARNING,
                category=exc.category.value,
                status_code=exc.status_code,
                provider_request_id=exc.provider_request_id,
            )
            await self._fail(begun, f"llm:{exc.category.value}")
            retry_after = exc.retry_after_seconds if exc.retryable else None
            raise AppError(
                ErrorCategory.DEPENDENCY_UNAVAILABLE,
                "generation_unavailable",
                "The language model is unavailable right now. Your message was saved; please retry."
                if exc.retryable
                else "The language model could not process this request. Your message was saved.",
                retry_after_seconds=int(retry_after) + 1 if retry_after is not None else None,
            ) from exc
        log_event(
            logger,
            "generation.completed",
            provider=response.provider,
            model_id=response.model_id,
            finish_reason=response.finish_reason.value,
            fallback_used=response.fallback_used,
            latency_ms=round(response.latency_ms, 2),
        )
        return response

    async def _fail(self, begun: StartedTurn, category: str) -> None:
        """Best effort: if this write fails too, the pending turn becomes stale
        and is taken over by a later request."""
        try:
            await self._store.fail(
                user_message_id=begun.user_message_id,
                attempt_id=begun.attempt_id,
                category=category,
            )
        except (SQLAlchemyError, OSError):
            log_event(logger, "persistence.failed", level=logging.ERROR, stage="mark_failed")
