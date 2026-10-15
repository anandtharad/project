"""Turn orchestrator: sequences one conversational turn end to end.

Lifecycle (spec §9.1; safety analysis arrives in Phase 5, retrieval in 3-4):
RECEIVED -> VALIDATED -> AUTHORIZED -> USER_MESSAGE_PERSISTED -> CONTEXT_LOADED
-> STATE_UPDATED -> DECISIONS_RESOLVED -> POLICY_CREATED -> CONTEXT_BUILT
-> RESPONSE_GENERATED -> RESPONSE_VALIDATED -> ASSISTANT_MESSAGE_PERSISTED
-> COMPLETED

Failure behavior (the user message is always preserved once persisted):
- unknown / foreign conversation -> 404, nothing persisted, no model call
- turn already in flight          -> 409 turn_in_progress
- state extraction fails          -> previous state kept, turn continues
- decision service fails          -> documented fallbacks, turn continues
- provider error                  -> turn marked failed, 503 dependency_unavailable
- turn deadline exceeded          -> turn marked failed, 504 timeout
- response fails validation       -> turn marked failed, 503 (no partial reply stored)
- assistant persistence fails     -> turn not reported complete, 503
- stale state write               -> reply kept, state write skipped and logged
Retrying with the same idempotency key re-attempts a failed turn without
creating a second user message.

Decisions run only after the ownership check in `begin`, and never control
access, persistence or safety handling (spec §3.3, §7.3).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError

from app.background import BackgroundTasks
from app.config import Effort, Settings
from app.context.builder import ContextBuilder
from app.decisions.schemas import DecisionSet
from app.decisions.service import DecisionService, build_decision_state
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
from app.memory.schemas import MemoryStatus
from app.memory.service import MemoryService, RetrievalResult
from app.observability import log_event
from app.orchestration.lifecycle import TurnStage, TurnTrace
from app.orchestration.persistence import (
    AssistantProvenance,
    CompletedTurn,
    SafetyWrite,
    StartedTurn,
    StateWrite,
    TurnAttemptLost,
    TurnStore,
    turn_superseded,
)
from app.orchestration.policy import ResponsePolicy, choose_policy
from app.orchestration.safety_flow import (
    guidance_for,
    is_safety_pathway,
    regeneration_note,
    safety_write,
)
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptTemplate
from app.safety.analysis import SafetyAssessment
from app.safety.factory import SafetyLayer
from app.safety.fallbacks import fallback_reply
from app.safety.output import OutputCheck
from app.safety.policy import SafetyDecision
from app.state.extractor import ExtractionResult, ExtractionStatus, StateExtractor
from app.state.schemas import ConversationState

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
        decisions: DecisionService,
        state_extractor: StateExtractor | None = None,
        memory: MemoryService | None = None,
        background: BackgroundTasks | None = None,
        safety: SafetyLayer | None = None,
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
        # Prompts from 1.1.0 carry a response-policy slot; 1.0.0 has none and
        # then runs without a policy (kept for baseline comparisons).
        self._policy_enabled = "response_guidance" in prompt.variables
        self._context_builder = context_builder
        self._validator = validator
        self._decisions = decisions
        self._state_extractor = state_extractor
        self._memory = memory
        self._background = background
        self._safety = safety

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
        try:
            return await self._run(begun, trace, trace_id, user_id, conversation_id, content)
        except (AppError, TurnAttemptLost):
            raise
        except Exception:
            # An unexpected error must not hold the conversation's pending
            # slot until it goes stale; mark the turn failed so a retry works.
            log_event(logger, "turn.unexpected_error", level=logging.ERROR)
            await self._fail(begun, "internal_error")
            raise

    async def _run(
        self,
        begun: StartedTurn,
        trace: TurnTrace,
        trace_id: str,
        user_id: uuid.UUID,
        conversation_id: uuid.UUID,
        content: str,
    ) -> TurnResult:
        previous = begun.state.state if begun.state is not None else None
        # Independent, so they run concurrently; none can fail the turn.
        extraction, recalled, assessment = await asyncio.gather(
            self._update_state(previous, begun, trace, trace_id),
            self._recall(user_id, content),
            self._assess(content, begun),
        )
        trace.mark(TurnStage.SAFETY_ASSESSED)
        trace.mark(TurnStage.MEMORIES_RETRIEVED)
        state = extraction.state if extraction is not None and extraction.state else previous
        trace.mark(TurnStage.STATE_UPDATED)

        # Recent messages only, exactly what scripts/eval_decisions.py measures:
        # thresholds chosen there stay valid for live turns.
        decisions = await self._decisions.decide(build_decision_state(begun.history))
        mode = decisions.acted_on("conversation_mode")
        if mode is not None and isinstance(mode.value, str):
            state = (state or ConversationState()).model_copy(
                update={"conversation_mode": mode.value}
            )
        trace.mark(TurnStage.DECISIONS_RESOLVED)

        safety: SafetyDecision | None = None
        if self._safety is not None and assessment is not None:
            if mode is not None and mode.value == "safety":
                assessment = assessment.with_decision_signal()
            safety = self._safety.policy.decide(assessment, self._safety.region)
        # Safety overrides the ordinary strategy (spec Phase 5 acceptance).
        policy = choose_policy(content, decisions)
        trace.mark(TurnStage.POLICY_CREATED)

        context = self._context_builder.build(
            system=self._system_prompt(policy, safety),
            history=begun.history,
            notes=state.notes() if state is not None and not state.is_empty() else None,
            memories=_memories_for_context(recalled),
        )
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

        content = response.content
        safety_record: SafetyWrite | None = None
        if self._safety is not None and assessment is not None and safety is not None:
            content, response, safety_record = await self._check_reply(
                response, request, assessment, safety, trace
            )
        trace.mark(TurnStage.SAFETY_CHECKED)

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
                    "notes_included": context.notes_included,
                },
                "policy": policy.summary() if self._policy_enabled else None,
                "decisions": _decisions_summary(decisions),
                "state": _state_summary(extraction, self._state_extractor),
                "memory": _memory_summary(recalled),
                "safety": _safety_summary(assessment, safety, safety_record),
                "trace_id": trace_id,
            },
        )
        state_write = self._state_write(begun, previous, state, extraction)
        try:
            completed = await self._store.complete(
                conversation_id=begun.conversation_id,
                user_message_id=begun.user_message_id,
                attempt_id=begun.attempt_id,
                content=content,
                provenance=provenance,
                state_write=state_write,
                decisions=list(decisions.results.values()),
                safety=safety_record,
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
        if self._memory is not None and self._background is not None:
            # Post-turn processing: after the reply is stored, never part of it.
            self._background.spawn(
                self._memory.process_turn(
                    user_id=user_id,
                    conversation_id=begun.conversation_id,
                    user_message_id=begun.user_message_id,
                    history=begun.history,
                    trace_id=trace_id,
                ),
                name="memory_extraction",
            )
        if state_write is not None and completed.state_version is None:
            log_event(logger, "state.write_rejected_stale", level=logging.WARNING)
        trace.mark(TurnStage.COMPLETED)
        log_event(
            logger,
            "turn.completed",
            conversation_id=str(conversation_id),
            provider=response.provider,
            model_id=response.model_id,
            prompt=self._prompt.key,
            policy_source=policy.source if self._policy_enabled else None,
            strategy=policy.strategy.value if self._policy_enabled and policy.strategy else None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            generation_ms=round(response.latency_ms, 2),
            total_ms=round(trace.elapsed_seconds() * 1000, 2),
        )
        return TurnResult(turn=completed, timings_ms=trace.timings_ms())

    async def _recall(self, user_id: uuid.UUID, content: str) -> RetrievalResult | None:
        if self._memory is None:
            return None
        return await self._memory.retrieve_for_turn(user_id, content)

    def _system_prompt(self, policy: ResponsePolicy, safety: SafetyDecision | None = None) -> str:
        if not self._policy_enabled:
            return self._prompt.render()
        if safety is not None and safety.overrides_strategy:
            guidance = guidance_for(self._prompt, safety)
        else:
            guidance = "\n\n".join(self._prompt.fragment(name) for name in policy.fragments)
        return self._prompt.render(response_guidance=guidance)

    async def _assess(self, content: str, begun: StartedTurn) -> SafetyAssessment | None:
        if self._safety is None:
            return None
        # The history ends with this message; the analyzer takes the earlier ones.
        return await self._safety.analyzer.assess(content, begun.history[:-1])

    async def _output_check(
        self, reply: str, system: str, assessment: SafetyAssessment, safety: SafetyDecision
    ) -> tuple[OutputCheck, str | None]:
        """Deterministic checks on every reply; the semantic check only on a
        safety pathway, only when those pass, and never blocking on failure."""
        assert self._safety is not None
        check = self._safety.validator.check(reply, safety, assessment, system_prompt=system)
        status: str | None = None
        checker = self._safety.output_checker
        if check.ok and is_safety_pathway(safety) and checker is not None:
            result = await checker.check(reply, safety)
            if result is not None:
                status = result.status.value
                if result.violation is not None:
                    check = OutputCheck((result.violation,))
        return check, status

    async def _check_reply(
        self,
        response: GenerationResponse,
        request: GenerationRequest,
        assessment: SafetyAssessment,
        safety: SafetyDecision,
        trace: TurnTrace,
    ) -> tuple[str, GenerationResponse, SafetyWrite]:
        """Bounded recovery (spec §23): a reply that fails the checks is
        regenerated once with a note naming the problems; if that fails too,
        or cannot be generated in time, the pathway's fixed fallback is stored.
        A failed regeneration never fails the turn."""
        check, status = await self._output_check(
            response.content, request.system, assessment, safety
        )
        content, recovery = response.content, "none"
        if not check.ok:
            log_event(
                logger,
                "safety.output_violation",
                level=logging.WARNING,
                codes=list(check.codes),
                pathway=safety.pathway.value,
            )
            retry_request = request.model_copy(
                update={"system": f"{request.system}\n\n{regeneration_note(check.codes)}"}
            )
            retried: GenerationResponse | None = None
            remaining = self._settings.turn_deadline_seconds - trace.elapsed_seconds()
            try:
                if remaining > 0:
                    async with asyncio.timeout(remaining):
                        retried = await self._llm.generate(retry_request)
            except (TimeoutError, LLMError) as exc:
                log_event(
                    logger,
                    "safety.regeneration_failed",
                    level=logging.WARNING,
                    exc_type=type(exc).__name__,
                )
            if retried is not None and self._validator.validate(retried).accepted:
                second, status = await self._output_check(
                    retried.content, request.system, assessment, safety
                )
                if second.ok:
                    response, content, recovery = retried, retried.content, "regenerated"
            if recovery == "none":
                content, recovery = fallback_reply(safety), "fallback"
            log_event(logger, "safety.recovered", recovery=recovery, pathway=safety.pathway.value)
        record = safety_write(
            assessment,
            safety,
            violations=check.codes,
            output_check_status=status,
            recovery=recovery,
        )
        return content, response, record

    async def _update_state(
        self,
        previous: ConversationState | None,
        begun: StartedTurn,
        trace: TurnTrace,
        trace_id: str,
    ) -> ExtractionResult | None:
        if self._state_extractor is None:
            return None
        # Leave most of the turn budget for the reply itself.
        remaining = (self._settings.turn_deadline_seconds - trace.elapsed_seconds()) / 2
        return await self._state_extractor.extract(
            previous,
            begun.history,
            trace_id=trace_id,
            conversation_id=str(begun.conversation_id),
            timeout_seconds=remaining,
        )

    def _state_write(
        self,
        begun: StartedTurn,
        previous: ConversationState | None,
        state: ConversationState | None,
        extraction: ExtractionResult | None,
    ) -> StateWrite | None:
        if state is None or state == previous:
            return None
        return StateWrite(
            state=state,
            expected_version=begun.state.version if begun.state is not None else 0,
            extraction=_state_summary(extraction, self._state_extractor) or {},
        )

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


def _safety_summary(
    assessment: SafetyAssessment | None,
    safety: SafetyDecision | None,
    record: SafetyWrite | None,
) -> dict[str, object] | None:
    if assessment is None or safety is None or record is None:
        return None
    return {
        **assessment.summary(),
        **safety.summary(),
        "output_violations": record.output_violations,
        "output_check_status": record.output_check_status,
        "recovery": record.recovery,
    }


def _decisions_summary(decisions: DecisionSet) -> dict[str, object]:
    return {
        "provider": decisions.provider,
        "latency_ms": decisions.latency_ms,
        "request_id": decisions.request_id,
        "results": {
            key: {
                "question_version": result.question_version,
                "value": result.value,
                "returned_value": result.returned_value,
                "confidence": result.confidence,
                "fallback_reason": result.fallback_reason.value if result.fallback_reason else None,
            }
            for key, result in decisions.results.items()
        },
    }


def _state_summary(
    extraction: ExtractionResult | None, extractor: StateExtractor | None
) -> dict[str, object] | None:
    if extraction is None or extractor is None:
        return None
    return {
        "status": extraction.status.value,
        "failure": extraction.failure,
        "model_id": extraction.model_id,
        "requested_model": extractor.model,
        "prompt": extractor.prompt.key,
        "structured_mode": extraction.structured_mode.value if extraction.structured_mode else None,
        "latency_ms": extraction.latency_ms,
        "updated": extraction.status is ExtractionStatus.UPDATED,
    }


_PENDING = MemoryStatus.PENDING_CONFIRMATION


def _memories_for_context(recalled: RetrievalResult | None) -> list[dict[str, str]] | None:
    if recalled is None or not recalled.memories:
        return None
    return [
        {
            "type": r.memory.memory_type.value,
            "content": r.memory.content,
            "noted": r.memory.created_at.date().isoformat(),
            # Held because it conflicts with an older memory (spec §10.6).
            **(
                {"status": "unconfirmed"}
                if r.memory.status is MemoryStatus.PENDING_CONFIRMATION
                else {}
            ),
        }
        for r in recalled.memories
    ]


def _memory_summary(recalled: RetrievalResult | None) -> dict[str, object] | None:
    if recalled is None:
        return None
    return {
        "mode": recalled.mode,
        "failure": recalled.failure,
        "used": [{"id": str(r.memory.id), "score": round(r.score, 4)} for r in recalled.memories],
    }
