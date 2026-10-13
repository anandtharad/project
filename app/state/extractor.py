"""Update the conversation state with one structured LLM call (spec §8.4).

Runs before the reply. The previous notes and recent messages travel in a
user-role message as data; the instruction is the versioned
`state_extraction` prompt. The reply must match EXTRACTION_JSON_SCHEMA and is
validated with pydantic; on any failure (provider error, timeout, invalid
JSON) the previous state is kept and the turn continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError

from app.llm.base import LLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    ResponseSchema,
    StructuredMode,
    TokenUsage,
    TraceMetadata,
)
from app.observability import log_event
from app.prompting import PromptTemplate
from app.state.schemas import EXTRACTION_JSON_SCHEMA, ConversationState, ExtractedState

logger = logging.getLogger("app.state")

_RECENT_MESSAGES = 8
_MESSAGE_CHARS = 1_500


class ExtractionStatus(StrEnum):
    UPDATED = "updated"
    FAILED = "failed"  # previous state kept


@dataclass(frozen=True)
class ExtractionResult:
    status: ExtractionStatus
    state: ConversationState | None
    failure: str | None = None
    model_id: str | None = None
    structured_mode: StructuredMode | None = None
    latency_ms: float | None = None
    usage: TokenUsage | None = None


def _escape(text: str) -> str:
    # Keep message text from closing the data tags it is wrapped in.
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def build_extraction_input(
    previous: ConversationState | None, history: Sequence[ChatMessage]
) -> str:
    notes = previous.notes() if previous is not None else {}
    lines = []
    for message in history[-_RECENT_MESSAGES:]:
        speaker = "Person" if message.role is ChatRole.USER else "Companion"
        lines.append(f"{speaker}: {_escape(message.content[:_MESSAGE_CHARS])}")
    return (
        "<previous_notes>\n"
        + _escape(json.dumps(notes, ensure_ascii=False))
        + "\n</previous_notes>\n\n<recent_messages>\n"
        + "\n\n".join(lines)
        + "\n</recent_messages>\n\nReturn the updated notes."
    )


class StateExtractor:
    def __init__(
        self,
        *,
        llm: LLMClient,
        model: str,
        prompt: PromptTemplate,
        params: GenerationParams,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._llm = llm
        self._model = model
        self._prompt = prompt
        self._system = prompt.render()
        self._params = params
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompt(self) -> PromptTemplate:
        return self._prompt

    async def extract(
        self,
        previous: ConversationState | None,
        history: Sequence[ChatMessage],
        *,
        trace_id: str,
        conversation_id: str,
        timeout_seconds: float | None = None,
    ) -> ExtractionResult:
        timeout = self._timeout if timeout_seconds is None else min(self._timeout, timeout_seconds)
        if timeout <= 0:
            # The turn budget is already spent; GenerationRequest rejects it.
            return self._failed("timeout")
        request = GenerationRequest(
            model=self._model,
            system=self._system,
            messages=[
                ChatMessage(role=ChatRole.USER, content=build_extraction_input(previous, history))
            ],
            max_output_tokens=self._max_output_tokens,
            timeout_seconds=timeout,
            params=self._params,
            trace=TraceMetadata(
                trace_id=trace_id,
                purpose=GenerationPurpose.STATE_EXTRACTION,
                conversation_id=conversation_id,
            ),
            response_schema=ResponseSchema(
                name="conversation_state", json_schema=EXTRACTION_JSON_SCHEMA
            ),
        )
        try:
            async with asyncio.timeout(timeout):
                response = await self._llm.generate(request)
        except TimeoutError:
            return self._failed("timeout")
        except LLMError as exc:
            return self._failed(f"llm:{exc.category.value}")
        try:
            extracted = ExtractedState.model_validate_json(response.content)
        except ValidationError:
            # Content is never logged: it is derived from the conversation.
            return self._failed("invalid_output", model_id=response.model_id)
        mode = previous.conversation_mode if previous is not None else None
        state = extracted.to_state(conversation_mode=mode)
        log_event(
            logger,
            "state.extracted",
            model_id=response.model_id,
            structured_mode=response.structured_mode.value if response.structured_mode else None,
            latency_ms=round(response.latency_ms, 2),
            fields=sorted(state.notes()),
        )
        return ExtractionResult(
            status=ExtractionStatus.UPDATED,
            state=state,
            model_id=response.model_id,
            structured_mode=response.structured_mode,
            latency_ms=round(response.latency_ms, 2),
            usage=response.usage,
        )

    def _failed(self, failure: str, *, model_id: str | None = None) -> ExtractionResult:
        log_event(logger, "state.extraction_failed", level=logging.WARNING, failure=failure)
        return ExtractionResult(
            status=ExtractionStatus.FAILED, state=None, failure=failure, model_id=model_id
        )
