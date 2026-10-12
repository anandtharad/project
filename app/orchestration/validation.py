"""Response-contract validation for generated replies.

Phase 1 checks the response contract only (complete, non-empty, bounded,
storable). Safety-policy validation is added in Phase 5; until then there is
no independent output safety check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.llm.schemas import FinishReason, GenerationResponse


class RejectionReason(StrEnum):
    REFUSED = "refused"
    TRUNCATED = "truncated"
    UNEXPECTED_FINISH = "unexpected_finish"
    EMPTY = "empty"
    TOO_LONG = "too_long"
    INVALID_CHARACTERS = "invalid_characters"


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: RejectionReason | None = None


class ResponseValidator:
    def __init__(self, *, max_characters: int) -> None:
        self._max_characters = max_characters

    def validate(self, response: GenerationResponse) -> ValidationResult:
        if response.finish_reason is FinishReason.REFUSAL:
            return ValidationResult(False, RejectionReason.REFUSED)
        if response.finish_reason is FinishReason.MAX_TOKENS:
            return ValidationResult(False, RejectionReason.TRUNCATED)
        if response.finish_reason is not FinishReason.STOP:
            return ValidationResult(False, RejectionReason.UNEXPECTED_FINISH)
        if not response.content.strip():
            return ValidationResult(False, RejectionReason.EMPTY)
        if len(response.content) > self._max_characters:
            return ValidationResult(False, RejectionReason.TOO_LONG)
        if "\x00" in response.content:
            return ValidationResult(False, RejectionReason.INVALID_CHARACTERS)
        return ValidationResult(True)
