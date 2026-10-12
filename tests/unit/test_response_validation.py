from __future__ import annotations

import pytest

from app.llm.providers.fake import FakeLLMClient
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    TraceMetadata,
)
from app.orchestration.validation import RejectionReason, ResponseValidator
from tests.helpers import make_response

validator = ResponseValidator(max_characters=50)


def test_accepts_complete_reply() -> None:
    assert validator.validate(make_response("That sounds frustrating.")).accepted


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (make_response("", finish_reason=FinishReason.REFUSAL), RejectionReason.REFUSED),
        (
            make_response("cut off mid", finish_reason=FinishReason.MAX_TOKENS),
            RejectionReason.TRUNCATED,
        ),
        (
            make_response("paused", finish_reason=FinishReason.OTHER),
            RejectionReason.UNEXPECTED_FINISH,
        ),
        (make_response("   \n "), RejectionReason.EMPTY),
        (make_response("x" * 51), RejectionReason.TOO_LONG),
        (make_response("bad\x00byte"), RejectionReason.INVALID_CHARACTERS),
    ],
)
def test_rejections(response: object, reason: RejectionReason) -> None:
    result = validator.validate(response)  # type: ignore[arg-type]
    assert not result.accepted
    assert result.reason is reason


def test_llm_error_retryability() -> None:
    assert LLMError(LLMErrorCategory.RATE_LIMITED, "x").retryable
    assert LLMError(LLMErrorCategory.CONNECTION, "x").retryable
    assert not LLMError(LLMErrorCategory.INVALID_REQUEST, "x").retryable
    assert not LLMError(LLMErrorCategory.AUTHENTICATION, "x").retryable


async def test_fake_provider_is_labeled_and_never_echoes_content() -> None:
    request = GenerationRequest(
        model="m",
        system="s",
        messages=[ChatMessage(role=ChatRole.USER, content="my secret plans")],
        max_output_tokens=100,
        timeout_seconds=1,
        trace=TraceMetadata(trace_id="t", purpose=GenerationPurpose.CONVERSATION),
    )
    response = await FakeLLMClient().generate(request)
    assert "development stub" in response.content
    assert "secret" not in response.content
    assert ResponseValidator(max_characters=1000).validate(response).accepted
