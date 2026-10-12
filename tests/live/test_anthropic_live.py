"""Live provider tests - real API calls, real cost. Opt-in only.

Run with:
    RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... uv run pytest tests/live -v

These check integration (request accepted, response normalized, context
used), not conversational quality, which belongs to the evaluation suite.
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.llm.factory import build_llm_client
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    TraceMetadata,
)
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="live provider tests are opt-in: set RUN_LIVE_LLM_TESTS=1 and provider credentials",
)


def live_request(settings: Settings, messages: list[ChatMessage]) -> GenerationRequest:
    return GenerationRequest(
        model=settings.generation_model,
        system=PromptRegistry().get("companion_system", settings.companion_prompt_version).render(),
        messages=messages,
        max_output_tokens=settings.max_output_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        params=GenerationParams(
            effort=settings.generation_effort, temperature=settings.generation_temperature
        ),
        trace=TraceMetadata(trace_id="live-test", purpose=GenerationPurpose.CONVERSATION),
    )


async def test_live_single_turn_produces_valid_reply() -> None:
    settings = Settings(generation_provider="anthropic")
    client = build_llm_client(settings)
    try:
        response = await client.generate(
            live_request(
                settings,
                [ChatMessage(role=ChatRole.USER, content="I've been feeling overwhelmed lately.")],
            )
        )
    finally:
        await client.aclose()
    assert (
        ResponseValidator(max_characters=settings.max_response_characters)
        .validate(response)
        .accepted
    ), response.finish_reason
    assert response.model_id
    assert response.usage.input_tokens and response.usage.output_tokens
    assert response.provider_request_id


async def test_live_multi_turn_context_is_used() -> None:
    settings = Settings(generation_provider="anthropic")
    client = build_llm_client(settings)
    try:
        response = await client.generate(
            live_request(
                settings,
                [
                    ChatMessage(role=ChatRole.USER, content="My dog Biscuit has been unwell."),
                    ChatMessage(role=ChatRole.ASSISTANT, content="I'm sorry to hear that."),
                    ChatMessage(
                        role=ChatRole.USER,
                        content="Remind me, what's my dog's name? Just the name, please.",
                    ),
                ],
            )
        )
    finally:
        await client.aclose()
    assert "Biscuit" in response.content
