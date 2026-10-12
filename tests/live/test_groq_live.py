"""Live Groq tests - real API calls against the free tier. Opt-in only.

Run with:
    RUN_LIVE_LLM_TESTS=1 GROQ_API_KEY=... uv run pytest tests/live/test_groq_live.py -v

Two calls per model (default 3 models = 6 calls), far below free-tier
limits. Override the models with LIVE_GROQ_MODELS="model-a,model-b".
These verify integration (accepted request, normalized reply, no leaked
reasoning, context used), not conversational quality.
"""

from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.llm.factory import build_llm_client
from app.llm.providers.registry import resolve_provider
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    GenerationResponse,
    TraceMetadata,
)
from app.orchestration.validation import ResponseValidator
from app.prompting import PromptRegistry

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="live provider tests are opt-in: set RUN_LIVE_LLM_TESTS=1 and GROQ_API_KEY",
)

DEFAULT_MODELS = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
MODELS = [m.strip() for m in os.environ.get("LIVE_GROQ_MODELS", DEFAULT_MODELS).split(",") if m]


async def generate(model: str, messages: list[ChatMessage]) -> GenerationResponse:
    settings = Settings(generation_provider="groq", generation_model=model)
    resolved = resolve_provider(settings)
    client = build_llm_client(settings, resolved)
    request = GenerationRequest(
        model=resolved.model,
        system=PromptRegistry().get("companion_system", settings.companion_prompt_version).render(),
        messages=messages,
        max_output_tokens=resolved.max_output_tokens,
        timeout_seconds=settings.generation_timeout_seconds,
        params=GenerationParams(effort=resolved.effort),
        trace=TraceMetadata(trace_id="live-groq", purpose=GenerationPurpose.CONVERSATION),
    )
    try:
        return await client.generate(request)
    finally:
        await client.aclose()


@pytest.mark.parametrize("model", MODELS)
async def test_live_single_turn_reply_is_valid(model: str) -> None:
    response = await generate(
        model, [ChatMessage(role=ChatRole.USER, content="I've been feeling overwhelmed lately.")]
    )
    verdict = ResponseValidator(max_characters=8000).validate(response)
    assert verdict.accepted, (verdict.reason, response.finish_reason, response.content[:200])
    assert "<think>" not in response.content.lower()
    assert response.model_id
    assert response.usage.input_tokens and response.usage.output_tokens
    assert response.provider_request_id


@pytest.mark.parametrize("model", MODELS)
async def test_live_multi_turn_context_is_used(model: str) -> None:
    response = await generate(
        model,
        [
            ChatMessage(role=ChatRole.USER, content="My dog Biscuit has been unwell."),
            ChatMessage(role=ChatRole.ASSISTANT, content="I'm sorry to hear that."),
            ChatMessage(
                role=ChatRole.USER, content="Remind me, what's my dog's name? Just the name."
            ),
        ],
    )
    assert "Biscuit" in response.content, response.content[:200]
