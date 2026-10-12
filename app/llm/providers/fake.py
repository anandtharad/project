"""DEVELOPMENT STUB - not a language model.

Returns a fixed, clearly labeled reply so the service can be run end to end
without provider credentials. It never echoes user content. Configuration
refuses this provider when APP_ENV=production.
"""

from __future__ import annotations

from app.llm.schemas import (
    ChatRole,
    FinishReason,
    GenerationRequest,
    GenerationResponse,
    TokenUsage,
)


class FakeLLMClient:
    @property
    def provider(self) -> str:
        return "fake"

    async def aclose(self) -> None:
        return None

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        user_turns = sum(1 for m in request.messages if m.role is ChatRole.USER)
        content = (
            "[development stub - no language model was called] "
            f"Received your message ({len(request.messages[-1].content)} characters); "
            f"{user_turns} user turn(s) are in context."
        )
        return GenerationResponse(
            content=content,
            provider=self.provider,
            requested_model=request.model,
            model_id="fake-stub",
            finish_reason=FinishReason.STOP,
            raw_finish_reason="stop",
            usage=TokenUsage(),
            latency_ms=0.0,
        )
