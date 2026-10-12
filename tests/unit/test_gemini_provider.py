"""Gemini adapter tests through the real `google-genai` SDK with a mocked httpx
transport (no network). Shapes follow the verified google-genai 2.29.0 contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.llm.providers.gemini_provider import GeminiLLMClient, model_generation, thinking_config
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    TraceMetadata,
)

Handler = Callable[[httpx.Request], httpx.Response]

SUCCESS: dict[str, Any] = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "PRIVATE: plan a warm reply", "thought": True},
                    {"text": "That sounds like a lot to carry.", "thoughtSignature": "c2ln"},
                ],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 40,
        "candidatesTokenCount": 9,
        "thoughtsTokenCount": 30,
        "totalTokenCount": 79,
    },
    "modelVersion": "gemini-3.6-flash",
    "responseId": "resp-abc123",
}


class Recorder:
    def __init__(self, status: int = 200, body: Any = None, headers: dict[str, str] | None = None):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = SUCCESS if body is None else body
        self.headers = headers or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, (dict, list)):
            return httpx.Response(self.status, json=self.body, headers=self.headers)
        return httpx.Response(self.status, text=self.body, headers=self.headers)

    @property
    def sent(self) -> dict[str, Any]:
        return dict(json.loads(self.requests[-1].content))


def make_client(handler: Handler, *, max_retries: int = 0) -> GeminiLLMClient:
    return GeminiLLMClient(
        api_key="test-key",
        max_retries=max_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def make_request(model: str = "gemini-3.6-flash", **params: Any) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        system="SYSTEM PROMPT",
        messages=[
            ChatMessage(role=ChatRole.USER, content="Rough week."),
            ChatMessage(role=ChatRole.ASSISTANT, content="What happened?"),
            ChatMessage(role=ChatRole.USER, content="Everything at once."),
        ],
        max_output_tokens=2048,
        timeout_seconds=5,
        params=GenerationParams(**params),
        trace=TraceMetadata(trace_id="t", purpose=GenerationPurpose.CONVERSATION),
    )


@pytest.mark.parametrize(
    ("model", "generation"),
    [
        ("gemini-3.6-flash", "3"),
        ("models/gemini-3.8-flash", "3"),
        ("gemini-flash-latest", "3"),
        ("gemini-2.5-flash", "2.5"),
        ("gemma-3-27b-it", "other"),
    ],
)
def test_model_generation(model: str, generation: str) -> None:
    assert model_generation(model) == generation


def test_thinking_config_mapping() -> None:
    assert thinking_config("gemini-3.6-flash", None) is None
    assert thinking_config("gemini-3.6-flash", "low").thinking_level.value == "LOW"  # type: ignore[union-attr]
    assert thinking_config("gemini-3.6-flash", "none").thinking_level.value == "LOW"  # type: ignore[union-attr]
    assert thinking_config("gemini-3.1-flash-lite", "none").thinking_level.value == "MINIMAL"  # type: ignore[union-attr]
    assert thinking_config("gemini-3.8-flash", "max").thinking_level.value == "HIGH"  # type: ignore[union-attr]
    assert thinking_config("gemini-2.5-flash", "none").thinking_budget == 0  # type: ignore[union-attr]
    assert thinking_config("gemini-2.5-pro", "none").thinking_budget == 128  # type: ignore[union-attr]
    assert thinking_config("gemma-3-27b-it", "high") is None


async def test_request_shape() -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request(effort="low"))
    request = recorder.requests[-1]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "test-key"
    assert request.headers["x-server-timeout"] == "5"
    body = recorder.sent
    assert body["contents"] == [
        {"parts": [{"text": "Rough week."}], "role": "user"},
        {"parts": [{"text": "What happened?"}], "role": "model"},
        {"parts": [{"text": "Everything at once."}], "role": "user"},
    ]
    assert body["systemInstruction"]["parts"] == [{"text": "SYSTEM PROMPT"}]
    generation = body["generationConfig"]
    assert generation["maxOutputTokens"] == 2048
    assert generation["thinkingConfig"] == {"thinking_level": "LOW"}
    assert "temperature" not in generation
    assert "include_thoughts" not in json.dumps(body)


async def test_response_excludes_thought_parts() -> None:
    response = await make_client(Recorder()).generate(make_request())
    assert response.content == "That sounds like a lot to carry."
    assert "PRIVATE" not in response.content
    assert response.provider == "gemini"
    assert response.model_id == "gemini-3.6-flash"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "resp-abc123"
    assert response.usage.input_tokens == 40
    assert response.usage.output_tokens == 39  # visible + thinking tokens


@pytest.mark.filterwarnings("ignore:BRAND_NEW_REASON is not a valid FinishReason")
@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("MAX_TOKENS", FinishReason.MAX_TOKENS),
        ("SAFETY", FinishReason.REFUSAL),
        ("PROHIBITED_CONTENT", FinishReason.REFUSAL),
        ("RECITATION", FinishReason.REFUSAL),
        ("OTHER", FinishReason.OTHER),
        ("BRAND_NEW_REASON", FinishReason.OTHER),
    ],
)
async def test_finish_reason_mapping(reason: str, expected: FinishReason) -> None:
    body = json.loads(json.dumps(SUCCESS))
    body["candidates"][0]["finishReason"] = reason
    response = await make_client(Recorder(body=body)).generate(make_request())
    assert response.finish_reason is expected


async def test_blocked_prompt_is_a_refusal() -> None:
    body = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}, "responseId": "r3"}
    response = await make_client(Recorder(body=body)).generate(make_request())
    assert response.finish_reason is FinishReason.REFUSAL
    assert response.raw_finish_reason == "BLOCKED:PROHIBITED_CONTENT"
    assert response.content == ""


async def test_thinking_exhausted_budget_gives_empty_truncated_reply() -> None:
    body = {
        "candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"promptTokenCount": 12, "thoughtsTokenCount": 256},
    }
    response = await make_client(Recorder(body=body)).generate(make_request())
    assert response.content == ""
    assert response.finish_reason is FinishReason.MAX_TOKENS
    assert response.usage.output_tokens == 256


def error_body(
    code: int, status: str, message: str, details: list[Any] | None = None
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if details is not None:
        error["details"] = details
    return {"error": error}


@pytest.mark.parametrize(
    ("status", "body", "category"),
    [
        (
            400,
            error_body(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."),
            LLMErrorCategory.AUTHENTICATION,
        ),
        (
            400,
            error_body(400, "INVALID_ARGUMENT", "Invalid value at contents"),
            LLMErrorCategory.INVALID_REQUEST,
        ),
        (401, error_body(401, "UNAUTHENTICATED", "bad auth"), LLMErrorCategory.AUTHENTICATION),
        (
            402,
            error_body(402, "RESOURCE_EXHAUSTED", "credits depleted"),
            LLMErrorCategory.PERMISSION,
        ),
        (403, error_body(403, "PERMISSION_DENIED", "denied"), LLMErrorCategory.PERMISSION),
        (404, error_body(404, "NOT_FOUND", "models/x is not found"), LLMErrorCategory.NOT_FOUND),
        (500, error_body(500, "INTERNAL", "oops"), LLMErrorCategory.SERVER_ERROR),
        (502, "<html>Bad Gateway</html>", LLMErrorCategory.SERVER_ERROR),
        (
            503,
            error_body(503, "UNAVAILABLE", "The model is overloaded."),
            LLMErrorCategory.OVERLOADED,
        ),
        (504, error_body(504, "DEADLINE_EXCEEDED", "late"), LLMErrorCategory.TIMEOUT),
    ],
)
async def test_api_errors_are_classified(
    status: int, body: Any, category: LLMErrorCategory
) -> None:
    with pytest.raises(LLMError) as info:
        await make_client(Recorder(status, body)).generate(make_request())
    assert info.value.category is category
    assert info.value.status_code == status
    assert "test-key" not in info.value.message


async def test_per_minute_quota_uses_retry_info_delay() -> None:
    body = error_body(
        429,
        "RESOURCE_EXHAUSTED",
        "You exceeded your current quota.",
        [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"},
        ],
    )
    with pytest.raises(LLMError) as info:
        await make_client(Recorder(429, body)).generate(make_request())
    assert info.value.category is LLMErrorCategory.RATE_LIMITED
    assert info.value.retryable
    assert info.value.retry_after_seconds == 17.0


async def test_daily_quota_is_not_retryable() -> None:
    body = error_body(
        429,
        "RESOURCE_EXHAUSTED",
        "Quota exceeded.",
        [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
            }
        ],
    )
    with pytest.raises(LLMError) as info:
        await make_client(Recorder(429, body)).generate(make_request())
    assert info.value.category is LLMErrorCategory.QUOTA_EXHAUSTED
    assert not info.value.retryable


async def test_transport_and_body_errors() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError) as timeout:
        await make_client(slow).generate(make_request())
    assert timeout.value.category is LLMErrorCategory.TIMEOUT
    with pytest.raises(LLMError) as conn:
        await make_client(refuse).generate(make_request())
    assert conn.value.category is LLMErrorCategory.CONNECTION
    with pytest.raises(LLMError) as invalid:
        await make_client(Recorder(200, "not json")).generate(make_request())
    assert invalid.value.category is LLMErrorCategory.RESPONSE_INVALID


async def test_retries_are_bounded_by_max_retries() -> None:
    calls: list[int] = []

    def overloaded(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json=error_body(503, "UNAVAILABLE", "overloaded"))

    with pytest.raises(LLMError):
        await make_client(overloaded, max_retries=1).generate(make_request())
    assert len(calls) == 2


async def test_vertex_env_cannot_switch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    recorder = Recorder()
    await make_client(recorder).generate(make_request())
    assert recorder.requests[-1].url.host == "generativelanguage.googleapis.com"
