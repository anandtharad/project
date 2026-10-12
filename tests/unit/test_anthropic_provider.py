"""Anthropic adapter tests against the real SDK with a mocked HTTP transport.

No network access: httpx2.MockTransport answers every request, so these
verify the exact request JSON/headers the SDK sends and how responses and
errors are normalized. Live behavior is covered separately in tests/live.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest

from app.llm.providers.anthropic_provider import FALLBACK_BETA, AnthropicLLMClient
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

Handler = Callable[[httpx2.Request], httpx2.Response]


def message_json(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "What's been feeling most overwhelming?"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 18},
    }
    body.update(overrides)
    return body


def make_client(handler: Handler, *, max_retries: int = 0, **kwargs: Any) -> AnthropicLLMClient:
    sdk = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="https://api.anthropic.test",
        max_retries=max_retries,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return AnthropicLLMClient(client=sdk, **kwargs)


def make_request(**params: Any) -> GenerationRequest:
    return GenerationRequest(
        model="claude-opus-5-5",
        system="SYSTEM PROMPT",
        messages=[
            ChatMessage(role=ChatRole.USER, content="I've been feeling overwhelmed lately."),
            ChatMessage(role=ChatRole.ASSISTANT, content="What's been weighing on you?"),
            ChatMessage(role=ChatRole.USER, content="Work, mostly."),
        ],
        max_output_tokens=4000,
        timeout_seconds=5,
        params=GenerationParams(**params),
        trace=TraceMetadata(trace_id="t1", purpose=GenerationPurpose.CONVERSATION),
    )


class Recorder:
    def __init__(self, response: httpx2.Response | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.response = response or httpx2.Response(
            200, json=message_json(), headers={"request-id": "req_abc"}
        )

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.response

    @property
    def body(self) -> dict[str, Any]:
        return dict(json.loads(self.requests[-1].content))


async def test_request_shape_with_effort_and_default_fallback() -> None:
    recorder = Recorder()
    client = make_client(recorder)
    await client.generate(make_request(effort="low"))

    sent = recorder.requests[-1]
    assert sent.url.path == "/v1/messages"
    assert FALLBACK_BETA in sent.headers["anthropic-beta"]
    body = recorder.body
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 4000
    assert body["system"] == "SYSTEM PROMPT"
    assert body["messages"] == [
        {"role": "user", "content": "I've been feeling overwhelmed lately."},
        {"role": "assistant", "content": "What's been weighing on you?"},
        {"role": "user", "content": "Work, mostly."},
    ]
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in body


async def test_fallback_off_and_no_effort_omits_fields() -> None:
    recorder = Recorder()
    client = make_client(recorder, refusal_fallback="off")
    await client.generate(make_request())
    assert "anthropic-beta" not in recorder.requests[-1].headers
    assert "fallbacks" not in recorder.body
    assert "output_config" not in recorder.body


async def test_configured_temperature_is_forwarded_for_older_models() -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request(temperature=0.3))
    assert recorder.body["temperature"] == 0.3


async def test_response_normalization_skips_thinking_blocks() -> None:
    client = make_client(Recorder())
    response = await client.generate(make_request())
    assert response.content == "What's been feeling most overwhelming?"
    assert response.finish_reason is FinishReason.STOP
    assert response.raw_finish_reason == "end_turn"
    assert response.model_id == "claude-opus-5-5"
    assert response.requested_model == "claude-opus-5-5"
    assert response.usage.input_tokens == 120
    assert response.usage.output_tokens == 18
    assert response.provider_request_id == "req_abc"
    assert response.fallback_used is False
    assert response.latency_ms >= 0


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", FinishReason.STOP),
        ("stop_sequence", FinishReason.STOP),
        ("max_tokens", FinishReason.MAX_TOKENS),
        ("refusal", FinishReason.REFUSAL),
        ("pause_turn", FinishReason.OTHER),
        ("tool_use", FinishReason.OTHER),
    ],
)
async def test_finish_reason_mapping(stop_reason: str, expected: FinishReason) -> None:
    recorder = Recorder(httpx2.Response(200, json=message_json(stop_reason=stop_reason)))
    response = await make_client(recorder).generate(make_request())
    assert response.finish_reason is expected


async def test_fallback_served_response_records_actual_model() -> None:
    body = message_json(
        model="claude-opus-5",
        content=[
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-5"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "Here is a reply."},
        ],
        usage={
            "input_tokens": 50,
            "output_tokens": 9,
            "iterations": [
                {
                    "type": "fallback_message",
                    "model": "claude-opus-5",
                    "input_tokens": 50,
                    "output_tokens": 9,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
            ],
        },
    )
    response = await make_client(Recorder(httpx2.Response(200, json=body))).generate(make_request())
    assert response.fallback_used is True
    assert response.model_id == "claude-opus-5"
    assert response.requested_model == "claude-opus-5-5"
    assert response.content == "Here is a reply."


def error_response(status: int, error_type: str, headers: dict[str, str] | None = None) -> Handler:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"type": "error", "error": {"type": error_type, "message": "nope"}},
            headers={"request-id": "req_err", **(headers or {})},
        )

    return handler


@pytest.mark.parametrize(
    ("status", "error_type", "category", "retryable"),
    [
        (400, "invalid_request_error", LLMErrorCategory.INVALID_REQUEST, False),
        (401, "authentication_error", LLMErrorCategory.AUTHENTICATION, False),
        (402, "billing_error", LLMErrorCategory.PERMISSION, False),
        (403, "permission_error", LLMErrorCategory.PERMISSION, False),
        (404, "not_found_error", LLMErrorCategory.NOT_FOUND, False),
        (413, "request_too_large", LLMErrorCategory.INVALID_REQUEST, False),
        (429, "rate_limit_error", LLMErrorCategory.RATE_LIMITED, True),
        (500, "api_error", LLMErrorCategory.SERVER_ERROR, True),
        (503, "api_error", LLMErrorCategory.OVERLOADED, True),
        (504, "api_error", LLMErrorCategory.TIMEOUT, True),
        (529, "overloaded_error", LLMErrorCategory.OVERLOADED, True),
    ],
)
async def test_http_errors_are_classified(
    status: int, error_type: str, category: LLMErrorCategory, retryable: bool
) -> None:
    client = make_client(error_response(status, error_type))
    with pytest.raises(LLMError) as info:
        await client.generate(make_request())
    assert info.value.category is category
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert info.value.provider_request_id == "req_err"
    assert "test-key" not in info.value.message


async def test_rate_limit_retry_after_is_captured() -> None:
    client = make_client(error_response(429, "rate_limit_error", {"retry-after": "7"}))
    with pytest.raises(LLMError) as info:
        await client.generate(make_request())
    assert info.value.retry_after_seconds == 7.0


async def test_connection_and_timeout_errors_are_classified() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(LLMError) as conn_info:
        await make_client(refuse).generate(make_request())
    assert conn_info.value.category is LLMErrorCategory.CONNECTION
    with pytest.raises(LLMError) as timeout_info:
        await make_client(slow).generate(make_request())
    assert timeout_info.value.category is LLMErrorCategory.TIMEOUT


async def test_transient_errors_are_retried_by_sdk_within_budget() -> None:
    calls: list[int] = []

    def flaky(_: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx2.Response(
                529,
                json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
                headers={"retry-after": "0"},
            )
        return httpx2.Response(200, json=message_json())

    response = await make_client(flaky, max_retries=1).generate(make_request())
    assert response.content
    assert len(calls) == 2


async def test_invalid_requests_are_not_retried() -> None:
    calls: list[int] = []

    def bad(_: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        return httpx2.Response(
            400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
        )

    with pytest.raises(LLMError):
        await make_client(bad, max_retries=2).generate(make_request())
    assert len(calls) == 1


async def test_missing_credentials_map_to_authentication_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    sdk = anthropic.AsyncAnthropic(
        base_url="https://api.anthropic.test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=message_json()))
        ),
    )
    with pytest.raises(LLMError) as info:
        await AnthropicLLMClient(client=sdk).generate(make_request())
    assert info.value.category is LLMErrorCategory.AUTHENTICATION
