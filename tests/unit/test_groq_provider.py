"""Groq adapter tests through the real `groq` SDK with a mocked httpx transport.

No network access. Request/response/error shapes follow the verified Groq
contract (groq SDK 1.7.0 source + Groq docs). Live behavior: tests/live.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import groq
import httpx
import pytest

from app.llm.providers.groq_provider import GroqLLMClient, model_family
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


def completion_json(
    content: str | None = "That sounds really hard.", **overrides: Any
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1760000000,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": "PRIVATE: user seems sad, respond warmly",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 20,
            "total_tokens": 62,
            "completion_tokens_details": {"reasoning_tokens": 8},
        },
        "x_groq": {"id": "req_groq_123"},
    }
    body.update(overrides)
    return body


class Recorder:
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response or httpx.Response(200, json=completion_json())

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.response

    @property
    def body(self) -> dict[str, Any]:
        return dict(json.loads(self.requests[-1].content))


def make_client(handler: Handler, *, max_retries: int = 0) -> GroqLLMClient:
    sdk = groq.AsyncGroq(
        api_key="gsk_test_not_real",
        max_retries=max_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return GroqLLMClient(api_key=None, client=sdk)


def make_request(model: str = "openai/gpt-oss-120b", **params: Any) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        system="SYSTEM PROMPT",
        messages=[
            ChatMessage(role=ChatRole.USER, content="I had a rough day."),
            ChatMessage(role=ChatRole.ASSISTANT, content="I'm sorry. Want to talk?"),
            ChatMessage(role=ChatRole.USER, content="Yes."),
        ],
        max_output_tokens=2048,
        timeout_seconds=5,
        params=GenerationParams(**params),
        trace=TraceMetadata(trace_id="t1", purpose=GenerationPurpose.CONVERSATION),
    )


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("openai/gpt-oss-120b", "gpt-oss"),
        ("openai/gpt-oss-20b", "gpt-oss"),
        ("qwen/qwen3.8-27b", "qwen"),
        ("some-vendor/other-model", "other"),
    ],
)
def test_model_family(model: str, family: str) -> None:
    assert model_family(model) == family


async def test_gpt_oss_request_hides_reasoning_and_clamps_effort() -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request(effort="none"))
    sent = recorder.requests[-1]
    assert sent.url.path == "/openai/v1/chat/completions"
    body = recorder.body
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["max_completion_tokens"] == 2048
    assert body["messages"] == [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "I had a rough day."},
        {"role": "assistant", "content": "I'm sorry. Want to talk?"},
        {"role": "user", "content": "Yes."},
    ]
    assert body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low"  # gpt-oss has no "none"
    for absent in ("reasoning_format", "temperature", "max_tokens", "stream", "service_tier"):
        assert absent not in body


@pytest.mark.parametrize(
    ("effort", "sent"), [("low", "low"), ("medium", "medium"), ("xhigh", "high"), ("max", "high")]
)
async def test_gpt_oss_effort_mapping(effort: str, sent: str) -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request(effort=effort))
    assert recorder.body["reasoning_effort"] == sent


async def test_qwen_request_uses_hidden_reasoning_format() -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request("qwen/qwen3.8-27b", effort="none"))
    body = recorder.body
    assert body["reasoning_format"] == "hidden"
    assert body["reasoning_effort"] == "none"
    assert "include_reasoning" not in body  # mutually exclusive with reasoning_format


async def test_unknown_family_without_effort_sends_no_reasoning_params() -> None:
    recorder = Recorder()
    await make_client(recorder).generate(make_request("vendor/plain-model", temperature=0.7))
    body = recorder.body
    for absent in ("reasoning_effort", "reasoning_format", "include_reasoning"):
        assert absent not in body
    assert body["temperature"] == 0.7


async def test_response_normalization_never_returns_reasoning_field() -> None:
    response = await make_client(Recorder()).generate(make_request())
    assert response.content == "That sounds really hard."
    assert "PRIVATE" not in response.content
    assert response.provider == "groq"
    assert response.model_id == "openai/gpt-oss-120b"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "req_groq_123"
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 20
    assert response.reasoning_removed is False


async def test_inline_think_blocks_are_stripped() -> None:
    body = completion_json(
        "<think>The user wants comfort.</think>\n\nThat sounds exhausting.",
        model="qwen/qwen3.8-27b",
    )
    response = await make_client(Recorder(httpx.Response(200, json=body))).generate(
        make_request("qwen/qwen3.8-27b")
    )
    assert response.content == "That sounds exhausting."
    assert response.reasoning_removed is True


async def test_truncated_reasoning_yields_empty_reply() -> None:
    body = completion_json("<think>still reasoning when the budget ran out")
    body["choices"][0]["finish_reason"] = "length"
    response = await make_client(Recorder(httpx.Response(200, json=body))).generate(make_request())
    assert response.content == ""
    assert response.finish_reason is FinishReason.MAX_TOKENS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("length", FinishReason.MAX_TOKENS),
        ("tool_calls", FinishReason.OTHER),
        ("content_filter", FinishReason.REFUSAL),
        ("something_new", FinishReason.OTHER),
    ],
)
async def test_finish_reason_mapping(raw: str, expected: FinishReason) -> None:
    body = completion_json()
    body["choices"][0]["finish_reason"] = raw
    response = await make_client(Recorder(httpx.Response(200, json=body))).generate(make_request())
    assert response.finish_reason is expected


async def test_missing_optional_fields_do_not_crash() -> None:
    body = completion_json()
    body.pop("x_groq")
    body.pop("usage")
    response = await make_client(Recorder(httpx.Response(200, json=body))).generate(make_request())
    assert response.provider_request_id == "chatcmpl-abc"
    assert response.usage.input_tokens is None


async def test_empty_choices_is_an_invalid_response() -> None:
    body = completion_json()
    body["choices"] = []
    with pytest.raises(LLMError) as info:
        await make_client(Recorder(httpx.Response(200, json=body))).generate(make_request())
    assert info.value.category is LLMErrorCategory.RESPONSE_INVALID


def error_handler(status: int, headers: dict[str, str] | None = None) -> Handler:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": "nope", "type": "x", "code": "y"}},
            headers=headers or {},
        )

    return handler


@pytest.mark.parametrize(
    ("status", "category", "retryable"),
    [
        (400, LLMErrorCategory.INVALID_REQUEST, False),
        (401, LLMErrorCategory.AUTHENTICATION, False),
        (403, LLMErrorCategory.PERMISSION, False),
        (404, LLMErrorCategory.NOT_FOUND, False),
        (413, LLMErrorCategory.INVALID_REQUEST, False),
        (422, LLMErrorCategory.INVALID_REQUEST, False),
        (429, LLMErrorCategory.RATE_LIMITED, True),
        (498, LLMErrorCategory.OVERLOADED, True),
        (500, LLMErrorCategory.SERVER_ERROR, True),
        (502, LLMErrorCategory.SERVER_ERROR, True),
        (503, LLMErrorCategory.OVERLOADED, True),
    ],
)
async def test_http_errors_are_classified(
    status: int, category: LLMErrorCategory, retryable: bool
) -> None:
    with pytest.raises(LLMError) as info:
        await make_client(error_handler(status)).generate(make_request())
    assert info.value.category is category
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert "gsk_" not in info.value.message


async def test_rate_limit_retry_after_is_captured() -> None:
    handler = error_handler(429, {"retry-after": "7", "x-request-id": "req_err"})
    with pytest.raises(LLMError) as info:
        await make_client(handler).generate(make_request())
    assert info.value.retry_after_seconds == 7.0
    assert info.value.provider_request_id == "req_err"


async def test_connection_and_timeout_errors() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as conn:
        await make_client(refuse).generate(make_request())
    assert conn.value.category is LLMErrorCategory.CONNECTION
    with pytest.raises(LLMError) as timeout:
        await make_client(slow).generate(make_request())
    assert timeout.value.category is LLMErrorCategory.TIMEOUT


async def test_sdk_retries_transient_errors_but_not_498() -> None:
    calls: list[int] = []

    def flaky(_: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="overloaded", headers={"retry-after": "0"})
        return httpx.Response(200, json=completion_json())

    assert (await make_client(flaky, max_retries=1).generate(make_request())).content
    assert len(calls) == 2

    capacity_calls: list[int] = []

    def capacity(_: httpx.Request) -> httpx.Response:
        capacity_calls.append(1)
        return httpx.Response(498, json={"error": {"message": "flex capacity"}})

    with pytest.raises(LLMError):
        await make_client(capacity, max_retries=2).generate(make_request())
    assert len(capacity_calls) == 1


async def test_sdk_debug_logging_never_records_message_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from app.observability import configure_logging

    configure_logging("DEBUG")
    # configure_logging replaces the root handlers; re-attach pytest's capture.
    logging.getLogger().addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG)
    logging.getLogger("app.test").debug("capture-sentinel")
    await make_client(Recorder()).generate(make_request())
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "capture-sentinel" in joined  # proves DEBUG records are being captured
    assert "I had a rough day." not in joined
    assert "SYSTEM PROMPT" not in joined
