"""OpenAI / OpenAI-compatible adapter tests through the real `openai` SDK with
a mocked httpx2 transport (no network). Includes the credential-leak guards:
OPENAI_API_KEY and OpenAI account headers must never reach other endpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from app.llm.providers.openai_provider import (
    KEYLESS_PLACEHOLDER,
    OpenAICompatibleLLMClient,
    effort_extra_body,
)
from app.llm.schemas import (
    ChatMessage,
    ChatRole,
    FinishReason,
    GenerationParams,
    GenerationPurpose,
    GenerationRequest,
    LLMError,
    LLMErrorCategory,
    ResponseSchema,
    StructuredMode,
    TraceMetadata,
)
from app.llm.structured import schema_instruction
from tests.unit.structured_fixtures import STATE_JSON, STATE_SCHEMA, fenced, state_schema

Handler = Callable[[httpx2.Request], httpx2.Response]


def completion_json(
    content: Any = "What's been weighing on you?", **message: Any
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6-luna-2026-09-01",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, **message},
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 12,
            "total_tokens": 42,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }


class Recorder:
    def __init__(self, response: httpx2.Response | None = None) -> None:
        self.requests: list[httpx2.Request] = []
        self.response = response or httpx2.Response(
            200, json=completion_json(), headers={"x-request-id": "req_oa_1"}
        )

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        return self.response

    @property
    def body(self) -> dict[str, Any]:
        return dict(json.loads(self.requests[-1].content))

    @property
    def headers(self) -> httpx2.Headers:
        return self.requests[-1].headers


def make_client(handler: Handler, **kwargs: Any) -> OpenAICompatibleLLMClient:
    options: dict[str, Any] = {
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
        "max_retries": 0,
    }
    options.update(kwargs)
    return OpenAICompatibleLLMClient(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), **options
    )


def make_request(model: str = "gpt-6-luna", **params: Any) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        system="SYSTEM PROMPT",
        messages=[
            ChatMessage(role=ChatRole.USER, content="Work is a lot."),
            ChatMessage(role=ChatRole.ASSISTANT, content="What part?"),
            ChatMessage(role=ChatRole.USER, content="My manager."),
        ],
        max_output_tokens=1000,
        timeout_seconds=5,
        params=GenerationParams(**params),
        trace=TraceMetadata(trace_id="t", purpose=GenerationPurpose.CONVERSATION),
    )


async def test_openai_request_shape() -> None:
    recorder = Recorder()
    client = make_client(recorder, system_role="developer", effort_style="reasoning_effort")
    await client.generate(make_request(effort="low"))
    assert str(recorder.requests[-1].url) == "https://api.openai.com/v1/chat/completions"
    body = recorder.body
    assert body["messages"][0] == {"role": "developer", "content": "SYSTEM PROMPT"}
    assert [m["role"] for m in body["messages"][1:]] == ["user", "assistant", "user"]
    assert body["max_completion_tokens"] == 1000
    assert body["reasoning_effort"] == "low"
    for absent in ("max_tokens", "temperature", "stream", "n"):
        assert absent not in body
    assert recorder.headers["authorization"] == "Bearer sk-test"


async def test_compatible_request_shape() -> None:
    recorder = Recorder()
    client = make_client(
        recorder,
        provider="qwen",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        max_tokens_param="max_tokens",
    )
    await client.generate(make_request("qwen-plus", effort="high", temperature=0.4))
    sent = recorder.requests[-1]
    assert sent.url.host == "dashscope-intl.aliyuncs.com"
    assert sent.url.path == "/compatible-mode/v1/chat/completions"
    body = recorder.body
    assert body["messages"][0]["role"] == "system"
    assert body["max_tokens"] == 1000
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
    assert body["temperature"] == 0.4


async def test_response_normalization() -> None:
    response = await make_client(Recorder()).generate(make_request())
    assert response.content == "What's been weighing on you?"
    assert response.provider == "openai"
    assert response.model_id == "gpt-6-luna-2026-09-01"
    assert response.requested_model == "gpt-6-luna"
    assert response.finish_reason is FinishReason.STOP
    assert response.provider_request_id == "req_oa_1"
    assert response.usage.input_tokens == 30
    assert response.usage.output_tokens == 12
    assert response.usage.cache_read_input_tokens == 4


async def test_refusal_is_reported_even_with_stop_finish() -> None:
    body = completion_json(None, refusal="I can't help with that.")
    response = await make_client(Recorder(httpx2.Response(200, json=body))).generate(make_request())
    assert response.finish_reason is FinishReason.REFUSAL
    assert response.content == ""


async def test_separate_reasoning_fields_are_never_returned() -> None:
    body = completion_json("Here for you.", reasoning_content="PRIVATE chain of thought")
    response = await make_client(Recorder(httpx2.Response(200, json=body))).generate(make_request())
    assert response.content == "Here for you."
    assert response.reasoning_removed is False


async def test_inline_reasoning_is_stripped() -> None:
    body = completion_json("<think>PRIVATE</think>Here for you.")
    response = await make_client(Recorder(httpx2.Response(200, json=body))).generate(make_request())
    assert response.content == "Here for you."
    assert response.reasoning_removed is True


async def test_none_key_never_falls_back_to_openai_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-must-not-leak")
    recorder = Recorder()
    client = make_client(
        recorder, provider="ollama", api_key=None, base_url="http://localhost:11434/v1"
    )
    await client.generate(make_request("llama3"))
    assert recorder.headers["authorization"] == f"Bearer {KEYLESS_PLACEHOLDER}"
    assert "sk-real-openai-key" not in str(recorder.headers)


async def test_openai_account_headers_stripped_for_third_parties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-secret")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-secret")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-Internal-Token: internal-secret")
    recorder = Recorder()
    client = make_client(
        recorder, provider="xai", api_key="xai-key", base_url="https://api.x.ai/v1"
    )
    await client.generate(make_request("grok-4"))
    headers = {k.lower(): v for k, v in recorder.headers.items()}
    assert "openai-organization" not in headers
    assert "openai-project" not in headers
    assert "x-internal-token" not in headers
    assert headers["authorization"] == "Bearer xai-key"


async def test_openai_account_headers_kept_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_ORG_ID", "org-mine")
    recorder = Recorder()
    await make_client(recorder, send_openai_headers=True).generate(make_request())
    assert recorder.headers["openai-organization"] == "org-mine"


def error_handler(status: int, headers: dict[str, str] | None = None) -> Handler:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            json={"error": {"message": "nope", "type": "x", "code": "y"}},
            headers=headers or {},
        )

    return handler


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (400, LLMErrorCategory.INVALID_REQUEST),
        (401, LLMErrorCategory.AUTHENTICATION),
        (403, LLMErrorCategory.PERMISSION),
        (404, LLMErrorCategory.NOT_FOUND),
        (409, LLMErrorCategory.SERVER_ERROR),
        (422, LLMErrorCategory.INVALID_REQUEST),
        (429, LLMErrorCategory.RATE_LIMITED),
        (500, LLMErrorCategory.SERVER_ERROR),
        (503, LLMErrorCategory.OVERLOADED),
        (504, LLMErrorCategory.TIMEOUT),
        (529, LLMErrorCategory.OVERLOADED),
    ],
)
async def test_http_errors_are_classified(status: int, category: LLMErrorCategory) -> None:
    with pytest.raises(LLMError) as info:
        await make_client(error_handler(status)).generate(make_request())
    assert info.value.category is category
    assert info.value.status_code == status
    assert "sk-test" not in info.value.message


async def test_retry_after_and_request_id_on_errors() -> None:
    handler = error_handler(429, {"retry-after": "3", "x-request-id": "req_err_9"})
    with pytest.raises(LLMError) as info:
        await make_client(handler).generate(make_request())
    assert info.value.retry_after_seconds == 3.0
    assert info.value.provider_request_id == "req_err_9"


async def test_connection_and_timeout_errors() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError) as conn:
        await make_client(refuse).generate(make_request())
    assert conn.value.category is LLMErrorCategory.CONNECTION
    with pytest.raises(LLMError) as timeout:
        await make_client(slow).generate(make_request())
    assert timeout.value.category is LLMErrorCategory.TIMEOUT


async def test_custom_headers_env_cannot_override_the_configured_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK applies OPENAI_CUSTOM_HEADERS after auth headers; an
    # Authorization line there must not replace the provider's own key.
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: Bearer sk-real-openai")
    recorder = Recorder()
    client = make_client(
        recorder, provider="xai", api_key="xai-key", base_url="https://api.x.ai/v1"
    )
    await client.generate(make_request("grok-4.3"))
    assert recorder.headers["authorization"] == "Bearer xai-key"
    assert "sk-real-openai" not in str(recorder.headers)


def test_enable_thinking_extra_body_mapping() -> None:
    assert effort_extra_body(None, "enable_thinking") is None
    assert effort_extra_body("high", "reasoning_effort") is None
    assert effort_extra_body("none", "enable_thinking") == {"enable_thinking": False}
    assert effort_extra_body("low", "enable_thinking") == {
        "enable_thinking": True,
        "thinking_budget": 1024,
    }


async def test_qwen_thinking_controls_travel_as_top_level_fields() -> None:
    recorder = Recorder()
    client = make_client(
        recorder,
        provider="qwen",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        effort_style="enable_thinking",
    )
    await client.generate(make_request("qwen-plus", effort="none"))
    body = recorder.body
    assert body["enable_thinking"] is False
    assert "reasoning_effort" not in body
    assert body["messages"][0]["role"] == "system"


async def test_list_content_keeps_only_text_parts() -> None:
    body = completion_json(
        [
            {"type": "thinking", "thinking": [{"type": "text", "text": "PRIVATE"}]},
            {"type": "text", "text": "I'm here with you."},
        ]
    )
    body["choices"][0]["finish_reason"] = "model_length"
    response = await make_client(Recorder(httpx2.Response(200, json=body))).generate(
        make_request("mistral-large-latest")
    )
    assert response.content == "I'm here with you."
    assert response.finish_reason is FinishReason.MAX_TOKENS


async def test_error_body_with_200_status_is_invalid_response() -> None:
    body = {"error": {"message": "upstream failed", "code": 502}}
    with pytest.raises(LLMError) as info:
        await make_client(Recorder(httpx2.Response(200, json=body))).generate(make_request())
    assert info.value.category is LLMErrorCategory.RESPONSE_INVALID


def coded_error(status: int, code: str, message: str) -> Handler:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"error": {"message": message, "type": "x", "code": code}}
        )

    return handler


@pytest.mark.parametrize(
    ("status", "code", "message", "category"),
    [
        (
            400,
            "data_inspection_failed",
            "Input data may contain inappropriate content.",
            LLMErrorCategory.CONTENT_FILTERED,
        ),
        (
            400,
            "Arrearage",
            "Access denied, please make sure your account is in good standing.",
            LLMErrorCategory.PERMISSION,
        ),
        (400, "invalid_api_key", "Incorrect API key provided.", LLMErrorCategory.AUTHENTICATION),
        (
            429,
            "insufficient_quota",
            "You exceeded your current quota, please check your plan and billing details.",
            LLMErrorCategory.QUOTA_EXHAUSTED,
        ),
        (
            429,
            "insufficient_quota",
            "Free allocated quota exceeded.",
            LLMErrorCategory.QUOTA_EXHAUSTED,
        ),
        (
            429,
            "insufficient_quota",
            "Allocated quota exceeded, please increase your quota limit.",
            LLMErrorCategory.RATE_LIMITED,
        ),
        (
            500,
            "RequestTimeOut",
            "Request timed out, please try again later.",
            LLMErrorCategory.TIMEOUT,
        ),
    ],
)
async def test_provider_specific_error_codes(
    status: int, code: str, message: str, category: LLMErrorCategory
) -> None:
    with pytest.raises(LLMError) as info:
        await make_client(coded_error(status, code, message)).generate(make_request())
    assert info.value.category is category


async def test_xai_string_error_body_with_api_key_message_is_auth() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            json={
                "code": "Client specified an invalid argument",
                "error": "Incorrect API key provided: xa***",
            },
        )

    with pytest.raises(LLMError) as info:
        await make_client(handler, provider="xai", base_url="https://api.x.ai/v1").generate(
            make_request("grok-4.3")
        )
    assert info.value.category is LLMErrorCategory.AUTHENTICATION


def test_thinking_budget_leaves_room_for_the_reply() -> None:
    body = effort_extra_body("high", "enable_thinking", max_output_tokens=4096)
    assert body == {"enable_thinking": True, "thinking_budget": 3072}
    small = effort_extra_body("max", "enable_thinking", max_output_tokens=1000)
    assert small == {"enable_thinking": True, "thinking_budget": 256}


# --- structured output ---

STRICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "conversation_state", "schema": STATE_SCHEMA, "strict": True},
}
INSTRUCTED_SYSTEM = "SYSTEM PROMPT\n\n" + schema_instruction(state_schema())


def structured_request(
    schema: ResponseSchema, model: str = "gpt-6-luna", **params: Any
) -> GenerationRequest:
    return make_request(model, **params).model_copy(update={"response_schema": schema})


def json_reply(content: Any) -> Recorder:
    return Recorder(httpx2.Response(200, json=completion_json(content)))


async def test_openai_strict_structured_output() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(
        recorder,
        system_role="developer",
        effort_style="reasoning_effort",
        structured_output="json_schema",
    )
    response = await client.generate(structured_request(state_schema(), effort="low"))
    body = recorder.body
    assert body["response_format"] == STRICT_FORMAT
    assert body["messages"][0] == {"role": "developer", "content": "SYSTEM PROMPT"}
    assert body["reasoning_effort"] == "low"
    assert body["max_completion_tokens"] == 1000
    assert response.content == STATE_JSON
    assert response.structured_mode is StructuredMode.NATIVE_STRICT


async def test_non_strict_json_schema_is_best_effort() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(recorder, structured_output="json_schema")
    response = await client.generate(structured_request(state_schema(strict=False)))
    assert recorder.body["response_format"]["json_schema"]["strict"] is False
    assert response.structured_mode is StructuredMode.NATIVE


async def test_json_object_mode_puts_schema_in_instruction() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(
        recorder,
        provider="deepseek",
        base_url="https://api.deepseek.com",
        max_tokens_param="max_tokens",
        structured_output="json_object",
    )
    response = await client.generate(structured_request(state_schema(), "deepseek-flash"))
    body = recorder.body
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": INSTRUCTED_SYSTEM}
    # JSON-object modes reject requests whose messages never mention JSON.
    assert "JSON" in body["messages"][0]["content"]
    assert response.structured_mode is StructuredMode.NATIVE


async def test_qwen_json_object_keeps_thinking_controls() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(
        recorder,
        provider="qwen",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        effort_style="enable_thinking",
        structured_output="json_object",
    )
    await client.generate(structured_request(state_schema(), "qwen-plus", effort="none"))
    body = recorder.body
    assert body["response_format"] == {"type": "json_object"}
    assert body["enable_thinking"] is False
    assert "provider" not in body


async def test_best_effort_json_schema_sends_schema_twice() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(
        recorder,
        provider="together",
        base_url="https://api.together.ai/v1",
        structured_output="json_schema_best_effort",
    )
    response = await client.generate(structured_request(state_schema(), "openai/gpt-oss-120b"))
    body = recorder.body
    assert body["response_format"] == STRICT_FORMAT
    assert body["messages"][0]["content"] == INSTRUCTED_SYSTEM
    assert "provider" not in body
    assert response.structured_mode is StructuredMode.NATIVE


async def test_openrouter_requires_endpoints_that_honor_response_format() -> None:
    recorder = json_reply(STATE_JSON)
    client = make_client(
        recorder,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        max_tokens_param="max_tokens",
        structured_output="json_schema_best_effort",
        structured_require_parameters=True,
    )
    response = await client.generate(structured_request(state_schema(), "openai/gpt-oss-120b"))
    body = recorder.body
    assert body["provider"] == {"require_parameters": True}
    assert body["response_format"] == STRICT_FORMAT
    assert response.structured_mode is StructuredMode.NATIVE


async def test_prompted_mode_strips_a_code_fence() -> None:
    recorder = json_reply(fenced(STATE_JSON))
    client = make_client(
        recorder,
        provider="openai_compatible",
        base_url="http://localhost:8080/v1",
        structured_output="prompt",
    )
    response = await client.generate(structured_request(state_schema(), "local-model"))
    body = recorder.body
    assert "response_format" not in body
    assert body["messages"][0] == {"role": "system", "content": INSTRUCTED_SYSTEM}
    assert response.content == STATE_JSON
    assert response.structured_mode is StructuredMode.PROMPTED


async def test_unfenced_prompted_reply_is_left_as_is() -> None:
    reply = '  {"topic": null, "subtopics": [], "user_goal": null}\n'
    client = make_client(json_reply(reply), structured_output="prompt")
    response = await client.generate(structured_request(state_schema()))
    assert response.content == reply


async def test_request_without_schema_is_unchanged_for_every_style() -> None:
    recorder = Recorder()
    client = make_client(
        recorder,
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        structured_output="json_schema_best_effort",
        structured_require_parameters=True,
    )
    response = await client.generate(make_request("openai/gpt-oss-120b"))
    body = recorder.body
    for absent in ("response_format", "provider"):
        assert absent not in body
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM PROMPT"}
    assert response.structured_mode is None


async def test_structured_refusal_is_reported() -> None:
    body = completion_json(None, refusal="I can't help with that.")
    client = make_client(Recorder(httpx2.Response(200, json=body)), structured_output="json_schema")
    response = await client.generate(structured_request(state_schema()))
    assert response.finish_reason is FinishReason.REFUSAL
    assert response.content == ""
    assert response.structured_mode is StructuredMode.NATIVE_STRICT


async def test_unsupported_schema_is_an_invalid_request() -> None:
    handler = coded_error(
        400,
        "invalid_json_schema",
        "Invalid schema for response_format 'conversation_state': "
        "'additionalProperties' is required to be supplied and to be false.",
    )
    client = make_client(handler, structured_output="json_schema")
    with pytest.raises(LLMError) as info:
        await client.generate(structured_request(state_schema()))
    assert info.value.category is LLMErrorCategory.INVALID_REQUEST
