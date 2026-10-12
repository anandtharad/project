# Architecture (as built: Phase 1)

This describes what exists today. Components from the specification that are
not built yet are listed at the end, with where they will plug in.

## Shape

A single modular FastAPI service over PostgreSQL. There are no queues, caches
or extra services. The logical pipeline from the specification maps onto
in-process components:

```text
HTTP request
  -> TraceMiddleware            trace ID, request log, unhandled-error -> generic 500
  -> BodySizeLimitMiddleware    413 above MAX_REQUEST_BYTES (declared or streamed)
  -> Route (FastAPI)            schema validation, dev auth, input length check
  -> TurnOrchestrator           lifecycle sequencing, deadline, failure mapping
       TurnStore.begin          tx 1: authorize + idempotency + persist user msg + load history
       ContextBuilder           system prompt + budgeted recent history
       LLMClient.generate       provider adapter (Anthropic / stub), bounded by turn deadline
       ResponseValidator        response contract
       TurnStore.complete       tx 2: persist assistant msg + mark turn completed
  -> JSON response
```

## Component responsibilities

| Component | Module | Owns | Does not own |
|---|---|---|---|
| API | `app/api/*` | validation, dev auth, HTTP error contract | conversational logic |
| Orchestrator | `app/orchestration/engine.py` | turn sequencing, deadline, failure mapping, provenance | SQL, provider details |
| Turn store | `app/orchestration/persistence.py` | transactions, ordering, idempotency, in-flight rule | generation |
| Context builder | `app/context/builder.py` | what the model sees, history budget | storage |
| LLM adapter | `app/llm/*` | provider requests, normalized responses and errors | authorization, persistence |
| Response validator | `app/orchestration/validation.py` | response contract (complete, non-empty, bounded) | safety policy (Phase 5) |
| Prompt registry | `app/prompting.py` | versioned, checksum-locked prompts | — |

Core invariant: nothing the model produces can trigger data access. The model
has no tools, and ownership is checked in SQL (`get_owned_conversation`)
before any history is read.

## Turn lifecycle and transactions

Stages are recorded per turn (`app/orchestration/lifecycle.py`) and returned as
`timing_ms` in development:

`RECEIVED → VALIDATED → AUTHORIZED → USER_MESSAGE_PERSISTED → CONTEXT_LOADED →
CONTEXT_BUILT → RESPONSE_GENERATED → RESPONSE_VALIDATED →
ASSISTANT_MESSAGE_PERSISTED → COMPLETED`

Phase 1 skips the specification's `INPUT_SAFETY_ANALYZED`, `STATE_UPDATED`,
`DECISIONS_RESOLVED` and `RETRIEVAL_PERFORMED_IF_NEEDED` stages. They arrive
with Phases 5, 2, 2 and 3/4.

No transaction is held open during generation:

1. **Begin** (short transaction): `SELECT … FOR UPDATE` on the conversation
   row, filtered by owner. Then resolve the idempotency key, refuse a second
   in-flight turn, allocate `last_sequence + 1`, insert the user message as
   `pending` with a fresh `attempt_id`, and read recent history.
2. **Generate** outside any transaction, inside
   `asyncio.timeout(TURN_DEADLINE_SECONDS - elapsed)`.
3. **Complete** (short transaction): lock the conversation and user message,
   verify the message is still `pending` *with this `attempt_id`*, insert the
   assistant message at the next sequence, and mark the turn `completed`.

Any failure after step 1 marks the turn `failed`, with a category such as
`timeout`, `llm:rate_limited`, `invalid_response:truncated` or
`persistence_error`. If even that write fails, the turn stays `pending`, ages
past `STALE_TURN_SECONDS`, and the next request takes it over.

## Data model (migration `0001`)

- `users(id, external_id UNIQUE, created_at)`
- `conversations(id, user_id → users ON DELETE CASCADE, status, last_sequence, created_at, updated_at)`
- `messages(id, conversation_id → conversations ON DELETE CASCADE, sequence, role, content, created_at,`
  user-turn fields `turn_status, idempotency_key, request_fingerprint, attempt_id, attempt_started_at, failure_category`,
  assistant provenance `reply_to_message_id → messages, provider, model_id, prompt_id, prompt_version, generation_metadata JSONB)`

Database-enforced invariants:
- `UNIQUE (conversation_id, sequence)`: total order and no duplicate positions.
- `UNIQUE (conversation_id, idempotency_key)`: persistent idempotency.
- `UNIQUE (reply_to_message_id)`: at most one assistant reply per user message.
- Partial `UNIQUE (conversation_id) WHERE turn_status = 'pending'`: at most one in-flight turn.
- A check constraint ties fields to roles. An assistant row must carry
  `reply_to_message_id`, `model_id`, `prompt_id` and `prompt_version`.

`generation_metadata` records requested and served model, prompt checksum,
finish reason, fallback use, provider request ID, token usage, latency,
generation parameters, context-budget stats and the trace ID. Every reply is
traceable to its exact model, prompt and settings.

## LLM adapter

`LLMClient` (`app/llm/base.py`) is a typed protocol over provider-neutral
`GenerationRequest`/`GenerationResponse` (`app/llm/schemas.py`). Provider SDK
objects never leave `app/llm/providers/`.

`AnthropicLLMClient` uses the official `anthropic` SDK (1.13) Messages API.
These details were verified against the SDK's installed type signatures:
- `output_config={"effort": …}` controls thinking depth. Current models think
  adaptively, so there is no thinking budget. Only `text` blocks are surfaced.
- `temperature`/`top_p`/`top_k` are gone from SDK 1.x and rejected by Opus
  4.7+. A configured temperature is forwarded via `extra_body` only for older
  models.
- `fallbacks="default"` plus the `server-side-fallback-2026-07-01` beta lets
  the provider retry on another model when the requested one declines for
  policy reasons. The model that actually served the reply is recorded in
  `model_id`, with `fallback_used`. Set `GENERATION_REFUSAL_FALLBACK=off` to
  disable it.
- A `refusal` or `max_tokens` finish is never stored as a reply. The
  validator rejects it.

**Retries.** The SDK retries connection errors, 408/409/429 and 5xx with
backoff that honors `retry-after`, up to `GENERATION_MAX_RETRIES`. It does
not retry invalid or authentication requests. Each attempt is bounded by
`GENERATION_TIMEOUT_SECONDS`, and the whole call by the remaining turn
deadline. Retries cannot duplicate turns, because the turn is completed at
most once per `attempt_id`, and the database allows at most one reply per
user message.

`FakeLLMClient` is a labeled development stub. It never echoes content and
is refused in production.

## Context assembly

The system instruction is the rendered versioned prompt and nothing else.
User content only travels in user-role messages. History is the most recent
`HISTORY_MAX_MESSAGES` messages, trimmed oldest-first to
`MAX_CONTEXT_TOKENS`. The token count is an estimate of about 4 characters per
token, not an exact provider count. The current message is always included,
the context always starts with a user turn, and adjacent same-role messages
(for example, a user message whose turn failed, followed by the next one) are
merged.

## Observability

Structured JSON logs carry a `trace_id`: `request.completed`,
`generation.completed/failed`, `response.validated`, `turn.completed`
(provider, model, prompt version, tokens, latency), `turn.replayed`,
`persistence.failed`, `request.unhandled_error` (exception type and
location only). Message and reply content, tokens and keys are never logged,
and a test enforces this. SDK/HTTP debug loggers are capped at WARNING.

## Not built yet, and where it plugs in

| Spec component | Phase | Integration point |
|---|---|---|
| Conversation state + Jev decisions | 2 | between `CONTEXT_LOADED` and `CONTEXT_BUILT`; `LLMClient.generate_structured` added then |
| Memory | 3 | new tables and service; authorized memories become a separate context section |
| Knowledge RAG (pgvector) | 4 | compose already runs the pgvector image; extension enabled by the Phase 4 migration |
| Input safety analysis, safety pathway, policy output checks | 5 | before `CONTEXT_BUILT` and inside `ResponseValidator` |
| Evaluation runner, experiment tables | 6 | `evaluation_runs` / `evaluation_results` |
| SFT data pipeline | 7 | offline tooling under `data/training` and `scripts/` |
