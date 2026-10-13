# Architecture (as built: Phases 1-2)

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
       TurnStore.begin          tx 1: authorize + idempotency + persist user msg + load history + state
       StateExtractor           structured LLM call: updated conversation state (or keep previous)
       DecisionService          one System One request (Laya / Jev / off), validated, thresholded
       choose_policy            code: safety signal > explicit request > decision > default
       ContextBuilder           system prompt + policy fragment, notes block, budgeted history
       LLMClient.generate       provider adapter, bounded by turn deadline
       ResponseValidator        response contract
       TurnStore.complete       tx 2: persist reply + decisions + versioned state, mark completed
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
| Prompt registry | `app/prompting.py` | versioned, checksum-locked prompts and fragments | — |
| State extractor | `app/state/*` | typed conversation state with provenance, one structured LLM call | persistence, policy |
| Decision service | `app/decisions/*` | versioned questions, System One adapter, answer validation, thresholds, fallbacks | access control, safety handling |
| Response policy | `app/orchestration/policy.py` | choosing the reply approach in code | wording (prompt fragments) |

Core invariant: nothing the model produces can trigger data access. The model
has no tools, and ownership is checked in SQL (`get_owned_conversation`)
before any history is read.

## Turn lifecycle and transactions

Stages are recorded per turn (`app/orchestration/lifecycle.py`) and returned as
`timing_ms` in development:

`RECEIVED → VALIDATED → AUTHORIZED → USER_MESSAGE_PERSISTED → CONTEXT_LOADED →
STATE_UPDATED → DECISIONS_RESOLVED → POLICY_CREATED → CONTEXT_BUILT →
RESPONSE_GENERATED → RESPONSE_VALIDATED → ASSISTANT_MESSAGE_PERSISTED → COMPLETED`

The specification's `INPUT_SAFETY_ANALYZED` and `RETRIEVAL_PERFORMED_IF_NEEDED`
stages arrive with Phases 5 and 3/4. State extraction and decisions never fail
a turn; see [decisions.md](decisions.md) for each stage's failure behavior.

No transaction is held open during generation:

1. **Begin** (short transaction): `SELECT … FOR UPDATE` on the conversation
   row, filtered by owner. Then resolve the idempotency key, refuse a second
   in-flight turn, allocate `last_sequence + 1`, insert the user message as
   `pending` with a fresh `attempt_id`, and read recent history.
2. **Generate** outside any transaction, inside
   `asyncio.timeout(TURN_DEADLINE_SECONDS - elapsed)`.
3. **Complete** (short transaction): lock the conversation and user message,
   verify the message is still `pending` *with this `attempt_id`*, insert the
   assistant message at the next sequence, insert the turn's decisions, write
   the new conversation state only if its version is still the one read in
   step 1, and mark the turn `completed`.

Any failure after step 1 marks the turn `failed`, with a category such as
`timeout`, `llm:rate_limited`, `invalid_response:truncated` or
`persistence_error`. If even that write fails, the turn stays `pending`, ages
past `STALE_TURN_SECONDS`, and the next request takes it over.

## Data model (migrations `0001`, `0002`)

- `users(id, external_id UNIQUE, created_at)`
- `conversations(id, user_id → users ON DELETE CASCADE, status, last_sequence, created_at, updated_at)`
- `messages(id, conversation_id → conversations ON DELETE CASCADE, sequence, role, content, created_at,`
  user-turn fields `turn_status, idempotency_key, request_fingerprint, attempt_id, attempt_started_at, failure_category`,
  assistant provenance `reply_to_message_id → messages, provider, model_id, prompt_id, prompt_version, generation_metadata JSONB)`
- `conversation_states(conversation_id PK → conversations ON DELETE CASCADE, schema_version, state JSONB, state_version > 0, source_message_id → messages ON DELETE SET NULL, extraction JSONB, updated_at)`
- `turn_decisions(id, conversation_id → conversations, user_message_id → messages, question_id, question_version, kind, value, returned_value, probabilities, confidence, provider, model_id, latency_ms, validation_status, fallback, fallback_reason, created_at)`, `UNIQUE (user_message_id, question_id)`

Database-enforced invariants:
- `UNIQUE (conversation_id, sequence)`: total order and no duplicate positions.
- `UNIQUE (conversation_id, idempotency_key)`: persistent idempotency.
- `UNIQUE (reply_to_message_id)`: at most one assistant reply per user message.
- Partial `UNIQUE (conversation_id) WHERE turn_status = 'pending'`: at most one in-flight turn.
- A check constraint ties fields to roles. An assistant row must carry
  `reply_to_message_id`, `model_id`, `prompt_id` and `prompt_version`.

`generation_metadata` records requested and served model, prompt checksum,
finish reason, fallback use, provider request ID, token usage, latency,
generation parameters, context-budget stats, the response policy, a decision
summary, state-extraction provenance and the trace ID. Every reply is
traceable to its exact model, prompt, policy and settings.

## LLM adapters

`LLMClient` (`app/llm/base.py`) is a typed protocol over provider-neutral
`GenerationRequest`/`GenerationResponse` (`app/llm/schemas.py`). Provider SDK
objects never leave `app/llm/providers/`.

**Presets and resolution.** `registry.py` defines one preset per supported
provider: adapter kind, endpoint, API-key setting, default model, output and
context limits, and request shape. At startup, `resolve_provider(settings)`
merges the preset with `GENERATION_*` overrides. Configuration errors (no
key, no model, missing SDK extra) fail startup with a message naming the
variable or install command. The orchestrator uses the resolved model,
effort and limits. See [providers.md](providers.md) for the full table.

**Adapters**, each verified against its official SDK source and docs:
- `anthropic_provider.py` (`anthropic` SDK): `output_config.effort`; optional
  server-side refusal fallback; the served model is recorded.
- `openai_provider.py` (`openai` SDK): OpenAI and every OpenAI-compatible
  preset (xAI, Qwen, DeepSeek, OpenRouter, Together, Mistral, Ollama,
  custom). Per-preset system role, max-tokens field, and effort style
  (`reasoning_effort`, DashScope `enable_thinking`, or none). It pins
  credentials so `OPENAI_*` environment variables cannot leak to third
  parties.
- `groq_provider.py` (`groq` SDK): per-model-family reasoning controls (gpt-oss
  `include_reasoning=false`; qwen `reasoning_format=hidden`).
- `gemini_provider.py` (`google-genai` SDK): explicit timeout and retries (the
  SDK sets neither), `thinking_level`/`thinking_budget`, thought parts
  dropped, blocked prompts reported as refusals.
- `chat_completions.py`: shared normalization and error classification for
  the two Chat Completions SDKs (`openai`, `groq`).

**Reasoning never reaches users.** Separate reasoning fields (`reasoning`,
`reasoning_content`, Gemini thought parts, Mistral thinking chunks) are never
read. Inline `<think>` blocks are stripped by `app/llm/reasoning.py`, and
`reasoning_removed` is recorded in provenance. A refusal, truncation, or
empty reply is never stored; the response validator rejects it.

**Errors** normalize to `LLMErrorCategory`: authentication (including
providers that report a bad key as HTTP 400), permission, not found,
rate limited (with retry-after), quota exhausted (non-retryable), content
filtered (provider moderation), overloaded, server error, timeout,
connection, invalid request, invalid response.

**Retries.** The `anthropic`, `openai` and `groq` SDKs retry transient
errors themselves, honoring `retry-after`, up to `GENERATION_MAX_RETRIES`.
The Gemini adapter configures the same bound explicitly. Each attempt is
limited by `GENERATION_TIMEOUT_SECONDS`, and the whole call by the remaining
turn deadline. Retries cannot duplicate turns: a turn completes at most once
per `attempt_id`, and the database allows at most one reply per user
message.

`FakeLLMClient` is a labeled development stub. It never echoes content and
is refused in production.

## Context assembly

The system instruction is the rendered versioned prompt plus the policy
fragment(s) chosen by code, and nothing else. User content, and the
conversation notes derived from it, only travel in user-role messages: the
notes open the first included user message in an escaped
`<conversation_notes>` block. History is the most recent
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
| Memory | 3 | new tables and service; authorized memories become a separate context section |
| Knowledge RAG (pgvector) | 4 | compose already runs the pgvector image; extension enabled by the Phase 4 migration |
| Input safety analysis, safety pathway, policy output checks | 5 | before `CONTEXT_BUILT` and inside `ResponseValidator` |
| Evaluation runner, experiment tables | 6 | `evaluation_runs` / `evaluation_results` |
| SFT data pipeline | 7 | offline tooling under `data/training` and `scripts/` |
