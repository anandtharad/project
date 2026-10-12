# Implementation status

Last updated: 2026-10-12. Spec: COMPANION-TECH-SPEC-001 v2.1.

## Phase summary

| Phase | Status |
|---|---|
| 0: Foundation | **Done** |
| 1: Basic conversation | **Done** (live LLM calls verified with Groq; other providers unverified, see below) |
| 2: State and Jev | Not started. **Blocked**: official Jev docs unreachable (see Blockers) |
| 3: Memory | Not started |
| 4: RAG | Not started |
| 5: Safety | Not started. **Required before any real-user exposure** |
| 6: Evaluation and observability | Not started (structured logging and provenance already exist) |
| 7: Fine-tuning preparation | Not started |

## Multi-provider LLM support (added on request)

Added at the user's request after Phase 1, to make the generative LLM
swappable across providers (spec FR-004, §5.2 model comparison). Text only;
speech-to-text and text-to-speech are separate.

| Item | Status |
|---|---|
| Provider presets and resolution (12 providers + stub) | Done; unit-tested |
| Anthropic adapter | Done; offline-tested |
| OpenAI / OpenAI-compatible adapter (OpenAI, xAI, Qwen, DeepSeek, OpenRouter, Together, Mistral, Ollama, custom) | Done; offline-tested through the real `openai` SDK |
| Groq adapter | Done; offline-tested through the real `groq` SDK |
| Gemini adapter | Done; offline-tested through the real `google-genai` SDK |
| Credential-leak protections (OpenAI SDK env fallbacks) | Done; tested |
| **Live Groq calls** | **Verified** on 2026-10-12: `tests/live/test_groq_live.py` passed 6/6 on the user's Windows machine (Python 3.12.3) for `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`, and a manual multi-turn chat through the API and PostgreSQL worked |
| Live calls to other providers | **Unverified**: no keys. This cloud environment's egress policy also blocks provider APIs |

An adversarial review of the multi-provider changes (4 reviewers, each
checked by an independent skeptic) confirmed 21 of 37 reported issues. All
of them are fixed and covered by tests, including:
- a temperature above 1 that passed startup but failed every turn;
- reasoning leaking through an orphan `</think>` tag;
- SDK debug logs recording request bodies;
- a stale `GENERATION_API_KEY` being sent to another provider;
- preset default effort being applied to models that reject it.

Provider contracts were verified against each official SDK's installed
source plus official documentation. Many docs pages were only reachable
through search summaries; facts that depend on them (current model IDs,
free-tier limits, some error codes) are marked as such in
[providers.md](providers.md) and must be confirmed with a live run.

## Repository inspection (spec §0.1)

The repository was empty (no commits) at the start. There was no existing
code to preserve and no conflicts with the specification.

## Phase 0 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Clean environment can be initialized from README | Met | `uv sync` from `uv.lock`; README steps followed in this environment |
| App starts and connects to PostgreSQL | Met | `/ready` → `database: ok` (local PG 16 and the compose `db` service) |
| Migrations run | Met | `test_upgrade_downgrade_upgrade_and_models_match`: up → down → up, plus a model/migration drift check |
| Health checks work | Met | `/health` (liveness, no dependencies) and `/ready` (DB + migration head) tests |
| Tests run without model credentials | Met | all non-live tests use a scripted LLM double or a mocked HTTP transport |

Docker Compose was verified end to end in this environment: `db` healthy,
`migrate` exited 0, `app` healthy, a conversation turn completed and
persisted with provenance. One caveat: this sandbox intercepts TLS, so the
image was built from the committed Dockerfile plus one extra layer that
trusts the sandbox CA. The committed Dockerfile itself has not been built
unmodified here.

## Phase 1 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Multi-turn context is preserved | Met (test double) | `test_multi_turn_context_is_preserved_in_order` asserts the exact ordered history sent to the model |
| Message ordering is correct | Met | per-conversation sequence under row lock; unique `(conversation_id, sequence)`; ordering and pagination tests |
| Response addresses the user's actual message | Met for context use (live, Groq) | `tests/live/test_groq_live.py` passed on all three Groq free-tier models: replies pass validation and recall earlier context (`Biscuit`). Conversational quality is measured in Phase 6 |
| Duplicate requests do not duplicate completed turns | Met | DB-enforced idempotency; replay, in-flight duplicate, and parallel-burst tests |
| Provider failures are controlled | Met | failure-matrix tests: timeout, rate limit, connection, auth, invalid request, refusal, truncation, empty, over-long, persistence failure |
| Model and prompt versions are recorded | Met | `provider`, `model_id` (served model), `prompt_id`, `prompt_version` columns plus `generation_metadata` (prompt checksum, params, usage, request ID, trace ID) |

## Test results (this environment)

`uv run pytest` with PostgreSQL 16 (`TEST_DATABASE_URL` set):

| Suite | Result |
|---|---|
| unit | 226 passed |
| integration | 38 passed |
| security | 43 passed |
| live | 8 skipped (opt-in; provider APIs unreachable from this environment) |
| **total** | **307 passed, 8 skipped** |

Without `TEST_DATABASE_URL`, the database-backed tests skip, each with an
explicit reason. Tests clear provider-related environment variables, so the
developer's shell cannot change their outcome.
`ruff check`, `ruff format --check`, and `mypy --strict` (app, scripts, tests)
are clean.

### Failure-handling matrix coverage (spec §23)

| Failure | Status | Test |
|---|---|---|
| Invalid input | Rejected before model and storage | `test_invalid_input_rejected_before_model_or_storage`, body-limit tests |
| Unauthorized conversation | 404, no content | `test_access_control.py` |
| LLM timeout | 504, user message kept | `test_turn_deadline_exceeded` |
| Database transaction failure | Not reported complete | `test_persistence_failure_is_not_reported_as_complete`, `test_double_database_failure_leaves_turn_recoverable`, `test_database_outage_returns_dependency_error` |
| Output validation failure | Safe failure, nothing stored | `test_invalid_generations_are_not_persisted` |
| Duplicate request | Resolved, no duplicate turn | `test_idempotency.py`, `test_concurrency.py` |
| Jev timeout / malformed result | Phase 2 | — |
| Embedding failure / RAG unavailable | Phase 4 | — |
| Memory store unavailable | Phase 3 | — |
| Safety dependency unavailable | Phase 5 | — |

## Verified vs unverified integrations

- **Groq (live):** verified on 2026-10-12 with real free-tier calls from the
  user's Windows machine; 6/6 live tests passed for `openai/gpt-oss-120b`,
  `openai/gpt-oss-20b` and `qwen/qwen3.8-27b`. That confirms Groq accepts the
  exact request shape (`max_completion_tokens`, `reasoning_effort`,
  `include_reasoning=false` for gpt-oss, `reasoning_format="hidden"` for
  qwen), replies normalize and pass validation with no reasoning text, usage
  and the request ID (`x_groq.id`) are recorded, and multi-turn context is
  used. The user then ran the full stack on Windows (compose PostgreSQL,
  `alembic upgrade head` and `uvicorn` on the host, `scripts/chat.py`) and
  held a working multi-turn chat through the HTTP API with Groq replies
  persisted. Still offline-only: error classification for real Groq errors
  (401, 429, 5xx).
- **Anthropic Messages API:** the adapter is written against the official
  `anthropic` 1.13.0 SDK's type signatures and tested through the real SDK
  with a mocked HTTP transport: exact request JSON, beta header, response
  normalization, error classification, SDK retry behavior. **No live call was
  made** because no API key was available in this environment. Run
  `RUN_LIVE_LLM_TESTS=1 uv run pytest tests/live` with a key to verify.
- **Server-side refusal fallback** (`fallbacks: "default"`): request shape
  verified offline. Whether your account or organization has the beta enabled
  is unverified. If it doesn't, the provider returns 400, which shows up as
  `generation_unavailable` on every turn. Set
  `GENERATION_REFUSAL_FALLBACK=off` in that case.

## Stubs and development-only pieces

- `GENERATION_PROVIDER=fake`: labeled stub, no model is called. Refused in production.
- Development bearer-token auth: not production authentication. Production
  startup is refused.
- `/docs` and `/openapi.json`: disabled in production.

## Deviations from the specification (with reasons)

- **Tables:** only `users`, `conversations` and `messages` exist. The other
  tables in §15.1 (states, memories, knowledge, safety events, evaluation)
  are created by their own phases' migrations, so there are no empty
  placeholder tables (§20: "do not create empty modules").
- **Response validator location:** `app/orchestration/validation.py` rather
  than `app/safety/validation.py`. Phase 1 validates the response *contract*
  only. Putting it under `safety/` would suggest safety checks exist. Phase 5
  adds the safety package.
- **`LLMClient.generate_structured`** is not defined yet. It is added in
  Phase 2 with its first consumer (state extraction), rather than shipped
  as a stub.
- **Idempotency key scope** is per conversation. Each conversation has
  exactly one owner, so this satisfies "same key and owner", and it lets a
  client reuse a key across conversations.
- **Model comparison (§5.2)** has not been run yet. It needs the evaluation
  framework (Phase 6) and live credentials. The default model
  (`claude-opus-5-5`, effort `low`) is a starting point, not a measured
  choice. The model is configurable, and every reply records the model that
  served it.

## Known limitations

- No independent safety layer (Phase 5). Do not use with real users.
- No per-user rate limiting or cost quotas. Concurrency is limited only per
  conversation.
- No conversation deletion endpoint and no retention policy yet.
- No streaming responses. Each reply is returned when complete.
- The context token budget is a character-based estimate, not provider token
  counts.
- Long conversations are truncated to the most recent messages. There is no
  summarization (later phase).
- A request cancelled mid-generation leaves its turn `pending` until
  `STALE_TURN_SECONDS` passes (150 s by default). Until then, new messages in
  that conversation get `409 turn_in_progress`.

## Blockers and open questions for Phase 2

1. **Jev documentation is unreachable.** `docs.typesafe.ai` and
   `www.typesafeai.org` both fail DNS resolution from this environment.
   Third-party articles confirm Jev exists (TypeSafe AI decision model;
   Choice/Score/Noul question types; early access since 2026-09-14), but the
   specification requires the *official* API contract before integration
   (§0.4, §7.1). Phase 2 needs one of these: network access to the official
   docs, the API reference provided directly, or a Jev API key for contract
   testing.
2. **LLM credentials** for live verification and the Phase 6 model comparison.
