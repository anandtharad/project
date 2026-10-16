# Implementation status

Last updated: 2026-10-16. Spec: COMPANION-TECH-SPEC-001 v2.1.

## Phase summary

| Phase | Status |
|---|---|
| 0: Foundation | **Done** |
| 1: Basic conversation | **Done** (live LLM calls verified with Groq; other providers unverified, see below) |
| 2: State and Jev | **Built and evaluated.** 4 of 5 acceptance criteria met; strategy decisions are wired and the policy measurably changes replies, but no decision model passes calibration yet, so decisions run in shadow mode (see Phase 2 acceptance). An LLM decision provider (`DECISION_PROVIDER=llm`, Groq `gpt-oss-20b`) now substitutes for Jev behind the same interface; evaluated live on dev (mode 0.70, strategy 0.58, retrieval AUROC 0.95, vs Laya 0.11 / 0.29 / 0.37), but no threshold reaches the 0.80 acted-on target, so it also runs in shadow mode (docs/decisions.md) |
| 3: Memory | **Built and evaluated.** All 5 acceptance criteria met on the synthetic evaluation set (see Phase 3 acceptance); retrieval and extraction quality are moderate and measured, not assumed |
| 4: RAG | **Built and evaluated** (docs/rag.md): 26 reviewed public-domain sources, versioned idempotent ingestion, hybrid retrieval with rank fusion and a relevance threshold, form-based routing v5 (never for casual chat, companion or conversation questions, or on safety pathways), escaped evidence block, citations that always resolve (100% on the reply model), sources in the API. Blind test: requests missed 11.1%, no evidence reaching the prompt for non-requests, retrieval hit rate 69.8%. Prompt 1.8.0 (knowledge fragments) helped on `gpt-oss-20b` and awaits confirmation on the reply model before it becomes the default |
| 5: Safety | **Built and wired into every turn** (docs/safety.md): pattern screen + LLM classifier, degraded mode, pathway policy overriding the strategy, verified regional resources, output checks with bounded recovery, `turn_safety` records. Not clinically validated; the person-facing texts need qualified review before real use |
| 6: Evaluation and observability | **Built and run** (docs/evaluation.md): experiment metadata on every run, evaluation tables and recorder, 216 golden end-to-end scenarios over all 16 spec categories (deterministic: guarantees 100%, routing and retrieval 97.3%), comparison reports with paired bootstrap intervals, 12 release gates with baselines (3 mandatory; all pass), public benchmarks imported without redistribution (safety; ESConv and EmpatheticDialogues), a blind pairwise judge rubric. Not yet run: golden live mode and the dialogue benchmark on the reply model (free-tier budget) |
| 7: Fine-tuning preparation | **Built, no data approved, no training run** (docs/fine-tuning.md): licence audit records with explicit approvals (none given: ESConv and EmpatheticDialogues are non-commercial, ESConv academic-only), a deterministic pipeline to canonical records (quality and safety filtering, strategy labels, dedup, leakage checks against every evaluation file, splits by conversation), and exports verified against official docs (TRL, OpenAI, ChatML, Llama 3) |

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
| Structured output (JSON Schema replies) in every adapter | Done; offline-tested through each SDK. Live on Groq: strict mode valid on `gpt-oss-120b` and `qwen3.8-27b`, **fails on `gpt-oss-20b`** (`json_validate_failed`) |
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
| unit | 733 passed |
| integration | 114 passed |
| security | 57 passed |
| live | 8 skipped (opt-in; run with `scripts/dev.py live-groq`) |
| **total** | **904 passed, 8 skipped** (2026-10-16, after the golden scenarios and routing v5) |

Live runs made from this environment once network access was allowed
(2026-10-13): Groq live tests, strict structured state extraction, the
decision and policy-effect evaluations, and a three-turn end-to-end chat
through the API with Groq and a local Laya server (docs/decisions.md).

A second adversarial review (5 lenses, each finding checked by an
independent skeptic) of the Phase 2 changes confirmed 15 issues; all are
fixed and covered by tests. They include: the explicit-request patterns read
"there are no solutions left" as refusing advice and could override the
danger guidance (now first-person requests only, and prompt 1.3.0 puts the
danger guidance first); curly apostrophes turned "I don't want advice" into
an advice request; an exhausted turn budget left a turn stuck as pending;
a NUL character in extracted state could fail the reply transaction; the
OpenAI-compatible adapter chose the structured-output mode from the reply
model instead of the requested one; and threshold coverage ignored failed
requests.

A third review of the Phase 3 changes (security and privacy, correctness
and concurrency, docs against code; three independent reviewers, each
finding reproduced) confirmed 29 distinct issues; all are fixed, the code
ones covered by tests.
The most important: the supporting quote was stored unscreened (it could
hold a diagnosis or a phone number) and is no longer stored; the "quote is
in the message" check accepted one-letter quotes and inferences; identifiers
had no code check; logs named which sensitive category was found; a
malformed embedding response, or OpenAI embedding retries, could fail or
time out a whole turn; retrieval looked only at the 500 newest memories;
settings changes, delete-all and deletes could race with a running
extraction; and lowering the retention period did not apply to existing
memories.

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
| Output validation failure | Contract failure: safe failure, nothing stored. Safety checks: one regeneration, then the pathway's fixed fallback | `test_invalid_generations_are_not_persisted`, `test_a_failing_reply_is_regenerated_once`, `test_a_reply_failing_twice_on_a_crisis_pathway_gets_the_fallback` |
| Duplicate request | Resolved, no duplicate turn | `test_idempotency.py`, `test_concurrency.py` |
| Jev timeout / malformed result | Documented fallback per question, turn completes | `test_service_failures_fall_back_for_every_question`, `test_slow_service_is_cut_off_at_the_decision_timeout`, `test_decision_outage_falls_back_and_the_turn_completes` |
| State extraction failure | Previous state kept, turn completes | `test_extraction_failure_keeps_previous_state_and_the_turn_completes`, `test_extractor_failures_keep_the_previous_state` |
| Stale state write | Rejected, reply kept | `test_stale_state_write_is_rejected_but_the_reply_is_kept` |
| Embedding failure | Full-text matching; memory stored without a vector and embedded later | `test_embedding_failure_falls_back_to_full_text_and_is_backfilled_later` |
| Memory store unavailable | No memories in the prompt, flagged; turn completes | `test_an_unavailable_memory_store_means_no_memories_and_the_turn_succeeds` |
| Memory extraction failure | Nothing stored; turn unaffected; one retry when the provider reports a schema failure or rejects the structured request | `test_a_failed_extraction_does_not_affect_the_turn`, `test_a_failed_structured_extraction_is_retried_once` |
| RAG unavailable | No passages; an evidence request is told sources are unavailable; never claims a source; turn completes | `test_a_retrieval_failure_never_claims_a_source` |
| Safety dependency unavailable | Degraded mode: patterns only, recorded; with no hit the turn is `unassessed`, never safe; strict output checks | `test_classifier_failure_is_degraded_mode_and_explicit_risk_still_routes`, `test_classifier_failure_with_no_signal_is_recorded_as_unassessed_not_safe`, `test_a_failed_semantic_check_does_not_block_the_reply` |

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
- **Local embeddings (fastembed, bge-small-en-v1.5):** verified here by
  running it (download, load, inference, thread safety, pgvector round trip)
  and through the memory evaluation. Not yet run on Windows; all its wheels
  exist for Windows x64 / Python 3.12.
- **OpenAI and Gemini embeddings:** built on the official SDKs (openai 3.29,
  google-genai 2.29) and tested against mocked transports. **No live call**
  (no keys).
- **pgvector:** verified against extension 0.6.0 (local) and 0.8.7 (the
  compose image).
- **LLM decision provider (`DECISION_PROVIDER=llm`):** verified live on
  Groq `openai/gpt-oss-20b` through the dev evaluation (125 conversations,
  2026-10-14): strict `json_schema` requests, 5 of 125 schema failures
  mapped to `invalid_response`, p50 0.65 s. Offline: the scripted LLM double,
  the fake provider end to end, and the real `groq` SDK on a mocked
  transport. Not yet exercised in a live HTTP turn.
- **Server-side refusal fallback** (`fallbacks: "default"`): request shape
  verified offline. Whether your account or organization has the beta enabled
  is unverified. If it doesn't, the provider returns 400, which shows up as
  `generation_unavailable` on every turn. Set
  `GENERATION_REFUSAL_FALLBACK=off` in that case.

## Stubs and development-only pieces

- `GENERATION_PROVIDER=fake`: labeled stub, no model is called. Refused in production.
- `EMBEDDING_PROVIDER=fake`: hashed bag of words for tests. Refused in production.
- Development bearer-token auth: not production authentication. Production
  startup is refused.
- `/docs` and `/openapi.json`: disabled in production.

## Deviations from the specification (with reasons)

- **Tables:** `users`, `conversations`, `messages`, `conversation_states`,
  `turn_decisions`, `memories` and `user_memory_settings` exist. The other
  tables in §15.1 (knowledge, safety events, evaluation) are created by
  their own phases' migrations, so there are no empty placeholder tables (§20). Evaluation
  results are JSON files under `eval/results/` until the Phase 6 tables.
  `turn_decisions` is an experiment-specific table (§15.1 allows them) for
  the per-decision records §7.6/§7.8 require.
- **Response validator location:** `app/orchestration/validation.py` rather
  than `app/safety/validation.py`. Phase 1 validates the response *contract*
  only; the Phase 5 safety checks of reply content are `app/safety/output.py`.
- **Structured generation** is a `response_schema` field on the existing
  `generate()` request rather than a separate `generate_structured` method:
  one code path per adapter, with the mode reported in `structured_mode`.
- **Jev configuration names:** `DECISION_PROVIDER=off|jev|laya|llm` replaces
  `JEV_ENABLED`, because the same interface also serves self-hosted Laya
  and the LLM substitute.
  `JEV_API_KEY`, `JEV_MODEL` and `JEV_TIMEOUT_SECONDS` (alias of
  `DECISION_TIMEOUT_SECONDS`) keep the spec's names.
- **Laya** was added at the user's request as a self-hosted, open-weights
  decision model that speaks Jev's protocol; it is not in the spec.
- **An LLM substitutes for Jev** (`DECISION_PROVIDER=llm`), because there
  is no Jev key and Laya zero-shot scored at or below the majority
  baseline. It sits behind the same typed `DecisionBackend` interface,
  answers the same versioned questions, and returns only typed answers
  (validated like Jev's), never text for the person, so §7.1 "Jev is not
  the response generator" still holds; the reply model and its prompt are
  unchanged. Its choice probabilities are normalised integer weights the
  model states, not Jev-style probabilities (docs/decisions.md), so its
  thresholds are kept per model and decision prompt, and it runs in shadow
  mode until evaluated. The default model (`openai/gpt-oss-20b`) differs
  from the reply model so the two do not share Groq's per-model limits.
- **Thresholds are keyed by calibration identity** (`jev:<model>`,
  `laya:<model>`, `llm:<provider>:<model>:decision_llm@<version>`) rather
  than by provider alone, so a model or prompt change never reuses another
  setup's thresholds (§7.7, §7.8 item 8).
- **Idempotency key scope** is per conversation. Each conversation has
  exactly one owner, so this satisfies "same key and owner", and it lets a
  client reuse a key across conversations.
- **Model comparison (§5.2)** has not been run yet. It needs the evaluation
  framework (Phase 6) and live credentials. The default model
  (`claude-opus-5-5`, effort `low`) is a starting point, not a measured
  choice. The model is configurable, and every reply records the model that
  served it.

## Known limitations

- The safety layer (Phase 5) is not clinically validated, its evaluation is
  indicative (AI-labelled and public benchmark data; imminent danger and abuse
  disclosures are barely covered), and its person-facing texts need review by
  a qualified person. Do not use with real users until that review is done.
- Per-user message limits (`USER_MESSAGES_PER_MINUTE`, `USER_MESSAGES_PER_DAY`)
  guard against runaway use; they are approximate under concurrent first
  messages in different conversations, and there is no token or cost quota
  per user.
- Conversations are deleted after `CONVERSATION_RETENTION_DAYS` (365) with
  no new message, or on request (`DELETE /v1/conversations/{id}`); both keep
  memories, which have their own retention. Deleting a memory does not
  delete the message it came from.
- Memory sensitivity screening relies on the extraction model's label, with
  a code backstop only for precise money amounts. Retrieval precision on the
  held-out set is about 0.6, so some irrelevant memories reach the prompt;
  the prompt tells the companion to use a memory only when it helps.
- Memory extraction is one more LLM call after each reply (in the
  background) and shares the provider's rate limits.
- No streaming responses. Each reply is returned when complete.
- The context token budget is a character-based estimate, not provider token
  counts.
- Long conversations are truncated to the most recent messages. There is no
  summarization (later phase).
- Decisions add latency: Laya on CPU takes about 2-5 s per turn (p95 ~5 s);
  `DECISION_TIMEOUT_SECONDS` bounds it (fallbacks after that). The LLM
  provider adds one hosted call per turn (about 1.5K tokens on
  `gpt-oss-20b`; on Groq's free tier about 5 turns a minute and ~130 a day
  before decisions fall back as `rate_limited`); its latency is unmeasured.
- State extraction adds one LLM call per turn (~0.7-1.2 s on Groq) and, on
  Groq's free tier, shares the reply model's per-minute token limit.
- The decision evaluation set is synthetic and AI-labelled (see its dataset
  card); results on it are indicative until humans review the labels.
- The `safety` conversation mode is a signal only, and Laya's recall of it
  on the evaluation set is 3 of 13. It is not a safety mechanism (Phase 5).
- A request cancelled mid-generation leaves its turn `pending` until
  `STALE_TURN_SECONDS` passes (150 s by default). Until then, new messages in
  that conversation get `409 turn_in_progress`.

## Phase 2 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Invalid decisions are rejected | Met | every answer validated against its question (label in options and most probable, probabilities over exactly the options summing to ~1, values in [0, 1]); invalid answers fall back, never repaired: `test_answers_that_do_not_fit_the_question_are_rejected` and related tests |
| Missing fields do not crash the app | Met | all state fields optional; missing answers fall back individually; an unreadable stored state is replaced: `test_invalid_and_missing_answers_fall_back_individually`, extraction failure tests |
| Jev outage has defined behavior | Met (offline; Laya outage also exercised) | timeout, connection, 429, 401/422, 5xx/529 and malformed responses each map to a fallback reason; the turn completes. The LLM provider maps every provider error category, truncated, unparseable and off-schema output the same way (`tests/unit/test_llm_decisions.py`) |
| Strategy decisions measurably affect responses | **Partly met** | the policy measurably changes replies on held-out data (judge recovers the forced strategy 0.80 vs 0.33 chance, prompt 1.2.0; docs/decisions.md). It acts today on explicit requests in the message. Decision-driven strategies are wired and tested, but Laya zero-shot is below the majority baseline for conversation_mode, at chance for retrieval, and only modestly above it for response_strategy (0.27 vs 0.17 on test), with no confidence threshold reaching the target accuracy, and Jev has no key yet, so decisions stay in shadow mode |
| No decision bypasses safety or authorization | Met | decisions run after the ownership check and only select fixed prompt fragments; `safety_response` is not selectable; a safety signal takes precedence: `test_decisions_never_run_for_a_foreign_conversation`, `test_safety_mode_signal_takes_precedence`, policy tests |

Phase 2 tasks (spec §21): state schema and persistence, state extraction,
decision adapter from the verified contract, initial Choice questions,
retrieval-routing decision (recorded only until Phase 4), strategy policy,
validation and fallbacks, and evaluation on labelled data are all built.
See [decisions.md](decisions.md).

## Phase 3 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Useful memories persist across sessions | Met | stored after the reply, retrieved into a later conversation: `test_a_useful_memory_persists_into_a_later_conversation`; live extraction recall 1.00 on the held-out split (prompt 1.1.0; 0.94 on dev); live end-to-end in Docker: a fact told in one conversation was recalled and used in the next (docs/memory.md) |
| Irrelevant content is not automatically stored | Met, with measured errors | code-side policy (explicit, a quote of two or more words in the message sharing a word with the memory, no identifiers, no sensitive categories by default, no instruction-like content, no duplicates, at most 3 per message); messages under 20 characters never reach the extractor; live: 3 of 39 held-out items with nothing to store had something stored, against 13 of 39 for the model's raw candidates; precision of what is stored 0.86. Remaining errors include 3 religion/legal details the model labelled not sensitive |
| Retrieval is relevance-filtered | Met | semantic threshold 0.49 chosen on dev; held-out test precision 0.62 and recall 0.74 (top 5 without a threshold: precision 0.40); expired and superseded memories never returned (0 in every run) |
| Deletion is reflected in future retrieval | Met | `test_deletion_is_reflected_in_future_retrieval` (the row and its embedding are gone), delete-all (also during a running extraction), expiry, retention and purge tests; live in Docker: after deletion the companion said it had no record of the dog's name |
| Unauthorized access is blocked | Met | `tests/security/test_memory_isolation.py`: another person's memory is a 404 for view, delete and confirm; delete-all and settings are per person; one person's memories never reach another's reply or extraction call |

Phase 3 tasks (spec §21): schema, candidate extraction and eligibility
policy, persistence and retrieval, duplicate and contradiction handling,
user inspection and deletion, retention, and cross-user security tests are
all built. See [memory.md](memory.md).

## Phase 6 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Configurations can be compared reproducibly | Met | every run records its metadata (model, prompts and checksums, settings, dataset checksum, commit); `scripts/eval_report.py` diffs two runs, with a paired bootstrap interval where items match; seeded sampling, cached replies |
| Failures trace to component/model/prompt/policy/retrieval versions | Met | `turn.completed` logs every component's version; turns store provider, model, prompt and checksum; per-turn `turn_safety` and `turn_decisions`; golden results name the failing step and check (routing v5 and a policy fix were traced this way) |
| Critical privacy/safety tests are mandatory | Met | mandatory gates: golden guarantees (privacy probes, deletion, failure status codes, explicit safety), citations resolving, no evidence for non-requests; plus the full test suite including `tests/security` |
| Metrics derive from actual runs | Met | every number in the docs points to a file in `eval/results/`; gates read those files; two measurement bugs (a missing prompt fragment, curly apostrophes) were found and the affected runs redone or rescored with a record |

## Phase 7 acceptance

| Criterion | Status | Evidence |
|---|---|---|
| Dataset preparation is reproducible | Met (fixtures) | the pipeline is deterministic; a build twice gives identical records and build id (`test_splits_are_by_conversation_and_builds_are_reproducible`); the manifest records file and evaluation-set checksums. Not yet run on real data: nothing is approved |
| Every example has provenance | Met | `TrainingRecord.provenance`: source, URL, licence, file SHA-256, conversation id, turn, synthetic flag, every transformation with its version |
| Held-out evaluation is isolated | Met | conversations used by any evaluation item are excluded; any example repeating an evaluation text or sharing an 8-word run with one is dropped (all 13 evaluation files indexed); EmpatheticDialogues valid/test are never read; splits are by conversation |
| No training run is required | Met | none was made |
| No unapproved data is used | Met | a build refuses unapproved sources and files whose SHA-256 differs from the audit; approvals need a name and date; none is given |

## Voice (companion prompt 1.5.0, added on request)

The developer found 1.4.0 robotic. 1.5.0 (default) rewrites the voice and
keeps the rules; blind judges preferred it in 64 of 75 judgements, including
31 of 36 on held-out public messages, with no hard-rule violations found
(docs/voice.md). Known gap: plans with arithmetic constraints are sometimes
wrong (a reply-model limitation the prompt did not fix). Judges were AI; human
review of a sample is still needed.

## Open questions and next steps

1. **A decision model that passes calibration.** The LLM provider on
   `gpt-oss-20b` gets close on dev (best acted-on accuracy 0.76 / 0.78
   against the 0.80 target). Options: a revised decision prompt or a
   stronger model evaluated on dev (`scripts/eval_decisions.py --pause 2
   --tokens-per-minute 7000`, `--resume` to continue; one split per day on
   Groq's free tier), a Jev API key (`DECISION_PROVIDER=jev`, then
   rerun `scripts/eval_decisions.py`), or fine-tuning Laya on labelled
   decisions (its notebook runs on Kaggle's free GPUs). Either way,
   thresholds come from the dev split and are confirmed on test before any
   question leaves shadow mode.
2. **Human review of the evaluation labels**, at least the ambiguous and
   adversarial rows.
3. **LLM credentials** for providers other than Groq, and the Phase 6 model
   comparison.
4. **Runs waiting for the reply model's daily budget** (Groq free tier,
   `gpt-oss-120b`): the 1.7.0 vs 1.8.0 generation comparison (20 queries),
   then the dialogue benchmark (`eval_dialogue.py run --split dev --limit
   40`, then two blind judges) and a live golden sample.
