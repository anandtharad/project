# Decisions, state and response policy (Phase 2)

How the application uses a decision model for bounded judgements (spec §7),
keeps conversation state (spec §8), and turns both into a response policy.

## Turn sequence

```
user message saved (ownership checked)
  -> state update      one structured LLM call (STATE_EXTRACTION)
  -> decisions         one request to the decision model, all questions
                       (state = latest message + up to 4 earlier messages)
  -> policy            plain code
  -> reply prompt      companion_system 1.3.0 + policy fragment(s)
  -> reply, validation, then reply + decisions + state saved in one transaction
```

Every new stage has a defined failure behavior, and none can fail the turn:

| Stage | On failure | Recorded as |
|---|---|---|
| State update (provider error, timeout, invalid JSON) | previous state kept | `generation_metadata.state.failure` |
| Decisions (timeout, outage, 429, 4xx, bad response) | each question gets its fallback | `turn_decisions.fallback_reason` |
| Single invalid or missing answer | that question falls back | `invalid_answer` / `missing_answer` |
| Stale state write (another turn stored a newer version) | reply kept, write skipped | log `state.write_rejected_stale` |

## Decision providers

One adapter (`app/decisions/backend.py`) on TypeSafe's official Python SDK
(`typesafe-sdk`, generated from `https://api.typesafe.ai/openapi.json`)
serves any server that implements `POST /v1/systemone`:

| `DECISION_PROVIDER` | Server | Notes |
|---|---|---|
| `off` (default) | none | every question uses its fallback |
| `laya` | self-hosted [Laya](https://huggingface.co/convaiinnovations/laya) (`laya-serve`, Apache-2.0) | runs locally, no data leaves the machine; needs `LAYA_API_KEY` (same key on server and app) |
| `jev` | TypeSafe Jev, `https://api.typesafe.ai` | hosted, early access; `JEV_API_KEY`; model pinned to `jev-1.13.0` |

`laya-serve` implements Jev's request and response shape; with
`LAYA_JEV_STRICT=1` (set by `scripts/dev.py laya` and the compose service)
its response matches the Jev contract exactly. Its `confidence` is computed
differently from Jev's (normalized entropy), so thresholds are per provider.

Run Laya locally (pinned `laya[serve]==0.4.6`, CPU):

```
uv run python scripts/dev.py laya                      # Windows/macOS/Linux, needs uv
docker compose --profile laya up -d laya               # or as a container
```

Then set in `.env`: `DECISION_PROVIDER=laya`, `LAYA_API_KEY=<long random>`,
`DECISION_TIMEOUT_SECONDS=5`. The first start downloads ~800 MB of weights.

Facts this integration relies on (verified 2026-10-12 against the SDK
source and docs.typesafe.ai): endpoint `POST /v1/systemone`, Bearer key,
request `{state, model, questions}`, answers keyed by question name; Choice
answers carry `choice`, `confidence`, `probabilities`; Score answers carry
`score`, `confidence`, `legend`, `probabilities`; Noul answers carry only
`noul` (the probability of yes) - no confidence is fabricated for it.
Errors 401/422/429/529; `jev-1.13.0` allows 255 options per Choice and a
64k-token context; TypeSafe states it does not train on user data.

## Questions (V1)

Defined in `app/decisions/questions.py`, versioned and locked by checksum
(`decisions/questions.lock.json`); a released question cannot be edited in
place.

| Question | Type | Answer space | Fallback | Downstream use |
|---|---|---|---|---|
| `conversation_mode@1.0.0` | Choice | casual, listening, exploration, reflection, problem_solving, planning, information, safety, other | `other` | stored in state; `safety` makes the policy conservative |
| `response_strategy@1.0.0` | Choice | reflect, validate, explore, clarify, summarize, provide_information, problem_solve, plan, check_in | `reflect` | sets the policy strategy |
| `knowledge_retrieval_useful@1.0.0` | Noul | probability of yes | 0 | recorded only until Phase 4 |

The decision model cannot choose `safety_response`; safety handling belongs
to the safety layer (Phase 5). Decisions never control access, persistence,
retention or rate limits (spec §7.3), and they run only after the ownership
check.

Every answer is validated against the question asked: the label must be one
of the options and the most probable one, probabilities must cover exactly
the options and sum to about 1, confidence and probabilities must be in
[0, 1]. Anything else is rejected, never repaired.

## Thresholds and shadow mode

A threshold is per question and provider (`app/decisions/thresholds.py`).
Without one, the question runs in **shadow mode**: its answer is validated
and stored (`returned_value`), but the application uses the fallback
(`fallback_reason = not_calibrated`). Thresholds are chosen on the dev split
of the evaluation set and confirmed on the held-out test split; see
Evaluation below.

## Response policy

`app/orchestration/policy.py`, in order of precedence:

1. **safety_signal**: a calibrated `conversation_mode = safety`. The prompt
   points the model to its danger guidance. A signal, not the safety layer.
2. **explicit_request**: the message says what the person wants: "I don't
   want advice / just need to vent" (validate, no advice), "what should I
   do? / any tips for...?" (problem_solve), "help me plan..." (plan), "can
   you summarize...?" (summarize).
3. **decision**: a calibrated `response_strategy`.
4. **default**: no constraint; the model chooses, exactly as in Phase 1.

Each policy maps to fixed text: fragments of the locked `companion_system`
prompt (1.3.0; `[fragments]` table, part of its checksum). The default
(open), reflect, validate and no-advice policies allow a complete reply with
neither advice nor a question; explore and clarify ask one question, and
problem_solve and plan offer options or steps. The prompt's danger guidance
always takes precedence over the chosen approach, including a no-advice
instruction (1.3.0), and the safety signal never adds "no advice".

Explicit requests are matched conservatively, as first-person requests to
the companion ("I don't want advice", "what should I do?", "can you help me
plan my week?", "can you summarize what we talked about?"). Mentions and
statements ("my boss asked me to summarize...", "there are no solutions
left") are not requests; curly apostrophes are normalized.

## Conversation state

`app/state/`. Fields: topic, subtopics, user_goal and emotional_context
(each with provenance `stated` or `inferred`; unknown values are not
stored), conversation_mode (from a calibrated decision only),
open_questions, recently_discussed. Every field may be absent.

- Extracted before each reply with the versioned `state_extraction` prompt
  and a strict JSON schema (no nulls or `anyOf`, so every structured-output
  mode can enforce it). It uses the reply model unless
  `STATE_EXTRACTION_MODEL` is set; `STATE_EXTRACTION_ENABLED=false` turns it
  off. Measured live on Groq (2026-10-13, 5 runs each, strict mode):
  `openai/gpt-oss-120b` 5/5 valid, `qwen/qwen3.8-27b` 3/3 valid when not
  rate-limited (its own, tighter limit), `openai/gpt-oss-20b` 0/5 (Groq
  rejected its own output as `json_validate_failed`). None adopted an
  instruction injected into the conversation.
- Stored in `conversation_states` with `state_version`; written in the same
  transaction as the reply only if the version read at the start of the turn
  is still current.
- State is derived from untrusted conversation text. It reaches the reply
  model only as a `<conversation_notes>` block at the start of the first
  user message (escaped so it cannot close its own block), never in the
  system prompt, and the prompt says to treat it as background, not
  instructions. It is never treated as verified personal facts.

## Evaluation

Dataset: `eval/datasets/decisions/v1/` (250 synthetic, AI-labelled
conversations; see its DATASET_CARD.md for provenance and caveats).

```
uv run python scripts/eval_decisions.py --split dev
uv run python scripts/eval_decisions.py --split test --thresholds-from eval/results/<dev file>.json
uv run python scripts/eval_policy_effect.py --examples 12
```

Results are written to `eval/results/` with the dataset checksum, question
checksums, and the served model. The evaluation builds the decision state
exactly as a live turn does (latest message and up to four earlier
messages; conversation notes are not sent to the decision model), so
thresholds chosen on dev apply unchanged to live turns.

## Results (2026-10-12/13, live)

### Laya decisions (`laya[serve]==0.4.6`, English checkpoint, zero-shot, CPU)

| Question | Dev (125) | Held-out test (125) | Majority baseline dev / test |
|---|---|---|---|
| conversation_mode: accuracy | 0.112 | 0.096 | 0.224 / 0.176 |
| response_strategy: accuracy | 0.288 | 0.269 | 0.229 / 0.168 |
| knowledge_retrieval_useful: AUROC | 0.373 | 0.498 | 0.5 = chance |
| `safety` mode recall | 1 of 7 | 2 of 6 | - |
| latency p50 / p95 | 2.3 s / 4.5 s | 1.8 s / 4.9 s | - |

No confidence threshold reaches the target (acted-on accuracy >= 0.80 with
at least 15% coverage and 10 examples) on dev, so **every Laya question runs
in shadow mode**: answers are validated and stored, and the policy ignores
them. This matches Laya's own model card ("near chance on typed-decisions
zero-shot ... a fast base to specialise, not a zero-shot decision engine").
Dev-only tuning of the state format did not change that: on a 60-example
dev subset, plain text (mode 0.22, strategy 0.29), latest-first text
(0.17, 0.26) and a turn list (0.10, 0.09) were all within noise of the
current JSON state (0.12, 0.26) and of the majority baseline, so the JSON
state (also Jev's documented shape) is kept.

What would change this: fine-tuning Laya on labelled decisions (its
fine-tuned checkpoint reaches 0.77 on its own benchmark), or Jev
(`DECISION_PROVIDER=jev`), re-evaluated with the same scripts. Jev has not
been evaluated: no API key yet.

Files: `eval/results/decisions-laya-english-{dev,test}-*.json`.

### Does the response policy change replies?

Groq `openai/gpt-oss-120b` writes the replies and judges them blind; 10
open-ended conversations per split without an explicit request; strategies
validate / problem_solve / explore (chance = 0.33).

| Prompt | Split | Judge recovers the forced strategy | 95% CI |
|---|---|---|---|
| 1.1.0 (soft fragments; judge `gpt-oss-20b`, 12 dev conversations incl. explicit requests) | dev | 0.33 | 0.20-0.50 |
| 1.2.0 (directive fragments) | dev | 0.77 | 0.59-0.88 |
| 1.2.0 | **held-out test** | **0.80** | **0.63-0.91** |

The default prompt is now 1.3.0. It differs from 1.2.0 only in safety wording found by the Phase 2
review: the danger guidance takes precedence over the chosen approach, and the no-advice fragment
says it never overrides it. The strategy fragments are unchanged. The effect was not re-measured on
1.3.0 because the run would exceed Groq's free-tier daily token limit for the reply model; rerun
`scripts/eval_policy_effect.py --split test` to confirm.

Held-out test, surface measures per strategy (10 replies each):

| Strategy | Asks a question | Advice wording | Mean words |
|---|---|---|---|
| validate | 10% | 10% | 44 |
| explore | 100% | 20% | 34 |
| problem_solve | 100% | 70% | 109 |
| open (no policy) | 90% | 20% | 55 |

So the policy measurably changes replies when it acts (spec acceptance:
"strategy decisions measurably affect responses"). Today it acts on
explicit requests in the message; decision-driven strategies are wired and
tested but wait for a decision model that passes its threshold. This
measures that replies change, not that they get better; reply quality is
Phase 6.

Files: `eval/results/policy-effect-*.json`.

### Live end-to-end (2026-10-13)

Three turns through the HTTP API with Groq (replies and state) and Laya
(decisions) on PostgreSQL: all completed; state reached version 3 with
correct provenance (`stated`), extracted with strict structured output; the
"I don't want advice, I just need to vent" turn got the validate + no-advice
policy and a reply with neither advice nor a question; the summary request
got an accurate summary; 9 decision rows stored in shadow mode. Turn time
3.7-4.9 s, of which Laya on CPU took 1.8-3.8 s.
