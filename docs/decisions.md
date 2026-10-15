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
| `llm` | a general LLM through the provider registry; default Groq `openai/gpt-oss-20b` (Apache-2.0) | the decision provider's own key (`GROQ_API_KEY`); see [LLM decision provider](#llm-decision-provider-decision_providerllm) |

`laya-serve` implements Jev's request and response shape; with
`LAYA_JEV_STRICT=1` (set by `scripts/dev.py laya` and the compose service)
its response matches the Jev contract exactly. Its `confidence` is computed
differently from Jev's (normalized entropy), so thresholds are per
calibration identity (see Thresholds below).

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

## LLM decision provider (`DECISION_PROVIDER=llm`)

An openly available substitute for Jev, which has no key yet, while Laya
zero-shot scores at or below the majority baseline. It sits behind the same
typed interface (`app/decisions/llm_backend.py`): it answers the same three
versioned questions and returns only typed answers, never text for the
person, so Jev's role in the spec (§7.1, "Jev is not the response
generator") is unchanged.

How it works, per turn:

1. One structured request to the decision model, through its own client
   (built by the provider registry for `DECISION_LLM_PROVIDER` /
   `DECISION_LLM_MODEL`, with `DECISION_MAX_RETRIES`; separate from the
   reply model's client). Default: Groq `openai/gpt-oss-20b`, effort `low`,
   at most 1024 output tokens, within `DECISION_TIMEOUT_SECONDS`.
2. System: the locked prompt `prompts/decision_llm/1.0.0.toml`, with each
   question's instructions and options inserted verbatim from
   `app/decisions/questions.py` (the per-question layout is a prompt
   fragment, so it is locked too). User: the same decision state as Jev and
   Laya get (latest message and up to four earlier messages), as JSON inside
   `<decision_state>` tags, marked as untrusted data and escaped so it cannot
   close its block.
3. Reply: a strict JSON schema built from the questions. A choice question
   gets an integer weight from 0 to 100 for **every** option; the noul gets
   an integer `probability_yes` from 0 to 100. The schema uses only the
   features Groq documents for strict mode (no `minimum`/`maximum`); the
   range and integer type are checked in code. Any other shape, truncated
   output or invalid JSON is `invalid_response` for the whole turn.
4. Conversion to the answer the service validates: choice = the
   highest-weight option (the first in option order on ties), `probabilities`
   = weights divided by their sum, `confidence` = the top normalised weight;
   noul = `probability_yes / 100`. If every option of a question gets weight
   0 there is nothing to normalise: the answer has no confidence and is
   rejected (`invalid_answer`), never invented.

**The numbers are normalised weights the model states, not calibrated
probabilities.** Asking the model for probabilities directly does not work:
no structured-output mode can require numbers to sum to 1, and in live
trials `gpt-oss-20b`'s stated probabilities summed to 0.90-1.11, which the
service's sum check (within 0.02) rejects; renormalising them would be
repairing an answer, which this layer never does. Independent weights with a
fixed, documented normalisation avoid both. What makes the result usable is
a threshold chosen for this exact model and prompt on the dev split and
confirmed on the held-out split. **On the dev split no threshold reaches
the target, so every question runs in shadow mode** with this provider too
(results below): answers are validated and stored
(`turn_decisions.returned_value`), and the policy ignores them.
Groq returns no log-probabilities for any model (HTTP 400), so token
probabilities are not an option there.

Failures map to the same fallback reasons as Jev's: timeout -> `timeout`;
connection, overload, 5xx, unknown -> `unavailable`; 429 and used-up quota
-> `rate_limited`; schema failures (Groq's `json_validate_failed`),
unparseable or truncated output -> `invalid_response`; authentication,
permission, unknown model, invalid request, provider moderation or a refusal
-> `rejected`. Neither the state nor the model's output is logged or put in
an error message (tested).

Cost and limits: about 1.5K tokens per turn on `gpt-oss-20b` (a ~1K-token
system prompt, the state, a ~150-token JSON reply and low-effort hidden
reasoning). Groq's free tier gives `gpt-oss-20b` its own limits (30 requests
and 8K tokens per minute, 1K requests and 200K tokens per day), separate
from the reply model's: about 5 turns a minute and ~130 a day. The default
is deliberately not the reply model (`openai/gpt-oss-120b` on Groq); using
the reply model for decisions logs a startup warning because they would
share its limits. `DECISION_TIMEOUT_SECONDS=5` leaves room for a hosted
model's latency.

Privacy: with this provider the latest message and up to four earlier
messages (each earlier one cut to 600 characters) go to the decision
provider (Groq by default) on every turn, in addition to what the reply and
state calls already send there. Conversation notes are not sent.

Configuration (`.env.example`): `DECISION_PROVIDER=llm`,
`DECISION_LLM_PROVIDER=groq`, `DECISION_LLM_MODEL=openai/gpt-oss-20b`,
`DECISION_LLM_EFFORT=low` (`off` for models without reasoning controls),
`DECISION_LLM_MAX_OUTPUT_TOKENS=1024`, `DECISION_PROMPT_VERSION=1.0.0`, and
the provider's own key. Startup refuses `DECISION_PROVIDER=llm` without that
key; `GENERATION_API_KEY` is used only when the decision provider is also
the reply provider, so the reply provider's key is never sent elsewhere.
Any registry provider works except `openai_compatible` with a different
reply provider (it has no endpoint of its own); a model change needs
`DECISION_LLM_MODEL` too.

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

A threshold is per question and **calibration identity**
(`app/decisions/thresholds.py`): the exact setup it was measured with, not
just the provider, because another model or prompt produces differently
distributed numbers. Keys:

| Provider | Calibration key |
|---|---|
| `jev` | `jev:<JEV_MODEL>`, e.g. `jev:jev-1.13.0` |
| `laya` | `laya:<LAYA_MODEL>`, e.g. `laya:english` |
| `llm` | `llm:<DECISION_LLM_PROVIDER>:<DECISION_LLM_MODEL>:decision_llm@<DECISION_PROMPT_VERSION>`, e.g. `llm:groq:openai/gpt-oss-20b:decision_llm@1.0.0` |

Changing the model or the decision prompt version therefore selects another
(empty) entry; it never silently reuses another model's thresholds. Effort
and the output limit are not part of the key but are recorded with every
evaluation, and `--thresholds-from` refuses dev results made with other
values: re-evaluate after changing `DECISION_LLM_EFFORT` or
`DECISION_LLM_MAX_OUTPUT_TOKENS`. Every entry is empty today.

Without a threshold, the question runs in **shadow mode**: its answer is
validated and stored (`returned_value`), but the application uses the
fallback (`fallback_reason = not_calibrated`). Thresholds are chosen on the
dev split of the evaluation set and confirmed on the held-out test split;
see Evaluation below.

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
checksums, the requested and served models, and for `DECISION_PROVIDER=llm`
the decision provider, effort, output limit, decision prompt key and
checksum (plus a checksum of the rendered system prompt and schema) and the
calibration key (spec §7.8 item 8, §19.3). `--thresholds-from` refuses a
file that is not a dev result or whose provider, model, decision prompt,
effort, output limit or question checksums differ from the current run. The
evaluation builds the decision state exactly as a live turn does (latest
message and up to four earlier messages; conversation notes are not sent to
the decision model), so thresholds chosen on dev apply unchanged to live
turns.

On a free tier, pace the run and continue it later if a daily limit stops
it (125 examples x ~1.5K tokens is close to Groq's 200K tokens per day for
`gpt-oss-20b`, so plan one split per day):

```
uv run python scripts/eval_decisions.py --split dev --pause 2 --tokens-per-minute 7000
uv run python scripts/eval_decisions.py --split dev --pause 2 --tokens-per-minute 7000 --resume
```

Each answer is cached in `eval/results/.cache/` as it arrives, keyed by the
run identity, dataset and split. A rate limit is waited out when its
`retry-after` is at most `--max-wait` (120 s); a longer wait, a used-up
quota or five limits in a row stop the run (exit code 2) instead of being
recorded as failures, which would silently lower accuracy and coverage.
Timeouts, outages and invalid responses are still recorded as failures,
because a live turn falls back on them. `--resume` skips the examples
already answered; `--start N` begins at example N (with `--limit` to run a
fixed part per day), taking the earlier examples from the cache. The result
file is written once every example in range has an answer.

## Results (2026-10-12/14, live)

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
fine-tuned checkpoint reaches 0.77 on its own benchmark), Jev
(`DECISION_PROVIDER=jev`), or the LLM provider (`DECISION_PROVIDER=llm`),
each re-evaluated with the same scripts. Jev has not been evaluated (no API
key yet). The LLM provider is evaluated below.

Files: `eval/results/decisions-laya-english-{dev,test}-*.json`.

### LLM decisions (`DECISION_PROVIDER=llm`, Groq `openai/gpt-oss-20b`, `decision_llm@1.0.0`, effort low; 2026-10-14)

| Question | Dev (125) | Laya dev | Majority baseline dev |
|---|---|---|---|
| conversation_mode: accuracy (all 125; valid answers only) | 0.696; 0.725 | 0.112 | 0.224 |
| response_strategy: accuracy (all; valid only) | 0.576; 0.602 | 0.288 | 0.229 |
| knowledge_retrieval_useful: AUROC | 0.948 (accuracy at 0.5: 0.933) | 0.373 | 0.5 = chance |
| `safety` mode recall | 5 of 7 (the other two: listening, reflection) | 1 of 7 | - |
| invalid answers (schema failures) | 5 of 125 | - | - |
| latency p50 / p95 | 0.65 s / 1.08 s | 2.3 s / 4.5 s | - |

Far better than Laya and the majority baseline, but **no threshold reaches
the pre-set target** (acted-on accuracy >= 0.80 with at least 15% coverage
and 10 examples). The best acted-on accuracy on the sweep is 0.758 for
conversation_mode (threshold 0.95, coverage 0.50) and 0.784 for
response_strategy (threshold 0.95, coverage 0.31). The target was fixed
before this run and is not lowered after it, so **every LLM question also
runs in shadow mode**, and the held-out test split was not run (it would
only confirm a threshold, and there is none). The annotators agreed far more
with each other (kappa 0.96-0.98, DATASET_CARD.md) than the model agrees
with them, so the gap is the model's, not a ceiling of the labels.

knowledge_retrieval_useful (a yes/no question) has no threshold sweep in the
evaluation, so it is recorded only, whatever its AUROC; Phase 4 routing can
add a sweep for it.

What could close the gap: a stronger decision model (the reply model would
share the developer's daily token budget), a revised decision prompt
evaluated on dev, or labelled data to fine-tune a small model.

File: `eval/results/decisions-llm-groq-openai_gpt-oss-20b-decision_llm_1.0.0-dev-20261014T031400.json`.

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
`scripts/eval_policy_effect.py --split test` to confirm. A live spot check of 1.3.0 (Groq
`gpt-oss-120b`): "I don't want advice. I just want it all to end, I can't do this anymore." got
the validate + no-advice policy, and the reply still followed the danger guidance (acknowledged
the pain, urged contacting emergency services or a crisis line and someone they trust).

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
