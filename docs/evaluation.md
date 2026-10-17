# Evaluation and observability (Phase 6)

How the system is measured, how two configurations are compared, and what
gates a release (spec §16, §19). Results are files in `eval/results/`, each
written by a run. No number in the docs comes from anywhere else.

## Suites

| Suite | Script | Data | What it measures | Needs |
|---|---|---|---|---|
| Decisions | `eval_decisions.py` | `eval/datasets/decisions/v1` | accuracy and calibration of each decision question (docs/decisions.md) | a decision provider |
| Policy effect | `eval_policy_effect.py` | decisions set | whether a response strategy changes the reply (a judge names the approach taken) | live model |
| Memory | `eval_memory.py retrieval / extraction` | `eval/datasets/memory/v1` | retrieval precision, recall and F1; extraction precision of stored memories (docs/memory.md) | database; live model for extraction |
| Voice | `eval_voice.py` | `eval/datasets/voice/v1` | how replies read: length, questions, lists, stock phrases (docs/voice.md) | live model |
| Safety | `eval_safety.py --mode patterns / classifier` | `eval/datasets/safety/v1` + public benchmarks | missed risk, false positives, invalid answers, classifier latency (docs/safety.md) | classifier mode: live model |
| Knowledge | `eval_rag.py routing / retrieval / generation` | `eval/datasets/rag/v1` | routing recall and unnecessary lookups, retrieval hit rate and abstention, citation validity and grounded wording (docs/rag.md) | database; live model for generation |
| Golden scenarios | `eval_golden.py --mode deterministic / live` | `eval/datasets/golden/v1` | 216 end-to-end conversations through the HTTP API, database and every layer | database; live model for live mode |
| Dialogue benchmarks | `eval_dialogue.py run / score` | ESConv and EmpatheticDialogues, downloaded | companion replies against the human reply at the same point, by proxies and a blind pairwise judge | live model |

Each dataset has a `DATASET_CARD.md`: origin, licence, labels, split and
limits. Choices are made on `dev`. `test` is held out and run once with
those choices fixed. When a test run leads to a change, the card says so,
and that test set is no longer called blind.

## Golden scenarios

A scenario is a short conversation of messages and actions. The actions
are: start a new conversation, delete all memories, have another user probe
the data, or make a dependency fail for the next turn. Each step carries
expectations, and every check is one of three kinds:

| Kind | Checks | Rule |
|---|---|---|
| **Guarantee** | HTTP status (503 after a reply-model failure, success after any other failure), privacy probe refused, memories gone after deletion, every citation resolves, explicit safety statements on the right pathway | must always hold: a mandatory gate |
| **Heuristic** | knowledge route, evidence reaching the prompt and its sources, safety pathway for wording only the classifier can judge | measured, with a floor |
| **Reply** | question count, lists, stock phrases, length, required and forbidden mentions, citing | live mode only |

**Deterministic mode** runs without a key:
- the development stub model;
- no safety classifier (degraded mode);
- no decision model;
- local embeddings.

The knowledge corpus is ingested into a temporary database. All 191
scenarios that have a deterministic part run in about a minute. **Live
mode** uses the configured provider. It caches replies per scenario, so a
run stopped by a rate limit resumes.

**Current result** (2026-10-16, `golden-deterministic-all-20261016T033433.json`):

| | |
|---|---|
| Scenarios checked | 191 of 216 (25 have reply checks only) |
| Guarantees held | **100%** of scenarios |
| Routing and retrieval checks | **97.3%**: the 5 failures are requests the library covers whose best passage scores just under the relevance threshold |

The first runs found router and policy bugs and 13 wrong expectations
(eval/datasets/golden/v1/DATASET_CARD.md). The test split's first run is
kept as it came out: guarantees 100%, routing and retrieval 83.5%.

Live mode has not been run on this set yet. A full run costs about 544
reply calls, plus classifier and extraction calls.

## Experiment metadata

Every Phase 6 run records `metadata` (`app/evaluation/metadata.py`):

- the provider, model, effort, temperature and token limits;
- every prompt's id, version and checksum;
- the decision provider and question versions, with the lock checksum;
- memory, embedding and knowledge settings, the knowledge manifest
  checksum, and the chunking;
- the dataset file and its checksum;
- the git commit, and whether the tree had changes;
- the Python version and the time.

Keys are never recorded. Older result files record the subset their script
wrote: model, prompt, split and dataset checksum.

Runs can also be stored in the database (`--record-db`, `evaluation_runs`
and `evaluation_results`; `app/evaluation/runs.py`). Items are stored as
ids, labels, metrics and a pass flag, never conversation text.

## Comparing configurations

```bash
uv run python scripts/eval_report.py eval/results/A.json eval/results/B.json --out report.md
```

The report covers three things:

- **Settings:** every setting that differs, from the metadata or the
  file's top-level fields.
- **Metrics:** every numeric metric in both files, with the difference.
- **Paired items:** for runs that score the same items (golden scenarios,
  safety predictions), the items that changed outcome and a bootstrap 95%
  interval for the difference in pass rate.

For example, the safety classifier against patterns alone: +14.1 points
correct, interval +6.4 to +21.8 (docs/safety.md).

**A comparison measures only what reaches the model.** The first 1.7.0 vs
1.8.0 comparison was void because the evaluation built its own system
prompt and left out the fragment under test. Evaluations now call the
engine's own `system_prompt()`.

## Judges

A model judge supplements deterministic checks and never replaces them
(spec §16.5).

- **Who judges.** Judges are Claude subagents, not the reply model.
- **What they see.** Replies are anonymised and shown in an order shuffled
  per item. The key is kept apart from the packets.
- **Agreement.** At least two judges are used, and their agreement is
  reported alongside the verdicts.

Rubrics are versioned files in `eval/rubrics/`. None is calibrated against
human judgements yet; results say so.

| Use | Rubric | Status |
|---|---|---|
| Voice, 1.4.0 vs 1.5.0 | in docs/voice.md | done: 3 judges, 31 of 36 held-out judgements preferred 1.5.0 |
| Groundedness of knowledge replies | `eval/rubrics/groundedness_v1.md` | done: 2 judges, 60 replies from 3 runs; verdict agreement 86.7% (kappa 0.78); found 2 misses in the code "no source" measure (docs/rag.md) |
| Dialogue benchmarks: companion vs human reply | `eval/rubrics/dialogue_pairwise_v1.md` | run on the reply model (dev, 40 items, 2026-10-17): see below |

## Public benchmarks

| Source | Licence | Use | Where |
|---|---|---|---|
| XSTest, SimpleSafetyTests, OR-Bench, Do-Not-Answer | CC BY 4.0; Do-Not-Answer CC BY-NC-SA 4.0 | safety screen | `import_safety_benchmarks.py` → `eval/datasets/safety/external/` |
| ESConv | CC BY-NC 4.0; "academic research use only" | reply comparison | `import_dialogue_benchmarks.py` → `eval/datasets/dialogue/external/` |
| EmpatheticDialogues | CC BY-NC 4.0 | reply comparison | same |
| DailyDialog | CC BY-NC-SA 4.0 | not imported: its download host did not resolve from the build environment | – |

Downloads are pinned by URL and SHA-256, kept in git-ignored folders, and
never committed. Results record ids and measures only. Nothing from them is
used for training (Phase 7 would need a fresh licence review).

## Release gates

`eval/gates.toml` lists each gate with its metric, the baseline from an
actual run, and a threshold. The thresholds were chosen on 2026-10-16,
before any further final evaluation. Mandatory gates are privacy, safety
and consistency checks, which must hold whatever the average quality
(spec §16.6).

```bash
uv run python scripts/check_gates.py           # exit 1 if a mandatory gate fails or has no result
uv run python scripts/check_gates.py --strict  # exit 1 if any gate fails
uv run pytest                                  # the test suite, tests/security included, is mandatory too
```

Each gate reads the latest result file matching its pattern. Where several
configurations write the same kind of file, a `require` filter applies: the
citation gate, for example, reads only runs on the default reply model.
All 12 gates pass as of 2026-10-16.

## Observability

Every request and turn writes structured JSON log events with the trace ID
(`app/observability.py`). The trace ID is taken when the event happens, so
background work keeps it. Events:

- **turn:** `turn.accepted`, `turn.completed` (provider, served model,
  prompt, policy source and strategy, tokens, generation and total time,
  the version of every component, knowledge status, citation count),
  `turn.replayed`, `turn.rate_limited`, `turn.attempt_lost`,
  `turn.unexpected_error`;
- **each layer:**
  - `safety.*`: assessed, classifier off or failed, output violation,
    recovered;
  - `decisions.resolved` and `decisions.skipped`;
  - `state.extracted` and `state.extraction_failed`;
  - `memory.retrieved` and `memory.processed`;
  - `knowledge.retrieved`, `knowledge.skipped`, `knowledge.timeout`,
    `knowledge.invalid_citations`;
  - `generation.completed` and `generation.failed`;
  - `response.validated`;
- **failures:** `dependency.unavailable`, `persistence.failed`,
  `background.failed`.

Message text is never logged. Each completed turn also stores the provider,
model, prompt version and checksum in the database, and its safety
assessment (`turn_safety`) and decisions (`turn_decisions`) in their own
tables. So a failure can be traced to a component, model, prompt, policy
or retrieval version (Phase 6 acceptance).

## Retention

| What | Where | Kept |
|---|---|---|
| Result files | `eval/results/*.json` (committed) | in git. Synthetic or public items, or ids only for licensed data |
| Model reply caches | `eval/results/.cache/` (ignored) | until deleted; safe to delete, runs will call the model again |
| Downloaded benchmarks, judge packets | `*/external/` (ignored) | until deleted; the importers re-download and verify them |
| Database runs | `evaluation_runs`, `evaluation_results` | until `RunRecorder.delete_run`. They hold no personal data |

Evaluation never reads production conversations.

## Dialogue benchmark on the reply model (2026-10-17)

Groq `openai/gpt-oss-120b`, companion prompt 1.7.0 (1.8.0 is identical
when no knowledge lookup runs, as here), 40 dev items: 20 ESConv, 20
EmpatheticDialogues. Two blind pairwise judges compared the companion's
reply with the human reply at the same point (`dialogue_pairwise_v1.md`;
replies shown as A and B in a seeded random order). File:
`eval/results/dialogue-dev-20261016T195212.json` (ids and measures only; the
dataset text stays in the git-ignored folder).

| | Judge A | Judge B | Where both agree (36 of 40) |
|---|---|---|---|
| Companion preferred | 72.5% | 82.5% | 80.6% |
| Tie | 2.5% | 2.5% | 2.8% |
| Human preferred | 25% | 15% | 16.7% |

Measured in code on the same replies: the companion's reply is a median
2.0 times the human reply's length (41.8 words on average); it asks a
question in 57.5% of replies (humans 47.5%) and suggests something in 20%
(humans 15%); 20% of replies contain a stock phrase; none uses lists or
headings.

Caveats: AI judges, not people, and AI judges are known to favour longer,
more elaborate replies; the length ratio says that bias may be at work. A
human-rated sample is still needed before reading this as quality.

