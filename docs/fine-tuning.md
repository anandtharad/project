# Fine-tuning preparation (Phase 7)

Fine-tuning is planned but deferred (spec §17). The system works with a
standard instruction model. This phase prepares data and plans experiments:
**no training run was made, and no data has been approved for one.**

## Sources and approval (spec §17.2)

`training/sources.toml` holds one audit record per source:

- dataset ID, URL, file URL and SHA-256;
- licence and permitted use, with the citation;
- origin and synthetic status;
- how its splits are used;
- quality annotations;
- the date the licence was checked;
- an approval.

| Source | Licence | Permitted use | Approved |
|---|---|---|---|
| ESConv | CC BY-NC 4.0 | non-commercial; the README adds "academic research use only" | **no** |
| EmpatheticDialogues | CC BY-NC 4.0 | non-commercial | **no** |

A build reads only approved sources, and an approval must name who approved
it and when (`approved_by`, `approved_on`). Approving is the project
owner's decision after reading the licence. Nothing is approved by default,
and the code never approves anything.

These are never training sources:
- **conversations with real people** (privacy);
- **evaluation sets** (held-out isolation, spec §16.3);
- **model replies from evaluation runs**.

```bash
uv run python scripts/import_dialogue_benchmarks.py     # downloads both, checks SHA-256
uv run python scripts/prepare_training_data.py audit    # licences, approvals, files, leakage index
# edit training/sources.toml to approve a source, then:
uv run python scripts/prepare_training_data.py build
```

## Pipeline (spec §17.3)

`app/training/pipeline.py`, deterministic. The same files, evaluation sets
and settings give byte-identical records and the same build id.

| Stage | What it does |
|---|---|
| Licence gate | refuses an unapproved source, or a file whose SHA-256 differs from the audited one |
| Conversations | ESConv: every conversation. EmpatheticDialogues: the official **train** split only (valid and test are evaluation sources and are never read) |
| Normalisation | same-speaker messages joined, NFKC, zero-width and control characters removed, whitespace collapsed; ED's `_comma_` restored |
| Examples | each supporter or listener message that answers the person is a target. History: up to 8 earlier messages, starting with the person, as at run time |
| Quality filter | drops, with a reason (below) |
| Safety screen | the deterministic screen (`app/safety/patterns.py`) on the message and the target. Any hit drops the example: safety is handled by the application's safety layer, which stays in place after any fine-tuning (spec §17.5) |
| Strategy labels | ESConv supporter strategies mapped to the companion's: Question → explore; Restatement and Reflection of feelings → reflect; Affirmation and Reassurance → validate; Providing Suggestions → problem_solve. Others: none. EmpatheticDialogues has no labels, so `strategy` is absent |
| Deduplication | the same message and target only once; the same reply word for word at most 3 times ("That sounds hard.") |
| Leakage check | drops any conversation used by an evaluation item, and any example whose message, history or target repeats an evaluation text, or shares an 8-word run with one |
| Split | by conversation (SHA-256 of source and id), one in ten to validation, so no conversation is in both splits |

**Quality filter.** Each dropped example is counted under one reason in
`manifest.json`:

| Reason | Why |
|---|---|
| `message_too_short`, `target_too_short`, `target_too_long` | fewer than 2 words in the message; target under 3 or over 200 words |
| `contact_or_link` | a URL, email address or phone number in the target: the companion never gives contact details from memory |
| `self_disclosure` | the supporter's own experience ("I went through the same thing", "my dad…", ESConv's Self-disclosure label). The companion never claims a life of its own, so these replies would teach the wrong thing |
| `unsupported_information` | ESConv's Information label: facts without evidence would teach unsupported claims |
| `crowdwork_artifact` | traces of the collection task ("this task", "survey", "bonus") |
| `advice_after_no_advice` | the person declined advice and the reply gave some anyway |
| `repeats_previous` | the target repeats the previous reply |

## Canonical record (spec §17.4)

`app/training/schema.py`, `TrainingRecord`, with no tokenizer or chat
template attached:

| Field | Content |
|---|---|
| `system` | the companion prompt **by reference** (`companion_system@<version>` and its checksum); exporters render it from the prompt registry, with the strategy's policy fragment as at run time |
| `history`, `message`, `target` | the conversation before the target, the person's message, the reply to learn |
| `state`, `strategy`, `memory`, `knowledge` | optional; none of these sources has state, memories or evidence, so they are empty |
| `absent` | lists exactly the empty optional fields (validated). Nothing is made up to fill one |
| `provenance` | source, URL, licence, file SHA-256, the source's conversation id, turn index, synthetic flag, every transformation with its version, pipeline version |
| `quality` | word counts, history length and truncation, the source's own labels (emotion, problem, survey scores) |
| `safety` | the screen version and the rules that fired (always none in kept records) |
| `split` | `train` or `validation` |

`manifest.json` records, per source:
- the file checksum and the approval;
- conversations and examples read;
- drops by reason;
- records per split.

It also records the settings, the evaluation files indexed (with their
checksums), the records' SHA-256 and the build id. `records.jsonl` holds
licensed text, so builds are kept in the git-ignored `training/build/`.

## Export

```bash
uv run python scripts/prepare_training_data.py export --build <id> --format trl
```

`app/training/export.py` writes `<format>-train.jsonl` and
`<format>-validation.jsonl` next to the build, and records their counts and
SHA-256 in the manifest. The system message is rendered from the referenced
prompt with the policy fragments a turn would get, using the engine's own
`system_prompt()`. The export refuses a prompt whose checksum changed since
the build.

Formats were checked against official documentation on 2026-10-16. Each
claim was re-checked by an independent reader:

| Format | Shape | Source |
|---|---|---|
| **`trl` (preferred)** | `{"prompt": [system, history…, user], "completion": [assistant]}`. TRL applies the model's own chat template, and with prompt-completion data trains on the completion only by default (`completion_only_loss=None` resolves to true) | `huggingface/trl` docs and code: `docs/source/sft_trainer.md`, `trl/trainer/sft_trainer.py`, `sft_config.py` (main and v1.16.0) |
| `openai` | `{"messages": [...]}`. The target has `"weight": 1`, earlier assistant messages `"weight": 0`. Weights are set on every assistant message because no documented default exists | OpenAI SFT guide, fine-tuning best practices, `openai/openai-openapi` schema. **OpenAI is winding down self-serve fine-tuning**: since 2026-07-02, organisations with no recent fine-tuned-model inference cannot create jobs (OpenAI deprecations page) |
| `chatml` | text: `<\|im_start\|>{role}\n{content}<\|im_end\|>\n` per turn; prompt ends `<\|im_start\|>assistant\n`; completion ends `<\|im_end\|>`. With a system message first, Qwen2.5's default system text is not inserted | `Qwen/Qwen2.5-7B-Instruct` `tokenizer_config.json`; qwen.readthedocs.io v2.5 |
| `llama3` | text: `<\|begin_of_text\|>` once, then `<\|start_header_id\|>{role}<\|end_header_id\|>\n\n{content}<\|eot_id\|>` per turn; prompt ends with the assistant header and its two newlines, as Meta's reference code and the official template render it (the web page's example omits them); completion ends `<\|eot_id\|>` | Meta's Llama 3 prompt-format page, `meta-llama/llama3` `tokenizer.py`, the Meta-Llama-3-8B-Instruct chat template |

**Not for Llama 3.1.** Its official chat template always adds "Cutting
Knowledge Date: December 2023" and "Today Date: …" lines to the system
message, with a hard-coded default date, and Meta's own documents disagree
on them. For Llama 3.1 and any other model, use `trl`, so the model's own
template is applied.

## Baseline failures that fine-tuning might address (spec §17.1, §17.5)

Measured, with the prompted and orchestrated system:

| Behaviour | Measured | Where | Fine-tuning data that would target it |
|---|---|---|---|
| Unsupported specifics without evidence | 1.9–2.6 per reply on the reply model (prompt 1.7.0) | docs/rag.md (groundedness) | library questions with ideal grounded answers and "no source" answers, written and reviewed for this purpose; not in ESConv or ED |
| Citing when evidence is given | 3 of 8 (reply model, 1.7.0) | docs/rag.md | the same |
| Ordinary-chat voice | fixed by prompting (1.5.0); stock phrases and question endings measured | docs/voice.md | ESConv / ED style, filtered as above, if approved |
| Strategy adherence | the policy changes replies (measured), decisions run in shadow mode | docs/decisions.md | ESConv strategy-labelled targets |
| Plans with arithmetic constraints | wrong at times | docs/voice.md | not a fine-tuning target for a reply model; a code check would be better |

## Planned experiments (spec §17.5)

Proceed only when all of these hold:
- a source is approved;
- the baseline failures above are re-measured on the model to be tuned;
- splits are clean (the leakage check passes);
- the evaluation suites are ready (docs/evaluation.md).

1. **Compare four systems under the same controls:**
   1. the base instruction model;
   2. the prompted and orchestrated baseline (this application);
   3. SFT on the canonical records;
   4. later, SFT plus preference optimisation.

   Each one keeps the safety layer, output checks and knowledge
   retrieval.
2. **Choose mixture weights by held-out evaluation, not equal by default.**
   Candidates:
   - ED only;
   - ESConv only;
   - 1:1;
   - 3:1 ED to ESConv.

   ESConv is all emotional support. Too much of it may make ordinary chat
   overly therapeutic, which the voice set and the golden casual scenarios
   measure.
3. **Judge on held-out sets only:**
   - golden scenarios (live, test split);
   - the voice held-out messages;
   - the dialogue benchmark test split (blind pairwise);
   - the safety set;
   - RAG generation and groundedness.

   Every release gate in `eval/gates.toml` must still hold, the mandatory
   ones without exception.
4. **Preference optimisation (DPO) needs a reviewed preference dataset.**
   It must not rank longer or more emotional replies higher by default.

Not yet decided: which open-weights model to tune, and where. The reply
model in use (`openai/gpt-oss-120b` on Groq) is served by a provider.
Whether and how it can be fine-tuned there has not been checked.
