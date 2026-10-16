# Dialogue benchmark items v1

Fixed-context reply items sampled from two public conversation datasets, for
comparing the companion's replies with what a person replied at the same
point (`app/evaluation/dialogue.py`, `scripts/eval_dialogue.py`). Spec §16.2
lists both as candidate datasets.

**Nothing from these datasets is committed.**
`scripts/import_dialogue_benchmarks.py` downloads the files to
`eval/datasets/dialogue/external/` (git-ignored), checks their SHA-256, and
writes `items.jsonl` there. Result files in `eval/results/` hold only item
ids and measures. Judge packets, which contain dataset text, stay in the
ignored folder.

## Sources

| | ESConv | EmpatheticDialogues |
|---|---|---|
| Paper | Liu et al., *Towards Emotional Support Dialog Systems*, ACL 2021 | Rashkin et al., *Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset*, ACL 2019 |
| File | `ESConv.json` from `raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/` | `empatheticdialogues.tar.gz` from `dl.fbaipublicfiles.com/parlai/empatheticdialogues/` (the URL in the repository README) |
| SHA-256 | `aa0556c5b330562ba009c1cd5137486bfa2a7255f33225a6524cd58f7efdd9af` | `56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d` |
| Licence | CC BY-NC 4.0 (repository LICENSE). The README adds: "Data and codes are for academic research use only." | CC BY-NC 4.0 (repository LICENSE) |
| Content | 1,300 conversations between a help-seeker and a supporter. Each conversation has problem, emotion and situation labels, and each supporter message has a strategy label (Question, Restatement or Paraphrasing, Reflection of feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others) | About 25k short conversations, each grounded in a situation and one of 32 emotions; the first speaker describes the situation and the second responds. Official train/valid/test split |

Licences were checked on 2026-10-16 from the repositories' LICENSE and
README files. Both licences are **non-commercial**. ESConv also says
**academic research use only**. This project uses them only to evaluate
replies, not to train. Whether a given use fits "academic research" is for
the project owner to judge. Do not redistribute the data, and do not train
on it without reviewing the terms again (Phase 7, spec §17.2).

**DailyDialog** (spec §16.2) was investigated but not imported. Its licence
is CC BY-NC-SA 4.0 (Hugging Face card `li2017dailydialog/daily_dialog`). That
repository holds only a loading script, which downloads from `yanran.li`, a
host that did not resolve from the build environment (2026-10-16). It is
also everyday chit-chat, which the companion's voice set already covers.

## Item format and sampling

One item is one reply point:

```json
{"id": "esconv-0346-19", "source": "esconv", "split": "dev",
 "context": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
 "turns": ["<the person's next message>"],
 "reference": {"reply": "<what the human replied>", "strategies": ["Question"]},
 "meta": {"problem": "...", "emotion": "...", "situation": "...", "cut": 19,
          "context_truncated": true},
 "licence": "..."}
```

- **Roles.** The help-seeker (ESConv) or the first speaker (ED) is `user`;
  the supporter or listener is `assistant`. Consecutive messages from one
  speaker are joined. The ED `_comma_` escapes are restored.
- **Cut points.** These are positions where the person's message has at
  least 5 words and the human reply has at least 4. Each conversation gives
  one seeded random cut.
- **Context.** This is the last 8 messages before the cut, and it always
  starts with the person. It is empty when the cut is the first message.
  `context_truncated` marks items where earlier messages were dropped.
- **Splits.** ED uses its official split (valid → `dev`, test → `test`).
  ESConv has no official split: the 1,300 conversations are shuffled with
  the seed and halved. Neither split takes items from the other split's
  conversations.
- **Size.** By default, 50 items per source and split: 200 in all
  (`--per-split`). The seed is 20261016. Two runs on the pinned files give
  the same `items.jsonl` (sha256 `617542e6…` at the default settings).

v1 at the defaults:

| | ESConv | ED |
|---|---|---|
| Items | 50 dev / 50 test | 50 dev / 50 test |
| Mean context messages | 6.3 (79 of 100 truncated) | 0.9 (58 of 100 have no context) |
| Median words, message / human reply | 17 / 18 | 17 / 10 |

## How it is used

`scripts/eval_dialogue.py run` replies to each item's history with the
configured model and companion prompt. The response policy is live, with
decisions off. Memory, knowledge and the safety classifier are not used. It
measures:

- **Deterministic proxies against the human reply:**
  - whether each reply asks a question;
  - whether it uses suggestion wording (the `ADVICE_WORDING` pattern of
    `app/evaluation/policy_effect.py`; for ESConv, the human side uses the
    strategy labels instead);
  - the companion-to-human length ratio;
  - rates conditioned on what the human did, for example "suggests when the
    human did not".
- **The voice measures** of `app/evaluation/voice.py`.
- **Safety flags.** It marks items whose message the deterministic safety
  screen flags (a real turn would take a safety pathway there).

`scripts/eval_dialogue.py score` adds blind pairwise judgements
(`eval/rubrics/dialogue_pairwise_v1.md`). For each item, the companion's
reply and the human reply appear as A and B in a seeded random order. The
key is kept apart from the packets. It records the win, tie and loss rates
per judge, the agreement between judges, and the rates on the items where
the judges agree.

No item comes from the 12 conversations whose opening messages were the
held-out public set for companion prompt 1.5.0 (docs/voice.md). This was
checked against `eval/results/voice-companion-1.4.0-vs-1.5.0-20261015.json`,
which records their ids only.

## Limits

- **One reply, not a conversation.** The history is the humans' own, so a
  reply is read at a point the companion did not steer to. Multi-turn
  behaviour is covered by the golden scenarios.
- **The human reply is a reference, not gold.** Both datasets are
  crowd-sourced, and some human replies are brief or off. Losing to a human
  reply means a judge preferred it; winning does not mean the reply is good.
- **The proxies are crude.** `?` counts as a question. The suggestion
  pattern catches "try" and "plan" in any sense.
- **Domain.** ESConv often discusses depression and other distress, so some
  messages trip the safety screen. Those are reported, not removed. ED
  covers everyday emotional situations.
- The judge is a model, calibrated against no human-reviewed set.
