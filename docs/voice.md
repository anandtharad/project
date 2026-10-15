# Voice: how replies read

The developer tested 1.4.0 and found it "very robotic". The problems were:

- every reply opened with a stock line ("It sounds like...", "I hear you");
- every reply ended with a question, often a menu of options;
- it gave advice and logistics (journaling, check-in routines) to someone who wanted to be heard;
- it used bulleted lists with bold despite the prompt forbidding them;
- replies were long for short messages.

Companion prompt **1.5.0** rewrites the voice, and it is now the default
(`COMPANION_PROMPT_VERSION`). The rules for honesty, danger, memories, notes
and instruction integrity are unchanged in substance.

## What 1.5.0 changes

- **Voice:** it writes the way a warm, perceptive friend texts back, not a
  coach, therapist or help desk. It answers the feeling under the message, in
  the person's own words.
- **Concrete rules**, because the reply model follows those better than
  style advice:
  - length matches the message (a line or two gets one to three sentences);
  - plain text only, unless the person asked for a plan, steps, a list or a
    summary;
  - at most one question, often none, and never a menu;
  - advice only when asked for or clearly wanted, and only after the person
    feels heard;
  - a short list of stock phrases to avoid.
- **It may offer to listen** ("you're not a bother here"), without suggesting
  it replaces the people in someone's life.
- **About itself:** asked how it is or how its day was, it says lightly that
  it is an AI with no days of its own. It never makes up a day, plans,
  feelings or experiences.
- **Danger replies are warmer** but direct. It says it's glad they told it and
  that they matter, offers to keep talking while they reach out, and gives the
  same guidance to contact emergency services or a crisis line and someone
  they trust.
- **Three short tone examples** (a burden worry, a drained day, a small win).
  They are labelled as tone only and are not about the developer's chat.
- **The approach fragments use the same voice.** The plan fragment asks the
  model to check a plan against the person's constraints.

## How it was chosen

The code is `app/evaluation/voice.py` (surface measures) and
`scripts/eval_voice.py` (generation). The reply model is Groq
`openai/gpt-oss-120b`. Decisions were off and the live response policy was
used.

1. **Design.** Four drafts were written independently (concrete rules,
   few-shot examples, persona first, self-check list). Each was checked
   against the hard requirements by an adversarial reviewer.
   - Two drafts had used the developer's own messages as examples. Those
     examples were replaced, both to keep the test fair and to keep the
     developer's words out of the repository.
2. **Screen on the developer's 3-turn chat.** Three blind judges ranked all
   five versions, and 1.4.0 came last for every judge.
3. **Top two on the scenario set.** The few-shot and persona drafts ran on
   `eval/datasets/voice/v1/dev.jsonl` (12 conversations, 14 replies:
   listening, casual, short, no-advice, plan requests, danger). The few-shot
   draft won.
   - Judges flagged two real problems in it: it claimed a day of its own, and
     its danger reply was more scripted than 1.4.0's.
   - A third flag (an "invented" interview) was a judging error. The judges
     had not been shown the scenario's earlier messages. Packets now include
     them.
4. **Revise and confirm.** The two real problems were fixed, and the result
   became 1.5.0. It was compared with 1.4.0 on three sets:
   - the scenario set;
   - the developer's chat;
   - **12 held-out real messages**, never used for tuning: opening messages
     sampled from ESConv and EmpatheticDialogues.

Judges were Claude subagents, not the reply model. Each saw the two versions
anonymised, in an order shuffled per judge and scenario, with the scenario's
earlier messages. Each ranked them, rated naturalness and helpfulness from 1
to 5, and listed hard-rule violations.

## Results (2026-10-15)

| Set | Judgements preferring 1.5.0 | Mean words (1.4.0 → 1.5.0) | Replies with a stock phrase | Replies ending with a question |
|---|---|---|---|---|
| Scenario set (12 conversations, 14 replies) | 30 of 36 | 126 → 80 | 64% → 21% | 71% → 36% |
| Developer's chat (3 replies) | 3 of 3 | 63 → 39 | 100% → 0% | 100% → 33% |
| Held-out public messages (12) | 31 of 36 | 53 → 34 | 83% → 25% | 100% → 25% |
| **All** | **64 of 75 (85%)** | | | |

- **Hard-rule violations:** judges found none in either version.
- **Lists or bold** on the scenario set: from 21% of replies to 7%. The one
  left is a plan, where a list is allowed (though bold is not).

Where 1.4.0 still won:

- **The weekly-timetable request.** 1.5.0's plan broke the person's
  constraints: six gym sessions instead of three. The constraint check in the
  plan fragment did not fix this; the reply model gets arithmetic wrong at low
  reasoning effort.
- **"How's your day going?"** Judges found the honest "I'm an AI, so I don't
  really have days" a little stiff. Honesty here is a requirement, so it stays.
- **One procrastination message** in the held-out set.

File: `eval/results/voice-companion-1.4.0-vs-1.5.0-20261015.json`. It contains:

- the full replies and judge reasons for the scenario set;
- measures and preference counts only for the developer's chat and the public
  messages, whose texts are not redistributed.

## Limits

- **Small sample.** 26 scenarios; the judges are AI models with correlated
  judgement, not people. This shows a strong, consistent preference, not
  proof that people will find it better. Human review of a sample is the next
  step (spec §16.5).
- **Same model throughout.** Replies were generated once per version on the
  same model. A different reply model may need its own check.
- **Surface measures are proxies** (`app/evaluation/voice.py`). A reply can
  pass all of them and still be cold.
- **Danger replies still name US numbers** (911, 988). Localised, verified
  resources are part of the safety layer (Phase 5).

## Run it

```
uv run python scripts/eval_voice.py --variant old=companion_system@1.4.0 \
    --variant new=companion_system@1.5.0 --scenarios eval/datasets/voice/v1/dev.jsonl
```

`--variant label=<path to a draft .toml>` compares a draft that is not yet in
`prompts/`. Replies are cached in `eval/results/.cache/`.

Each person message costs one reply call (about 2-3K tokens on gpt-oss-120b).
On Groq's free tier that comes out of the same daily allowance as your own
chats, so keep runs small.
