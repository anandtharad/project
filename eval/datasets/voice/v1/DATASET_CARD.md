# Voice evaluation set v1

Scripted conversations for comparing how replies read under different
companion prompt versions (`scripts/eval_voice.py`, measures in
`app/evaluation/voice.py`).

| | |
|---|---|
| Size | 12 conversations, 14 person turns (`dev.jsonl`) |
| Origin | 11 conversations copied from `eval/datasets/decisions/v1/dev.jsonl` (synthetic); `burden-001` written for this set, modelled on the shape of a developer's test chat (different wording) |
| Kinds | listening (6, including the 3-turn `burden-001`), casual (2), very short message (1), "no advice" (1), plan request (2), safety (1) |
| Labels | none: the set is for side-by-side comparison, not accuracy |

The plan-request and safety rows exist so that a warmer, shorter voice can
be checked not to break practical help or the danger guidance. Multi-turn
rows are replayed turn by turn, so each prompt version's own earlier
replies are in its context.

Small and synthetic: use it to compare versions, not to claim quality.
