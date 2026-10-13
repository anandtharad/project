# Decision evaluation set v1

Labelled conversations for evaluating the Phase 2 decision questions
(`conversation_mode@1.0.0`, `response_strategy@1.0.0`,
`knowledge_retrieval_useful@1.0.0`; wording in `app/decisions/questions.py`).

| | |
|---|---|
| Size | 250 conversations: `dev.jsonl` 125 (thresholds and prompt tuning), `test.jsonl` 125 (held out, reporting only) |
| Language | English |
| Created | 2026-10-12, for this project |
| Origin | **Synthetic.** Written by AI agents from scenario briefs; no real conversations or personal data |
| Labels | **AI-labelled, not human-reviewed, not clinically validated** |
| License | Same as this repository; contains no third-party data |

## How it was made

1. Five scenario families, 50 conversations each: `casual` (small talk,
   check-ins), `emotional` (stress, relationships, loneliness, grief),
   `practical` (problems, decisions, planning, summaries), `information`
   (factual and how-to questions, and near-misses), `hard` (very short or
   ambiguous messages, topic changes, long multi-turn conversations, prompt
   injection, sarcasm, and non-graphic safety cases with sad-but-not-safety
   contrasts). Each author agent also gave the labels it intended.
2. Two annotator agents labelled all 250 conversations blind (without the
   intended labels), using the question wording as the guide.
3. The gold label is the majority of the three votes (intended, annotator 1,
   annotator 2). `response_strategy` is `null` when the gold mode is `safety`:
   safety handling is not the decision model's job, so it is not scored.
4. Split: within each family, odd ids go to `dev`, even ids to `test`.

All three votes are kept in each row (`votes`) so labels can be audited.

## Agreement

| Question | Annotator 1 vs 2 (kappa) | Intended vs annotators (kappa) | No majority |
|---|---|---|---|
| conversation_mode | 0.977 | 0.875 / 0.889 | 0 |
| response_strategy | 0.957 | 0.861 / 0.856 | 0 |
| knowledge_retrieval_useful | 0.967 | 0.933 / 0.922 | 0 |

**Read these numbers with care.** The annotators are AI models given the
same guide, so their errors are correlated; high agreement shows the labels
are consistent with the written criteria, not that people would agree. The
strategy label in particular is a judgement call. A human review of at least
the `ambiguous` and `adversarial` rows is needed before results on this set
are treated as more than indicative.

## Label distribution (gold)

- conversation_mode: information 50, problem_solving 38, listening 38,
  casual 35, planning 31, exploration 28, reflection 15, safety 13, other 2
- response_strategy: provide_information 47, validate 39, plan 24,
  problem_solve 24, check_in 23, reflect 23, explore 23, summarize 17,
  clarify 17, not scored (safety) 13
- knowledge_retrieval_useful: false 191, true 59
- difficulty: clear 131, ambiguous 89, adversarial 30

## Row format

```json
{"id": "emotional-007", "family": "emotional",
 "conversation": [{"role": "user", "content": "..."}],
 "labels": {"conversation_mode": "listening", "response_strategy": "validate",
            "knowledge_retrieval_useful": false},
 "votes": {"intended": {...}, "annotator_1": {...}, "annotator_2": {...}},
 "difficulty": "clear", "tags": ["work_stress", "advice_refused"]}
```

## Intended use and limits

- For evaluating and choosing thresholds for the decision questions above.
  Do not tune prompts or thresholds on `test.jsonl`.
- Small: per-label results on a few examples (for example `safety`, 13 in
  total) are noisy.
- Synthetic text is cleaner and more explicit than real conversations.
- Safety rows exist only to check that the decision layer does not miss
  obvious cases; they are not a safety evaluation (spec Phase 5).
