# Memory evaluation set v1

Labelled data for evaluating Phase 3 memory (spec §10.8): what should be
stored from a message (extraction), and which stored memories help reply to
a later message (retrieval).

| | |
|---|---|
| Size | Extraction: 150 messages (`extraction_dev.jsonl` 76, `extraction_test.jsonl` 74). Retrieval: 45 scenarios (`retrieval_dev.jsonl` 23, `retrieval_test.jsonl` 22) |
| Language | English (mostly British) |
| Created | 2026-10-13, for this project |
| Origin | **Synthetic.** Written by AI agents from a written labelling guide; no real conversations or personal data. Phone numbers use the UK drama range (07700 900xxx) |
| Labels | **AI-labelled, not human-reviewed** |
| License | Same as this repository; contains no third-party data |

## How it was made

1. **Guide.** One memory policy, the same for authors and annotators:
   store only what the person explicitly says about themselves that will
   help in future conversations (episodic, semantic, preference). Do not
   store small talk, questions, momentary moods, inferences, opinions about
   the world, details only about other people, or instructions to the AI.
   Sensitive categories are labelled and not stored by default. Identifiers
   and secrets are never stored. This matches `app/memory/policy.py` and
   prompt `memory_extraction@1.0.0` (the set was written before 1.1.0).
2. **Extraction items.** Two author agents wrote 75 items each. One wrote
   mostly positive and mixed messages (`ext-pos-*`). The other wrote mostly
   negative and hard ones (`ext-neg-*`): small talk, questions, momentary
   emotions, inference traps, other people, sensitive disclosures,
   identifiers, prompt injection, sarcasm, hypotheticals, corrections, and
   borderline positives. Each item has 0-3 earlier messages, and each author
   gave the memories it intended.
3. **Retrieval scenarios.** One author agent wrote 45 scenarios. Each has
   5-10 stored memories (some expired or superseded, which never count as
   relevant) and a new message. The scenarios include direct references,
   indirect relevance, preferences that apply to a kind of request, lexical
   traps, paraphrases with no shared words, superseded facts, expired
   events, and 8 scenarios with nothing relevant.
4. **Blind labels.** Two annotator agents labelled every item and scenario
   without seeing the intended labels.
5. **Gold = majority of three** (intended, annotator 1, annotator 2).
   - Extraction: memories from the three sources are grouped when their
     bge-small-en-v1.5 cosine similarity is at least 0.8. A group proposed by
     at least two sources is gold, and every phrasing in it is kept (`also`).
     Memories proposed by one source only are listed in `minority`; a
     prediction matching them counts as neither right nor wrong.
     Sensitive categories with at least two votes are kept.
   - Retrieval: an active memory is relevant when at least two of the three
     voted for it.
6. **Normalization.** The scenario authors wrote memories as "User has ...".
   The extractor stores "has ..." (prompt: one third-person sentence
   without the name), so "User " was removed and "User's" became "Their".
   The authored text is kept in `original_content`. This was done after the
   first dev run and before any test run.
7. **Split.** Odd ids go to `dev` (thresholds, prompt and policy work), even
   ids to `test` (reporting only).

All votes are kept in each row (`votes`).

## Agreement

| | Annotator 1 vs 2 | Intended vs annotators |
|---|---|---|
| Extraction: anything to store? (kappa, 150 items) | 1.00 | 1.00 / 1.00 |
| Extraction: memory-level F1 (matched at 0.8) | 0.98 | |
| Retrieval: memory relevant? (kappa, 218 active memories) | 0.96 | 0.97 / 0.95 |

The majority changed the intended memories on 5 extraction items (on four,
two intended memories were merged into the one both annotators wrote; on
one, the annotators' phrasing replaced the intended one) and the intended
relevant set on 2 scenarios. There were 7 minority-only memories, almost all
granularity choices (for example, one annotator added a second memory for a
detail the other two left inside the first).

**Read these numbers with care.** The annotators are AI models given the
same guide, so their errors are correlated. High agreement shows the labels
follow the written guide, not that people would agree. Human review is
needed before the results are treated as more than indicative.

## Contents

- Extraction gold memories: 97 (semantic 44, episodic 33, preference 20);
  items with something to store: 76 of 150. Dev: 52 memories in 41 of 76
  items; test: 45 memories in 35 of 74.
- Sensitive categories present (gold): health 13, identifier 10, finance 4,
  legal 4, mental health 3, politics 3, religion 2, sexuality 2, ethnicity 1.
- Retrieval: 218 active, 48 expired and 12 superseded memories; 75
  relevant (dev 33, test 42); scenarios with nothing relevant: dev 6,
  test 2.

## Row formats

Extraction:

```json
{"id": "ext-pos-006", "family": "positive",
 "context": [{"role": "assistant", "content": "..."}],
 "message": "I've been vegan for about eight years, so please ...",
 "expected": [{"content": "has been vegan for about eight years",
               "memory_type": "semantic", "lifetime": "long",
               "also": ["Has been vegan for about eight years"]}],
 "minority": [], "sensitive_categories": [], "tags": ["positive", "multi-fact"],
 "votes": {"intended": ["..."], "annotator_1": ["..."], "annotator_2": ["..."]}}
```

Retrieval:

```json
{"id": "ret-001",
 "memories": [{"key": "m1", "content": "has a job interview ... this Friday.",
               "original_content": "User has a job interview ... this Friday.",
               "memory_type": "episodic", "days_ago": 6, "status": "active"}],
 "message": "Well, the interview's done ...", "relevant": ["m1", "m3"],
 "tags": ["direct_reference"], "votes": {"intended": ["m1", "m3"], "...": []}}
```

## Intended use and limits

- For evaluating memory extraction (with `scripts/eval_memory.py
  extraction`) and retrieval (`scripts/eval_memory.py retrieval`). Tune on
  `dev` only.
- Small: per-tag results on a few items are noisy. Confidence intervals
  are reported.
- Synthetic messages are cleaner and more explicit than real ones, and each
  is judged on its own; real memory sets grow over months.
- Matching a predicted memory to gold by embedding similarity is an
  approximation (see `app/evaluation/memory.py`). Per-item outputs are
  kept so that errors can be audited by hand.
