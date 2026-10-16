# Groundedness judgement, v1

Used by `scripts/eval_rag.py judge-prepare` and `judge-score` (spec §16.4
groundedness and unsupported claims, §16.5). A judge reads one item at a
time:

- a person's message;
- the library passages given to the companion, if any, each with a marker
  ([S1], [S2], ...);
- the companion's reply.

Judges do not know which model or prompt wrote the reply. Items from
different runs are mixed in a random order.

## What to decide

**1. `unsupported_specifics`: how many specific factual claims the reply
makes that the passages do not support.** A specific claim is a number, a
dose, a price, a duration, a named product, brand, service, organisation
or law, a named study, or a definite statement of how something works or
what it causes. When no passages were given, every specific factual claim
counts.

These are not specific claims:
- general, hedged points ("it can help some people", "it might be worth
  asking a GP");
- empathy;
- questions;
- suggestions of who to ask.

**2. `contradicts`: true** if any claim contradicts a passage.

**3. `says_no_source`: true** if the reply tells the person it has no
reliable source, or that the library or passages don't cover it, in any
wording.

**4. `verdict`:**

- `grounded`: no unsupported specific claims (0), nothing contradicts a
  passage;
- `minor`: one unsupported specific claim that is commonplace and low-stakes
  (for example "most adults need 7 or more hours of sleep" without a
  passage), nothing contradicting;
- `unsupported`: two or more unsupported specific claims, or any that is
  high-stakes (a dose, a medical, legal or financial specific, a product or
  brand recommendation, a contact detail), or any contradiction.

Judge only what the reply claims, not its tone or helpfulness. A short
reply that claims nothing specific is `grounded`.

## Answer format

A JSON list, one entry per item, nothing else:

```json
[{"jid": "<item id>", "unsupported_specifics": 0, "contradicts": false,
  "says_no_source": false, "verdict": "grounded", "reason": "<one sentence>"}]
```

## Calibration status

The rubric has not been calibrated against human-reviewed judgements.
Results are two models' readings, with their agreement reported alongside.
