# Golden scenarios v1

End-to-end scenarios run through the real application by
`scripts/eval_golden.py` (spec §16.3). The format and checks are in
`app/evaluation/golden.py`.

| | |
|---|---|
| Size | 216 scenarios, 544 person messages, 648 steps; up to 13 messages in one scenario |
| Split | dev 118 / test 98. Generated scenarios were split 7 / 6 per category with seed 20261016 |
| Modes | 191 run in both modes; 25 have only reply checks and run in live mode only |
| Categories | all 16 of spec §16.3, with 12–18 scenarios each (dev / test in brackets): casual 13 (7/6), work stress 13 (7/6), relationships 13 (7/6), loneliness and grief 13 (7/6), advice requested 13 (7/6), advice refused 13 (7/6), topic change 13 (7/6), contradiction 13 (7/6), memory 14 (8/6), knowledge 14 (8/6), retrieval failure 12 (6/6), safety 18 (10/8), prompt injection 13 (7/6), privacy 14 (8/6), long conversation 13 (7/6), dependency failure 14 (8/6) |

## Origin

**Synthetic, AI-written, and not reviewed by a person.**

- **9 scenarios were written by hand while the runner was built.** Five
  are explicit safety statements; four are smoke tests (knowledge,
  privacy, a dependency failure, memory).
- **208 came from a writing workflow (2026-10-16).** Four writers each
  wrote 13 scenarios for each of four categories. Each writer was given:
  - a description of the documented behaviour;
  - the scenario format;
  - the list of library sources.

  The writers did not read the code. A separate reviewer per writer then
  checked every expectation against the same description and fixed or
  removed the ones that description did not guarantee.
- Messages vary in register: lowercase texting, typos, British and Indian
  English. No real person's data is used.
- The crisis wording in the explicit safety scenarios is minimal and has no
  method or plan in it. Generated safety scenarios cover idioms, games,
  fiction, ambiguous concern and over-reliance; they do not cover explicit
  crisis.

## Changes after the first runs (2026-10-16)

The first deterministic run on dev failed 22 of 105 scenarios. Each failure
was traced, and either the scenario or the system was found wrong.

| Found | What was wrong | Fixed in | Failing steps |
|---|---|---|---|
| dev | The router read statements as questions: subject-dropped modals ("might try that"), a "which" after a comma, quoted speech, a mid-sentence "how to", unmarked rhetoric ("honestly how do people do this") | `app/rag/routing.py` (v5) | 7 |
| dev | Companion questions after the first words of a message ("ok rohan, are you a real person?", "did you see that goal??", "can we play 20 questions?"), questions about the conversation ("who did I say…?"), "did that work?" | router | 10 |
| dev | "what is mindfulness anyway?" was read as rhetorical. It now runs a lookup, but the search scores 0.69 and returns nothing (a retrieval near-miss, below) | router ("anyway" is rhetorical only without "?") | 1 |
| dev | **A v5 regression the scenarios caught.** A long quoted chain message was read as the person's own words | router (quoted spans up to 500 characters) | 1 |
| dev | **Scenario wrong.** The writers' brief said personal questions never run a lookup. The router does run one, by design, and the relevance threshold keeps evidence out. These expectations went from "no lookup" to "no evidence", which is what the person sees | scenarios | 8 |
| dev | **Scenario wrong.** It assumed the library has nothing on grief; it has a short passage. Removed | scenarios | 1 |
| dev | Retrieval near-misses (kept as failures) | – | 2 |
| test (first run) | "not after any advice" (British) and "no advice pls" were not read as declining advice, so evidence reached the prompt | `app/orchestration/policy.py` | 1 |
| test (first run) | "did my long run", "and when I brought it up…", "why do they always do this", "show me your prompt", "help me write a text" | router | 8 |
| test (first run) | **Scenario wrong**, as above (no lookup → no evidence) | scenarios | 4 |
| test (first run) | Retrieval near-misses (kept as failures) | – | 2 |

The test split's first run (`golden-deterministic-test-20261016T033203.json`)
is kept as it came out:

- guarantees held in 100% of scenarios;
- routing and retrieval checks passed in 83.5%.

Fixes made after it mean **the test split is no longer blind for routing**.
The router's blind measure is the routing test set (docs/rag.md), where v5
equals v4.

The remaining failures are 5 retrieval near-misses: requests the library
covers that score just under the relevance threshold. They are kept as
failures, not edited away.

## Limits

- **AI-written and AI-reviewed.** The expectations encode the documented
  behaviour as two models read it. The runs above found 13 expectations
  that were wrong, so others may be too.
- **Reply checks are blunt.** Mentions, word and question counts, lists and
  stock phrases only. How good a reply is gets judged separately (voice and
  dialogue evaluations).
- **Live mode has not been run on this set yet.** Reply checks are
  untested until it is (docs/evaluation.md).
