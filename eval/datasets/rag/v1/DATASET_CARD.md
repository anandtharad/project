# Knowledge retrieval evaluation set v1

Data for evaluating Phase 4 knowledge retrieval (`app/rag/`, docs/rag.md):

- **Routing:** should a message trigger a lookup?
- **Retrieval:** does a lookup find the right sources, and nothing when the
  library has no answer?
- **Generation:** are replies grounded, and are citations valid?

Everything here is **synthetic and AI-labelled, not human-reviewed**. It
contains no real user messages.

## Routing: `routing.jsonl`

One message per line: `id`, `text`, `retrieve` (should a lookup run),
`kind` (casual, reflection, factual, evidence, source_followup,
current_info, off_topic, or unlabelled), `source` and `split`.

| Source | Split | Items | How it was made |
|---|---|---|---|
| `authored-1` | dev | 55 (25 retrieve) | Written by the developer's AI assistant while designing the router; `kind` not labelled |
| `authored-2` | dev | 35 (15 retrieve) | Written at the same time, held back, then used to check and fix the router |
| `blind-1` | dev | 90 (40 retrieve) | Written by a separate AI agent from a written label definition, without seeing the router. First used as a blind test; the router was then fixed on it, so it is now development data |
| `blind-2` | **test** | 100 (45 retrieve) | Written the same way by another agent, which saw neither the router nor the other sets. Only results and ids are looked at, never the texts, so it stays a valid test |

**Label definition** (given to the blind authors).

- **True:** an explicit request for general information, an explanation,
  techniques or tips, facts, evidence or sources on a library topic.
  Questions about where the companion's last claim came from also count.
- **False:** casual chat, venting, describing feelings or events,
  rhetorical or emotional questions, questions about the companion or about
  specific people, off-topic requests, and "no advice" messages.

The authors were told not to write about self-harm, suicide, abuse or
crisis, which are covered by the safety set.

**What `retrieve=true` means for an off-topic question.** An off-topic
request ("how do I reverse a string in python") is labelled false: the
library cannot answer it. The router runs a lookup for it anyway, and the
relevance threshold is expected to find nothing. So the routing evaluation
reports both:

- lookups run on negatives;
- evidence actually given for negatives (`--with-retrieval`), which is what
  matters for replies.

## Retrieval: `retrieval.jsonl`

One query per line: `id`, `query`, `type` (answerable, unanswerable,
off_topic), `relevant_sources` (source keys), `expected_facts` (short
phrases copied verbatim from an answering passage), `split` and `origin`.

**How it was made** (workflow on 2026-10-15). The writers saw only the
passages, never the retrieval code.

1. Four writer agents, each given the passages of one group of sources,
   wrote answerable queries in a casual chat style, paraphrased rather than
   copied.
2. A verifier per group read the whole corpus. It corrected
   `relevant_sources` to every source that substantially answers the query,
   fixed expected facts so they appear verbatim, and rejected ambiguous or
   unanswered queries.
3. A fifth agent read the whole corpus and wrote unanswerable queries
   (on-topic, but checked not to be answered) and off-topic queries.
4. Code then:
   - checked every expected fact against the listed sources' passages,
     dropping facts that do not appear and items left with none;
   - removed duplicate queries;
   - assigned a deterministic split stratified by type (alternating after
     sorting by a hash of the query).

Counts are in the result files. Relevance is labelled per source, not per
chunk, so the labels stay valid when chunking settings change. The fact
check supplies the chunk-level signal.

## Use

- Choose thresholds and chunking on `dev`.
- Report `test` once, with the dev choices unchanged.

The corpus passages behind the labels are public-domain U.S. government text
(see `data/knowledge/manifest.toml`).
