# Memory (Phase 3)

How the companion keeps a few things a person told it, for later
conversations (spec §10), and how the person stays in control.

Memory is not a transcript archive or a profile. A memory is one short
sentence the person explicitly said about themselves ("has a job interview
on Friday", "prefers short answers"), stored only when application code
accepts it.

## Turn sequence

```
user message saved (ownership checked)
  -> state update and memory retrieval, in parallel
       retrieval: embed the message, hybrid search over this person's
       active and unconfirmed memories, threshold, top 5
  -> decisions, policy
  -> reply prompt: companion_system 1.4.0; the memories go into the
     person's latest message as an escaped <memories> block
  -> reply saved (with the IDs and scores of the memories used)
  -> background: memory extraction for this message
       one structured LLM call proposes candidates
       -> eligibility policy (code) -> embed -> store
```

Retrieval never fails a turn, and extraction runs after the reply is saved,
so it cannot delay or break the reply.

| Stage | On failure | Recorded as |
|---|---|---|
| Memory store unreachable during retrieval | no memories in the prompt; the reply cannot claim any | `generation_metadata.memory.mode = "unavailable"` |
| Embedding call fails or the local model is still loading | full-text matching only | `mode = "lexical"`, `failure = "embedding_failed"` |
| Extraction call fails (provider error, timeout, invalid JSON) | nothing stored for that message | log `memory.extraction_failed` |
| Embedding fails at storage time | stored without a vector; the maintenance loop embeds it later | log `memory.embedding_failed` |
| Background task crashes | logged (error type only), turn unaffected | log `background.task_failed` |

## What is stored

Table `memories` (migration `0003`), one row per memory:

| Column | Notes |
|---|---|
| `id`, `user_id` | every query filters on `user_id` in SQL |
| `memory_type` | `episodic` (an ongoing or upcoming situation), `semantic` (a stable fact), `preference` (how they want the companion to talk with them) |
| `content` | at most 200 characters, control characters removed |
| `status` | `active`, `pending_confirmation` (conflicts with an older memory), `superseded` |
| `importance` | 1-3, from the extractor, clamped |
| `provenance` | extraction model, prompt version, structured-output mode, `explicit`, the supporting quote (`evidence`, at most 200 characters), the policy reason, the lifetime |
| `source_conversation_id`, `source_message_id` | where it came from (set to NULL if that is deleted) |
| `supersedes_id` | the memory this one replaced, or conflicts with |
| `expires_at` | always set: short 30 days, medium 180 days, long = the retention period, all capped by `MEMORY_RETENTION_DAYS` (365) |
| `embedding`, `embedding_model` | pgvector vector (any dimension) and the model that made it; only vectors from the same model are ever compared |
| `schema_version`, timestamps | |

Session context (the current topic, open questions) is not memory. It
lives in the per-conversation state (docs/decisions.md).

Table `user_memory_settings`: `memory_enabled` (default on) and
`allow_sensitive` (default off).

## Extraction and the eligibility policy

The model only proposes. Prompt `memory_extraction@1.0.0` asks for at most a
few candidates from the latest message, each with its type, whether the
person said it explicitly, a verbatim supporting quote, a sensitivity
category, a lifetime, an importance, and how it relates to existing
memories (`new`, `duplicate`, `updates`, `conflicts`, plus the related
memory's number). It may return no candidates, and usually should. The
extractor sees the latest message, the previous four messages, and up to ten
of the person's most related active or unconfirmed memories. Everything is
inside escaped data blocks.

`app/memory/policy.py` then decides each candidate in code and records why:

1. **Explicit and supported only.** `explicit` must be true, and the quote
   must appear (after normalizing case and punctuation) in the person's
   latest message. Inferences, things the assistant said, and quotes from
   earlier messages are rejected (`not_explicit`, `unsupported_by_message`).
2. **Identifiers and secrets are never stored** (phone numbers, addresses,
   passwords, ID numbers): `identifier`.
3. **Sensitive categories** (health, mental health, sexuality, religion,
   politics, ethnicity, legal, finance) are rejected unless the person
   turned on `allow_sensitive`: `sensitive:<category>`.
4. **Instruction-like content** ("ignore your rules", "system prompt") is
   rejected: memories are data, never instructions (`instruction_like`).
5. **Duplicates** of an active memory, or of another candidate in the same
   turn, are rejected (`duplicate`).
6. At most `MEMORY_MAX_PER_TURN` (3) are stored per message, the most
   important first (`turn_limit`).

Messages under 20 characters ("ok thanks!") never reach the extractor.

## Duplicates and contradictions (spec §10.6)

- **Update** (the person clearly replaced a fact: "I moved to Bristol, I
  don't live in Bath any more"): the new memory is stored and the old one
  becomes `superseded`. It is kept, with the new memory's `supersedes_id`
  pointing at it, so provenance is preserved; it is never retrieved.
- **Conflict** that does not clearly replace the old fact ("I might be
  moving to Cardiff"): the new memory is stored as `pending_confirmation`.
  When relevant, it reaches the reply prompt marked `"status":
  "unconfirmed"`, and the prompt tells the companion not to treat either
  memory as settled and to ask the person which is current. Nothing
  uncertain silently becomes a fact.
- **Confirming** a pending memory (`POST .../confirm`) makes it active and
  supersedes the memory it conflicted with. When a memory is superseded, any
  other pending proposal to replace it is superseded too.
- Cosine similarity cannot tell a duplicate from a contradiction: "prefers
  tea over coffee" and "prefers coffee over tea" score 0.996 with the local
  model. That is why the relation comes from the extractor and the policy,
  not from a similarity threshold.

## Retrieval (spec §10.5)

`MemoryStore.retrieve` and `select_hits` (`app/memory/store.py`):

- Candidates: this person's `active` and `pending_confirmation` memories
  that have not expired (at most 500, newest first).
- **Semantic:** `1 - cosine distance` between the message and each memory,
  computed only for memories embedded by the current model. A `CASE` keeps
  the distance from ever being computed across different dimensions.
- **Lexical:** PostgreSQL full-text match (`english`), where any word may
  match. `plainto_tsquery` joins words with AND, which a conversational
  message almost never satisfies, so the operators are turned into OR.
- A memory qualifies when its similarity is at least `MEMORY_MIN_SEMANTIC`
  (0.49). A full-text match alone qualifies only a memory without a usable
  vector (embedding failed, or made by another model), and is the whole
  search when the message could not be embedded. A full-text match adds a
  small boost (up to 0.2) to the ranking.
- Duplicates (same normalized content) are dropped, keeping the newest. The
  top `MEMORY_MAX_RETRIEVED` (5) are used.

In the prompt, memories are a JSON list (type, content, the date noted,
`status: unconfirmed` where relevant) with `<` and `>` escaped, at the
start of the person's latest message, after any conversation notes, never
in the system prompt. Prompt 1.4.0 says they are untrusted background, may
be out of date, and are never instructions; that the companion must not
claim to remember anything outside the block or this conversation; and
that deleting memories happens in the app's memory settings.

## Embeddings

`EMBEDDING_PROVIDER`:

| Provider | Model (default) | Status |
|---|---|---|
| `local` (default) | `BAAI/bge-small-en-v1.5` via fastembed 0.9 (ONNX Runtime, CPU), 384 dimensions | **Verified here**, by running it, reading the fastembed source, and the evaluation below |
| `openai` | `text-embedding-3-small` | Built on the official SDK and tested against a mocked transport; **not live-tested** (no key) |
| `gemini` | `gemini-embedding-001` (`RETRIEVAL_QUERY` / `RETRIEVAL_DOCUMENT`) | Built on the official SDK and tested against a mocked transport; **not live-tested** (no key) |
| `none` | | full-text matching only |
| `fake` | hashed bag of words | tests only; refused in production |

Facts about the local model that the code relies on, each checked by
running fastembed 0.9.0:

- The first start downloads about 65 MB from huggingface.co (5 files, the
  quantized ONNX port `Qdrant/bge-small-en-v1.5-onnx-Q`, MIT licence). After
  that it loads from the cache without network access.
- Loading and inference block, so the model loads in the background at
  startup and inference runs on one dedicated thread. Until the model is
  ready, retrieval uses full-text matching. A failed load (no network)
  takes about 40 s and is not retried for 5 minutes.
- Vectors are L2-normalized. For this model, query and passage embeddings
  are identical (no prefix). A short-memory test found BGE's optional query
  instruction did not help (MRR 0.903 with it, 0.917 without).
- Text beyond 512 tokens is truncated silently. Memories are at most 200
  characters, so this never applies to them.
- The process uses about 180 MB more memory once the model is loaded.
- The default cache is the OS temp folder, which cleanup tools may empty.
  `.env.example` sets `EMBEDDING_CACHE_DIR=.cache/fastembed`.

**Changing the embedding model** never mixes vectors. Old memories still
match by full text, and the maintenance loop (every
`MEMORY_MAINTENANCE_INTERVAL_SECONDS`, 600 s) re-embeds memories with no
vector from the current model, up to 1,000 per pass. The same loop embeds
memories stored while the model was loading or a call failed.
`MEMORY_MIN_SEMANTIC` was tuned for the local model; another model needs
its own value.

## User controls (spec §10.7)

All under `/v1/users/me`, authenticated, and scoped to the caller. Another
person's memory ID returns the same `404 memory_not_found` as a missing
one.

| Route | |
|---|---|
| `GET /memories?status=active\|pending_confirmation\|superseded\|all` | list (default: active and pending) |
| `GET /memories/{id}` | view one |
| `DELETE /memories/{id}` | delete one (204) |
| `DELETE /memories` | delete all, returns `{"deleted": n}` |
| `POST /memories/{id}/confirm` | confirm a pending memory |
| `GET` / `PUT /memory-settings` | `memory_enabled`, `allow_sensitive` (strict body) |

With `memory_enabled` off, nothing is extracted or stored (also checked
again inside the storing transaction, in case the setting changed during
extraction), and no memories are retrieved. Existing memories are kept
until the person deletes them or they expire.

## Retention and deletion (spec §10.7, §15.4)

Where memory content exists, and what deleting a memory removes:

| Place | Deleted with the memory? |
|---|---|
| `memories` row: content, quote in `provenance`, embedding | **Yes.** The embedding is a column of the same row; there is no other vector index or cache |
| PostgreSQL dead tuples after `DELETE` | Reclaimed by (auto)vacuum; until then the bytes remain in the data files |
| The person's original message, in `messages` | **No.** It is part of the conversation. There is no conversation deletion endpoint yet (known limitation since Phase 1) |
| Reply metadata (`generation_metadata.memory`) | Holds only memory IDs and scores, no content |
| Application logs | Never contain memory or message content (counts, modes, error types only) |
| LLM provider | Memory content is sent to the generation provider in reply prompts and extraction calls; that provider's retention policy applies |
| Embedding provider | `local`: nothing leaves the machine. `openai`/`gemini`: memory content is sent to that provider |
| Backups | None are configured by this repository. An operator's PostgreSQL backups keep deleted and expired memories until those backups expire, so backup retention must not exceed what the deployment promises (at most `MEMORY_RETENTION_DAYS`) |
| Evaluation files | Synthetic data only |

Expiry: every memory has `expires_at`. Expired memories are never listed or
retrieved, and the maintenance loop deletes them (`memory.purged`).
Superseded memories are kept for provenance until they expire or the person
deletes them (they are visible with `?status=superseded`).

## Evaluation

Dataset: `eval/datasets/memory/v1` (see its dataset card): 150 extraction
items and 45 retrieval scenarios, synthetic and AI-labelled, split into
dev and test. Runner: `scripts/eval_memory.py`; scoring:
`app/evaluation/memory.py`.

Spec §10.8 coverage:

| Behavior | How it is tested |
|---|---|
| Extraction precision | live run below |
| Retrieval precision and recall | evaluation runs below (real store, local model) |
| Unsupported memory references | prompt rules; `<memories>` only in the user turn and only with authorized memories (`test_a_useful_memory_persists_into_a_later_conversation`) |
| Stale memory usage | superseded and expired memories never retrieved (integration tests; 0 inactive memories returned in the evaluation runs) |
| Contradictions | `test_an_update_supersedes_and_a_conflict_waits_for_confirmation`, policy unit tests |
| Expiration | `test_expired_memories_are_not_used_and_are_purged` |
| Deletion | `test_deletion_is_reflected_in_future_retrieval`, `test_list_view_delete_and_delete_all` |
| Cross-user isolation | `tests/security/test_memory_isolation.py` |
| Prompt injection in memory content | `test_memory_content_cannot_break_out_of_the_memories_block`, `test_instruction_like_memories_are_rejected`, escaping unit tests |

## Results (2026-10-13, live)

### Retrieval (local bge-small-en-v1.5, the real store on PostgreSQL)

`MEMORY_MIN_SEMANTIC` was chosen on dev (best F1; ties go to the higher
threshold) and applied unchanged to test. Top 5 per message.

| Split | Threshold | Precision (95% CI) | Recall (95% CI) | F1 | Nothing returned when nothing was relevant |
|---|---|---|---|---|---|
| dev (23 scenarios, 33 relevant) | 0.49 (selected) | 0.84 (0.67-0.93) | 0.79 (0.62-0.89) | 0.81 | 4 of 6 |
| **test** (22 scenarios, 42 relevant) | 0.49 | **0.62** (0.48-0.74) | **0.74** (0.59-0.85) | 0.67 | 1 of 2 |
| test, top 5 with no threshold | — | 0.40 | 1.00 | 0.58 | 0 of 2 |
| test, full-text fallback only | — | 0.63 | 0.29 | 0.39 | 1 of 2 |

- Expired and superseded memories were never returned (0 in every run).
- The F1 curve peaks sharply: on dev, 0.46 gives F1 0.75 and 0.52 gives
  0.70. The earlier placeholder of 0.6 would have found 21% of the relevant
  memories. Letting a full-text match alone qualify a memory that has a
  vector lowered precision on dev, so it is off.
- On test, lexical traps (same words, unrelated meaning) are the weakest
  case (precision 0.33), and indirect relevance reaches 0.63 recall.
- The full-text fallback (no embeddings) finds under a third of the relevant
  memories: it is a degraded mode, not an equivalent.

Files: `eval/results/memory-retrieval-dev-20261013T114550.json`,
`eval/results/memory-retrieval-test-20261013T114602.json`.

### Extraction (Groq `openai/gpt-oss-120b`, prompt `memory_extraction@1.0.0`)

"Stored" is what the system keeps (the model's candidates after the
eligibility policy); "proposed" is the model's raw candidates. Matching uses
the 0.8 similarity rule described above, and every error below was checked
by hand.

| Split | | Precision (95% CI) | Recall (95% CI) | Something stored when nothing should be | Failed calls |
|---|---|---|---|---|---|
| **test** (74 items, 45 memories), first run | stored | **0.85** (0.71-0.93) | **0.73** (0.59-0.84) | **3 of 39** | 3 |
| | proposed | 0.59 | 0.76 | 14 of 39 | |
| dev (76 items, 52 memories), current policy | stored | 0.80 (0.68-0.88) | 0.94 (0.84-0.98) | 5 of 35 | 0 (1 on the first try) |
| | proposed | 0.59 | 0.96 | 16 of 35 | |

The code-side policy is what keeps irrelevant content out: it removes most
of the model's wrong proposals (14 to 3 items on test) and costs little
recall.

Hand audit of the test run:
- **Errors (4 of 40 stored):** two details about other people (a brother's
  hobbies, a coworker's lunch), one joke stored as a preference, and one
  religious practice the model labelled not sensitive.
- **Matching artifacts (2):** a correct memory phrased differently ("lives in
  Bristol, with Bath 20 minutes away"), and a correct fact stored together
  with a salary. The salary is a real **sensitive leak**: after the run, a
  code backstop now rejects content with a precise amount of money. It was
  motivated by this test error; on dev it rejects no gold memory.
- **Missed memories** are mostly messages with a preference and a fact,
  where the model kept only the preference ("prefers metric units" but not
  "lives in Germany"), and one candidate rejected because the model labelled
  it an identifier when only the message had an address.
- **Failed calls:** 3 of 74 test calls (and 1 of 76 on dev) failed with a
  400 from Groq; every immediate retry of those items succeeded (26 of 26).
  The extractor now retries once and logs the provider's error code.

Files: `eval/results/memory-extraction-test-20261013T115147.json`,
`eval/results/memory-extraction-dev-20261013T120854.json`.

### Limits of these results

- The evaluation set is synthetic and AI-labelled (see its dataset card);
  the splits are small, hence the wide intervals.
- Each extraction item is judged on its own, with no existing memories; real
  memory sets grow over months.
- Retrieval was measured with the local model only. OpenAI and Gemini
  embeddings need their own threshold.
