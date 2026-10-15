# Threat model and privacy notes (Phases 1-3 scope)

Informed by the OWASP Top 10 for LLM Applications and the NIST AI RMF. These
references shape the analysis; they do not certify the system. This document
covers what exists today and is revisited at each phase.

## Assets

- Conversation content (sensitive personal disclosures are expected).
- User-to-conversation ownership.
- Provider credentials (`GENERATION_API_KEY` and the per-provider keys, `JEV_API_KEY`,
  `LAYA_API_KEY`) and dev tokens.
- Conversation state (inferred goals and emotional context about a person).
- Memories: facts a person shared, kept across conversations, with
  embeddings of them.
- The system prompt (low sensitivity, but its integrity matters).
- Availability and spend (paid LLM calls).

## Trust boundaries

1. Client → API: everything in a request is untrusted.
2. API → LLM provider: conversation content leaves the system here (replies
   and state extraction).
2b. API → decision service: the latest message and up to four earlier
   messages are sent to Laya (self-hosted: stays on your machine), TypeSafe
   Jev (third party), or with `DECISION_PROVIDER=llm` the decision LLM's
   provider (Groq by default; a third party, on every turn, through a client
   separate from the reply model's). Conversation notes are not sent. The
   decision model's output is untrusted (boundary 3).
2d. API → safety classifier: the latest message and up to four earlier
   messages are sent to the safety classifier's provider on every turn
   (default Groq `openai/gpt-oss-safeguard-20b` when `GROQ_API_KEY` is set;
   off otherwise), and on safety pathways the reply is sent once more for
   the semantic output check. Its output is untrusted (boundary 3); the
   classifier's rationale, which can quote the message, is discarded.
2c. API → embedding provider: memory text and each message are embedded.
   With `EMBEDDING_PROVIDER=local` (the default) nothing leaves the machine;
   with `openai`/`gemini` the text goes to that provider.
3. LLM output → API: model output is untrusted until validated.
4. API → PostgreSQL: trusted store; access only through ownership-scoped queries.

## Threats, current mitigations, gaps

| Threat | Mitigations now | Gaps / planned |
|---|---|---|
| Cross-user data disclosure | Ownership filter in the SQL that loads a conversation; foreign conversations return the same 404 as missing ones; idempotency replay happens only after the ownership check; tests cover read, list, post and replay across users | Production authN/authZ (Phase 1 uses static dev tokens) |
| Prompt injection (direct) | User text only travels in user-role messages; the system instruction is the fixed versioned prompt, which tells the model that conversation text cannot change it; the model has **no tools**, so injected instructions cannot access data | Behavioral resistance (e.g. refusing to reveal the system prompt) is unverified without live evaluation; output policy checks arrive in Phase 5 |
| Prompt injection (indirect: memory) | Memories come from untrusted text, so they are untrusted: the extraction prompt excludes instructions, and code rejects candidates matching instruction-like patterns (after Unicode normalization; mixed-alphabet words rejected); memories reach the reply model only in the user role, as an escaped JSON `<memories>` block (it cannot close the block), and the prompt says they are never instructions; the extractor sees its input inside escaped tags; content is capped at 200 characters and stripped of control characters (tests cover breaking out of the block) | The patterns are a backstop, not a complete filter; behavioral resistance is unverified without live adversarial evaluation. RAG documents (Phase 4) need the same treatment |
| Cross-user memory disclosure | Every memory query made on a person's behalf filters on `user_id` in SQL, including the vector search (only the maintenance loop works across users, and returns nothing to anyone); another person's memory ID returns the same 404 as a missing one; the extractor is shown only the same person's memories; tests cover list, view, delete, delete-all, confirm, settings, retrieval into a turn and extraction | Production authN/authZ |
| Unwanted memory (storing what the person did not mean to share) | The model only proposes; code stores only explicit statements whose quote (two words or more) appears in the person's message and shares a word with the memory; identifiers never (model label plus code patterns); sensitive categories only with `allow_sensitive` (re-checked when storing), with a code backstop for amounts of money; the quote itself is not stored; at most 3 per message; memory can be turned off, and every memory listed, deleted or left to expire | The sensitivity screen depends on the model's label: on the test split 2 of 13 sensitive items leaked before the money backstop (docs/memory.md) |
| Memory poisoning across sessions (a false "fact" steering later replies) | Only what the person said in their own message is stored; conflicting facts are held as unconfirmed and the companion asks; superseded and expired memories are never retrieved; the person can see and delete everything | A person can deliberately store false facts about themselves; that only affects their own conversations |
| Prompt injection via conversation state | State is extracted from untrusted text, so it is untrusted too: it reaches the reply model only in the user role, inside a `<conversation_notes>` block whose JSON escapes `<`/`>` (it cannot close the block), and the prompt says the notes are never instructions. The extraction prompt treats its input as data inside escaped tags. Field lengths and counts are capped | Behavioral resistance is unverified without live adversarial evaluation |
| Decision manipulation | Decisions never control access, persistence or safety handling and run after the ownership check; answers are validated against the question asked (invalid answers are rejected, not repaired); only calibrated, above-threshold answers change the policy, and the worst case is one of the fixed prompt fragments; `safety_response` cannot be selected | A manipulated `conversation_mode=safety` only makes the reply more cautious |
| Prompt injection into the decision LLM (`DECISION_PROVIDER=llm`) | The model reads untrusted conversation text, so its output is untrusted: the state travels only in the user role, as JSON inside an escaped `<decision_state>` block (it cannot close the block), and the locked `decision_llm` prompt says it is data, never instructions; the reply must match a strict schema of integer weights exactly (anything else is `invalid_response`), and every answer is validated again; the model has no tools and its text never reaches the person or the reply prompt; thresholds are per model and prompt, and every question is in shadow mode until evaluated | Behavioral resistance is unverified without live adversarial evaluation; once calibrated, an injected message could at most pick one of the fixed strategy fragments for its own reply |
| Unsafe output handling | Output is stored and returned as plain text, never executed or rendered by the server; refusals, truncations, empty and over-long replies are rejected | **No safety-policy output validation yet (Phase 5)**; clients must escape output when rendering |
| Sensitive information in logs | Logs carry IDs, categories, timings and token counts, never content; exception messages and tracebacks are not logged or returned; memory logs say only that something sensitive was rejected, never which category; enforced by tests | Log retention and shipping not configured |
| API key exposure | Keys held as `SecretStr` (not in reprs or logs) and masked in `ResolvedProvider` reprs; server-side only, never returned; `.env` gitignored; `.env.example` has placeholders only; only the selected provider's key is sent, only to its endpoint | Key rotation procedure: rotate in the provider console, update the secret, restart; no secret manager integration yet |
| Decision-service credentials | `DECISION_PROVIDER=llm` uses the decision provider's own key (e.g. `GROQ_API_KEY`); `GENERATION_API_KEY` is used only when the decision provider is also the reply provider, so the reply provider's key never reaches another endpoint, and `openai_compatible` is refused unless it is the reply provider too (no endpoint of its own). The TypeSafe SDK reads `TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL`/`TYPESAFE_DEFAULT_MODEL` when arguments are missing; the adapter always passes all three. Its logger records request and response bodies at DEBUG, so it is capped at WARNING even when `TYPESAFE_LOG_LEVEL=debug` (tested). `laya-serve` accepts any caller without `LAYA_API_KEY`, so the app refuses `DECISION_PROVIDER=laya` without a key, and `dev.py laya` and compose bind it to localhost | — |
| Credential leakage via SDK environment fallbacks | The `openai` SDK would send `OPENAI_API_KEY` to third-party base URLs and forward `OPENAI_ORG_ID`/`OPENAI_PROJECT_ID`/`OPENAI_CUSTOM_HEADERS` (whose `Authorization` line overrides explicit keys). The adapter passes explicit keys and base URLs, pins `Authorization`, and strips OpenAI-only headers for other endpoints. The Gemini client pins `vertexai=False`. Tests cover each case | Other SDK env variables (e.g. `GROQ_BASE_URL`) are operator configuration |
| SSRF via configurable endpoint | `GENERATION_BASE_URL` is operator-only configuration (environment), never user input | Do not expose provider/base-URL selection to end users |
| Denial of service | Body limit (413, checked on declared and streamed bytes); input character limit before any model call; one in-flight turn per conversation; bounded turn deadline; DB statement timeout | **No per-user rate limiting yet**: one user can create many conversations and run turns in parallel across them |
| Retry / cost amplification | SDK retries bounded (`GENERATION_MAX_RETRIES`) and only for transient errors; total time bounded by the turn deadline; idempotent replays never call the model; at most one reply per user message (DB-enforced) | Per-user cost budget / quotas (later phase) |
| Malicious or oversized input | Strict schema (unknown fields rejected, string type enforced), NUL rejected, length limits, UUID path validation | — |
| Dev auth misuse | Constant-time token comparison; production startup refused while dev auth is the only mode | Real identity provider required before any deployment |
| Dependency compromise | Pinned lockfile (`uv.lock`); pinned container base images; provider SDK isolated behind an adapter; `typesafe-sdk` checked to be TypeSafe's own package (publisher, repository, docs links); Laya pinned to `laya[serve]==0.4.6` and run as a separate process, not imported by the app | No automated vulnerability scanning yet. Laya releases very often (44 releases in a month) and downloads model weights from Hugging Face at first start; the weight revision is not pinned by this repository |
| Dataset poisoning | N/A until evaluation and training data exist (Phases 6/7) | Provenance and license tracking planned |

## Privacy notes

- **What is stored:** message content in plaintext in PostgreSQL; generation
  metadata (model, prompt version, token usage, provider request ID). From
  Phase 2: a conversation state per conversation (topic, goals and emotional
  context marked stated or inferred, open questions) and per-turn decision
  results. These are inferences about the person; they are deleted with the
  conversation (cascade) but have no separate view or delete API yet. From
  Phase 3: memories (short sentences the person said about themselves and
  an embedding of each), which the person can list, delete
  one by one or all at once, and turn off.
- **Third-party processing:** conversation content (system prompt plus recent
  history) is sent to the configured LLM provider on every turn: Anthropic,
  OpenAI, Groq, Google, xAI, Alibaba Cloud, DeepSeek, OpenRouter (and the
  upstream provider it routes to), Together, or Mistral; with Ollama it stays
  on your own server. With `DECISION_PROVIDER=jev`, the latest message and
  up to four earlier messages are also sent to
  TypeSafe on every turn (TypeSafe states it does not train on user data;
  zero data retention is offered to enterprise customers). With `llm` the
  same messages go to the decision LLM's provider on every turn (Groq by
  default; its terms need the same review as the reply provider's, see
  below). With `laya` they stay on your machine. Retrieved memories travel in the reply prompt, and
  the extraction call sends the latest messages and related memories, both
  to the same LLM provider.
  Review that provider's commercial terms, data retention, data residency and
  training policies before handling real user data. Free tiers (e.g. Groq,
  Gemini) can have different data-use terms from paid tiers. Server-side refusal fallback
  (`GENERATION_REFUSAL_FALLBACK=default`) may route a declined request to
  another Anthropic model under the same terms.
- **Retention and deletion:** memories expire (at most
  `MEMORY_RETENTION_DAYS`, 365 by default) and are purged by the maintenance
  loop; the person can delete them at any time, and deleting a memory
  deletes its embedding (same row). The full table of where memory content
  lives is in [memory.md](memory.md#retention-and-deletion-spec-107-154).
  Conversations and messages still have **no** retention policy or deletion
  endpoint: they are kept until deleted manually, and deleting a memory does
  not delete the message it came from. Deleting a user cascades to their
  conversations, messages and memories at the database level, but no API
  exposes this yet.
- **Backups:** none are configured by this repository. Any backup an operator
  adds must follow the same retention and deletion policy, since deleting a
  row does not remove it from existing backups.
- **Encryption:** relies on infrastructure (disk encryption, TLS termination
  in front of the app). Compose binds ports to 127.0.0.1 only.

## Safety boundary (Phase 5, docs/safety.md)

Prompts are not a safety mechanism on their own. The safety layer adds:

- **Independent input analysis:** a pattern screen plus an LLM classifier.
- **Pathways:** constrained replies that override the ordinary strategy.
- **Verified resources:** rendered by code; the reply may offer no other
  contacts.
- **Output checks with bounded recovery.**
- **Per-turn records** without text.

A failed classifier is degraded mode, never "safe". Injected instructions in
the message cannot switch off a pathway: the classifier treats the
conversation as data, and explicit pattern hits are not downgraded.

**Remaining gaps:**

- The layer is not clinically validated.
- Its evaluation barely covers imminent danger and abuse disclosures.
- The person-facing texts need review by a qualified person.
- The classifier model is a provider preview.

The system must not be used with real users until that review is done.
