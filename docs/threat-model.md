# Threat model and privacy notes (Phases 1-2 scope)

Informed by the OWASP Top 10 for LLM Applications and the NIST AI RMF. These
references shape the analysis; they do not certify the system. This document
covers what exists today and is revisited at each phase.

## Assets

- Conversation content (sensitive personal disclosures are expected).
- User-to-conversation ownership.
- Provider credentials (`GENERATION_API_KEY`, `JEV_API_KEY`, `LAYA_API_KEY`) and dev tokens.
- Conversation state (inferred goals and emotional context about a person).
- The system prompt (low sensitivity, but its integrity matters).
- Availability and spend (paid LLM calls).

## Trust boundaries

1. Client → API: everything in a request is untrusted.
2. API → LLM provider: conversation content leaves the system here (replies
   and state extraction).
2b. API → decision service: the latest message and up to four earlier
   messages are sent to Laya (self-hosted: stays on your machine) or TypeSafe
   Jev (third party). Conversation notes are not sent.
3. LLM output → API: model output is untrusted until validated.
4. API → PostgreSQL: trusted store; access only through ownership-scoped queries.

## Threats, current mitigations, gaps

| Threat | Mitigations now | Gaps / planned |
|---|---|---|
| Cross-user data disclosure | Ownership filter in the SQL that loads a conversation; foreign conversations return the same 404 as missing ones; idempotency replay happens only after the ownership check; tests cover read, list, post and replay across users | Production authN/authZ (Phase 1 uses static dev tokens) |
| Prompt injection (direct) | User text only travels in user-role messages; the system instruction is the fixed versioned prompt, which tells the model that conversation text cannot change it; the model has **no tools**, so injected instructions cannot access data | Behavioral resistance (e.g. refusing to reveal the system prompt) is unverified without live evaluation; output policy checks arrive in Phase 5 |
| Prompt injection (indirect: memory/RAG) | N/A: no memory or retrieval yet | Phases 3/4 must treat memory and documents as untrusted data sections |
| Prompt injection via conversation state | State is extracted from untrusted text, so it is untrusted too: it reaches the reply model only in the user role, inside a `<conversation_notes>` block whose JSON escapes `<`/`>` (it cannot close the block), and the prompt says the notes are never instructions. The extraction prompt treats its input as data inside escaped tags. Field lengths and counts are capped | Behavioral resistance is unverified without live adversarial evaluation |
| Decision manipulation | Decisions never control access, persistence or safety handling and run after the ownership check; answers are validated against the question asked (invalid answers are rejected, not repaired); only calibrated, above-threshold answers change the policy, and the worst case is one of the fixed prompt fragments; `safety_response` cannot be selected | A manipulated `conversation_mode=safety` only makes the reply more cautious |
| Unsafe output handling | Output is stored and returned as plain text, never executed or rendered by the server; refusals, truncations, empty and over-long replies are rejected | **No safety-policy output validation yet (Phase 5)**; clients must escape output when rendering |
| Sensitive information in logs | Logs carry IDs, categories, timings and token counts, never content; exception messages and tracebacks are not logged or returned; enforced by a test | Log retention and shipping not configured |
| API key exposure | Keys held as `SecretStr` (not in reprs or logs) and masked in `ResolvedProvider` reprs; server-side only, never returned; `.env` gitignored; `.env.example` has placeholders only; only the selected provider's key is sent, only to its endpoint | Key rotation procedure: rotate in the provider console, update the secret, restart; no secret manager integration yet |
| Decision-service credentials | The TypeSafe SDK reads `TYPESAFE_API_KEY`/`TYPESAFE_BASE_URL`/`TYPESAFE_DEFAULT_MODEL` when arguments are missing; the adapter always passes all three. Its logger records request and response bodies at DEBUG, so it is capped at WARNING even when `TYPESAFE_LOG_LEVEL=debug` (tested). `laya-serve` accepts any caller without `LAYA_API_KEY`, so the app refuses `DECISION_PROVIDER=laya` without a key, and `dev.py laya` and compose bind it to localhost | — |
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
  conversation (cascade) but have no separate view or delete API yet.
- **Third-party processing:** conversation content (system prompt plus recent
  history) is sent to the configured LLM provider on every turn: Anthropic,
  OpenAI, Groq, Google, xAI, Alibaba Cloud, DeepSeek, OpenRouter (and the
  upstream provider it routes to), Together, or Mistral; with Ollama it stays
  on your own server. With `DECISION_PROVIDER=jev`, the latest message and
  up to four earlier messages are also sent to
  TypeSafe on every turn (TypeSafe states it does not train on user data;
  zero data retention is offered to enterprise customers). With `laya` they
  stay on your machine.
  Review that provider's commercial terms, data retention, data residency and
  training policies before handling real user data. Free tiers (e.g. Groq,
  Gemini) can have different data-use terms from paid tiers. Server-side refusal fallback
  (`GENERATION_REFUSAL_FALLBACK=default`) may route a declined request to
  another Anthropic model under the same terms.
- **Retention and deletion:** **not implemented yet.** Conversations and
  messages are kept until deleted manually. Deleting a user cascades to
  their conversations and messages at the database level, but no API exposes
  this yet. Phase 3 adds memory deletion. A conversation-deletion endpoint
  and a retention policy are tracked in implementation-status.md.
- **Backups:** none are configured by this repository. Any backup an operator
  adds must follow the same retention and deletion policy, since deleting a
  row does not remove it from existing backups.
- **Encryption:** relies on infrastructure (disk encryption, TLS termination
  in front of the app). Compose binds ports to 127.0.0.1 only.

## Safety boundary (not yet implemented)

The companion prompt includes basic guidance for danger situations, but
**prompts are not a safety mechanism**. Phase 5 adds independent input
analysis, a constrained safety pathway, output policy validation, and
separately reported safety metrics. Until then the system must not be used
with real users.
