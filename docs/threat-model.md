# Threat model and privacy notes (initial: Phase 1 scope)

Informed by the OWASP Top 10 for LLM Applications and the NIST AI RMF. These
references shape the analysis; they do not certify the system. This document
covers what exists today and is revisited at each phase.

## Assets

- Conversation content (sensitive personal disclosures are expected).
- User-to-conversation ownership.
- Provider credentials (`GENERATION_API_KEY`) and dev tokens.
- The system prompt (low sensitivity, but its integrity matters).
- Availability and spend (paid LLM calls).

## Trust boundaries

1. Client → API: everything in a request is untrusted.
2. API → LLM provider: conversation content leaves the system here.
3. LLM output → API: model output is untrusted until validated.
4. API → PostgreSQL: trusted store; access only through ownership-scoped queries.

## Threats, current mitigations, gaps

| Threat | Mitigations now | Gaps / planned |
|---|---|---|
| Cross-user data disclosure | Ownership filter in the SQL that loads a conversation; foreign conversations return the same 404 as missing ones; idempotency replay happens only after the ownership check; tests cover read, list, post and replay across users | Production authN/authZ (Phase 1 uses static dev tokens) |
| Prompt injection (direct) | User text only travels in user-role messages; the system instruction is the fixed versioned prompt, which tells the model that conversation text cannot change it; the model has **no tools**, so injected instructions cannot access data | Behavioral resistance (e.g. refusing to reveal the system prompt) is unverified without live evaluation; output policy checks arrive in Phase 5 |
| Prompt injection (indirect: memory/RAG) | N/A: no memory or retrieval yet | Phases 3/4 must treat memory and documents as untrusted data sections |
| Unsafe output handling | Output is stored and returned as plain text, never executed or rendered by the server; refusals, truncations, empty and over-long replies are rejected | **No safety-policy output validation yet (Phase 5)**; clients must escape output when rendering |
| Sensitive information in logs | Logs carry IDs, categories, timings and token counts, never content; exception messages and tracebacks are not logged or returned; enforced by a test | Log retention and shipping not configured |
| API key exposure | Key held as `SecretStr` (not in reprs or logs), server-side only, never returned; `.env` gitignored; `.env.example` has placeholders only | Key rotation procedure: rotate in the provider console, update the secret, restart; no secret manager integration yet |
| Denial of service | Body limit (413, checked on declared and streamed bytes); input character limit before any model call; one in-flight turn per conversation; bounded turn deadline; DB statement timeout | **No per-user rate limiting yet**: one user can create many conversations and run turns in parallel across them |
| Retry / cost amplification | SDK retries bounded (`GENERATION_MAX_RETRIES`) and only for transient errors; total time bounded by the turn deadline; idempotent replays never call the model; at most one reply per user message (DB-enforced) | Per-user cost budget / quotas (later phase) |
| Malicious or oversized input | Strict schema (unknown fields rejected, string type enforced), NUL rejected, length limits, UUID path validation | — |
| Dev auth misuse | Constant-time token comparison; production startup refused while dev auth is the only mode | Real identity provider required before any deployment |
| Dependency compromise | Pinned lockfile (`uv.lock`); pinned container base images; provider SDK isolated behind an adapter | No automated vulnerability scanning yet |
| Dataset poisoning | N/A until evaluation and training data exist (Phases 6/7) | Provenance and license tracking planned |

## Privacy notes

- **What is stored:** message content in plaintext in PostgreSQL; generation
  metadata (model, prompt version, token usage, provider request ID). No
  inferred attributes, profiles or memories are stored in Phase 1.
- **Third-party processing:** conversation content (system prompt plus recent
  history) is sent to the configured LLM provider (Anthropic) on every turn.
  Review the provider's commercial terms, data retention and training
  policies before handling real user data. Server-side refusal fallback
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
