# API reference (v1)

Base path `/v1`. JSON in and out. A machine-readable schema is served at
`/openapi.json` (and `/docs`) outside production.

## Authentication

> **Development authentication only.** Requests carry
> `Authorization: Bearer <token>`, where the token is one of the static
> `DEV_AUTH_TOKENS` pairs. There is no token issuance, expiry, rotation or
> revocation. Production authentication is not implemented, and the service
> refuses to start with `APP_ENV=production`.

A missing or invalid token returns `401 unauthorized` before any data access.

## Endpoints

### `GET /health`
Liveness. Never touches dependencies. `200 {"status": "ok"}`.

### `GET /ready`
Readiness: the database is reachable and the schema is at the expected
migration head. Returns `200` with `"status": "ready"`, or `503` with
`"status": "not_ready"`, plus per-check results:
```json
{"status": "ready", "checks": {"database": "ok", "migrations": "ok",
 "llm_provider": "anthropic", "prompt": "companion_system@1.0.0"}}
```
The LLM check only reports which provider is configured. Credentials are not
verified without making a paid call.

### `POST /v1/conversations`
Creates a conversation owned by the caller. `201`:
```json
{"conversation_id": "…uuid…", "created_at": "2026-10-12T05:44:00Z", "status": "active"}
```

### `GET /v1/conversations/{conversation_id}`
`200` with `conversation_id`, `created_at`, `updated_at`, `status`, `message_count`.

### `DELETE /v1/conversations/{conversation_id}`
204 No Content. Deletes the conversation with its messages, conversation
state, decisions and every other per-turn record. Memories learned from it
are kept (their link to the conversation is cleared); delete them under
`/v1/users/me/memories`. `409 turn_in_progress` while a reply is being
generated (retry after it finishes); a turn abandoned longer than
`STALE_TURN_SECONDS` does not block deletion. Another user's conversation
returns the same 404 as a missing one.

### `POST /v1/conversations/{conversation_id}/messages`
Submits a user message and returns the assistant's reply.

Request (unknown fields are rejected):
```json
{"content": "I've been feeling overwhelmed lately.", "idempotency_key": "msg-0001"}
```
- `content`: required string, not blank, no NUL characters, at most
  `MAX_INPUT_CHARACTERS` (default 8000).
- `idempotency_key`: optional, 1–128 characters of `[A-Za-z0-9_.:-]`.
  Recommended on every request.

Response, `201` for a new turn or `200` for a replay:
```json
{
  "conversation_id": "…",
  "user_message_id": "…",
  "assistant_message_id": "…",
  "assistant_content": "What's been feeling most overwhelming?",
  "created_at": "2026-10-12T05:44:00.66Z",
  "replayed": false,
  "timing_ms": {"received": 0.0, "…": 0, "completed": 1840.2}
}
```
`timing_ms` is development-only (`EXPOSE_TIMING_METADATA`, forced off in
production). Internal reasoning, safety analysis and prompts are never
returned.

### `GET /v1/conversations/{conversation_id}/messages?after_sequence=0&limit=50`
Messages in conversation order (`limit` is 1–200). Each item has
`message_id`, `sequence`, `role` (`user`/`assistant`), `content`,
`created_at`, and, on user messages only, `status`
(`pending`/`completed`/`failed`). If more messages exist,
`next_after_sequence` gives the value to pass as `after_sequence`.

### Memories: `/v1/users/me/...`

Each person's own memories (spec §10.7; design in [memory.md](memory.md)).
Another person's memory ID returns the same `404 memory_not_found` as a
missing one.

| Method and path | Result |
|---|---|
| `GET /v1/users/me/memories?status=` | `200 {"memories": [...]}`. `status`: `active`, `pending_confirmation`, `superseded` or `all`; default active and pending. Expired memories are never listed |
| `GET /v1/users/me/memories/{memory_id}` | `200` one memory |
| `DELETE /v1/users/me/memories/{memory_id}` | `204`; the row and its embedding are deleted |
| `DELETE /v1/users/me/memories` | `200 {"deleted": n}` |
| `POST /v1/users/me/memories/{memory_id}/confirm` | `200`; a pending memory becomes active and supersedes the memory it conflicted with |
| `GET /v1/users/me/memory-settings` | `200 {"memory_enabled": true, "allow_sensitive": false}` |
| `PUT /v1/users/me/memory-settings` | the same body, both fields required and strictly boolean |

A memory: `memory_id`, `memory_type` (`episodic`, `semantic`,
`preference`), `content`, `status`, `importance` (1-3), `created_at`,
`updated_at`, `expires_at`, `source_conversation_id`, `supersedes_id`.

## Idempotency

Idempotency is enforced in PostgreSQL (a unique constraint on
`(conversation_id, idempotency_key)`). Keys are scoped to a conversation, and
a conversation has exactly one owner.

| Situation for the same key | Result |
|---|---|
| Turn completed | `200`, the original IDs and reply, `replayed: true`, no model call |
| Turn still in progress | `409 turn_in_progress` with `Retry-After` |
| Turn failed | The same user message is re-attempted (no duplicate user message) |
| Turn failed, and the conversation has moved on | `409 turn_superseded` |
| Key reused with different content | `409 idempotency_key_reused` |
| Requests without a key | Each is a new turn |

## Concurrency

Only one turn can be in flight per conversation. A message sent while another
is being generated gets `409 turn_in_progress` (`Retry-After: 2`). Ordering
uses a per-conversation sequence counter assigned under a row lock, never
timestamps. A pending turn older than `STALE_TURN_SECONDS` (by the database
clock) counts as abandoned, for example after a crashed worker, and the next
request takes it over. Different conversations run in parallel.

## Errors

Every error has the same shape:
```json
{"error": {"category": "conflict", "code": "turn_in_progress",
           "message": "Another message in this conversation is still being processed. Retry shortly.",
           "trace_id": "4f1c…"}}
```
`trace_id` matches the `X-Request-ID` response header. A well-formed inbound
`X-Request-ID` is honored. Stack traces, secrets and submitted content are
never included.

| HTTP | category | codes |
|---|---|---|
| 401 | `unauthorized` | `unauthorized` |
| 404 | `not_found` | `conversation_not_found` (also returned for another user's conversation), `memory_not_found` (likewise), `route_not_found` |
| 409 | `conflict` | `turn_in_progress`, `turn_superseded`, `idempotency_key_reused` |
| 413 | `invalid_request` | `request_too_large` (body over `MAX_REQUEST_BYTES`) |
| 422 | `invalid_request` | `validation_failed` (with field `details`), `content_too_long` |
| 429 | `rate_limited` | `too_many_messages` (per-user limit, `USER_MESSAGES_PER_MINUTE` / `USER_MESSAGES_PER_DAY`; nothing is stored; `Retry-After` says when the oldest counted message leaves the window) |
| 500 | `internal_error` | `internal_error` |
| 503 | `dependency_unavailable` | `generation_unavailable`, `invalid_generation`, `persistence_failed`, `database_unavailable` |
| 504 | `timeout` | `generation_timeout` |

On every `503`/`504` from a message submission, the user message has been
saved (status `failed`) and no assistant reply was stored. Retrying with the
same idempotency key re-attempts the turn. `Retry-After` is set when retrying
is likely to help.

The `forbidden` category is reserved for later phases; no endpoint returns
it yet.
