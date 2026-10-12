# Companion AI

An English-first conversational companion built as a decision-controlled LLM
application: application code owns authorization, orchestration, persistence
and enforcement; the LLM only produces conversational language.

> **Status: Phase 0 (foundation) and Phase 1 (basic conversation) are done.**
> This is a development build. It has **no independent safety layer yet**
> (Phase 5) and only **development authentication**. Do not expose it to real
> users. See [docs/implementation-status.md](docs/implementation-status.md).

What works now: text in → validated → persisted conversation history →
instruction LLM (Anthropic Claude via adapter) → validated response →
persisted assistant turn. It has idempotent retries, one in-flight turn per
conversation, bounded timeouts, controlled provider failures, and model and
prompt provenance on every reply.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/) 0.11+
- Docker with Compose v2.24+ (or a local PostgreSQL 16)
- An Anthropic API key for real replies. Without one, set
  `GENERATION_PROVIDER=fake` for a labeled stub that calls no model.

## Quick start (Docker Compose)

```bash
cp .env.example .env
# Edit .env:
#   DEV_AUTH_TOKENS=alice:<long random token>
#   GENERATION_API_KEY=<your Anthropic key>   (or GENERATION_PROVIDER=fake)
docker compose up --build
```

Compose starts PostgreSQL (pgvector image), runs `alembic upgrade head` once,
then starts the API on `http://127.0.0.1:8000`.

```bash
curl -s localhost:8000/ready
TOKEN=<your dev token>
CID=$(curl -s -X POST localhost:8000/v1/conversations \
  -H "Authorization: Bearer $TOKEN" | python3 -c 'import sys,json;print(json.load(sys.stdin)["conversation_id"])')
curl -s -X POST localhost:8000/v1/conversations/$CID/messages \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"content": "I have been feeling overwhelmed lately.", "idempotency_key": "msg-1"}'
```

Or chat interactively: `uv run python scripts/chat.py --token $TOKEN`.

Interactive API docs (non-production only): `http://127.0.0.1:8000/docs`.
Written reference: [docs/api.md](docs/api.md).

## Local development (app on the host)

```bash
uv sync                                   # creates .venv from uv.lock
cp .env.example .env                      # then edit as above
docker compose up -d db                   # or point DATABASE_URL at your own PostgreSQL 16
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --reload
```

## Tests and checks

```bash
uv run pytest tests/unit                  # no external dependencies

# Integration + security tests need PostgreSQL. The role must be allowed to
# CREATE DATABASE; each run creates and drops its own throwaway database.
export TEST_DATABASE_URL=postgresql+asyncpg://companion:companion@localhost:5432/companion
uv run pytest                             # unit + integration + security

# Live provider tests: opt-in, real API calls and cost.
RUN_LIVE_LLM_TESTS=1 ANTHROPIC_API_KEY=... uv run pytest tests/live -v

uv run ruff check . && uv run ruff format --check . && uv run mypy app scripts tests
```

Without `TEST_DATABASE_URL`, database-backed tests are skipped with an
explicit reason rather than silently passing.

## Configuration

Every setting is an environment variable (or a `.env` entry). See
[.env.example](.env.example) for the full list with defaults; `app/config.py`
is the source of truth. Important ones:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` |
| `DEV_AUTH_TOKENS` | `handle:token,…` development bearer tokens (not production auth) |
| `GENERATION_PROVIDER` | `anthropic` or `fake` (development stub) |
| `GENERATION_MODEL` | defaults to `claude-opus-5-5` |
| `GENERATION_API_KEY` | optional; otherwise the SDK uses `ANTHROPIC_API_KEY` or an `ant auth login` profile |
| `GENERATION_EFFORT` | `low`…`max` thinking effort, default `low` for chat (to be tuned by evaluation) |
| `GENERATION_REFUSAL_FALLBACK` | `default` enables the provider's server-side fallback when the model declines; `off` disables it |
| `TURN_DEADLINE_SECONDS` / `STALE_TURN_SECONDS` | overall turn budget / when a pending turn counts as abandoned |
| `MAX_INPUT_CHARACTERS`, `MAX_REQUEST_BYTES` | input limits enforced before any model call |

`APP_ENV=production` refuses to start until production authentication exists.

## Prompts

Prompts are versioned files under `prompts/<id>/<version>.toml` with intended
use, variables, and changelog. `prompts/prompts.lock.json` pins each released
template's SHA-256, so the app refuses to start if a released prompt was edited
in place. To change behavior, add a new version file, run
`uv run python scripts/lock_prompts.py`, and select it with
`COMPANION_PROMPT_VERSION`.

## Layout

```
app/
  api/            HTTP: routes, dev auth, error contract, middleware
  orchestration/  turn lifecycle, transactional turn store, response validation
  llm/            provider-neutral schemas + Anthropic and stub providers
  context/        context assembly with a history budget
  db/             SQLAlchemy models, engine, ownership-scoped queries
  prompting.py    versioned prompt registry
migrations/       Alembic migrations
prompts/          versioned prompt files + checksum lock
tests/            unit / integration / security / live
docs/             architecture, API, threat model, implementation status
```

## Documentation

- [Architecture](docs/architecture.md)
- [API reference](docs/api.md)
- [Threat model and privacy notes](docs/threat-model.md)
- [Implementation status and known limitations](docs/implementation-status.md)
