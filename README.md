# Companion AI

An English-first conversational companion built as a decision-controlled LLM
application: application code owns authorization, orchestration, persistence
and enforcement; the LLM only produces conversational language.

> **Status: Phases 0-1 are done; Phase 2 (state and decisions) is built and
> evaluated, see the status page for what is and isn't met.** This is a
> development build. It has **no independent safety layer yet** (Phase 5)
> and only **development authentication**. Do not expose it to real users.
> See [docs/implementation-status.md](docs/implementation-status.md).

What works now: text in → validated → persisted conversation history →
conversation state updated (structured LLM call) → bounded decisions from a
decision model (self-hosted Laya or TypeSafe Jev; shadow mode until
calibrated) → response policy in code → instruction LLM → validated
response → reply, decisions and versioned state persisted together. It has
idempotent retries, one in-flight turn per conversation, bounded timeouts,
controlled provider failures, and model, prompt and policy provenance on
every reply. Details: [docs/decisions.md](docs/decisions.md).

The LLM is swappable: Anthropic, OpenAI, Groq, Google Gemini, xAI (Grok),
Qwen (Alibaba Model Studio), DeepSeek, OpenRouter, Together, Mistral, Ollama,
or any OpenAI-compatible endpoint. See [docs/providers.md](docs/providers.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/) 0.11+
- Docker with Compose v2.24+ (or a local PostgreSQL 16)
- An API key for the LLM provider you choose (for example `GROQ_API_KEY`).
  Without one, set `GENERATION_PROVIDER=fake` for a labeled stub that calls
  no model.

## Quick start (Docker Compose)

```bash
cp .env.example .env
# Edit .env:
#   DEV_AUTH_TOKENS=alice:<long random token>
#   GENERATION_PROVIDER=groq        (or anthropic, openai, gemini, xai, qwen, ...)
#   GROQ_API_KEY=<your key>         (the key variable for your provider)
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

Or chat interactively: `uv run python scripts/chat.py --token <your dev token>`.

Interactive API docs (non-production only): `http://127.0.0.1:8000/docs`.
Written reference: [docs/api.md](docs/api.md).

## Local development (app on the host)

```bash
uv sync --all-extras                      # creates .venv from uv.lock (all provider SDKs)
cp .env.example .env                      # then edit as above
docker compose up -d db                   # or point DATABASE_URL at your own PostgreSQL 16
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --reload
```

## Tests and checks

The helper script works the same on Windows, macOS and Linux, so you never
need `export` or `$env:` syntax:

```bash
uv run python scripts/dev.py doctor        # check Python, .env, keys (masked), SDKs, Docker, database
uv run python scripts/dev.py unit          # offline unit tests, no keys or database needed
docker compose up -d db                    # start PostgreSQL for the full suite
uv run python scripts/dev.py test          # unit + integration + security tests
uv run python scripts/dev.py groq-models   # models your Groq key can use
uv run python scripts/dev.py live-groq     # live Groq tests: 6 small free-tier calls
uv run python scripts/dev.py live-groq --models openai/gpt-oss-20b
uv run python scripts/dev.py laya          # self-hosted Laya decision server (optional)
uv run python scripts/eval_decisions.py --split dev       # decision accuracy (needs a decision provider)
uv run python scripts/eval_policy_effect.py --split dev   # do strategies change replies? (live LLM)
```

`test` uses the compose database by default. Set `TEST_DATABASE_URL` to use a
different one; the role must be allowed to CREATE DATABASE. Keys are read
from `.env` or the environment and are never printed.

Lint and types: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy app scripts tests`.

## Windows

Everything runs natively in PowerShell, with no WSL needed.

1. Install [Git](https://git-scm.com/download/win), Python 3.12
   ([python.org](https://www.python.org/downloads/)), and
   [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install uv, either way works:
   `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   or `py -m pip install uv`. Then open a new terminal.
3. Clone the repo and set up:
   ```powershell
   git clone https://github.com/anandtharad/project.git
   cd project
   git checkout claude/wonderful-ride-qnd48q
   uv sync --all-extras
   Copy-Item .env.example .env
   notepad .env        # set GROQ_API_KEY=... and DEV_AUTH_TOKENS=me:some-long-token
   ```
4. Run the helper commands above, for example
   `uv run python scripts/dev.py doctor`.

Windows notes:
- Edit `.env` with Notepad (UTF-8). Do not create it with `echo ... > .env`
  in Windows PowerShell 5.1: that writes UTF-16, which the app reads but
  `docker compose` cannot parse. `dev.py doctor` warns about it; convert with
  `$c = Get-Content .env; [IO.File]::WriteAllLines("$PWD\.env", $c)`.
- In Windows PowerShell, `curl` is an alias for `Invoke-WebRequest`. Use
  `curl.exe` or `Invoke-RestMethod http://localhost:8000/ready`, or just
  `uv run python scripts/dev.py doctor`.
- If PostgreSQL is already installed locally and uses port 5432, move the
  compose database to another port. Put both lines in `.env`, so compose,
  `alembic` and `uvicorn` all use it in every terminal:
  ```
  POSTGRES_PORT=55432
  DATABASE_URL=postgresql+asyncpg://companion:companion@localhost:55432/companion
  ```
  The test commands read `TEST_DATABASE_URL` from the environment only:
  `$env:TEST_DATABASE_URL = "postgresql+asyncpg://companion:companion@localhost:55432/companion"`.
- A `$env:NAME = "value"` setting lasts for the whole PowerShell window.
  Clear one with `Remove-Item Env:NAME`.
- Multi-line commands use a backtick (`` ` ``) instead of `\`, or put the
  whole command on one line.

## Configuration

Every setting is an environment variable (or a `.env` entry). See
[.env.example](.env.example) for the full list with defaults; `app/config.py`
is the source of truth. Important ones:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` |
| `DEV_AUTH_TOKENS` | `handle:token,…` development bearer tokens (not production auth) |
| `GENERATION_PROVIDER` | `anthropic`, `openai`, `groq`, `gemini`, `xai`, `qwen`, `deepseek`, `openrouter`, `together`, `mistral`, `ollama`, `openai_compatible`, or `fake` (development stub) |
| `GENERATION_MODEL` | optional; defaults to the provider preset's model |
| `GROQ_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, … | the selected provider's key ([full list](docs/providers.md)); `GENERATION_API_KEY` overrides it |
| `GENERATION_EFFORT` | `none`…`max` reasoning/thinking depth, mapped per provider; unset = provider default |
| `MAX_OUTPUT_TOKENS`, `MAX_CONTEXT_TOKENS` | optional; provider defaults (Groq's are sized for its free tier) |
| `TURN_DEADLINE_SECONDS` / `STALE_TURN_SECONDS` | overall turn budget / when a pending turn counts as abandoned |
| `MAX_INPUT_CHARACTERS`, `MAX_REQUEST_BYTES` | input limits enforced before any model call |
| `STATE_EXTRACTION_ENABLED`, `STATE_EXTRACTION_MODEL` | conversation-state update before each reply (default on; the reply model) |
| `DECISION_PROVIDER` | `off` (default), `laya` (self-hosted, `LAYA_API_KEY`) or `jev` (`JEV_API_KEY`) |
| `COMPANION_PROMPT_VERSION` | reply prompt; `1.2.0` carries the response-policy slot (`1.0.0` runs without a policy) |

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
  orchestration/  turn lifecycle, transactional turn store, response policy, validation
  decisions/      decision questions, System One adapter (Laya / Jev), validation, thresholds
  state/          conversation state schema and structured extraction
  evaluation/     decision and policy-effect evaluation metrics
  llm/            provider-neutral schemas, provider presets, and adapters
                  (Anthropic, OpenAI-compatible, Groq, Gemini, dev stub)
  context/        context assembly with a history budget
  db/             SQLAlchemy models, engine, ownership-scoped queries
  prompting.py    versioned prompt registry
migrations/       Alembic migrations
prompts/          versioned prompt files + checksum lock
decisions/        decision-question checksum lock
eval/             evaluation datasets (with dataset cards) and results
docker/           optional Laya decision-server image
tests/            unit / integration / security / live
docs/             architecture, API, threat model, implementation status
```

## Documentation

- [Architecture](docs/architecture.md)
- [LLM providers](docs/providers.md)
- [Decisions, state and response policy](docs/decisions.md)
- [API reference](docs/api.md)
- [Threat model and privacy notes](docs/threat-model.md)
- [Implementation status and known limitations](docs/implementation-status.md)
