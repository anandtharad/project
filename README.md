# Companion AI

An English-first conversational companion built as a decision-controlled LLM
application: application code owns authorization, orchestration, persistence
and enforcement; the LLM only produces conversational language.

> **Status: Phases 0-1 are done; Phases 2 (state and decisions), 3
> (memory), 4 (knowledge retrieval), 5 (safety) and 6 (evaluation) are
> built and evaluated, see the status page for what is and isn't met.** This is a development
> build: the safety layer is **not clinically validated**, and
> authentication is **development-only**. Do not expose it to real users.
> See [docs/implementation-status.md](docs/implementation-status.md).

What works now: text in → validated → persisted conversation history →
conversation state updated (structured LLM call) → bounded decisions from a
decision model (self-hosted Laya or TypeSafe Jev; shadow mode until
calibrated) → response policy in code → instruction LLM → validated
response → reply, decisions and versioned state persisted together →
memories proposed in the background, accepted by code, and retrieved in
later conversations (the person can list, delete or turn them off). Every
turn also passes a safety layer (input analysis, pathways with verified
crisis contacts, output checks), and information requests are grounded in a
small reviewed library with citations that resolve. It has
idempotent retries, one in-flight turn per conversation, bounded timeouts,
controlled provider failures, and model, prompt and policy provenance on
every reply. Each layer and the whole conversation are measured, with
release gates (docs/evaluation.md). Details:
[docs/decisions.md](docs/decisions.md), [docs/memory.md](docs/memory.md),
[docs/safety.md](docs/safety.md), [docs/rag.md](docs/rag.md),
[docs/evaluation.md](docs/evaluation.md).

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

Or chat interactively: `uv run python scripts/chat.py --token <your dev token>`, where the
token is the text after the colon in `DEV_AUTH_TOKENS` (for `me:abc123`, `--token abc123`).

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
uv run python scripts/eval_memory.py retrieval --split dev   # memory retrieval (local model, needs the database)
uv run python scripts/eval_memory.py extraction --split dev  # memory extraction (live LLM)
uv run python scripts/eval_golden.py --mode deterministic    # 216 end-to-end scenarios, no key needed
uv run python scripts/eval_report.py eval/results/A.json eval/results/B.json   # compare two runs
uv run python scripts/check_gates.py                       # release gates (eval/gates.toml)
```

All evaluation suites, datasets, judges and gates: [docs/evaluation.md](docs/evaluation.md).

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
- Knowledge lookups need the reviewed corpus in the database once (and
  again after `git pull` changes `data/knowledge`):
  `uv run python scripts/ingest_knowledge.py`. Without it, information
  requests are answered without sources.
- Memory search uses a local embedding model. The first start downloads it
  (about 65 MB from huggingface.co) into `EMBEDDING_CACHE_DIR`
  (`.cache\fastembed` with `.env.example`); until it is ready, memories are
  matched by text only. onnxruntime needs the Microsoft Visual C++
  Redistributable (2019 or later), which most Windows machines already
  have. Hugging Face may warn that symlinks are unavailable; that only
  costs disk space. To run without it: `EMBEDDING_PROVIDER=none`.

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
| `USER_MESSAGES_PER_MINUTE` / `USER_MESSAGES_PER_DAY` | per-user message limits across all conversations (20 / 1000; `0` turns a limit off); over the limit, `429 too_many_messages` with `Retry-After` |
| `MAX_INPUT_CHARACTERS`, `MAX_REQUEST_BYTES` | input limits enforced before any model call |
| `STATE_EXTRACTION_ENABLED`, `STATE_EXTRACTION_MODEL` | conversation-state update before each reply (default on; the reply model) |
| `DECISION_PROVIDER` | `off` (default), `laya` (self-hosted, `LAYA_API_KEY`), `jev` (`JEV_API_KEY`) or `llm` (a separate LLM, default Groq `openai/gpt-oss-20b`; `DECISION_LLM_*`, see docs/decisions.md) |
| `MEMORY_ENABLED`, `MEMORY_EXTRACTION_MODEL` | memory across conversations (default on; extraction uses the reply model) |
| `EMBEDDING_PROVIDER`, `EMBEDDING_CACHE_DIR` | `local` (default, no key), `openai`, `gemini` or `none`; where the local model is cached |
| `MEMORY_MIN_SEMANTIC`, `MEMORY_RETENTION_DAYS` | retrieval threshold (0.49, tuned for the local model) and the longest a memory is kept (365 days) |
| `CONVERSATION_RETENTION_DAYS` | a conversation with no new message for this long is deleted with its turns (365 days; memories are kept) |
| `COMPANION_PROMPT_VERSION` | reply prompt; `1.7.0` (default) is the 1.5.0 voice (docs/voice.md) with the safety pathway fragments (1.6.0) and the evidence rules for knowledge passages; `1.4.0` is the previous voice (`1.0.0` runs without a policy) |
| `KNOWLEDGE_ENABLED`, `KNOWLEDGE_MIN_SEMANTIC` | knowledge lookups for information requests (default on; needs `scripts/ingest_knowledge.py` once) and the relevance threshold, tuned for the local embedding model; see docs/rag.md |
| `SAFETY_REGION`, `SAFETY_CLASSIFIER_PROVIDER`, `SAFETY_CLASSIFIER_MODEL` | crisis resources (`IN`, `US`, `GB` or generic) and the safety classifier (`auto`: Groq `openai/gpt-oss-safeguard-20b` when `GROQ_API_KEY` is set); see docs/safety.md |
| `MEMORY_PROMPT_VERSION` | memory-extraction prompt (see docs/memory.md for the evaluated versions) |

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
  memory/         memory schema, eligibility policy, store, retrieval, extraction
  safety/         taxonomy, pattern screen, classifier, pathway policy, resources, output checks
  rag/            knowledge extraction, chunking, ingestion, hybrid retrieval, routing, citations
  embeddings/     embedding clients (local fastembed, OpenAI, Gemini)
  evaluation/     metrics per suite, golden scenarios, dialogue benchmarks, run metadata and recorder
  llm/            provider-neutral schemas, provider presets, and adapters
                  (Anthropic, OpenAI-compatible, Groq, Gemini, dev stub)
  context/        context assembly with a history budget
  db/             SQLAlchemy models, engine, ownership-scoped queries
  prompting.py    versioned prompt registry
migrations/       Alembic migrations
data/knowledge/   reviewed knowledge manifest and extracted-text snapshots
prompts/          versioned prompt files + checksum lock
decisions/        decision-question checksum lock
eval/             evaluation datasets (with dataset cards), results, rubrics and release gates
docker/           optional Laya decision-server image
tests/            unit / integration / security / live
docs/             architecture, API, threat model, implementation status
```

## Documentation

- [Architecture](docs/architecture.md)
- [LLM providers](docs/providers.md)
- [Decisions, state and response policy](docs/decisions.md)
- [Memory](docs/memory.md)
- [Safety layer](docs/safety.md)
- [Knowledge retrieval](docs/rag.md)
- [Voice (how replies read)](docs/voice.md)
- [Evaluation and observability](docs/evaluation.md)
- [API reference](docs/api.md)
- [Threat model and privacy notes](docs/threat-model.md)
- [Implementation status and known limitations](docs/implementation-status.md)
