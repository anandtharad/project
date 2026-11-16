# Supabase as the database

The app talks to PostgreSQL through SQLAlchemy (the Python library, not a
database). Supabase is hosted PostgreSQL, so nothing in the code changes: the
app needs a `DATABASE_URL` that points at it. Facts below are from Supabase's
documentation as read on 2026-11-16 (connecting to Postgres, pgvector,
row-level security, pricing). **Not yet run against a real Supabase project.**

## Setup

1. Create a project. Note the database password.
2. Enable pgvector: Dashboard → Database → Extensions → search "vector" →
   enable. Supabase puts it in the `extensions` schema. (Migration 0003 runs
   `CREATE EXTENSION IF NOT EXISTS vector`, which then does nothing.)
3. Choose the connection (Dashboard → Connect):
   - **Direct connection**, port 5432:
     `db.<project-ref>.supabase.co`. Supabase recommends it for persistent
     servers like this app, but on the Free plan it is IPv6 only.
   - **Shared pooler, session mode**, port 5432:
     `aws-<index>-<region>.pooler.supabase.com`, user `postgres.<project-ref>`.
     Use it when your network has no IPv6 (common on home connections).
   - Not the transaction-mode pooler (port 6543): it does not support the
     prepared statements the app's driver (asyncpg) uses.
4. In `.env`, with `+asyncpg` and SSL:
   ```
   DATABASE_URL=postgresql+asyncpg://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres?ssl=require
   ```
   or, for the session pooler,
   ```
   DATABASE_URL=postgresql+asyncpg://postgres.<project-ref>:<password>@aws-<index>-<region>.pooler.supabase.com:5432/postgres?ssl=require
   ```
   URL-encode special characters in the password.
5. Create the tables and load the knowledge library:
   ```
   uv run alembic upgrade head
   uv run python scripts/ingest_knowledge.py
   ```
6. Start the app and check `GET /ready` (database and migrations "ok").

Keep the local Docker database for tests: the test suite creates and drops
databases, which a hosted project should not be used for.

## Security: the Data API

Supabase can expose tables in the `public` schema, where the app's tables
live, through its auto-generated Data API, reachable with the project's
publishable key. Migration 0009 enables row-level security on every table
with no policies, so that API returns nothing; Supabase: "no data is
accessible through the API when using a publishable key, until you create
policies." The app connects as `postgres`, the tables' owner, which bypasses
row-level security, so it is unaffected. Do not add policies, and do not
use Supabase's client libraries against these tables: the app's own API,
with its ownership checks, is the only way in. Keep the database password
and the service role key server-side.

## Free plan limits (pricing page, 2026-11-16)

- 500 MB database. Conversations, memories and the knowledge library are
  small, but watch it as users grow.
- **Free projects are paused after one week of inactivity**: the app then
  fails `/ready` until the project is restored from the dashboard.
- Two active projects.

Authentication is Clerk's, not Supabase's (docs/api.md), so a paused
database is the only Supabase dependency.
