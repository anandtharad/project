"""row-level security on every table

Hosted PostgreSQL such as Supabase can expose tables in the `public` schema
through an auto-generated Data API. With row-level security enabled and no
policies, that API returns nothing (Supabase's RLS guide). The application
connects as the tables' owner, which bypasses row-level security (it is not
FORCEd), so the app itself is unaffected; ownership checks stay in its
queries. Harmless on a plain PostgreSQL.

Revision ID: 0009
Revises: 0008
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    "users",
    "conversations",
    "messages",
    "conversation_states",
    "turn_decisions",
    "turn_safety",
    "memories",
    "user_memory_settings",
    "knowledge_documents",
    "knowledge_chunks",
    "evaluation_runs",
    "evaluation_results",
    "alembic_version",
)


def upgrade() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')


def downgrade() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')
