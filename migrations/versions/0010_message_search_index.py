"""full-text search index on messages

The agent's search_conversation_history tool (app/agent/tools.py) searches
the person's own messages with to_tsvector('english', content); this GIN
expression index serves that query.

Revision ID: 0010
Revises: 0009
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_messages_content_fts ON messages "
        "USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX ix_messages_content_fts")
