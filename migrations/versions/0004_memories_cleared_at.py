"""user_memory_settings.memories_cleared_at

Records the last "delete all memories", so an extraction already running for
an earlier message cannot store a memory after it.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-13 12:40:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "user_memory_settings",
        sa.Column("memories_cleared_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_memory_settings", "memories_cleared_at")
