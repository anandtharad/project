"""knowledge_documents.published_on (spec §11.3: publication date where available)

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-15 13:10:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("knowledge_documents", sa.Column("published_on", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_documents", "published_on")
