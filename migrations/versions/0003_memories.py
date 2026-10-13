"""memories

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-13 06:44:02.957742
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Ships with the compose image (pgvector/pgvector); needs a superuser or a
    # database where the extension is already installed.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "user_memory_settings",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("memory_enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("allow_sensitive", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_memory_settings_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_user_memory_settings")),
    )
    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("memory_type", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("importance", sa.SmallInteger(), nullable=False),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_conversation_id", sa.Uuid(), nullable=True),
        sa.Column("source_message_id", sa.Uuid(), nullable=True),
        sa.Column("supersedes_id", sa.Uuid(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "memory_type IN ('episodic', 'semantic', 'preference')",
            name=op.f("ck_memories_memory_type_valid"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'pending_confirmation', 'superseded')",
            name=op.f("ck_memories_status_valid"),
        ),
        sa.CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)",
            name=op.f("ck_memories_embedding_has_model"),
        ),
        sa.CheckConstraint("importance BETWEEN 1 AND 3", name=op.f("ck_memories_importance_range")),
        sa.ForeignKeyConstraint(
            ["source_conversation_id"],
            ["conversations.id"],
            name=op.f("fk_memories_source_conversation_id_conversations"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_message_id"],
            ["messages.id"],
            name=op.f("fk_memories_source_message_id_messages"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"],
            ["memories.id"],
            name=op.f("fk_memories_supersedes_id_memories"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_memories_user_id_users"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_memories")),
    )
    op.create_index("ix_memories_user_status", "memories", ["user_id", "status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_memories_user_status", table_name="memories")
    op.drop_table("memories")
    op.drop_table("user_memory_settings")
