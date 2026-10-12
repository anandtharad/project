"""initial conversation schema

Revision ID: 0001
Revises:
Create Date: 2026-10-12 05:36:51.870217
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("external_id", name=op.f("uq_users_external_id")),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default=sa.text("'active'"), nullable=False
        ),
        sa.Column("last_sequence", sa.Integer(), server_default=sa.text("0"), nullable=False),
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
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_conversations_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_index(op.f("ix_conversations_user_id"), "conversations", ["user_id"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("turn_status", sa.String(length=16), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("attempt_id", sa.Uuid(), nullable=True),
        sa.Column("attempt_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_category", sa.String(length=64), nullable=True),
        sa.Column("reply_to_message_id", sa.Uuid(), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("prompt_id", sa.String(length=128), nullable=True),
        sa.Column("prompt_version", sa.String(length=32), nullable=True),
        sa.Column("generation_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "(role = 'user' AND turn_status IN ('pending', 'completed', 'failed') AND reply_to_message_id IS NULL) OR (role = 'assistant' AND turn_status IS NULL AND idempotency_key IS NULL AND reply_to_message_id IS NOT NULL AND model_id IS NOT NULL AND prompt_id IS NOT NULL AND prompt_version IS NOT NULL)",
            name=op.f("ck_messages_role_fields"),
        ),
        sa.CheckConstraint("role IN ('user', 'assistant')", name=op.f("ck_messages_role_valid")),
        sa.CheckConstraint("sequence > 0", name=op.f("ck_messages_sequence_positive")),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["reply_to_message_id"],
            ["messages.id"],
            name=op.f("fk_messages_reply_to_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_messages_conversation_idempotency_key"
        ),
        sa.UniqueConstraint(
            "conversation_id", "sequence", name="uq_messages_conversation_sequence"
        ),
        sa.UniqueConstraint("reply_to_message_id", name=op.f("uq_messages_reply_to_message_id")),
    )
    op.create_index(
        "uq_messages_one_pending_turn",
        "messages",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("turn_status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_messages_one_pending_turn",
        table_name="messages",
        postgresql_where=sa.text("turn_status = 'pending'"),
    )
    op.drop_table("messages")
    op.drop_index(op.f("ix_conversations_user_id"), table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("users")
