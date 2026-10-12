"""conversation state and turn decisions

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-12 17:00:39.626815
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_states",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("state", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("source_message_id", sa.Uuid(), nullable=True),
        sa.Column("extraction", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state_version > 0", name=op.f("ck_conversation_states_state_version_positive")
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_conversation_states_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_message_id"],
            ["messages.id"],
            name=op.f("fk_conversation_states_source_message_id_messages"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("conversation_id", name=op.f("pk_conversation_states")),
    )
    op.create_table(
        "turn_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("user_message_id", sa.Uuid(), nullable=False),
        sa.Column("question_id", sa.String(length=64), nullable=False),
        sa.Column("question_version", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("returned_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("probabilities", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("provider", sa.String(length=16), nullable=False),
        sa.Column("model_id", sa.String(length=128), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("validation_status", sa.String(length=16), nullable=False),
        sa.Column("fallback", sa.Boolean(), nullable=False),
        sa.Column("fallback_reason", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('choice', 'score', 'noul')", name=op.f("ck_turn_decisions_kind_valid")
        ),
        sa.CheckConstraint(
            "validation_status IN ('valid', 'invalid', 'missing')",
            name=op.f("ck_turn_decisions_validation_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_turn_decisions_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_message_id"],
            ["messages.id"],
            name=op.f("fk_turn_decisions_user_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_turn_decisions")),
        sa.UniqueConstraint(
            "user_message_id", "question_id", name="uq_turn_decisions_message_question"
        ),
    )
    op.create_index(
        op.f("ix_turn_decisions_conversation_id"),
        "turn_decisions",
        ["conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_turn_decisions_conversation_id"), table_name="turn_decisions")
    op.drop_table("turn_decisions")
    op.drop_table("conversation_states")
