"""turn_safety: the safety layer's record for each completed turn

Labels, statuses and rule ids only, never message text (spec §12, §19.2).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15 11:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "turn_safety",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("user_message_id", sa.Uuid(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("pathway", sa.String(length=32), nullable=False),
        sa.Column("assessed", sa.Boolean(), nullable=False),
        sa.Column("degraded", sa.Boolean(), nullable=False),
        sa.Column("classifier_status", sa.String(length=16), nullable=False),
        sa.Column("classifier_model", sa.String(length=128), nullable=True),
        sa.Column("classifier_category", sa.String(length=32), nullable=True),
        sa.Column("pattern_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("decision_signal", sa.Boolean(), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("patterns_version", sa.String(length=32), nullable=False),
        sa.Column("region", sa.String(length=8), nullable=False),
        sa.Column("output_violations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_check_status", sa.String(length=16), nullable=True),
        sa.Column("recovery", sa.String(length=16), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "recovery IN ('none', 'regenerated', 'fallback')",
            name=op.f("ck_turn_safety_recovery_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_turn_safety_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_message_id"],
            ["messages.id"],
            name=op.f("fk_turn_safety_user_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_turn_safety")),
        sa.UniqueConstraint("user_message_id", name=op.f("uq_turn_safety_user_message_id")),
    )
    op.create_index(
        op.f("ix_turn_safety_conversation_id"), "turn_safety", ["conversation_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_turn_safety_conversation_id"), table_name="turn_safety")
    op.drop_table("turn_safety")
