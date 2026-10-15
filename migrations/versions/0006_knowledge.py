"""knowledge_documents and knowledge_chunks (Phase 4, RAG)

Reviewed sources with provenance and licence, one row per version; chunks
with a generated English full-text vector and an optional embedding.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15 12:30:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_key", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("publisher", sa.String(length=128), nullable=False),
        sa.Column("licence", sa.String(length=128), nullable=False),
        sa.Column("licence_url", sa.Text(), nullable=True),
        sa.Column("attribution", sa.Text(), nullable=True),
        sa.Column("reviewed_on", sa.Date(), nullable=True),
        sa.Column("topics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("review_status", sa.String(length=16), nullable=False),
        sa.Column("content_checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "review_status IN ('approved', 'retired')",
            name=op.f("ck_knowledge_documents_review_status_valid"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_documents")),
        sa.UniqueConstraint("source_key", "version", name="uq_knowledge_documents_source_version"),
    )
    op.create_index(
        "uq_knowledge_documents_one_current",
        "knowledge_documents",
        ["source_key"],
        unique=True,
        postgresql_where=sa.text("review_status = 'approved'"),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_checksum", sa.String(length=64), nullable=False),
        sa.Column(
            "search",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', heading || ' ' || text)", persisted=True),
            nullable=True,
        ),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.CheckConstraint(
            "(embedding IS NULL) = (embedding_model IS NULL)",
            name=op.f("ck_knowledge_chunks_embedding_has_model"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["knowledge_documents.id"],
            name=op.f("fk_knowledge_chunks_document_id_knowledge_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunks")),
        sa.UniqueConstraint("document_id", "ordinal", name="uq_knowledge_chunks_document_ordinal"),
    )
    op.create_index(
        op.f("ix_knowledge_chunks_document_id"), "knowledge_chunks", ["document_id"], unique=False
    )
    op.create_index(
        "ix_knowledge_chunks_search", "knowledge_chunks", ["search"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_search", table_name="knowledge_chunks")
    op.drop_index(op.f("ix_knowledge_chunks_document_id"), table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_index("uq_knowledge_documents_one_current", table_name="knowledge_documents")
    op.drop_table("knowledge_documents")
