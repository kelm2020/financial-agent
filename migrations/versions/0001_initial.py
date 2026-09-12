"""Initial Postgres, pgvector, conversations, knowledge and audit schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.Uuid(), primary_key=True),
        sa.Column("customer_id", sa.String(length=32), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("channel", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "last_turn_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint("channel IN ('chat', 'voice')", name="ck_conversations_channel"),
    )
    op.create_index(
        "ix_conversations_customer_last_turn",
        "conversations",
        ["customer_id", "last_turn_at"],
    )

    op.create_table(
        "kb_chunks",
        sa.Column("chunk_id", sa.String(length=128), primary_key=True),
        sa.Column("document_id", sa.String(length=128), nullable=False),
        sa.Column("topic", sa.String(length=64), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("policy_version", sa.String(length=32), nullable=False),
        sa.Column("valid_from", sa.Date(), nullable=False),
        sa.Column("valid_until", sa.Date(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column(
            "content_search",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('spanish', coalesce(heading, '') || ' ' || content)", persisted=True
            ),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    # The vector dimension and HNSW index are set by F2 when the embedding model is selected.
    op.create_index(
        "ix_kb_chunks_content_search", "kb_chunks", ["content_search"], postgresql_using="gin"
    )
    op.create_index(
        "ix_kb_chunks_topic_validity", "kb_chunks", ["topic", "valid_from", "valid_until"]
    )

    op.create_table(
        "audit_events",
        sa.Column(
            "event_id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")
        ),
        sa.Column("conversation_id", sa.Uuid(), nullable=True),
        sa.Column("customer_ref_hash", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("integrity_hmac", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("kb_chunks")
    op.drop_table("conversations")
