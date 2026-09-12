"""Complete the Phase 2 knowledge schema and add the HNSW index."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_phase2_knowledge"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "kb_chunks",
        sa.Column("section_id", sa.String(length=64), nullable=False, server_default=""),
    )
    op.add_column(
        "kb_chunks",
        sa.Column("contextualized_content", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "kb_chunks",
        sa.Column("status", sa.String(length=16), nullable=False, server_default="approved"),
    )
    op.add_column(
        "kb_chunks",
        sa.Column(
            "audience",
            postgresql.ARRAY(sa.String(length=32)),
            nullable=False,
            server_default=sa.text("ARRAY[]::varchar[]"),
        ),
    )
    op.create_check_constraint(
        "ck_kb_chunks_status",
        "kb_chunks",
        "status IN ('approved', 'draft', 'retired')",
    )
    op.create_index("ix_kb_chunks_section_id", "kb_chunks", ["section_id"])
    op.execute(
        """
        CREATE INDEX ix_kb_chunks_embedding_hnsw
        ON kb_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_table(
        "kb_index_metadata",
        sa.Column("index_name", sa.String(length=64), primary_key=True),
        sa.Column("kb_version", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("corpus_sha256", sa.String(length=64), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.CheckConstraint("embedding_dimensions > 0", name="ck_kb_embedding_dimensions"),
        sa.CheckConstraint("chunk_count >= 0", name="ck_kb_chunk_count"),
    )


def downgrade() -> None:
    op.drop_table("kb_index_metadata")
    op.drop_index("ix_kb_chunks_embedding_hnsw", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_section_id", table_name="kb_chunks")
    op.drop_constraint("ck_kb_chunks_status", "kb_chunks", type_="check")
    op.drop_column("kb_chunks", "audience")
    op.drop_column("kb_chunks", "status")
    op.drop_column("kb_chunks", "contextualized_content")
    op.drop_column("kb_chunks", "section_id")
