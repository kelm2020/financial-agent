"""Application-computed Spanish lexemes, retrieval parity and stricter kb columns.

`content_search` stops being a generated column. Postgres' `spanish` configuration neither
folds accents ("acreditación" ≠ "acreditacion") nor stems the unaccented forms, so lexemes are
computed once in Python (accent folding + Snowball Spanish, `app.rag.text.tokenize`) and stored
with the `simple` configuration. Both stores therefore match on exactly the same tokens.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_kb_lexical_parity"
down_revision: str | None = "0002_phase2_knowledge"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TOPICS = "('negociacion', 'medios_pago', 'escalamiento', 'faq')"


def upgrade() -> None:
    # The index is derived data and its lexemes cannot be computed in SQL: clear it so the
    # application refuses to start until `make ingest` rebuilds it (assert_index_current).
    op.execute("DELETE FROM kb_chunks")
    op.execute("DELETE FROM kb_index_metadata")

    op.drop_index("ix_kb_chunks_content_search", table_name="kb_chunks")
    op.drop_column("kb_chunks", "content_search")
    op.add_column("kb_chunks", sa.Column("content_search", postgresql.TSVECTOR(), nullable=False))
    op.create_index(
        "ix_kb_chunks_content_search", "kb_chunks", ["content_search"], postgresql_using="gin"
    )

    op.add_column(
        "kb_chunks",
        sa.Column(
            "applicable_segments",
            postgresql.ARRAY(sa.String(length=32)),
            nullable=False,
            server_default=sa.text("ARRAY[]::varchar[]"),
        ),
    )
    for column in ("section_id", "contextualized_content", "status", "audience",
                   "applicable_segments"):  # fmt: skip
        op.alter_column("kb_chunks", column, server_default=None)
    op.create_check_constraint("ck_kb_chunks_section_id", "kb_chunks", "section_id <> ''")
    op.create_check_constraint(
        "ck_kb_chunks_contextualized_content", "kb_chunks", "contextualized_content <> ''"
    )
    op.create_check_constraint("ck_kb_chunks_topic", "kb_chunks", f"topic IN {_TOPICS}")


def downgrade() -> None:
    op.execute("DELETE FROM kb_chunks")
    op.execute("DELETE FROM kb_index_metadata")
    op.drop_constraint("ck_kb_chunks_topic", "kb_chunks", type_="check")
    op.drop_constraint("ck_kb_chunks_contextualized_content", "kb_chunks", type_="check")
    op.drop_constraint("ck_kb_chunks_section_id", "kb_chunks", type_="check")
    op.alter_column("kb_chunks", "section_id", server_default="")
    op.alter_column("kb_chunks", "contextualized_content", server_default="")
    op.alter_column("kb_chunks", "status", server_default="approved")
    op.alter_column("kb_chunks", "audience", server_default=sa.text("ARRAY[]::varchar[]"))
    op.drop_column("kb_chunks", "applicable_segments")
    op.drop_index("ix_kb_chunks_content_search", table_name="kb_chunks")
    op.drop_column("kb_chunks", "content_search")
    op.add_column(
        "kb_chunks",
        sa.Column(
            "content_search",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('spanish', coalesce(heading, '') || ' ' || content)", persisted=True
            ),
        ),
    )
    op.create_index(
        "ix_kb_chunks_content_search", "kb_chunks", ["content_search"], postgresql_using="gin"
    )
