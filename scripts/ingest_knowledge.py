from __future__ import annotations

import asyncio
from datetime import date

from app.rag.factory import embedding_client
from app.rag.ingest import ingest_corpus
from app.rag.store import PgVectorHybridStore
from config.settings import Settings, get_settings


async def run(settings: Settings | None = None, *, effective_on: date | None = None) -> None:
    resolved = settings or get_settings()
    # The CLI edge is the only place allowed to read the wall clock.
    reference = effective_on or date.today()
    async with (
        embedding_client(resolved, allow_network=True) as embeddings,
        PgVectorHybridStore(resolved.database_url) as store,
    ):
        metadata = await ingest_corpus(store, embeddings, effective_on=reference)
    print(
        f"Indexed {metadata.chunk_count} chunks with {metadata.embedding_model} "
        f"({metadata.corpus_sha256[:12]})"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
