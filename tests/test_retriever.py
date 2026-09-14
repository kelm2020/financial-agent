from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

import app.rag.text as text_module
from app.rag.config import load_embedding_config, load_reranker_config
from app.rag.corpus import (
    applicable_segments,
    chunk_document,
    corpus_sha256,
    load_corpus,
    parse_document,
)
from app.rag.embeddings import (
    CachedEmbeddingClient,
    EmbeddingCacheError,
    HashingEmbeddingClient,
    OpenAIEmbeddingClient,
)
from app.rag.evaluation import (
    Calibration,
    RetrievalDataset,
    calibrate_min_dense_score,
    calibrate_min_rerank_score,
    evaluate_retriever,
    load_retrieval_dataset,
)
from app.rag.factory import (
    RERANK_CACHE_PATH,
    build_retriever,
    embedding_client,
    reranker_client,
)
from app.rag.ingest import assert_index_current, ingest_corpus
from app.rag.models import IndexMetadata, KnowledgeDocument, SearchHit
from app.rag.rerank import CachedReranker, CohereReranker, RerankCacheError
from app.rag.retriever import PolicyRetriever, dense_evidence, no_evidence_action
from app.rag.store import InMemoryHybridStore, _bm25_scores, _row_to_hit, lexical_query
from app.rag.text import cosine, hashing_embedding, tokenize, words
from scripts.build_embedding_cache import cache_texts
from tests.rag_support import (
    REFERENCE_DATE,
    FixedScoreReranker,
    NamedHashingEmbeddingClient,
    OracleReranker,
    cached_embeddings,
    hit,
    offline_settings,
)

ROOT = Path(__file__).parents[1]
KB = ROOT / "kb"


def corpus() -> list:  # type: ignore[type-arg]
    return load_corpus(effective_on=REFERENCE_DATE)


@pytest.fixture
async def hashing_index() -> tuple[InMemoryHybridStore, HashingEmbeddingClient]:
    store = InMemoryHybridStore()
    embeddings = HashingEmbeddingClient(dimensions=64)
    await ingest_corpus(store, embeddings, effective_on=REFERENCE_DATE)
    return store, embeddings


@pytest.fixture
async def semantic_index() -> tuple[InMemoryHybridStore, CachedEmbeddingClient]:
    store = InMemoryHybridStore()
    embeddings = cached_embeddings()
    await ingest_corpus(store, embeddings, effective_on=REFERENCE_DATE)
    return store, embeddings


# -------------------------------------------------------------------------------- corpus


def test_canonical_corpus_has_35_stable_sections() -> None:
    chunks = corpus()
    assert len(chunks) == 35
    assert len({chunk.chunk_id for chunk in chunks}) == 35
    assert {chunk.document_id for chunk in chunks} == {"POL-NEG", "PAY-MET", "ESC", "FAQ"}
    assert (chunks[0].section_id, chunks[-1].section_id) == ("POL-NEG-001", "FAQ-015")


def test_chunks_carry_segment_aware_context() -> None:
    by_section = {chunk.section_id: chunk for chunk in corpus()}
    discounts = by_section["POL-NEG-003"]
    assert discounts.applicable_segments == (
        "mora_temprana",
        "mora_media",
        "mora_tardia",
        "prejudicial",
    )
    assert "«Quitas de interés por segmento»" in discounts.contextualized_content
    assert "mora tardía" in discounts.contextualized_content
    assert by_section["FAQ-001"].applicable_segments == ()
    assert by_section["FAQ-001"].contextualized_content.endswith("aplica a todos los clientes.")
    assert applicable_segments("Casos PREJUDICIALES y mora tardía") == (
        "mora_tardia",
        "prejudicial",
    )


def test_corpus_date_is_explicit() -> None:
    with pytest.raises(TypeError):
        load_corpus()  # type: ignore[call-arg]


def test_parse_document_rejects_bad_frontmatter(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    missing.write_text("## DOC-001 · Título\nTexto", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter"):
        parse_document(missing)
    malformed = tmp_path / "malformed.md"
    malformed.write_text("---\n[]\n---\n## DOC-001 · Título\nTexto", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata"):
        parse_document(malformed)
    unclosed = tmp_path / "unclosed.md"
    unclosed.write_text("---\ndoc_id: FAQ\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front matter inválido"):
        parse_document(unclosed)


def test_chunker_rejects_unknown_documents_missing_and_malformed_headings() -> None:
    unknown = KnowledgeDocument(
        doc_id="OTHER",
        titulo="Otro",
        version="1",
        status="approved",
        effective_from=date(2026, 1, 1),
        effective_to=None,
        audiencia=("agente",),
    )
    with pytest.raises(ValueError, match="topic"):
        chunk_document(unknown, "## OTHER-001 · Texto\nContenido")
    document, _ = parse_document(KB / "04-faq.md")
    with pytest.raises(ValueError, match="secciones"):
        chunk_document(document, "Sin headings")
    with pytest.raises(ValueError, match="encabezado ## inválido"):
        chunk_document(document, "## FAQ-001 · Bien\nTexto\n## FAQ-002 sin separador\nTexto")


def test_large_section_splits_with_overlap_and_stable_part_ids() -> None:
    document, _ = parse_document(KB / "04-faq.md")
    body = "## FAQ-999 · Muy larga\n" + " ".join(f"palabra{index}" for index in range(20))
    chunks = chunk_document(document, body, max_words=8, overlap=2)
    assert [chunk.chunk_id.rsplit("-", 1)[-1] for chunk in chunks] == ["1", "2", "3"]
    assert chunks[0].content.split()[-2:] == chunks[1].content.split()[:2]
    assert chunks[-1].content.split()[-1] == "palabra19"


def test_corpus_filters_draft_future_and_expired_documents(tmp_path: Path) -> None:
    template = """---
doc_id: FAQ
titulo: Test
version: "1.0"
status: {status}
effective_from: {start}
effective_to: {end}
audiencia: [agente]
---
## FAQ-001 · Test
Contenido.
"""
    for name, status, start, end in (
        ("draft", "draft", "2026-01-01", "null"),
        ("future", "approved", "2027-01-01", "null"),
        ("expired", "approved", "2025-01-01", "2025-12-31"),
    ):
        (tmp_path / f"{name}.md").write_text(
            template.format(status=status, start=start, end=end), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="vacío"):
        load_corpus(tmp_path, effective_on=REFERENCE_DATE)


def test_corpus_hash_is_stable_and_content_sensitive(tmp_path: Path) -> None:
    first, second = tmp_path / "a.md", tmp_path / "b.md"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    before = corpus_sha256([second, first])
    assert before == corpus_sha256([first, second])
    second.write_text("changed", encoding="utf-8")
    assert corpus_sha256([first, second]) != before


# --------------------------------------------------------------------------- text / lexical


def test_lexical_normalization_has_no_query_tuned_vocabulary() -> None:
    # Regression guard for the F2 finding: the old synonym table and content-word stopwords
    # were derived from the held-out queries.
    assert not hasattr(text_module, "_SYNONYMS")
    leaked = {"verdad", "hacen", "tengo", "algo", "tienen", "real", "llego", "junto",
              "entrada", "tarda", "descuento", "alguien", "cuantas", "pagar"}  # fmt: skip
    assert not leaked & text_module._STOPWORDS
    assert tokenize("¿Cuánto TARDA la acreditación?") == tokenize("cuanto tarda la acreditacion")
    assert tokenize("vencimientos") == tokenize("vencimiento")
    assert tokenize("no llego a pagar") == ("no", "lleg", "pag")


def test_lexical_query_is_stemmed_accent_folded_and_tsquery_safe() -> None:
    assert lexical_query("¿Cuánto tarda la acreditación? 'x' & (y) | ! se acreditan") == (
        "cuant | tard | acredit"
    )
    assert lexical_query("¿?") == ""
    assert words("Débito—automático") == ("debito", "automatico")
    assert tokenize("acreditación") == tokenize("acreditacion") == tokenize("se acreditan")


def test_low_level_scoring_edges() -> None:
    assert _bm25_scores((), [("documento",)]) == [0.0]
    assert _bm25_scores(("consulta",), ()) == []
    assert hashing_embedding("", dimensions=3) == (0.0, 0.0, 0.0)
    assert cosine((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert cosine((2.0, 0.0), (0.5, 0.5)) == pytest.approx(0.7071067811865475)  # not a dot product
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


# ------------------------------------------------------------------------------- store


async def test_in_memory_store_rejects_shape_mismatches() -> None:
    chunk = corpus()[0]
    metadata = IndexMetadata(
        kb_version="1",
        embedding_model="test",
        embedding_dimensions=2,
        corpus_sha256="a" * 64,
        chunk_count=1,
    )
    store = InMemoryHybridStore()
    with pytest.raises(ValueError, match="exactamente"):
        await store.replace_index([chunk], [], metadata)
    with pytest.raises(ValueError, match="chunk_count"):
        await store.replace_index([], [], metadata)
    with pytest.raises(ValueError, match="dimensión"):
        await store.replace_index([chunk], [[1.0]], metadata)


async def test_store_filters_topic_and_validity(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    vector = (await embeddings.embed(["plazos de acreditación"]))[0]
    hits = await store.search(
        "plazos de acreditación", vector, topic="medios_pago",
        effective_on=REFERENCE_DATE, limit=10, candidate_limit=20, rrf_k=60,
    )  # fmt: skip
    assert hits
    assert all(hit.chunk.topic == "medios_pago" for hit in hits)
    before_validity = await store.search(
        "plazos", vector, topic="any", effective_on=date(2025, 1, 1),
        limit=10, candidate_limit=20, rrf_k=60,
    )  # fmt: skip
    assert before_validity == []


async def test_dense_score_is_real_for_lexical_only_candidates(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    vector = (await embeddings.embed(["débito automático"]))[0]
    hits = await store.search(
        "débito automático", vector, topic="any", effective_on=REFERENCE_DATE,
        limit=35, candidate_limit=1, rrf_k=60,
    )  # fmt: skip
    lexical_only = [hit for hit in hits if hit.dense_rank is None]
    assert lexical_only
    assert all(hit.dense_score != 0 for hit in lexical_only)


def test_postgres_row_mapping_preserves_citations_and_clamps_scores() -> None:
    chunk = corpus()[2]
    row = {
        **chunk.model_dump(mode="python"),
        "audience": list(chunk.audience),
        "applicable_segments": list(chunk.applicable_segments),
        "lexical_score": 1,
        "dense_score": "1.0000002",
        "lexical_rank": 1,
        "dense_rank": None,
        "rrf_score": 0.5,
    }
    mapped = _row_to_hit(row)
    assert mapped.chunk == chunk
    assert (mapped.lexical_rank, mapped.dense_rank, mapped.dense_score) == (1, None, 1.0)


# ------------------------------------------------------------------------------ ingest


async def test_ingest_records_verifiable_metadata(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    metadata = await assert_index_current(store, embeddings)
    assert (metadata.chunk_count, metadata.embedding_dimensions) == (35, 64)
    assert metadata.embedding_model == "local-hashing"


async def test_index_current_detects_each_drift(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient], tmp_path: Path
) -> None:
    store, embeddings = hashing_index
    with pytest.raises(RuntimeError, match="no fue indexada"):
        await assert_index_current(InMemoryHybridStore(), embeddings)
    copied = tmp_path / "kb"
    copied.mkdir()
    for source in KB.glob("*.md"):
        (copied / source.name).write_bytes(source.read_bytes())
    (copied / "04-faq.md").write_text("cambio", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corpus"):
        await assert_index_current(store, embeddings, root=copied)
    with pytest.raises(RuntimeError, match="modelo"):
        await assert_index_current(store, NamedHashingEmbeddingClient("otro-modelo", 64))
    with pytest.raises(RuntimeError, match="dimensión"):
        await assert_index_current(store, NamedHashingEmbeddingClient("local-hashing", 8))


# --------------------------------------------------------------------------- retriever


def _retriever(
    store: InMemoryHybridStore, embeddings: HashingEmbeddingClient, **kwargs: object
) -> PolicyRetriever:
    options: dict[str, object] = {"min_rrf_score": 0.35, "min_dense_score": -1.0, **kwargs}
    return PolicyRetriever(store, embeddings, **options)  # type: ignore[arg-type]


async def test_retrieval_returns_citations_and_evidence(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    result = await _retriever(store, embeddings).search(
        "plazos de acreditación de la transferencia",
        topic="medios_pago",
        effective_on=REFERENCE_DATE,
    )
    assert result.status == "ok"
    assert result.source_chunk_ids[0] == "PAY-MET-002"
    assert set(result.source_chunk_ids) == {hit.chunk.section_id for hit in result.hits}
    assert result.evidence_score == dense_evidence(result.hits)
    assert result.on_no_evidence is None


async def test_empty_query_and_empty_store_abstain_with_action(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    empty_query = await _retriever(store, embeddings).search(
        "   ", topic="faq", effective_on=REFERENCE_DATE
    )
    empty_store = await _retriever(InMemoryHybridStore(), embeddings).search(
        "anticipo", topic="negociacion", effective_on=REFERENCE_DATE
    )
    assert (empty_query.status, empty_query.on_no_evidence) == ("no_evidence", "ofrecer_derivacion")
    assert (empty_store.status, empty_store.on_no_evidence) == ("no_evidence", "derivar")
    assert empty_store.evidence_score is None


@pytest.mark.parametrize(
    ("options", "reranker"),
    [
        ({"min_rrf_score": 1.1}, None),
        ({"min_dense_score": 1.1}, None),
        ({"min_rerank_score": 0.9, "evidence_gate": "rerank"}, FixedScoreReranker(0.2)),
    ],
)
async def test_each_gate_forces_abstention_and_keeps_evidence_for_traces(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
    options: dict[str, object],
    reranker: FixedScoreReranker | None,
) -> None:
    store, embeddings = hashing_index
    result = await _retriever(store, embeddings, reranker=reranker, **options).search(
        "anticipo", topic="negociacion", effective_on=REFERENCE_DATE
    )
    assert (result.status, result.hits, result.on_no_evidence) == ("no_evidence", (), "derivar")
    assert result.evidence_score is not None


async def test_reranker_sees_all_candidates_and_truncates_to_limit(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    reranker = FixedScoreReranker(0.8)
    result = await _retriever(store, embeddings, reranker=reranker).search(
        "cuotas", topic="any", effective_on=REFERENCE_DATE, limit=2
    )
    assert reranker.calls == 1
    assert result.status == "ok"
    assert len(result.hits) == 2
    assert all(hit.rerank_score == 0.8 for hit in result.hits)


async def test_generation_retrieval_prefers_recall_and_keeps_grounding_evidence(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    retriever = _retriever(store, embeddings, min_rrf_score=2.0, min_dense_score=2.0)

    gated = await retriever.search("anticipo", topic="negociacion", effective_on=REFERENCE_DATE)
    generated = await retriever.search_for_generation(
        "anticipo", topic="negociacion", effective_on=REFERENCE_DATE
    )
    empty = await retriever.search_for_generation(
        " ", topic="negociacion", effective_on=REFERENCE_DATE
    )
    no_documents = await _retriever(InMemoryHybridStore(), embeddings).search_for_generation(
        "anticipo", topic="negociacion", effective_on=REFERENCE_DATE
    )

    assert gated.status == "no_evidence"
    assert generated.status == "ok"
    assert generated.hits and generated.source_chunk_ids
    assert generated.evidence_score is not None
    assert (empty.status, no_documents.status) == ("no_evidence", "no_evidence")


async def test_generation_retrieval_reports_reranker_score(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    retriever = _retriever(
        store,
        embeddings,
        reranker=FixedScoreReranker(0.12),
        evidence_gate="rerank",
        min_rerank_score=0.99,
    )

    result = await retriever.search_for_generation(
        "anticipo", topic="negociacion", effective_on=REFERENCE_DATE
    )

    assert result.status == "ok"
    assert result.evidence_score == 0.12


def test_no_evidence_action_is_graded_by_risk() -> None:
    faq_hit = hit(corpus()[-1])
    policy_hit = hit(corpus()[0])
    assert no_evidence_action("negociacion", []) == "derivar"
    assert no_evidence_action("escalamiento", []) == "derivar"
    assert no_evidence_action("faq", []) == "ofrecer_derivacion"
    assert no_evidence_action("medios_pago", []) == "ofrecer_derivacion"
    assert no_evidence_action("any", []) == "derivar"
    assert no_evidence_action("any", [faq_hit]) == "ofrecer_derivacion"
    assert no_evidence_action("any", [policy_hit]) == "derivar"
    assert dense_evidence([]) == -1.0
    assert dense_evidence([hit(corpus()[0], dense=0.2), hit(corpus()[1], dense=0.7)]) == 0.7


# ------------------------------------------------------------------ embeddings and cache


async def test_embedding_cache_persists_rounds_and_reuses_vectors(tmp_path: Path) -> None:
    class CountingEmbeddings(HashingEmbeddingClient):
        calls = 0

        async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
            type(self).calls += 1
            return [
                tuple(value + 1e-9 for value in vector) for vector in await super().embed(texts)
            ]

    path = tmp_path / "cache.json"
    cached = CachedEmbeddingClient(CountingEmbeddings(8), path, precision=4)
    first = await cached.embed(["anticipo", "cuotas", "anticipo"])
    assert first[0] == first[2]
    assert all(value == round(value, 4) for value in first[0])
    assert await cached.embed(["cuotas"]) == first[1:2]
    assert CountingEmbeddings.calls == 1

    offline = CachedEmbeddingClient(None, path, model_name="local-hashing", dimensions=8)
    assert await offline.embed(["cuotas", "anticipo"]) == [first[1], first[0]]
    with pytest.raises(EmbeddingCacheError, match="Faltan 1 embeddings"):
        await offline.embed(["texto nuevo"])


async def test_embedding_cache_retain_prunes_unused_entries(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cached = CachedEmbeddingClient(HashingEmbeddingClient(8), path)
    await cached.embed(["viejo", "vigente"])
    assert await cached.retain(["vigente"]) == 1
    fresh = CachedEmbeddingClient(None, path, model_name="local-hashing", dimensions=8)
    assert await fresh.retain(["vigente"]) == 0
    stored = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert list(stored) == [hashlib.sha256(b"vigente").hexdigest()]


def test_offline_cache_needs_model_and_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="offline"):
        CachedEmbeddingClient(None, tmp_path / "cache.json")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "JSON válido"),
        ("[]", "formato"),
        ('{"model":"local-hashing","dimensions":8,"entries":[]}', "formato"),
        ('{"model":"other","dimensions":8,"entries":{}}', "generado con other/8"),
        ('{"model":"local-hashing","dimensions":4,"entries":{}}', "generado con"),
        ('{"model":"local-hashing","dimensions":8,"entries":{"k":[0.1]}}', "otra dimensión"),
    ],
)
async def test_embedding_cache_never_silently_discards_a_bad_file(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "cache.json"
    path.write_text(content, encoding="utf-8")
    cached = CachedEmbeddingClient(HashingEmbeddingClient(8), path)
    with pytest.raises(EmbeddingCacheError, match=message):
        await cached.embed(["consulta"])
    assert path.read_text(encoding="utf-8") == content


async def test_openai_embedding_adapter_batches_orders_and_validates() -> None:
    requests: list[list[str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert (payload["model"], payload["dimensions"]) == ("embed", 2)
        assert request.headers["Authorization"] == "Bearer secret"
        requests.append(payload["input"])
        data = [
            {"index": index, "embedding": [float(index), 1.0]}
            for index in range(len(payload["input"]))
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    embeddings = OpenAIEmbeddingClient(
        api_key="secret", model="embed", dimensions=2, client=client, batch_size=2
    )
    assert await embeddings.embed(["a", "b", "c"]) == [(0.0, 1.0), (1.0, 1.0), (0.0, 1.0)]
    assert requests == [["a", "b"], ["c"]]
    await embeddings.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_openai_embedding_adapter_rejects_bad_shape_and_closes_owned_client() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    embeddings = OpenAIEmbeddingClient(api_key="secret", model="embed", dimensions=2, client=client)
    with pytest.raises(ValueError, match="forma"):
        await embeddings.embed(["a"])
    await client.aclose()
    owned = OpenAIEmbeddingClient(api_key="secret", model="embed", dimensions=2)
    await owned.aclose()
    assert owned._client.is_closed


# ------------------------------------------------------------------------------ reranker


async def test_cohere_reranker_reorders_by_relevance_and_clamps() -> None:
    chunks = corpus()[:3]

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v2/rerank"
        assert (payload["model"], payload["query"], payload["top_n"]) == ("rerank-x", "q", 3)
        assert payload["documents"] == [chunk.embedding_text for chunk in chunks]
        return httpx.Response(
            200,
            json={"results": [
                {"index": 2, "relevance_score": 1.2},
                {"index": 0, "relevance_score": 0.4},
                {"index": 1, "relevance_score": -0.1},
            ]},
        )  # fmt: skip

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    reranker = CohereReranker(api_key="k", model="rerank-x", client=client)
    reranked = await reranker.rerank("q", [hit(chunk) for chunk in chunks])
    assert [item.chunk.section_id for item in reranked] == [
        "POL-NEG-003",
        "POL-NEG-001",
        "POL-NEG-002",
    ]
    assert [item.rerank_score for item in reranked] == [1.0, 0.4, 0.0]
    assert await reranker.rerank("q", []) == []
    await reranker.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_cohere_reranker_retries_rate_limits_then_gives_up() -> None:
    chunks = corpus()[:1]
    statuses = iter([429, 429, 200])
    waits: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 429:
            headers = {"Retry-After": "7"} if not waits else {"Retry-After": "soon"}
            return httpx.Response(429, headers=headers)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.3}]})

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    reranker = CohereReranker(api_key="k", model="m", client=client, sleep=fake_sleep)
    assert [h.rerank_score for h in await reranker.rerank("q", [hit(chunks[0])])] == [0.3]
    assert waits == [7.0, 4.0]  # Retry-After honoured; exponential fallback when unparsable

    async def always_limited(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    limited = CohereReranker(
        api_key="k",
        model="m",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(always_limited), base_url="https://t"
        ),
        max_retries=2,
        sleep=fake_sleep,
    )
    with pytest.raises(httpx.HTTPStatusError):
        await limited.rerank("q", [hit(chunks[0])])
    assert waits[2:] == [2.0, 4.0]
    await client.aclose()


async def test_cohere_reranker_rejects_incomplete_results_and_closes_owned_client() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://test")
    reranker = CohereReranker(api_key="k", model="m", client=client)
    with pytest.raises(ValueError, match="inesperado"):
        await reranker.rerank("q", [hit(chunk) for chunk in corpus()[:2]])
    await client.aclose()
    owned = CohereReranker(api_key="k", model="m")
    assert owned.model_name == "m"
    await owned.aclose()
    assert owned._client.is_closed


# ------------------------------------------------------------------------ config/factory


def test_model_ids_come_from_models_yaml(tmp_path: Path) -> None:
    assert load_embedding_config().model == "text-embedding-3-large"
    assert load_embedding_config().dimensions == 1536
    assert load_reranker_config().provider == "cohere"
    path = tmp_path / "models.yaml"
    path.write_text("tiers: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"tiers\.embedding"):
        load_embedding_config(path)


async def test_factory_is_offline_without_key_or_permission() -> None:
    async with embedding_client(offline_settings(), allow_network=True) as offline:
        assert offline._delegate is None
    keyed = offline_settings(openai_api_key=SecretStr("sk"))
    async with embedding_client(keyed, allow_network=False) as forbidden:
        assert forbidden._delegate is None
    async with embedding_client(keyed, allow_network=True) as live:
        assert isinstance(live._delegate, OpenAIEmbeddingClient)
    assert live._delegate._client.is_closed


async def test_factory_reranker_is_live_only_with_key_and_permission(tmp_path: Path) -> None:
    cache = tmp_path / "rerank_cache.json"
    keyed = offline_settings(cohere_api_key=SecretStr("k"))
    async with reranker_client(offline_settings(), allow_network=True, cache_path=cache) as none:
        assert none is None  # no key and no committed cache: nothing to measure
    async with reranker_client(keyed, allow_network=True, cache_path=cache) as live:
        assert live is not None and not live.is_offline
        assert live.model_name == load_reranker_config().model
        delegate = live._delegate
    assert isinstance(delegate, CohereReranker) and delegate._client.is_closed
    cache.write_text('{"model": "rerank-v3.5", "entries": {}}', encoding="utf-8")
    async with reranker_client(keyed, allow_network=False, cache_path=cache) as offline:
        assert offline is not None and offline.is_offline


def test_retriever_thresholds_come_from_settings(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, _ = hashing_index
    settings = offline_settings(
        rag_min_rrf_score=0.4, rag_min_dense_score=0.6, rag_min_rerank_score=0.7
    )
    retriever = build_retriever(store, cached_embeddings(), settings)
    assert retriever._evidence_gate == "dense"
    rerank_settings = offline_settings(rag_evidence_gate="rerank")
    assert build_retriever(store, cached_embeddings(), rerank_settings)._evidence_gate == "dense"
    with_reranker = build_retriever(
        store, cached_embeddings(), rerank_settings, reranker=FixedScoreReranker(0.1)
    )
    assert with_reranker._evidence_gate == "rerank"
    assert (retriever._min_rrf_score, retriever._min_dense_score, retriever._min_rerank_score) == (
        0.4,
        0.6,
        0.7,
    )
    assert (
        build_retriever(store, cached_embeddings(), settings, min_dense_score=-1.0)._min_dense_score
        == -1.0
    )


# ------------------------------------------------------------ datasets and measured quality


def test_splits_are_separate_and_test_is_anexo_e1() -> None:
    dev, test = load_retrieval_dataset("dev"), load_retrieval_dataset("test")
    assert (len(test.positive), len(test.negative)) == (6, 3)
    assert test.positive[0].expected_section_ids == ("POL-NEG-004",)

    def queries(dataset: RetrievalDataset) -> set[str]:
        return {case.query for case in dataset.positive} | {case.query for case in dataset.negative}

    assert not queries(dev) & queries(test)
    covered = {section for case in dev.positive for section in case.expected_section_ids}
    assert len(covered) >= 30


def test_dataset_contract_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split=test"):
        load_retrieval_dataset("dev", ROOT / "evals" / "retrieval.yaml")
    base = {"split": "dev", "effective_on": "2026-09-12",
            "negative": [{"id": "N", "query": "q", "topic": "any"}]}  # fmt: skip
    with pytest.raises(ValidationError, match="únicos"):
        RetrievalDataset.model_validate(
            {
                **base,
                "positive": [
                    {"id": "N", "query": "q", "topic": "faq", "expected_section_ids": ["FAQ-001"]}
                ],
            }
        )
    with pytest.raises(ValidationError):
        RetrievalDataset.model_validate(
            {
                **base,
                "positive": [{"id": "P", "query": "q", "topic": "faq", "expected_section_ids": []}],
            }
        )


async def test_committed_cache_covers_corpus_and_both_splits() -> None:
    vectors = await cached_embeddings().embed(cache_texts())
    assert len(vectors) == len(cache_texts())


async def test_measured_retrieval_quality_is_reproducible_offline(
    semantic_index: tuple[InMemoryHybridStore, CachedEmbeddingClient],
) -> None:
    """Pins the numbers published in the README; changing them must be deliberate."""
    store, embeddings = semantic_index
    retriever = build_retriever(store, embeddings, offline_settings())
    ranking = build_retriever(store, embeddings, offline_settings(), min_dense_score=-1.0)
    test_split, dev_split = load_retrieval_dataset("test"), load_retrieval_dataset("dev")

    gated_test = await evaluate_retriever(retriever, test_split)
    assert (gated_test.recall_at_3, round(gated_test.mrr, 2), gated_test.abstentions) == (
        0.5,
        0.33,
        3,
    )
    assert {case.id for case in gated_test.failures} == {"R-02", "R-03", "R-06"}

    ungated_test = await evaluate_retriever(ranking, test_split)
    assert (round(ungated_test.recall_at_3, 2), round(ungated_test.mrr, 2)) == (0.83, 0.58)
    assert ungated_test.abstentions == 0

    gated_dev = await evaluate_retriever(retriever, dev_split)
    assert (round(gated_dev.recall_at_3, 2), gated_dev.abstentions) == (0.72, 9)


async def test_settings_threshold_is_the_dev_calibration(
    semantic_index: tuple[InMemoryHybridStore, CachedEmbeddingClient],
) -> None:
    store, embeddings = semantic_index
    retriever = build_retriever(store, embeddings, offline_settings(), min_dense_score=-1.0)
    calibration = await calibrate_min_dense_score(retriever, load_retrieval_dataset("dev"))
    assert (calibration.gate, calibration.threshold) == (
        "dense",
        offline_settings().rag_min_dense_score,
    )
    assert (calibration.positives_answered, calibration.negatives_abstained) == (24, 9)
    with pytest.raises(ValueError, match="split dev"):
        await calibrate_min_dense_score(retriever, load_retrieval_dataset("test"))


async def test_rerank_calibration_uses_the_cross_encoder_score_on_dev_only(
    semantic_index: tuple[InMemoryHybridStore, CachedEmbeddingClient],
) -> None:
    store, embeddings = semantic_index
    dev = load_retrieval_dataset("dev")
    oracle = OracleReranker({case.query: case.expected_section_ids for case in dev.positive})
    retriever = build_retriever(
        store, embeddings, offline_settings(), reranker=oracle, evidence_gate="rerank"
    )
    calibration = await calibrate_min_rerank_score(retriever, dev)
    assert (calibration.gate, calibration.threshold) == ("rerank", 0.5)
    assert (calibration.positives_answered, calibration.negatives_abstained) == (32, 10)
    with pytest.raises(ValueError, match="split dev"):
        await calibrate_min_rerank_score(retriever, load_retrieval_dataset("test"))
    without_reranker = build_retriever(store, embeddings, offline_settings())
    with pytest.raises(ValueError, match="requiere un retriever con reranker"):
        await calibrate_min_rerank_score(without_reranker, dev)


async def test_calibration_refuses_scores_without_variation() -> None:
    empty = PolicyRetriever(
        InMemoryHybridStore(), HashingEmbeddingClient(8), min_rrf_score=0.0, min_dense_score=0.0,
        reranker=FixedScoreReranker(0.5),
    )  # fmt: skip
    with pytest.raises(ValueError, match="no varían"):
        await calibrate_min_rerank_score(empty, load_retrieval_dataset("dev"))


async def test_evidence_gate_is_explicit_and_independent_of_reranking(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index
    rerank_gate = await _retriever(
        store, embeddings, reranker=FixedScoreReranker(0.8), min_dense_score=1.1,
        evidence_gate="rerank",
    ).search("cuotas", topic="any", effective_on=REFERENCE_DATE)  # fmt: skip
    assert (rerank_gate.status, rerank_gate.evidence_score) == ("ok", 0.8)
    dense_gate = await _retriever(
        store, embeddings, reranker=FixedScoreReranker(0.8), min_dense_score=1.1
    ).search("cuotas", topic="any", effective_on=REFERENCE_DATE)
    assert dense_gate.status == "no_evidence"  # reranker orders, dense similarity decides
    with pytest.raises(ValueError, match="requiere un reranker"):
        _retriever(store, embeddings, evidence_gate="rerank")


async def test_rrf_gate_uses_fused_ranking_not_the_reranked_top(
    hashing_index: tuple[InMemoryHybridStore, HashingEmbeddingClient],
) -> None:
    store, embeddings = hashing_index

    class PromoteLast(FixedScoreReranker):
        async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
            last = hits[-1].model_copy(update={"rerank_score": 0.9})
            return [last, *(h.model_copy(update={"rerank_score": 0.1}) for h in hits[:-1])]

    retriever = _retriever(
        store, embeddings, reranker=PromoteLast(0.0), min_rrf_score=0.4, evidence_gate="rerank"
    )
    result = await retriever.search("cuotas", topic="any", effective_on=REFERENCE_DATE)
    assert result.status == "ok"
    assert result.hits[0].rrf_score < 0.4  # a low-fused document promoted by the reranker


async def test_cached_reranker_persists_scores_and_fails_offline_on_miss(tmp_path: Path) -> None:
    path = tmp_path / "rerank.json"
    chunks = corpus()[:3]
    delegate = FixedScoreReranker(0.7)
    live = CachedReranker(delegate, path)
    first = await live.rerank("q", [hit(chunk) for chunk in chunks[:2]])
    assert [item.rerank_score for item in first] == [0.7, 0.7]
    await live.rerank("q", [hit(chunk) for chunk in chunks[:2]])
    assert delegate.calls == 1  # second call fully served from cache

    offline = CachedReranker(None, path, model_name="fixed-score")
    assert [i.rerank_score for i in await offline.rerank("q", [hit(c) for c in chunks[:2]])] == [
        0.7,
        0.7,
    ]
    with pytest.raises(RerankCacheError, match="Faltan 1 scores"):
        await offline.rerank("q", [hit(chunks[2])])
    with pytest.raises(RerankCacheError, match="Faltan 2 scores"):
        await offline.rerank("otra consulta", [hit(c) for c in chunks[:2]])
    assert await offline.retain_used() == 0
    fresh = CachedReranker(None, path, model_name="fixed-score")
    assert await fresh.retain_used() == 2
    assert json.loads(path.read_text(encoding="utf-8"))["entries"] == {}


async def test_cached_reranker_orders_by_cached_score(tmp_path: Path) -> None:
    chunks = corpus()[:3]

    class Ascending(FixedScoreReranker):
        async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
            return [h.model_copy(update={"rerank_score": i / 10}) for i, h in enumerate(hits)]

    cached = CachedReranker(Ascending(0.0), tmp_path / "rerank.json")
    reranked = await cached.rerank("q", [hit(chunk) for chunk in chunks])
    assert [item.chunk.section_id for item in reranked] == [
        "POL-NEG-003",
        "POL-NEG-002",
        "POL-NEG-001",
    ]
    assert [item.rerank_score for item in reranked] == [0.2, 0.1, 0.0]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json", "JSON válido"),
        ("[]", "formato"),
        ('{"model":"other","entries":{}}', "generado con other"),
    ],
)
async def test_cached_reranker_never_discards_a_bad_file(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "rerank.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RerankCacheError, match=message):
        await CachedReranker(FixedScoreReranker(0.5), path).rerank("q", [hit(corpus()[0])])
    assert path.read_text(encoding="utf-8") == content
    with pytest.raises(ValueError, match="model_name"):
        CachedReranker(None, path)


def committed_reranker() -> CachedReranker:
    return CachedReranker(None, RERANK_CACHE_PATH, model_name=load_reranker_config().model)


async def test_committed_rerank_cache_reproduces_the_cohere_measurement(
    semantic_index: tuple[InMemoryHybridStore, CachedEmbeddingClient],
) -> None:
    """Pins the reranked numbers in the README, offline, from data/rerank_cache.json."""
    store, embeddings = semantic_index
    reranker = committed_reranker()
    settings = offline_settings()
    test_split, dev_split = load_retrieval_dataset("test"), load_retrieval_dataset("dev")

    gated = await evaluate_retriever(
        build_retriever(store, embeddings, settings, reranker=reranker), test_split
    )
    assert (gated.recall_at_3, round(gated.mrr, 2), gated.abstentions) == (0.5, 0.42, 3)
    ranking = build_retriever(store, embeddings, settings, reranker=reranker, min_dense_score=-1.0)
    ungated = await evaluate_retriever(ranking, test_split)
    assert (ungated.recall_at_3, round(ungated.mrr, 2), ungated.abstentions) == (1.0, 0.81, 0)
    dev = await evaluate_retriever(
        build_retriever(store, embeddings, settings, reranker=reranker), dev_split
    )
    assert (dev.recall_at_3, round(dev.mrr, 2), dev.abstentions) == (0.75, 0.70, 9)


async def test_settings_rerank_threshold_and_gate_choice_come_from_dev(
    semantic_index: tuple[InMemoryHybridStore, CachedEmbeddingClient],
) -> None:
    store, embeddings = semantic_index
    settings = offline_settings()
    dev = load_retrieval_dataset("dev")
    rerank = await calibrate_min_rerank_score(
        build_retriever(
            store, embeddings, settings, reranker=committed_reranker(), evidence_gate="rerank"
        ),
        dev,
    )
    dense = await calibrate_min_dense_score(
        build_retriever(store, embeddings, settings, min_dense_score=-1.0), dev
    )
    assert rerank.threshold == settings.rag_min_rerank_score
    assert (rerank.positives_answered, rerank.negatives_abstained) == (22, 9)

    def balanced(calibration: Calibration) -> float:
        return (
            calibration.positives_answered / calibration.positive_cases
            + calibration.negatives_abstained / calibration.negative_cases
        )

    chosen = "dense" if balanced(dense) >= balanced(rerank) else "rerank"
    assert chosen == settings.rag_evidence_gate
