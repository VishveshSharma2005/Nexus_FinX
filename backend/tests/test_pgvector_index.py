"""The same index contract, run against Postgres + pgvector.

Skipped when no database is reachable, so a machine without Docker still gets
a green suite -- and, more importantly, does not get a green suite that
implies this path was exercised when it was not.

To run it:

    docker compose up -d db
    FINX_DATABASE_URL=postgresql://finx:finx@localhost:5432/finx pytest -k pgvector
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.core.contracts import RetrievalFilter, SourceKind
from app.providers.embeddings import HashingEmbeddingProvider
from tests.index_contract import make_chunk


def _reachable(url: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(_dsn(url), connect_timeout=2):
            return True
    except Exception:  # noqa: BLE001 - any failure means "not available here"
        return False


def _dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


DATABASE_URL = get_settings().database_url

pytestmark = pytest.mark.skipif(
    not _reachable(DATABASE_URL),
    reason=f"no Postgres at {DATABASE_URL} (run: docker compose up -d db)",
)


@pytest.fixture
async def index():
    from app.rag.pgvector_index import PgVectorIndex

    store = PgVectorIndex(DATABASE_URL, dimension=512)
    with store._connection.cursor() as cursor:  # noqa: SLF001 - test fixture teardown
        cursor.execute("TRUNCATE chunks, embeddings CASCADE")
    yield store
    store.close()


@pytest.mark.asyncio
async def test_versions_share_one_embedding_row(index) -> None:
    """The rule the whole schema exists for, checked on the real database."""
    embedder = HashingEmbeddingProvider()
    text = "11.2 The Borrower shall not prepay during the Lock-in Period of twelve months."

    v1 = make_chunk("doc-1", 1, text)
    vectors = await embedder.embed_documents([v1.text])
    await index.upsert([v1], vectors)

    v2 = make_chunk("doc-1", 2, text)
    known = await index.existing_hashes([v2.content_hash])
    assert known, "identical text must be recognised as already embedded"

    await index.upsert([v2], [None])
    stats = index.stats()
    assert stats["chunks"] == 2
    assert stats["embeddings"] == 1


@pytest.mark.asyncio
async def test_retrieval_is_scoped_to_one_version(index) -> None:
    embedder = HashingEmbeddingProvider()
    old = make_chunk("doc-1", 1, "11.2 A lock-in period of twelve months applies to this Loan.")
    new = make_chunk("doc-1", 2, "11.2 No lock-in period applies to this Loan.")
    await index.upsert([old, new], await embedder.embed_documents([old.text, new.text]))

    query = await embedder.embed_query("lock-in period")
    hits = await index.search(
        query,
        query_text="lock-in period",
        filters=RetrievalFilter(document_id="doc-1", version=2),
        top_k=10,
    )
    assert hits
    assert all(hit.chunk.version == 2 for hit in hits)


@pytest.mark.asyncio
async def test_applicability_is_filtered_in_sql(index) -> None:
    """An inapplicable circular must be excluded before it is ever scored."""
    from datetime import date

    from app.core.contracts import Applicability, DocumentMetadata, LenderClass

    embedder = HashingEmbeddingProvider()
    ucb_only = make_chunk(
        "ucb-circular",
        1,
        "Penal charges shall be reasonable and disclosed in the Key Facts Statement.",
        source_kind=SourceKind.RBI_CORPUS,
        applicability=Applicability(lender_classes=(LenderClass.COOPERATIVE_BANK,)),
    )
    await index.upsert([ucb_only], await embedder.embed_documents([ucb_only.text]))

    query = await embedder.embed_query("penal charges")
    hits = await index.search(
        query,
        query_text="penal charges",
        filters=RetrievalFilter(
            metadata=DocumentMetadata(lender_class=LenderClass.NBFC),
            as_of=date(2026, 1, 1),
        ),
        top_k=10,
    )
    assert not hits, "a UCB-only circular must not surface for an NBFC loan"
