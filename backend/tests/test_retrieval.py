"""Retrieval: the applicability filter, version scoping, and hybrid ranking.

The goldset in tests/eval measures answer quality. These pin the properties
that must hold regardless of how well ranking happens to be tuned.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.core.contracts import (
    Applicability,
    DocumentMetadata,
    LenderClass,
    LoanType,
    SourceKind,
)
from app.providers.embeddings import HashingEmbeddingProvider
from app.rag.index import LocalVectorIndex
from app.rag.retrieve import retrieve, retrieve_for_answer
from tests.index_contract import make_chunk


@pytest.fixture
async def index(tmp_path):
    store = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()

    chunks = [
        make_chunk(
            "loan-1",
            1,
            "11.2 A lock-in period of twelve months applies before prepayment.",
            ordinal=0,
        ),
        make_chunk(
            "loan-1",
            1,
            "9.1 Penal charges of 2% per month are levied on any overdue amount.",
            ordinal=1,
        ),
        make_chunk(
            "loan-1",
            2,
            "11.1 The Borrower may prepay at any time without any lock-in period.",
            ordinal=2,
        ),
        make_chunk(
            "ucb-only",
            1,
            "Penal charges shall be reasonable and disclosed in the Key Facts Statement.",
            source_kind=SourceKind.RBI_CORPUS,
            applicability=Applicability(lender_classes=(LenderClass.COOPERATIVE_BANK,)),
        ),
        make_chunk(
            "future-directions",
            1,
            "An RE shall not levy pre-payment charges on floating rate loans to individuals.",
            ordinal=1,
            source_kind=SourceKind.RBI_CORPUS,
            applicability=Applicability(effective_from=date(2026, 1, 1)),
        ),
        make_chunk(
            "applies-now",
            1,
            "Penal charges shall not be capitalised nor added to the rate of interest.",
            ordinal=2,
            source_kind=SourceKind.RBI_CORPUS,
            applicability=Applicability(
                lender_classes=(LenderClass.HOUSING_FINANCE_COMPANY, LenderClass.NBFC),
                effective_from=date(2024, 4, 1),
            ),
        ),
    ]
    await store.upsert(chunks, await embedder.embed_documents([c.text for c in chunks]))
    yield store
    store.close()


HFC = DocumentMetadata(loan_type=LoanType.HOME, lender_class=LenderClass.HOUSING_FINANCE_COMPANY)


async def _search(index, question, **kwargs):
    return await retrieve(question, index=index, embedder=HashingEmbeddingProvider(), **kwargs)


@pytest.mark.asyncio
async def test_a_circular_that_does_not_bind_this_lender_never_appears(index) -> None:
    """A UCB-only rule must not reach an NBFC borrower, however well it matches."""
    result = await _search(
        index, "penal charges", metadata=HFC, source_kinds=(SourceKind.RBI_CORPUS,)
    )
    assert result.passages
    assert all(p.chunk.document_id != "ucb-only" for p in result.passages)


@pytest.mark.asyncio
async def test_a_circular_not_yet_in_force_never_appears(index) -> None:
    """The 2025 Directions do not govern a loan signed before they took effect."""
    result = await _search(
        index,
        "prepayment charges",
        metadata=HFC,
        as_of=date(2025, 11, 30),
        source_kinds=(SourceKind.RBI_CORPUS,),
    )
    assert all(p.chunk.document_id != "future-directions" for p in result.passages)


@pytest.mark.asyncio
async def test_the_same_circular_appears_once_it_is_in_force(index) -> None:
    result = await _search(
        index,
        "prepayment charges",
        metadata=HFC,
        as_of=date(2026, 1, 14),
        source_kinds=(SourceKind.RBI_CORPUS,),
    )
    assert any(p.chunk.document_id == "future-directions" for p in result.passages)


@pytest.mark.asyncio
async def test_unknown_metadata_widens_rather_than_excludes(index) -> None:
    """An undetected lender class must not hide regulation that does bind.

    Being wrong this way costs precision. Being wrong the other way hides a
    rule the borrower is entitled to know about.
    """
    result = await _search(
        index, "penal charges", metadata=DocumentMetadata(), source_kinds=(SourceKind.RBI_CORPUS,)
    )
    assert any(p.chunk.document_id == "ucb-only" for p in result.passages)


@pytest.mark.asyncio
async def test_retrieval_is_scoped_to_one_version(index) -> None:
    result = await _search(index, "lock-in period", document_id="loan-1", version=2)
    assert result.passages
    assert all(p.chunk.version == 2 for p in result.passages)
    assert not any("twelve months" in p.chunk.text for p in result.passages)


@pytest.mark.asyncio
async def test_confidence_is_absolute_not_relative(index) -> None:
    """The gate's signal must mean the same thing across different questions.

    Normalising it would make the top hit score ~1.0 for every query, including
    one the corpus cannot answer.
    """
    strong = await _search(index, "penal charges of 2% per month on overdue amounts")
    weak = await _search(index, "cryptocurrency mining rigs and share trading")
    assert strong.confidence > weak.confidence


@pytest.mark.asyncio
async def test_an_answer_always_includes_the_borrowers_own_clauses(index) -> None:
    """Regulation alone is not an answer about someone's loan."""
    result = await retrieve_for_answer(
        "are penal charges allowed under RBI rules",
        index=index,
        embedder=HashingEmbeddingProvider(),
        document_id="loan-1",
        version=1,
        metadata=HFC,
        as_of=date(2026, 1, 14),
    )
    assert any(p.chunk.source_kind is SourceKind.USER_DOCUMENT for p in result.passages)


@pytest.mark.asyncio
async def test_every_passage_carries_a_citation(index) -> None:
    result = await _search(index, "penal charges", document_id="loan-1", version=1)
    for passage in result.passages:
        assert passage.chunk.citation.chunk_id == passage.chunk.chunk_id
