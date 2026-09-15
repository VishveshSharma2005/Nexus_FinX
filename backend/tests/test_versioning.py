"""Versioning: deduplication, amendment resolution, and version isolation.

Three properties are load-bearing and are tested against the real documents,
not fixtures, because the real ones are what broke the first implementation:

* **Deduplication.** An unchanged clause must not be embedded twice.
* **Resolution.** A delta amendment's unlisted clauses carry forward; the
  clauses it deletes do not.
* **Isolation.** Retrieval scoped to one version can never return a clause
  from another. This is the one where a bug is actively dangerous: a lock-in
  the lender removed still appearing in a v2 answer would tell a borrower they
  are trapped when they are free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.core.contracts import DocumentKind, ParseSource, RetrievalFilter, SourceKind
from app.core.text import content_hash
from app.parsers.docx.parser import DocxParser
from app.providers.embeddings import HashingEmbeddingProvider
from app.rag.chunk import chunk_version, is_boilerplate
from app.rag.index import LocalVectorIndex
from app.rag.ingest import ingest_document
from app.rag.versioning import (
    ClauseOperation,
    diff_versions,
    resolve_amendment,
    resolve_standalone,
)

SAMPLES = get_settings().sample_corpus_dir
AGREEMENT_V1 = SAMPLES / "home_loan_agreement_A.docx"
AGREEMENT_V2 = SAMPLES / "home_loan_agreement_A_v2.docx"
DOCUMENT_ID = "loan-KHFL-HL-2026-0041872"

DOCX_MEDIA = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

pytestmark = pytest.mark.skipif(
    not (AGREEMENT_V1.is_file() and AGREEMENT_V2.is_file()),
    reason="real sample agreements not present",
)


def _parse(path: Path, version: int):
    return DocxParser().parse(
        ParseSource(
            content=path.read_bytes(),
            filename=path.name,
            media_type=DOCX_MEDIA,
            document_id=DOCUMENT_ID,
            version=version,
        )
    )


@pytest.fixture(scope="module")
def v1():
    return resolve_standalone(_parse(AGREEMENT_V1, 1))


@pytest.fixture(scope="module")
def amendment():
    return _parse(AGREEMENT_V2, 2)


@pytest.fixture(scope="module")
def v2(v1, amendment):
    return resolve_amendment(v1, amendment)


# --- The amendment is recognised as one -------------------------------------


def test_the_revision_is_identified_as_an_amendment(amendment) -> None:
    assert amendment.kind is DocumentKind.AMENDMENT
    assert amendment.amends is not None
    assert len(amendment.amends.evidence) >= 2, "the call must rest on more than one cue"


def test_the_amendment_names_the_agreement_it_revises(amendment) -> None:
    target = amendment.amends
    assert target.loan_account_number == "KHFL/HL/2026/0041872"
    assert target.base_dated is not None
    assert target.effective_from is not None
    assert target.effective_from > target.base_dated


def test_the_original_is_not_mistaken_for_an_amendment() -> None:
    original = _parse(AGREEMENT_V1, 1)
    assert original.kind is DocumentKind.LOAN_AGREEMENT
    assert original.amends is None


@pytest.mark.asyncio
async def test_an_amendment_cannot_be_ingested_without_its_base(tmp_path) -> None:
    """Absence of a base is an error, never a silently thin version.

    Indexing an amendment alone would produce a "version 2" holding only the
    handful of clauses it reprints, and every question about the rest of the
    loan would then be answered from nothing.
    """
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    with pytest.raises(ValueError, match="no base version"):
        await ingest_document(
            path=AGREEMENT_V2,
            parser=DocxParser(),
            embedder=HashingEmbeddingProvider(),
            index=index,
            document_id=DOCUMENT_ID,
            version=2,
        )
    index.close()


def test_resolving_a_restatement_needs_no_base() -> None:
    """A document that restates itself in full is already its own version."""
    standalone = resolve_standalone(_parse(AGREEMENT_V1, 1))
    assert len(standalone.clauses) == len(_parse(AGREEMENT_V1, 1).clauses)


# --- Resolution -------------------------------------------------------------


def test_the_amendment_is_smaller_than_the_version_it_produces(v1, amendment, v2) -> None:
    """The point of the whole exercise: a 15-clause document yields a full loan."""
    assert len(amendment.clauses) < len(v1.clauses)
    assert len(v2.clauses) >= len(v1.clauses) - 5


def test_unlisted_clauses_carry_forward(v1, v2) -> None:
    """Security and insurance are untouched by the amendment and must survive.

    Reading a delta amendment as a restatement would delete them, leaving a
    borrower asking "what did I pledge?" with no answer at all.
    """
    for number in ("6.1", "6.2", "6.3", "7.1", "7.2", "12.1"):
        original = _clause(v1, number)
        carried = _clause(v2, number)
        assert carried is not None, f"clause {number} vanished from v2"
        assert carried.content_hash == original.content_hash


def test_revised_clauses_are_replaced(v2) -> None:
    rate = _clause(v2, "3.1")
    penal = _clause(v2, "9.1")
    assert "8.60%" in rate.text
    assert "1.5%" in penal.text


def test_deleted_clauses_do_not_survive(v1, v2) -> None:
    """The lock-in the lender removed must be gone from v2.

    If this regresses, FinX tells a borrower they are locked in for twelve
    months when the amendment freed them.
    """
    assert _clause(v1, "11.2") is not None, "the fixture should have a lock-in in v1"
    assert "Lock-in Period" in _clause(v1, "11.2").text

    assert _clause(v2, "11.2") is None
    assert _clause(v2, "11.4") is None
    assert _clause(v2, "11.5") is None

    deleted = {c.number for c in v2.changes if c.operation is ClauseOperation.DELETED}
    assert {"11.2", "11.4", "11.5"} <= deleted


def test_a_statement_that_the_rest_is_unchanged_does_not_replace_anything(v1, v2) -> None:
    """Regression, and a subtle one.

    The amendment's clause 9.2 reads "All other provisions of Clause 9 ...
    remain unchanged". It borrows the original's numbering but enacts nothing.
    Treating it as a revision of base clause 9.2 deleted the real 9.2 -- the
    prohibition on capitalising penal charges -- from the version.
    """
    original = _clause(v1, "9.2")
    resolved = _clause(v2, "9.2")
    assert resolved is not None
    assert resolved.content_hash == original.content_hash
    assert "not be capitalised" in resolved.text or "capitalis" in resolved.text

    rate = _clause(v2, "3.2")
    assert rate.content_hash == _clause(v1, "3.2").content_hash


def test_a_partly_reprinted_section_is_reported_rather_than_merged(v2) -> None:
    """The revised fee schedule shows only changed rows, so the rest still
    stands. That ambiguity is surfaced, not resolved by guesswork."""
    assert v2.notes, "a partial restatement must produce a note"
    assert any("only in part" in note for note in v2.notes)


def test_resolution_is_deterministic(v1, amendment) -> None:
    first = resolve_amendment(v1, amendment)
    second = resolve_amendment(v1, amendment)
    assert [c.content_hash for c in first.clauses] == [c.content_hash for c in second.clauses]


def test_clause_ids_belong_to_the_version_that_produced_them(v2) -> None:
    for clause in v2.clauses:
        assert clause.clause_id.startswith(f"{DOCUMENT_ID}-v2-")


# --- Deduplication ----------------------------------------------------------


def test_most_clauses_are_reused_rather_than_re_embedded(v2) -> None:
    reuse = len(v2.reused_hashes)
    fresh = len(v2.embedding_required_hashes)
    assert reuse > fresh * 3, f"expected mostly reuse, got {reuse} reused vs {fresh} new"


def test_reflowed_text_hashes_the_same_but_a_changed_rate_does_not() -> None:
    a = "9.1 Penal charges of 2% per\nmonth on the overdue amount."
    b = "9.1 Penal charges of 2%  per month on the overdue amount."
    c = "9.1 Penal charges of 1.5% per month on the overdue amount."
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


@pytest.mark.asyncio
async def test_ingesting_a_revision_embeds_only_what_changed(tmp_path) -> None:
    """The rule, measured end to end against the real documents."""
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()
    parser = DocxParser()

    first = await ingest_document(
        path=AGREEMENT_V1,
        parser=parser,
        embedder=embedder,
        index=index,
        document_id=DOCUMENT_ID,
        version=1,
    )
    second = await ingest_document(
        path=AGREEMENT_V2,
        parser=parser,
        embedder=embedder,
        index=index,
        document_id=DOCUMENT_ID,
        version=2,
        base=first.resolved,
    )

    assert first.embeddings_reused == 0
    assert second.embeddings_reused > second.embeddings_computed
    assert second.reuse_ratio > 0.7

    stats = index.stats()
    assert stats["chunks"] > stats["embeddings"], "versions must share embedding rows"
    assert stats["shared_embeddings"] > 0
    index.close()


@pytest.mark.asyncio
async def test_re_ingesting_the_same_version_is_idempotent(tmp_path) -> None:
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()
    parser = DocxParser()

    async def ingest():
        return await ingest_document(
            path=AGREEMENT_V1,
            parser=parser,
            embedder=embedder,
            index=index,
            document_id=DOCUMENT_ID,
            version=1,
        )

    await ingest()
    before = index.stats()
    again = await ingest()
    after = index.stats()

    assert after == before, "re-uploading the same file must not duplicate it"
    assert again.embeddings_computed == 0
    index.close()


# --- Version isolation ------------------------------------------------------


@pytest.mark.asyncio
async def test_a_v1_clause_never_reaches_a_v2_query(tmp_path) -> None:
    """The isolation rule, tested on the clause where it matters most."""
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()
    parser = DocxParser()

    first = await ingest_document(
        path=AGREEMENT_V1,
        parser=parser,
        embedder=embedder,
        index=index,
        document_id=DOCUMENT_ID,
        version=1,
    )
    await ingest_document(
        path=AGREEMENT_V2,
        parser=parser,
        embedder=embedder,
        index=index,
        document_id=DOCUMENT_ID,
        version=2,
        base=first.resolved,
    )

    question = "Is there a lock-in period before I can prepay the loan?"
    query = await embedder.embed_query(question)

    v1_hits = await index.search(
        query,
        query_text=question,
        filters=RetrievalFilter(document_id=DOCUMENT_ID, version=1),
        top_k=25,
    )
    v2_hits = await index.search(
        query,
        query_text=question,
        filters=RetrievalFilter(document_id=DOCUMENT_ID, version=2),
        top_k=25,
    )

    assert all(p.chunk.version == 1 for p in v1_hits)
    assert all(p.chunk.version == 2 for p in v2_hits)

    assert any("Lock-in Period" in p.chunk.text for p in v1_hits), (
        "v1 genuinely contains a lock-in, so the query should find it there"
    )
    assert not any("Lock-in Period of twelve" in p.chunk.text for p in v2_hits), (
        "the deleted lock-in leaked into v2"
    )
    index.close()


@pytest.mark.asyncio
async def test_retrieval_can_be_scoped_to_the_borrowers_document_alone(tmp_path) -> None:
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()

    await ingest_document(
        path=AGREEMENT_V1,
        parser=DocxParser(),
        embedder=embedder,
        index=index,
        document_id=DOCUMENT_ID,
        version=1,
    )
    query = await embedder.embed_query("penal charges")
    hits = await index.search(
        query,
        query_text="penal charges",
        filters=RetrievalFilter(source_kinds=(SourceKind.USER_DOCUMENT,)),
        top_k=10,
    )
    assert hits
    assert all(p.chunk.source_kind is SourceKind.USER_DOCUMENT for p in hits)
    index.close()


# --- The zero-token diff ----------------------------------------------------


def test_the_version_diff_costs_no_tokens_and_finds_the_real_changes(v1, v2) -> None:
    diff = diff_versions(v1, v2)
    assert not diff.is_empty
    assert diff.unchanged_count > len(diff.added)

    added = " ".join(clause.text for clause in diff.added)
    removed = " ".join(clause.text for clause in diff.removed)
    assert "8.60%" in added and "8.35%" in removed
    assert "Lock-in Period" in removed


def test_diffing_a_version_against_itself_finds_nothing(v1) -> None:
    assert diff_versions(v1, v1).is_empty


# --- Chunking ---------------------------------------------------------------


def test_web_navigation_is_dropped_at_chunk_time_not_at_parse_time() -> None:
    """The parser keeps everything; chunking decides what is worth indexing."""
    assert is_boilerplate("MORE LINKS :")
    assert is_boilerplate("E-LMS")
    assert is_boilerplate("RSS")
    assert not is_boilerplate(
        "11.3 Any prepayment made after the Lock-in Period shall attract a charge of 3%."
    )


def test_chunking_preserves_the_clause_hash_for_whole_clauses(v1) -> None:
    """Otherwise carried-forward clauses would look new and be re-embedded."""
    chunks, _ = chunk_version(v1)
    by_id = {clause.clause_id: clause for clause in v1.clauses}
    whole = [c for c in chunks if not c.chunk_id.endswith(("-p0", "-p1", "-p2"))]
    assert whole
    for chunk in whole:
        assert chunk.content_hash == by_id[chunk.clause_id].content_hash


def _clause(version, number: str):
    for clause in version.clauses:
        if clause.number == number:
            return clause
    return None


def _settings() -> Settings:
    return get_settings()
