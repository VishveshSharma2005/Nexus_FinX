"""Turning resolved clauses into indexable chunks.

A chunk is what gets embedded, retrieved and cited, so this module decides
three things:

* **Granularity.** A clause is the default unit, because a citation naming a
  clause is worth more than one naming a page. Only a clause too large to
  embed usefully is split, and then with overlap so a sentence spanning the
  split is still findable from either side.
* **What is worth indexing at all.** The RBI circulars in this corpus were
  saved from a web page, so they carry navigation furniture -- "MORE LINKS",
  "E-LMS", "RSS" -- that the parser is right to keep (a parser must not
  discard text) and that retrieval is right to drop. Filtering belongs here,
  not there.
* **What travels with the text.** Every chunk carries its citation and, for
  corpus chunks, the applicability record retrieval filters on before
  reranking.

The content hash comes from the clause and is *not* recomputed per chunk when
a clause is whole, so a clause carried forward between versions keeps its
hash, and therefore its embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.contracts import (
    Applicability,
    Chunk,
    Citation,
    Clause,
    DocumentKind,
    ParsedDocument,
    SourceKind,
)
from app.core.corpus import CorpusDocument
from app.core.text import content_hash
from app.rag.versioning import ResolvedVersion

# Chunks above this are split. Sized so that eight of them plus a short pinned
# summary comfortably fit a single turn's context.
MAX_CHUNK_CHARS = 1800

# Carried into the next chunk so a sentence cut by a split is still retrievable.
CHUNK_OVERLAP_CHARS = 200

# Below this a chunk carries no answerable content.
MIN_CHUNK_CHARS = 25


@dataclass(frozen=True)
class ChunkingReport:
    """What chunking did, so ingest can report rather than assert."""

    produced: int
    split_clauses: int
    skipped_boilerplate: int
    skipped_short: int


# Web furniture from RBI pages saved to PDF. Matched on the whole chunk, so a
# clause that merely mentions one of these words is unaffected.
_BOILERPLATE = re.compile(
    r"^(more links\s*:?|e-lms|rss|follow rbi|sitemap|search|notifications?|"
    r"master circulars?|master directions?|feedback|disclaimer|"
    r"important websites?|tenders?|faqs?|contact us|useful links)\W*$",
    re.IGNORECASE,
)

# A chunk that is mostly a run of these is navigation, not regulation.
_NAVIGATION_TOKENS = frozenset(
    {
        "home",
        "about us",
        "notifications",
        "press releases",
        "speeches",
        "publications",
        "statistics",
        "regulatory reporting",
        "complaints",
        "rss",
        "e-lms",
        "sitemap",
        "more links",
        "follow rbi",
    }
)


def is_boilerplate(text: str) -> bool:
    """Whether a chunk is site furniture rather than document content."""
    stripped = " ".join(text.split())
    if _BOILERPLATE.match(stripped):
        return True

    # Several navigation labels and little else.
    lowered = stripped.lower()
    hits = sum(1 for token in _NAVIGATION_TOKENS if token in lowered)
    return hits >= 3 and len(stripped) < 400


def _split_text(text: str) -> list[str]:
    """Split an oversized clause on sentence boundaries, with overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    sentences = re.split(r"(?<=[.;:])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the budget is cut on whitespace rather
        # than dropped.
        while len(sentence) > MAX_CHUNK_CHARS:
            cut = sentence.rfind(" ", 0, MAX_CHUNK_CHARS) or MAX_CHUNK_CHARS
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        if len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS and current:
            parts.append(current.strip())
            current = current[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
        current = f"{current} {sentence}".strip()

    if current.strip():
        parts.append(current.strip())
    return [part for part in parts if part]


def _clause_citation(
    clause: Clause,
    *,
    document_id: str,
    version: int,
    title: str | None,
) -> Citation:
    return Citation(
        chunk_id="",  # filled in by the caller once the chunk id is known
        source_kind=SourceKind.USER_DOCUMENT,
        document_id=document_id,
        version=version,
        page=clause.page_start,
        clause_number=clause.number,
        title=title,
    )


def chunk_version(
    resolved: ResolvedVersion,
    *,
    title: str | None = None,
) -> tuple[list[Chunk], ChunkingReport]:
    """Chunk one resolved version of a borrower's document.

    Chunk ids are derived from the clause id, so re-chunking the same version
    is stable and idempotent -- re-ingesting a document must not create a
    second copy of it.
    """
    chunks: list[Chunk] = []
    split_clauses = skipped_short = skipped_boilerplate = 0

    for clause in resolved.clauses:
        text = clause.text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            skipped_short += 1
            continue
        if is_boilerplate(text):
            skipped_boilerplate += 1
            continue

        parts = _split_text(text)
        if len(parts) > 1:
            split_clauses += 1

        for index, part in enumerate(parts):
            chunk_id = clause.clause_id if len(parts) == 1 else f"{clause.clause_id}-p{index}"
            # A whole clause keeps the clause's hash, which is what lets a
            # carried-forward clause reuse its stored embedding. A split part
            # is new text and is hashed on its own.
            part_hash = clause.content_hash if len(parts) == 1 else content_hash(part)

            citation = _clause_citation(
                clause, document_id=resolved.document_id, version=resolved.version, title=title
            ).model_copy(update={"chunk_id": chunk_id})

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=resolved.document_id,
                    version=resolved.version,
                    clause_id=clause.clause_id,
                    ordinal=len(chunks),
                    text=part,
                    content_hash=part_hash,
                    source_kind=SourceKind.USER_DOCUMENT,
                    citation=citation,
                )
            )

    return chunks, ChunkingReport(
        produced=len(chunks),
        split_clauses=split_clauses,
        skipped_boilerplate=skipped_boilerplate,
        skipped_short=skipped_short,
    )


def chunk_corpus_document(
    parsed: ParsedDocument,
    entry: CorpusDocument,
) -> tuple[list[Chunk], ChunkingReport]:
    """Chunk an RBI circular, attaching the applicability retrieval filters on.

    Applicability comes from the manifest rather than from the text. The
    circular states its addressees in prose; the manifest states them in the
    same vocabulary the borrower's document metadata uses, which is what makes
    the two comparable before reranking.
    """
    applicability: Applicability = entry.applicability
    chunks: list[Chunk] = []
    split_clauses = skipped_short = skipped_boilerplate = 0

    for clause in parsed.clauses:
        text = clause.text.strip()
        if len(text) < MIN_CHUNK_CHARS:
            skipped_short += 1
            continue
        if is_boilerplate(text):
            skipped_boilerplate += 1
            continue

        parts = _split_text(text)
        if len(parts) > 1:
            split_clauses += 1

        for index, part in enumerate(parts):
            chunk_id = f"{entry.id}-c{clause.ordinal:04d}" + (
                f"-p{index}" if len(parts) > 1 else ""
            )
            part_hash = clause.content_hash if len(parts) == 1 else content_hash(part)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=entry.id,
                    version=1,
                    clause_id=clause.clause_id,
                    ordinal=len(chunks),
                    text=part,
                    content_hash=part_hash,
                    source_kind=SourceKind.RBI_CORPUS,
                    applicability=applicability,
                    citation=Citation(
                        chunk_id=chunk_id,
                        source_kind=SourceKind.RBI_CORPUS,
                        document_id=entry.id,
                        version=1,
                        page=clause.page_start,
                        clause_number=clause.number,
                        title=entry.title,
                        circular_id=entry.rbi_reference,
                        source_url=entry.source_url,
                    ),
                )
            )

    return chunks, ChunkingReport(
        produced=len(chunks),
        split_clauses=split_clauses,
        skipped_boilerplate=skipped_boilerplate,
        skipped_short=skipped_short,
    )


def pinned_summary(parsed: ParsedDocument, resolved: ResolvedVersion) -> str:
    """A short standing description of the document, sent with every turn.

    Deliberately a handful of lines. Each chat turn sends the retrieved
    clauses plus this, never the whole document, so the cost of a conversation
    does not grow with the size of the agreement.
    """
    meta = parsed.metadata
    facts = [
        f"Document: {parsed.title or parsed.document_id}",
        f"Type: {parsed.kind.value.replace('_', ' ')}",
        f"Version: {resolved.version}",
    ]
    if meta.loan_type:
        facts.append(f"Loan type: {meta.loan_type.value}")
    if meta.borrower_type:
        facts.append(f"Borrower: {meta.borrower_type.value}")
    if meta.lender_class:
        facts.append(f"Lender class: {meta.lender_class.value.replace('_', ' ')}")
    if meta.agreement_date:
        facts.append(f"Dated: {meta.agreement_date.isoformat()}")
    if parsed.kind is DocumentKind.AMENDMENT and parsed.amends:
        if parsed.amends.effective_from:
            facts.append(f"Revision effective: {parsed.amends.effective_from.isoformat()}")
        facts.append("This version revises an earlier agreement; unlisted clauses carry forward.")
    facts.append(f"Clauses in force: {len(resolved.clauses)}")
    return "\n".join(facts)
