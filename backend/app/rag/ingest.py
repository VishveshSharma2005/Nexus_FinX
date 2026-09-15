"""Lane 1: parse, resolve, chunk, embed, index.

One place where a document becomes searchable, and the only place that decides
what is embedded and what is reused. Everything here depends on interfaces --
a parser, an embedding provider, an index -- so none of it changes when any of
those are swapped.

The rule it exists to enforce: a clause whose content hash is already stored is
never embedded again. It gains a chunk row for the new version, pointing at the
embedding that already exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.core.contracts import (
    Chunk,
    DocumentKind,
    DocumentParser,
    EmbeddingProvider,
    ParsedDocument,
    ParseSource,
    VectorIndex,
)
from app.core.corpus import CorpusManifest, load_manifest
from app.rag.chunk import chunk_corpus_document, chunk_version, pinned_summary
from app.rag.versioning import ResolvedVersion, resolve_amendment, resolve_standalone

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass
class IngestResult:
    document_id: str
    version: int
    parsed: ParsedDocument
    resolved: ResolvedVersion
    chunks_written: int
    embeddings_computed: int
    embeddings_reused: int
    summary: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def reuse_ratio(self) -> float:
        total = self.embeddings_computed + self.embeddings_reused
        return self.embeddings_reused / total if total else 0.0


async def _embed_new_only(
    chunks: list[Chunk],
    embedder: EmbeddingProvider,
    index: VectorIndex,
) -> tuple[list[list[float] | None], int, int]:
    """Embed only text the index does not already hold.

    Deduplicates within the batch as well as against storage, because a
    document can repeat a clause verbatim and paying twice in one upload is
    the same waste as paying twice across versions.
    """
    hashes = [chunk.content_hash for chunk in chunks]
    known = await index.existing_hashes(hashes)

    needed: list[str] = []
    seen: set[str] = set()
    for content_hash in hashes:
        if content_hash in known or content_hash in seen:
            continue
        seen.add(content_hash)
        needed.append(content_hash)

    by_hash = {chunk.content_hash: chunk.text for chunk in chunks}
    computed: dict[str, list[float]] = {}
    if needed:
        vectors = await embedder.embed_documents([by_hash[h] for h in needed])
        computed = dict(zip(needed, vectors, strict=True))

    aligned: list[list[float] | None] = [computed.get(h) for h in hashes]
    return aligned, len(needed), len(hashes) - len(needed)


async def ingest_document(
    *,
    path: Path,
    parser: DocumentParser,
    embedder: EmbeddingProvider,
    index: VectorIndex,
    document_id: str,
    version: int,
    base: ResolvedVersion | None = None,
) -> IngestResult:
    """Ingest one version of one borrower document.

    ``base`` is the resolved previous version. It is required when the upload
    turns out to be an amendment: without it, the amendment's unlisted clauses
    have nothing to carry forward from, and indexing it alone would produce a
    version missing most of the loan's terms.
    """
    source = ParseSource(
        content=path.read_bytes(),
        filename=path.name,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        document_id=document_id,
        version=version,
    )
    if not parser.supports(source):
        raise ValueError(f"Parser {parser.name!r} does not support {path.name}")

    parsed = parser.parse(source)
    notes = [warning.message for warning in parsed.warnings]

    if parsed.kind is DocumentKind.AMENDMENT:
        if base is None:
            raise ValueError(
                f"{path.name} revises an earlier agreement, but no base version was supplied. "
                "Indexing it on its own would produce a version missing the clauses it "
                "carries forward."
            )
        resolved = resolve_amendment(base, parsed)
        notes.extend(resolved.notes)
    else:
        resolved = resolve_standalone(parsed)

    chunks, report = chunk_version(resolved, title=parsed.title)
    if report.skipped_boilerplate:
        notes.append(f"Skipped {report.skipped_boilerplate} boilerplate block(s) at chunk time.")

    vectors, computed, reused = await _embed_new_only(chunks, embedder, index)
    await index.upsert(chunks, vectors)

    return IngestResult(
        document_id=document_id,
        version=version,
        parsed=parsed,
        resolved=resolved,
        chunks_written=len(chunks),
        embeddings_computed=computed,
        embeddings_reused=reused,
        summary=pinned_summary(parsed, resolved),
        notes=notes,
    )


async def ingest_corpus(
    *,
    corpus_dir: Path,
    parser_for,
    embedder: EmbeddingProvider,
    index: VectorIndex,
    manifest: CorpusManifest | None = None,
) -> list[IngestResult]:
    """Ingest the RBI corpus, tagging each chunk with its applicability."""
    manifest = manifest or load_manifest(corpus_dir)
    results: list[IngestResult] = []

    for entry in manifest.documents:
        path = corpus_dir / entry.file
        source = ParseSource(
            content=path.read_bytes(),
            filename=path.name,
            media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
            document_id=entry.id,
            version=1,
            kind_hint=DocumentKind.RBI_CIRCULAR,
        )
        parser = parser_for(source)
        parsed = parser.parse(source)

        chunks, report = chunk_corpus_document(parsed, entry)
        vectors, computed, reused = await _embed_new_only(chunks, embedder, index)
        await index.upsert(chunks, vectors)

        results.append(
            IngestResult(
                document_id=entry.id,
                version=1,
                parsed=parsed,
                resolved=resolve_standalone(parsed),
                chunks_written=len(chunks),
                embeddings_computed=computed,
                embeddings_reused=reused,
                notes=[f"Skipped {report.skipped_boilerplate} boilerplate block(s)."]
                if report.skipped_boilerplate
                else [],
            )
        )

    return results
