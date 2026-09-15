"""Shared helpers for testing any VectorIndex implementation.

Not a test module itself. It exists so the local index and the pgvector index
are checked against the same expectations rather than against two hand-written
sets that can quietly drift apart.
"""

from __future__ import annotations

from app.core.contracts import Applicability, Chunk, Citation, SourceKind
from app.core.text import content_hash


def make_chunk(
    document_id: str,
    version: int,
    text: str,
    *,
    ordinal: int = 0,
    source_kind: SourceKind = SourceKind.USER_DOCUMENT,
    applicability: Applicability | None = None,
) -> Chunk:
    """A chunk whose content hash comes from its text, as ingest produces.

    The hash must not include the version: two versions holding identical text
    have to collide, because that collision is what makes the embedding
    reusable.
    """
    chunk_id = f"{document_id}-v{version}-c{ordinal:04d}"
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        version=version,
        clause_id=f"{document_id}-c{ordinal:04d}",
        ordinal=ordinal,
        text=text,
        content_hash=content_hash(text),
        source_kind=source_kind,
        applicability=applicability or Applicability(),
        citation=Citation(
            chunk_id=chunk_id,
            source_kind=source_kind,
            document_id=document_id,
            version=version,
            page=1,
        ),
    )
