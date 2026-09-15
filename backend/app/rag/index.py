"""Vector index implementations.

Two, behind :class:`app.core.contracts.VectorIndex`: SQLite with numpy for a
demo that depends on no external service, and pgvector for the shape this
would take in production. Retrieval imports neither.

The schema is what makes token-efficient versioning real rather than aspirational:

    embeddings        content_hash -> vector          (one row per distinct text)
    chunks            chunk_id     -> content_hash    (one row per version)

An embedding is keyed by the hash of its text, not by the chunk. So a clause
that survives into a new version produces a new ``chunks`` row -- it needs its
own id, page and version so citations name the right version -- while pointing
at the embedding that already exists. Re-ingesting a revised agreement embeds
only genuinely changed clauses, and "add a new document-version pointer to it"
is a foreign key rather than a promise.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np

from app.core.contracts import (
    Applicability,
    Chunk,
    Citation,
    RetrievalFilter,
    RetrievedPassage,
    SourceKind,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT PRIMARY KEY,
    dimension    INTEGER NOT NULL,
    vector       BLOB    NOT NULL,
    model        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL REFERENCES embeddings(content_hash),
    document_id   TEXT NOT NULL,
    version       INTEGER NOT NULL,
    clause_id     TEXT,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    source_kind   TEXT NOT NULL,
    applicability TEXT NOT NULL,
    citation      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks(document_id, version);
CREATE INDEX IF NOT EXISTS chunks_by_hash    ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS chunks_by_source  ON chunks(source_kind);
"""


def _matches_applicability(applicability: Applicability, filters: RetrievalFilter) -> bool:
    """Whether a corpus chunk governs the document being asked about.

    Applied before reranking, and deliberately asymmetric:

    * An empty axis on the chunk means the circular does not restrict on it,
      so it matches anything.
    * An unknown value on the document means FinX could not determine it, and
      an unknown must *widen* the candidate set. Excluding regulation because
      a borrower's document did not state its lender class would hide rules
      that do in fact bind.

    So a passage is filtered out only when both sides state a value and the
    values disagree. Being wrong in that direction costs recall; being wrong
    in the other direction puts inapplicable regulation in front of a
    borrower as though it governed their loan.
    """
    metadata = filters.metadata
    if metadata is not None:
        if (
            applicability.loan_types
            and metadata.loan_type is not None
            and metadata.loan_type not in applicability.loan_types
        ):
            return False
        if (
            applicability.borrower_types
            and metadata.borrower_type is not None
            and metadata.borrower_type not in applicability.borrower_types
        ):
            return False
        if (
            applicability.lender_classes
            and metadata.lender_class is not None
            and metadata.lender_class not in applicability.lender_classes
        ):
            return False

    as_of: date | None = filters.as_of
    if as_of is not None:
        if applicability.effective_from and as_of < applicability.effective_from:
            return False
        if applicability.effective_to and as_of > applicability.effective_to:
            return False

    return True


class LocalVectorIndex:
    """SQLite + numpy. No server, no extension, no Docker.

    Exact search rather than approximate: this corpus is thousands of chunks,
    not millions, and an exact scan removes a whole class of "why did it not
    retrieve that clause" question from the demo.
    """

    name = "local"

    def __init__(self, path: Path | str, *, model: str = "unknown") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._model = model
        # The API serves a request on one thread and streams the response from
        # another, so the connection has to outlive the thread that opened it.
        # SQLite's own serialized threading mode makes that safe for reads; the
        # lock below serialises writes, which it does not.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    # -- writing -----------------------------------------------------------

    async def existing_hashes(self, content_hashes: list[str]) -> dict[str, str]:
        """Map ``content_hash -> chunk_id`` for text already embedded.

        A hit means the embedding can be reused and does not need paying for
        again.
        """
        if not content_hashes:
            return {}

        found: dict[str, str] = {}
        for batch in _batched(content_hashes, 500):
            placeholders = ",".join("?" * len(batch))
            rows = self._connection.execute(
                f"""
                SELECT e.content_hash AS content_hash,
                       (SELECT chunk_id FROM chunks c
                         WHERE c.content_hash = e.content_hash LIMIT 1) AS chunk_id
                  FROM embeddings e
                 WHERE e.content_hash IN ({placeholders})
                """,
                list(batch),
            ).fetchall()
            for row in rows:
                found[row["content_hash"]] = row["chunk_id"] or ""
        return found

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks, and store an embedding only for text not already held.

        ``embeddings`` may be shorter than ``chunks``: the caller passes
        vectors only for the hashes that needed embedding. Anything already
        present keeps the vector it has.
        """
        vectors = dict(zip((chunk.content_hash for chunk in chunks), embeddings, strict=False))

        with self._lock, self._connection:
            for content_hash, vector in vectors.items():
                if vector is None:
                    continue
                array = np.asarray(vector, dtype=np.float32)
                self._connection.execute(
                    """
                    INSERT INTO embeddings (content_hash, dimension, vector, model)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(content_hash) DO NOTHING
                    """,
                    (content_hash, int(array.size), array.tobytes(), self._model),
                )

            for chunk in chunks:
                self._connection.execute(
                    """
                    INSERT INTO chunks (chunk_id, content_hash, document_id, version,
                                        clause_id, ordinal, text, source_kind,
                                        applicability, citation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        content_hash = excluded.content_hash,
                        text         = excluded.text,
                        citation     = excluded.citation
                    """,
                    (
                        chunk.chunk_id,
                        chunk.content_hash,
                        chunk.document_id,
                        chunk.version,
                        chunk.clause_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.source_kind.value,
                        chunk.applicability.model_dump_json(),
                        chunk.citation.model_dump_json(),
                    ),
                )

    async def add_version_pointer(self, chunk_id: str, document_id: str, version: int) -> None:
        """Record that existing text also appears in another version."""
        row = self._connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)

        citation = json.loads(row["citation"])
        citation["version"] = version
        citation["document_id"] = document_id
        new_id = f"{document_id}-v{version}-{row['clause_id'] or row['chunk_id']}"
        citation["chunk_id"] = new_id

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO chunks (chunk_id, content_hash, document_id, version, clause_id,
                                    ordinal, text, source_kind, applicability, citation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO NOTHING
                """,
                (
                    new_id,
                    row["content_hash"],
                    document_id,
                    version,
                    row["clause_id"],
                    row["ordinal"],
                    row["text"],
                    row["source_kind"],
                    row["applicability"],
                    json.dumps(citation),
                ),
            )

    # -- reading -----------------------------------------------------------

    def _candidates(self, filters: RetrievalFilter) -> list[sqlite3.Row]:
        sql = [
            "SELECT c.*, e.vector AS vector, e.dimension AS dimension",
            "FROM chunks c JOIN embeddings e ON e.content_hash = c.content_hash",
            "WHERE 1 = 1",
        ]
        params: list[object] = []

        # Version scoping is a hard SQL predicate, not a post-filter. A v1
        # clause must not be able to reach a v2 answer even by accident.
        if filters.document_id is not None:
            sql.append("AND c.document_id = ?")
            params.append(filters.document_id)
        if filters.version is not None:
            sql.append("AND c.version = ?")
            params.append(filters.version)
        if filters.source_kinds:
            placeholders = ",".join("?" * len(filters.source_kinds))
            sql.append(f"AND c.source_kind IN ({placeholders})")
            params.extend(kind.value for kind in filters.source_kinds)

        return self._connection.execute(" ".join(sql), params).fetchall()

    async def search(
        self,
        embedding: list[float],
        *,
        query_text: str,
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievedPassage]:
        rows = self._candidates(filters)
        if not rows:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm

        passages: list[RetrievedPassage] = []
        for row in rows:
            applicability = Applicability.model_validate_json(row["applicability"])

            # The applicability filter runs here, before any scoring, so an
            # inapplicable passage is never even a candidate for the context.
            if row["source_kind"] == SourceKind.RBI_CORPUS.value and not _matches_applicability(
                applicability, filters
            ):
                continue

            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if vector.size != query.size:
                logger.warning(
                    "Skipping chunk %s: stored %d dimensions, query has %d. "
                    "The index was built with a different embedding model.",
                    row["chunk_id"],
                    vector.size,
                    query.size,
                )
                continue

            vector_norm = float(np.linalg.norm(vector))
            score = float(np.dot(query, vector / vector_norm)) if vector_norm else 0.0

            chunk = Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                version=row["version"],
                clause_id=row["clause_id"],
                ordinal=row["ordinal"],
                text=row["text"],
                content_hash=row["content_hash"],
                source_kind=SourceKind(row["source_kind"]),
                applicability=applicability,
                citation=Citation.model_validate_json(row["citation"]),
            )
            passages.append(RetrievedPassage(chunk=chunk, score=score, dense_score=score))

        passages.sort(key=lambda passage: passage.score, reverse=True)
        return passages[:top_k]

    # -- introspection, for the debug view and the seed script -------------

    def stats(self) -> dict[str, int]:
        embeddings = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        chunks = self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        shared = self._connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT content_hash FROM chunks GROUP BY content_hash HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        return {"embeddings": embeddings, "chunks": chunks, "shared_embeddings": shared}

    def versions_of(self, document_id: str) -> list[int]:
        rows = self._connection.execute(
            "SELECT DISTINCT version FROM chunks WHERE document_id = ? ORDER BY version",
            (document_id,),
        ).fetchall()
        return [row["version"] for row in rows]


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
