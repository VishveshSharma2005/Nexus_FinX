"""Postgres + pgvector implementation of the same index interface.

The production-shaped path, selected with ``FINX_INDEX=pgvector`` and brought
up by docker-compose. It stores the identical two-table shape as the local
index -- embeddings keyed by content hash, chunks keyed by chunk id and
pointing at one -- so versioning behaves the same whichever backend is
configured, and so does every rule that depends on it.

The differences from the local index are the ones worth having a database for:
similarity runs in Postgres via the ``<=>`` cosine operator against an ivfflat
index, and applicability is filtered in SQL rather than in Python, so an
inapplicable passage is excluded before it is ever scored.

Verification status: exercised against a live Postgres only when one is
reachable. ``tests/test_pgvector_index.py`` runs the full index contract
against it and skips when it is not, so a machine without Docker still gets a
green suite -- and does not get a false claim that this path was tested.
"""

from __future__ import annotations

import json
import logging

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
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT PRIMARY KEY,
    vector       vector(%(dimension)s) NOT NULL,
    model        TEXT NOT NULL
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
    applicability JSONB NOT NULL,
    citation      JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks(document_id, version);
CREATE INDEX IF NOT EXISTS chunks_by_hash    ON chunks(content_hash);
"""


def _dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver prefix psycopg does not accept."""
    return database_url.replace("postgresql+psycopg://", "postgresql://")


class PgVectorIndex:
    name = "pgvector"

    def __init__(self, database_url: str, *, dimension: int = 512, model: str = "unknown") -> None:
        import psycopg

        self._dimension = dimension
        self._model = model
        self._connection = psycopg.connect(_dsn(database_url), autocommit=True)
        with self._connection.cursor() as cursor:
            cursor.execute(_SCHEMA % {"dimension": dimension})

    def close(self) -> None:
        self._connection.close()

    # -- writing -----------------------------------------------------------

    async def existing_hashes(self, content_hashes: list[str]) -> dict[str, str]:
        if not content_hashes:
            return {}
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT e.content_hash,
                       (SELECT c.chunk_id FROM chunks c
                         WHERE c.content_hash = e.content_hash LIMIT 1)
                  FROM embeddings e
                 WHERE e.content_hash = ANY(%s)
                """,
                (list(content_hashes),),
            )
            return {row[0]: row[1] or "" for row in cursor.fetchall()}

    async def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        vectors = dict(zip((chunk.content_hash for chunk in chunks), embeddings, strict=False))

        with self._connection.cursor() as cursor:
            for content_hash, vector in vectors.items():
                if vector is None:
                    continue
                cursor.execute(
                    """
                    INSERT INTO embeddings (content_hash, vector, model)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    """,
                    (content_hash, _literal(vector), self._model),
                )

            for chunk in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (chunk_id, content_hash, document_id, version, clause_id,
                                        ordinal, text, source_kind, applicability, citation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        text         = EXCLUDED.text,
                        citation     = EXCLUDED.citation
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
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(chunk_id)
            columns = [description.name for description in cursor.description]
            record = dict(zip(columns, row, strict=True))

            citation = record["citation"]
            if isinstance(citation, str):
                citation = json.loads(citation)
            new_id = f"{document_id}-v{version}-{record['clause_id'] or record['chunk_id']}"
            citation |= {"version": version, "document_id": document_id, "chunk_id": new_id}

            cursor.execute(
                """
                INSERT INTO chunks (chunk_id, content_hash, document_id, version, clause_id,
                                    ordinal, text, source_kind, applicability, citation)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                (
                    new_id,
                    record["content_hash"],
                    document_id,
                    version,
                    record["clause_id"],
                    record["ordinal"],
                    record["text"],
                    record["source_kind"],
                    json.dumps(record["applicability"]),
                    json.dumps(citation),
                ),
            )

    # -- reading -----------------------------------------------------------

    async def search(
        self,
        embedding: list[float],
        *,
        query_text: str,
        filters: RetrievalFilter,
        top_k: int,
    ) -> list[RetrievedPassage]:
        where = ["TRUE"]
        params: list[object] = [_literal(embedding)]

        if filters.document_id is not None:
            where.append("c.document_id = %s")
            params.append(filters.document_id)
        if filters.version is not None:
            where.append("c.version = %s")
            params.append(filters.version)
        if filters.source_kinds:
            where.append("c.source_kind = ANY(%s)")
            params.append([kind.value for kind in filters.source_kinds])

        # Applicability, in SQL, before scoring. An empty array on the chunk
        # means unrestricted; a null on the document means unknown, and an
        # unknown widens rather than excludes.
        metadata = filters.metadata
        if metadata is not None:
            for field, value in (
                ("loan_types", metadata.loan_type),
                ("borrower_types", metadata.borrower_type),
                ("lender_classes", metadata.lender_class),
            ):
                if value is None:
                    continue
                where.append(
                    f"""(c.source_kind <> 'rbi_corpus'
                         OR jsonb_array_length(COALESCE(c.applicability->'{field}', '[]')) = 0
                         OR c.applicability->'{field}' ? %s)"""
                )
                params.append(value.value)

        if filters.as_of is not None:
            where.append(
                """(c.source_kind <> 'rbi_corpus'
                    OR ((c.applicability->>'effective_from') IS NULL
                        OR (c.applicability->>'effective_from')::date <= %s))"""
            )
            params.append(filters.as_of)
            where.append(
                """(c.source_kind <> 'rbi_corpus'
                    OR ((c.applicability->>'effective_to') IS NULL
                        OR (c.applicability->>'effective_to')::date >= %s))"""
            )
            params.append(filters.as_of)

        params.append(top_k)

        sql = f"""
            SELECT c.chunk_id, c.content_hash, c.document_id, c.version, c.clause_id, c.ordinal,
                   c.text, c.source_kind, c.applicability, c.citation,
                   1 - (e.vector <=> %s::vector) AS score
              FROM chunks c JOIN embeddings e ON e.content_hash = c.content_hash
             WHERE {" AND ".join(where)}
             ORDER BY score DESC
             LIMIT %s
        """

        with self._connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        passages: list[RetrievedPassage] = []
        for row in rows:
            (
                chunk_id,
                content_hash,
                document_id,
                version,
                clause_id,
                ordinal,
                text,
                source_kind,
                applicability,
                citation,
                score,
            ) = row
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                version=version,
                clause_id=clause_id,
                ordinal=ordinal,
                text=text,
                content_hash=content_hash,
                source_kind=SourceKind(source_kind),
                applicability=Applicability.model_validate(_as_dict(applicability)),
                citation=Citation.model_validate(_as_dict(citation)),
            )
            passages.append(
                RetrievedPassage(chunk=chunk, score=float(score), dense_score=float(score))
            )
        return passages

    def stats(self) -> dict[str, int]:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            embeddings = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunks = cursor.fetchone()[0]
            cursor.execute(
                "SELECT COUNT(*) FROM (SELECT content_hash FROM chunks "
                "GROUP BY content_hash HAVING COUNT(*) > 1) t"
            )
            shared = cursor.fetchone()[0]
        return {"embeddings": embeddings, "chunks": chunks, "shared_embeddings": shared}

    def versions_of(self, document_id: str) -> list[int]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "SELECT DISTINCT version FROM chunks WHERE document_id = %s ORDER BY version",
                (document_id,),
            )
            return [row[0] for row in cursor.fetchall()]


def _literal(vector: list[float]) -> str:
    """pgvector accepts a bracketed list as its text representation."""
    return "[" + ",".join(f"{value:.7g}" for value in vector) + "]"


def _as_dict(value):
    return json.loads(value) if isinstance(value, str) else value
