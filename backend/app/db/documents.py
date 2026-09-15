"""Where uploaded documents are remembered between requests.

Small on purpose. The index holds the chunks; this holds what a chat turn needs
that the chunks do not: the pinned summary, the metadata retrieval filters the
RBI corpus against, and the resolved clause set a later amendment will be
carried forward from.

SQLite, in the same file as the local index, so a demo survives a restart and
so "which documents do I have?" is answerable without re-parsing anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.core.contracts import Clause, DocumentKind, DocumentMetadata

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id  TEXT NOT NULL,
    version      INTEGER NOT NULL,
    title        TEXT,
    filename     TEXT,
    kind         TEXT NOT NULL,
    summary      TEXT NOT NULL,
    metadata     TEXT NOT NULL,
    clauses      TEXT NOT NULL,
    warnings     TEXT NOT NULL,
    uploaded_at  TEXT NOT NULL,
    PRIMARY KEY (document_id, version)
);
"""


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    version: int
    title: str
    filename: str
    kind: DocumentKind
    summary: str
    metadata: DocumentMetadata
    clauses: tuple[Clause, ...]
    warnings: tuple[str, ...]
    uploaded_at: str

    @property
    def as_of(self) -> date | None:
        """The date the RBI corpus should be filtered against for this loan."""
        return self.metadata.agreement_date


class DocumentStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Shared across request and streaming threads; see LocalVectorIndex.
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save(self, document: StoredDocument) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (document_id, version, title, filename, kind, summary,
                                       metadata, clauses, warnings, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, version) DO UPDATE SET
                    title=excluded.title, filename=excluded.filename, kind=excluded.kind,
                    summary=excluded.summary, metadata=excluded.metadata,
                    clauses=excluded.clauses, warnings=excluded.warnings,
                    uploaded_at=excluded.uploaded_at
                """,
                (
                    document.document_id,
                    document.version,
                    document.title,
                    document.filename,
                    document.kind.value,
                    document.summary,
                    document.metadata.model_dump_json(),
                    json.dumps([clause.model_dump(mode="json") for clause in document.clauses]),
                    json.dumps(list(document.warnings)),
                    document.uploaded_at,
                ),
            )

    def _row(self, row: sqlite3.Row) -> StoredDocument:
        return StoredDocument(
            document_id=row["document_id"],
            version=row["version"],
            title=row["title"] or "",
            filename=row["filename"] or "",
            kind=DocumentKind(row["kind"]),
            summary=row["summary"],
            metadata=DocumentMetadata.model_validate_json(row["metadata"]),
            clauses=tuple(Clause.model_validate(c) for c in json.loads(row["clauses"])),
            warnings=tuple(json.loads(row["warnings"])),
            uploaded_at=row["uploaded_at"],
        )

    def get(self, document_id: str, version: int) -> StoredDocument | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ? AND version = ?",
            (document_id, version),
        ).fetchone()
        return self._row(row) if row else None

    def latest(self, document_id: str) -> StoredDocument | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ? ORDER BY version DESC LIMIT 1",
            (document_id,),
        ).fetchone()
        return self._row(row) if row else None

    def versions(self, document_id: str) -> list[int]:
        rows = self._connection.execute(
            "SELECT version FROM documents WHERE document_id = ? ORDER BY version",
            (document_id,),
        ).fetchall()
        return [row["version"] for row in rows]

    def list_all(self) -> list[StoredDocument]:
        rows = self._connection.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC, version DESC"
        ).fetchall()
        return [self._row(row) for row in rows]
