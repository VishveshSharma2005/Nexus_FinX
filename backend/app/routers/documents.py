"""Upload and inspect documents.

The HTTP face of Lane 1. Everything it does is delegated to the ingest
pipeline; this module's job is to decide what the browser is told, including
what went wrong.
"""

from __future__ import annotations

import logging
import re
import tempfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.core.contracts import DocumentKind, ParseSource
from app.core.deps import get_embedding_provider, get_parser_for, get_vector_index
from app.db.documents import DocumentStore, StoredDocument
from app.rag.ingest import MEDIA_TYPES, ingest_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Versions of one loan must share a document id, or they are two unrelated
# uploads and an amendment has nothing to carry forward from. The loan account
# number is what the documents themselves use to say "these are the same loan".
_LOAN_ACCOUNT = re.compile(
    r"loan (?:account|a/c)(?:\s*no\.?|\s*number)?\s*:?\s*([A-Z0-9][A-Z0-9/\-]{5,40})",
    re.I,
)


@lru_cache(maxsize=4)
def _store(path: str) -> DocumentStore:
    """One connection per database file, not one per request."""
    return DocumentStore(path)


def get_store(settings: Settings = Depends(get_settings)) -> DocumentStore:
    return _store(str(settings.local_index_path))


class UploadResponse(BaseModel):
    document_id: str
    version: int
    title: str
    kind: str
    clause_count: int
    chunks_written: int
    embeddings_computed: int
    embeddings_reused: int
    reuse_percent: int
    warnings: list[str]
    summary: str


class DocumentSummary(BaseModel):
    document_id: str
    version: int
    title: str
    filename: str
    kind: str
    clause_count: int
    versions: list[int]


def _document_key(text: str, fallback: str) -> str:
    """A stable identity for the loan, so v2 lands on top of v1."""
    if match := _LOAN_ACCOUNT.search(text):
        slug = re.sub(r"[^A-Za-z0-9]+", "-", match.group(1)).strip("-")
        return f"loan-{slug}"
    return f"doc-{re.sub(r'[^A-Za-z0-9]+', '-', Path(fallback).stem).strip('-').lower()}"


@router.post("", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_store),
) -> UploadResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 20 MB.")

    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in MEDIA_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {suffix or '(none)'}. Upload a PDF or a Word .docx file."
            ),
        )

    source = ParseSource(
        content=content, filename=file.filename or f"upload{suffix}", media_type=MEDIA_TYPES[suffix]
    )
    parser = get_parser_for(source, settings)

    # Parsed once here only to learn the loan's identity and whether it revises
    # something already held. Ingest parses again with the resolved id, which
    # is cheap and keeps ingest the single place that writes to the index.
    probe = parser.parse(source)
    probe_text = "\n".join(clause.text for clause in probe.clauses[:12])
    resolved_id = document_id or _document_key(probe_text, source.filename)

    existing = store.versions(resolved_id)
    version = (max(existing) + 1) if existing else 1

    # Re-uploading a file already held is not a new version of the loan. Without
    # this, uploading the original twice produces a phantom v2 identical to v1,
    # and every later version number is wrong.
    incoming = {clause.content_hash for clause in probe.clauses}
    for held in existing:
        stored = store.get(resolved_id, held)
        if stored and {clause.content_hash for clause in stored.clauses} == incoming:
            return UploadResponse(
                document_id=resolved_id,
                version=stored.version,
                title=stored.title,
                kind=stored.kind.value,
                clause_count=len(stored.clauses),
                chunks_written=0,
                embeddings_computed=0,
                embeddings_reused=len(stored.clauses),
                reuse_percent=100,
                warnings=[
                    f"This file is already held as version {stored.version}; "
                    "nothing was re-indexed.",
                    *stored.warnings,
                ],
                summary=stored.summary,
            )

    base = None
    if probe.kind is DocumentKind.AMENDMENT:
        previous = store.latest(resolved_id)
        if previous is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "This document revises an earlier agreement, but the original has not been "
                    "uploaded. Upload the original agreement first, or its unchanged clauses "
                    "would be missing from this version."
                ),
            )
        from app.rag.versioning import ResolvedVersion

        base = ResolvedVersion(
            document_id=resolved_id, version=previous.version, clauses=previous.clauses
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / source.filename
        path.write_bytes(content)
        try:
            result = await ingest_document(
                path=path,
                parser=parser,
                embedder=get_embedding_provider(settings),
                index=get_vector_index(settings),
                document_id=resolved_id,
                version=version,
                base=base,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    store.save(
        StoredDocument(
            document_id=resolved_id,
            version=version,
            title=result.parsed.title or source.filename,
            filename=source.filename,
            kind=result.parsed.kind,
            summary=result.summary,
            metadata=result.parsed.metadata,
            clauses=result.resolved.clauses,
            warnings=tuple(result.notes),
            uploaded_at=datetime.now(UTC).isoformat(),
        )
    )

    total = result.embeddings_computed + result.embeddings_reused
    return UploadResponse(
        document_id=resolved_id,
        version=version,
        title=result.parsed.title or source.filename,
        kind=result.parsed.kind.value,
        clause_count=len(result.resolved.clauses),
        chunks_written=result.chunks_written,
        embeddings_computed=result.embeddings_computed,
        embeddings_reused=result.embeddings_reused,
        reuse_percent=round(100 * result.embeddings_reused / total) if total else 0,
        warnings=result.notes,
        summary=result.summary,
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(store: DocumentStore = Depends(get_store)) -> list[DocumentSummary]:
    seen: dict[str, DocumentSummary] = {}
    for document in store.list_all():
        if document.document_id in seen:
            continue
        seen[document.document_id] = DocumentSummary(
            document_id=document.document_id,
            version=document.version,
            title=document.title,
            filename=document.filename,
            kind=document.kind.value,
            clause_count=len(document.clauses),
            versions=store.versions(document.document_id),
        )
    return list(seen.values())


@router.get("/{document_id}/clauses")
def clauses(
    document_id: str,
    version: int | None = None,
    store: DocumentStore = Depends(get_store),
) -> dict:
    document = store.get(document_id, version) if version is not None else store.latest(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="No such document version.")
    return {
        "document_id": document.document_id,
        "version": document.version,
        "title": document.title,
        "warnings": list(document.warnings),
        "clauses": [
            {
                "clause_id": clause.clause_id,
                "number": clause.number,
                "heading": clause.heading,
                "page": clause.page_start,
                "text": clause.text,
            }
            for clause in document.clauses
        ],
    }
