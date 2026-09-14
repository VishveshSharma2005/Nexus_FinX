"""Loading and validation of the RBI corpus manifest.

The manifest is not documentation -- it is the data the applicability filter
runs on. A wrong ``lender_classes`` entry here means a passage that does not
govern the borrower's loan reaches the model, which is the exact failure this
project exists to prevent. So it is parsed into typed models and validated,
and a malformed manifest raises rather than degrading quietly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.contracts import Applicability, DocumentKind


class FilenameMismatch(BaseModel):
    """Recorded when a corpus file's name disagrees with its contents.

    Kept in the manifest rather than fixed by renaming: the file on disk is the
    evidence, and silently renaming it would hide that the catalogued identity
    was derived from the text rather than the filename.
    """

    model_config = ConfigDict(frozen=True)

    filename_implies: str
    actual_content: str
    action_taken: str


class CorpusDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    file: str
    title: str
    rbi_reference: str
    circular_number: str
    issued_on: date
    document_kind: DocumentKind = DocumentKind.RBI_CIRCULAR
    verified_from_pdf: bool = False
    addressees: tuple[str, ...] = ()
    applicability: Applicability
    topics: tuple[str, ...] = ()
    source_url: str | None = None
    notes: str | None = None
    filename_mismatch: FilenameMismatch | None = None

    @model_validator(mode="after")
    def _effective_dates_are_ordered(self) -> CorpusDocument:
        start, end = self.applicability.effective_from, self.applicability.effective_to
        if start and end and end < start:
            raise ValueError(f"{self.id}: effective_to {end} precedes effective_from {start}")
        return self

    def is_in_force_on(self, as_of: date) -> bool:
        """Whether this document's obligations bind a loan dated ``as_of``.

        A circular issued before a loan but effective after it does not govern
        that loan, however similar its text looks to the question.
        """
        start, end = self.applicability.effective_from, self.applicability.effective_to
        if start and as_of < start:
            return False
        return not (end and as_of > end)


class CorpusManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    documents: tuple[CorpusDocument, ...]

    @model_validator(mode="after")
    def _ids_and_files_are_unique(self) -> CorpusManifest:
        for field in ("id", "file"):
            values = [getattr(doc, field) for doc in self.documents]
            duplicates = {v for v in values if values.count(v) > 1}
            if duplicates:
                raise ValueError(f"Duplicate {field} in manifest: {sorted(duplicates)}")
        return self

    def by_id(self, document_id: str) -> CorpusDocument:
        for doc in self.documents:
            if doc.id == document_id:
                return doc
        raise KeyError(document_id)


def load_manifest(corpus_dir: Path) -> CorpusManifest:
    """Parse ``manifest.json`` and confirm every catalogued file exists."""
    manifest_path = corpus_dir / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = CorpusManifest.model_validate(raw)

    missing = [doc.file for doc in manifest.documents if not (corpus_dir / doc.file).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references files not on disk: {missing}")

    return manifest
