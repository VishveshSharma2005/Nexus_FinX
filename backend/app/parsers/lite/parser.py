"""LiteParser: PyMuPDF text extraction, OCR fallback, clause segmentation.

One concrete implementation of :class:`app.core.contracts.DocumentParser`, and
nothing outside ``app.parsers`` may import it. It is deliberately the
replaceable part of the system: a production parser with table and layout
understanding can take its place without the RAG layer noticing.

Design commitments, all of them contract invariants:

* Determinism -- the document id derives from the content bytes, so the same
  input yields the same clause ids and hashes on every run.
* Never raise on a document it said it supports. A page that cannot be read
  becomes a warning attached to the result, because a parse that fails loudly
  on page 9 of 12 is less useful to a borrower than one that returns 11 pages
  and says so.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import fitz

from app.core.contracts import (
    Clause,
    DocumentKind,
    DocumentMetadata,
    ParsedDocument,
    ParseSource,
    ParseWarning,
)
from app.parsers.lite import detect
from app.parsers.lite.ocr import MIN_TEXT_CHARS_PER_PAGE, OcrEngine
from app.parsers.lite.segment import (
    Line,
    SegmentationStrategy,
    segment,
    strip_running_headers,
)
from app.parsers.lite.text import clean_extracted_text, content_hash, document_fingerprint

logger = logging.getLogger(__name__)

SUPPORTED_MEDIA_TYPES = frozenset({"application/pdf", "application/x-pdf"})

# A document that parses to very few clauses is usually a segmentation failure
# rather than a genuinely short agreement, and the user should be told.
SUSPICIOUSLY_FEW_CLAUSES = 3

# Beyond this, a "clause" is really an unsegmented block -- typically a
# schedule or an annex table. It still indexes, but a citation pointing at it
# is too coarse to be worth much, so the document says so.
OVERSIZED_CLAUSE_CHARS = 4000

# Headings that identify nothing: labels the segmenter introduced, and the
# navigation furniture of PDFs saved from a web page.
UNINFORMATIVE_TITLES = frozenset(
    {
        "preamble",
        "search",
        "notifications",
        "master circulars",
        "master directions",
        "annex",
        "annexure",
        "schedule",
        "more links :",
        "more links",
        "e-lms",
        "rss",
        "follow rbi",
    }
)


class LiteParser:
    """PyMuPDF + Tesseract implementation of the parser contract."""

    name = "lite"

    def __init__(self, *, ocr_enabled: bool = True, tesseract_cmd: str = "") -> None:
        self._ocr = OcrEngine(enabled=ocr_enabled, tesseract_cmd=tesseract_cmd)

    # -- contract ----------------------------------------------------------

    def supports(self, source: ParseSource) -> bool:
        if source.media_type.lower() in SUPPORTED_MEDIA_TYPES:
            return True
        return source.filename.lower().endswith(".pdf")

    def parse(self, source: ParseSource) -> ParsedDocument:
        warnings: list[ParseWarning] = []
        document_id = source.document_id or f"doc-{document_fingerprint(source.content)}"

        try:
            document = fitz.open(stream=source.content, filetype="pdf")
        except Exception as exc:
            logger.warning("Could not open %s: %s", source.filename, exc)
            return ParsedDocument(
                document_id=document_id,
                version=source.version,
                kind=source.kind_hint or DocumentKind.UNKNOWN,
                title=source.filename,
                page_count=0,
                parser_name=self.name,
                warnings=(
                    ParseWarning(
                        code="unreadable_document",
                        message=(
                            f"The file could not be opened as a PDF ({type(exc).__name__}). "
                            "No content was extracted."
                        ),
                    ),
                ),
                parsed_at=datetime.now(UTC),
            )

        with document:
            lines, page_warnings = self._extract_lines(document)
            warnings.extend(page_warnings)
            page_count = document.page_count

        lines, chrome = strip_running_headers(lines, page_count)
        if chrome:
            # Reported, not silent: if this ever removes real text, the warning
            # is what makes that visible instead of mysterious.
            warnings.append(
                ParseWarning(
                    code="running_headers_removed",
                    message=(
                        "Repeated page header/footer text was excluded from clause "
                        f"segmentation: {'; '.join(repr(c) for c in chrome[:3])}"
                        + (f" and {len(chrome) - 3} more" if len(chrome) > 3 else "")
                    ),
                )
            )

        result = segment(lines)
        warnings.extend(
            ParseWarning(code=f"segmentation_{result.strategy.value}", message=note)
            for note in result.notes
        )

        clauses = tuple(
            Clause(
                clause_id=f"{document_id}-v{source.version}-c{ordinal:04d}",
                ordinal=ordinal,
                heading=segmented.heading,
                number=segmented.number,
                text=segmented.text,
                page_start=segmented.page_start,
                page_end=segmented.page_end,
                content_hash=content_hash(segmented.text),
                is_ocr=segmented.is_ocr,
            )
            for ordinal, segmented in enumerate(result.clauses)
        )

        if not clauses:
            warnings.append(
                ParseWarning(
                    code="no_clauses",
                    message=(
                        "No readable text was extracted from this document, so nothing can be "
                        "cited from it."
                    ),
                )
            )
        elif len(clauses) < SUSPICIOUSLY_FEW_CLAUSES and page_count > 1:
            warnings.append(
                ParseWarning(
                    code="few_clauses",
                    message=(
                        f"Only {len(clauses)} clause(s) were found across {page_count} pages. "
                        "The document's structure may not have been recognised, so citations "
                        "may be imprecise."
                    ),
                )
            )

        oversized = [c for c in clauses if len(c.text) > OVERSIZED_CLAUSE_CHARS]
        if oversized:
            warnings.append(
                ParseWarning(
                    code="oversized_clauses",
                    message=(
                        f"{len(oversized)} block(s) exceeded {OVERSIZED_CLAUSE_CHARS:,} characters "
                        "without an internal clause boundary, typically a schedule or annex table. "
                        "Citations into these will name a page rather than a clause."
                    ),
                    page=oversized[0].page_start,
                )
            )

        full_text = "\n".join(clause.text for clause in clauses)
        kind = detect.detect_kind(full_text, hint=source.kind_hint)

        return ParsedDocument(
            document_id=document_id,
            version=source.version,
            kind=kind,
            title=self._title(document_id, source, clauses),
            page_count=page_count,
            clauses=clauses,
            metadata=self._metadata(full_text, source, kind),
            parser_name=self.name,
            warnings=tuple(warnings),
            parsed_at=datetime.now(UTC),
        )

    # -- internals ---------------------------------------------------------

    def _extract_lines(self, document) -> tuple[list[Line], list[ParseWarning]]:
        """Read every page, falling back to OCR for image-only pages."""
        lines: list[Line] = []
        warnings: list[ParseWarning] = []

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            is_ocr = False

            try:
                raw = page.get_text("text")
            except Exception as exc:
                logger.warning("Text extraction failed on page %s: %s", page_number, exc)
                raw = ""
                warnings.append(
                    ParseWarning(
                        code="page_extraction_failed",
                        message=f"Text extraction failed on this page ({type(exc).__name__}).",
                        page=page_number,
                    )
                )

            if len(raw.strip()) < MIN_TEXT_CHARS_PER_PAGE:
                outcome = self._ocr.image_to_text(page)
                if outcome.text.strip():
                    raw = outcome.text
                    is_ocr = outcome.used_ocr
                if outcome.warning_code:
                    warnings.append(
                        ParseWarning(
                            code=outcome.warning_code,
                            message=outcome.warning_message or "",
                            page=page_number,
                        )
                    )

            cleaned = clean_extracted_text(raw)
            if not cleaned:
                continue

            lines.extend(
                Line(text=line, page=page_number, is_ocr=is_ocr) for line in cleaned.split("\n")
            )
            # Preserve the page break so paragraph fallback does not weld the
            # last line of one page to the first line of the next.
            lines.append(Line(text="", page=page_number, is_ocr=is_ocr))

        return lines, warnings

    @staticmethod
    def _title(document_id: str, source: ParseSource, clauses: tuple[Clause, ...]) -> str:
        """Prefer the document's own first heading over the upload filename."""
        for clause in clauses[:5]:
            candidate = (clause.heading or clause.text.split("\n")[0]).strip()
            # A composed heading such as "SEARCH / NOTIFICATIONS" is only as
            # useful as its most meaningful part, and both of those parts are
            # navigation chrome from a PDF saved off a web page.
            parts = [part.strip() for part in candidate.split(" / ") if part.strip()]
            meaningful = [part for part in parts if part.lower() not in UNINFORMATIVE_TITLES]
            if not meaningful:
                continue
            title = " / ".join(meaningful)
            if 6 <= len(title) <= 120:
                return title
        return source.filename or document_id

    @staticmethod
    def _metadata(full_text: str, source: ParseSource, kind: DocumentKind) -> DocumentMetadata:
        """Detected metadata, with any caller-supplied hint taking precedence.

        Detection is skipped entirely for RBI circulars. The lender classes
        named in a circular are its *addressees*, not the lender on anyone's
        loan, and inferring borrower metadata from regulation would corrupt the
        very fields the applicability filter compares against. For corpus
        documents that metadata comes from the manifest instead.
        """
        hint = source.metadata_hint
        detected = kind is not DocumentKind.RBI_CIRCULAR

        def resolve(hinted, detector):
            if hinted is not None:
                return hinted
            return detector(full_text) if detected else None

        return DocumentMetadata(
            loan_type=resolve(hint.loan_type if hint else None, detect.detect_loan_type),
            borrower_type=resolve(
                hint.borrower_type if hint else None, detect.detect_borrower_type
            ),
            lender_class=resolve(hint.lender_class if hint else None, detect.detect_lender_class),
            purpose=hint.purpose if hint else None,
            agreement_date=(
                hint.agreement_date
                if hint and hint.agreement_date
                else detect.detect_agreement_date(full_text)
            ),
            language=hint.language if hint else "en",
            extra=dict(hint.extra) if hint else {},
        )


__all__ = ["LiteParser", "SegmentationStrategy"]
