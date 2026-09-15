"""DocxParser: Word documents, read as documents rather than as flat text.

A second implementation of :class:`app.core.contracts.DocumentParser`, added
because the real demo agreements are ``.docx``. It shares the segmenter and the
hashing rules with the PDF parser, so versioning and citations behave
identically whichever format a borrower uploads.

Three things a .docx carries that a PDF does not, and that flattening to text
would throw away:

* **Tables are real.** The fee schedule and the Key Facts Statement rate table
  are ``w:tbl`` elements. Reading them cell-by-cell turns
  "Prepayment Charge | 3% of principal prepaid | On external refinance" into
  three unrelated fragments, and the row is the unit of meaning. Rows are kept
  intact, which is what lets ``app.risk`` extract fees deterministically later.
* **Headings are declared.** Word marks them with a style, so sections are
  known rather than guessed from capitalisation.
* **Page breaks are explicit, and nothing else is.** Word paginates at render
  time, so a .docx has no inherent page numbers. Only author-inserted breaks
  are in the file. Pages are derived from those and the document says so,
  rather than implying a precision that is not there.
"""

from __future__ import annotations

import logging
import zipfile
from datetime import UTC, datetime

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.core.contracts import (
    Clause,
    DocumentKind,
    DocumentMetadata,
    ParsedDocument,
    ParseSource,
    ParseWarning,
)
from app.parsers.lite import detect
from app.parsers.lite.parser import (
    OVERSIZED_CLAUSE_CHARS,
    SUSPICIOUSLY_FEW_CLAUSES,
    UNINFORMATIVE_TITLES,
)
from app.parsers.lite.segment import Line, segment, strip_running_headers
from app.parsers.lite.text import clean_extracted_text, content_hash, document_fingerprint

logger = logging.getLogger(__name__)

DOCX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    }
)

# Word's own marker for an author-inserted page break.
# Below this many distinct top-level numbers there is no sequence to have a
# gap in, only stray digits that happen to start a line.
MIN_NUMBERS_FOR_GAP_CHECK = 5

_PAGE_BREAK = qn("w:br")
_BREAK_TYPE = qn("w:type")


class DocxParser:
    """python-docx implementation of the parser contract."""

    name = "docx"

    def supports(self, source: ParseSource) -> bool:
        if source.media_type.lower() in DOCX_MEDIA_TYPES:
            return True
        return source.filename.lower().endswith(".docx")

    def parse(self, source: ParseSource) -> ParsedDocument:
        warnings: list[ParseWarning] = []
        document_id = source.document_id or f"doc-{document_fingerprint(source.content)}"

        try:
            import io

            document = docx.Document(io.BytesIO(source.content))
        except (zipfile.BadZipFile, ValueError, KeyError, OSError) as exc:
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
                            f"The file could not be opened as a Word document "
                            f"({type(exc).__name__}). No content was extracted."
                        ),
                    ),
                ),
                parsed_at=datetime.now(UTC),
            )

        lines, page_count, table_count = self._read_body(document)

        if not lines:
            warnings.append(
                ParseWarning(
                    code="no_clauses",
                    message=(
                        "No readable text was found in this document, so nothing can be "
                        "cited from it."
                    ),
                )
            )

        warnings.append(
            ParseWarning(
                code="pages_derived_from_breaks",
                message=(
                    "A Word document has no fixed pagination, so page numbers here are "
                    f"derived from the {page_count - 1} explicit page break(s) in the file "
                    "and may differ from what Word displays. Clause numbers are the "
                    "reliable citation for this document."
                ),
            )
        )

        lines, chrome = strip_running_headers(lines, page_count)
        if chrome:
            warnings.append(
                ParseWarning(
                    code="running_headers_removed",
                    message=(
                        "Repeated page header/footer text was excluded from clause "
                        f"segmentation: {'; '.join(repr(c) for c in chrome[:3])}"
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
                page_start=min(segmented.page_start, page_count),
                page_end=min(segmented.page_end, page_count),
                content_hash=content_hash(segmented.text),
                is_ocr=False,
            )
            for ordinal, segmented in enumerate(result.clauses)
        )

        warnings.extend(self._structural_warnings(clauses, page_count, table_count))

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

    def _read_body(self, document: DocxDocument) -> tuple[list[Line], int, int]:
        """Walk paragraphs and tables in document order.

        python-docx exposes paragraphs and tables as separate collections, which
        loses their relative order. Walking the body XML keeps a fee table in
        the place its surrounding clause refers to.
        """
        lines: list[Line] = []
        page = 1
        table_count = 0

        for element in document.element.body.iterchildren():
            if element.tag == qn("w:p"):
                paragraph = Paragraph(element, document)
                page += self._page_breaks_in(element)
                text = clean_extracted_text(paragraph.text)
                if text:
                    lines.append(Line(text=text, page=page, is_heading=self._is_heading(paragraph)))
            elif element.tag == qn("w:tbl"):
                table_count += 1
                lines.extend(self._read_table(Table(element, document), page))
                lines.append(Line(text="", page=page))

        return lines, max(page, 1), table_count

    @staticmethod
    def _page_breaks_in(element) -> int:
        return sum(1 for br in element.iter(_PAGE_BREAK) if br.get(_BREAK_TYPE) == "page")

    @staticmethod
    def _is_heading(paragraph: Paragraph) -> bool:
        """Whether Word declares this paragraph a heading.

        Passed to the segmenter as a flag rather than folded into the text. An
        earlier version upper-cased styled headings so that the shared,
        capitalisation-based segmenter would recognise them, which meant the
        Key Facts Statement's own row labels came back as
        "(III) SANCTIONED LOAN AMOUNT". A system whose whole purpose is to
        quote a borrower's document accurately cannot rewrite it on the way in.
        """
        style = paragraph.style
        name = (style.name or "") if style is not None else ""
        return name.lower().startswith("heading")

    @staticmethod
    def _read_table(table: Table, page: int) -> list[Line]:
        """Render each row as one line, preserving the row as a unit.

        "Prepayment Charge | 3% of principal prepaid | On external refinance"
        only means anything whole. Flattening it into separate cells is how a
        fee silently detaches from its amount.
        """
        lines: list[Line] = []
        for row in table.rows:
            cells = [" ".join(cell.text.split()) for cell in row.cells]
            # Word repeats a cell's text across a horizontal merge.
            deduplicated: list[str] = []
            for cell in cells:
                if not deduplicated or cell != deduplicated[-1]:
                    deduplicated.append(cell)
            text = " | ".join(cell for cell in deduplicated if cell)
            if text:
                lines.append(Line(text=text, page=page))
        return lines

    @staticmethod
    def _structural_warnings(
        clauses: tuple[Clause, ...], page_count: int, table_count: int
    ) -> list[ParseWarning]:
        warnings: list[ParseWarning] = []

        if clauses and len(clauses) < SUSPICIOUSLY_FEW_CLAUSES:
            warnings.append(
                ParseWarning(
                    code="few_clauses",
                    message=(
                        f"Only {len(clauses)} clause(s) were found. The document's "
                        "structure may not have been recognised."
                    ),
                )
            )

        oversized = [c for c in clauses if len(c.text) > OVERSIZED_CLAUSE_CHARS]
        if oversized:
            warnings.append(
                ParseWarning(
                    code="oversized_clauses",
                    message=(
                        f"{len(oversized)} block(s) exceeded {OVERSIZED_CLAUSE_CHARS:,} "
                        "characters without an internal clause boundary."
                    ),
                    page=oversized[0].page_start,
                )
            )

        # A gap in top-level clause numbering is worth surfacing: it is usually
        # an authoring slip, but it can also mean a section failed to extract,
        # and the borrower should not have to notice that themselves.
        #
        # Only when there is a real sequence to have a gap in. A Key Facts
        # Statement labels its rows "(i), (ii), (iii)" and carries no clause
        # numbers at all; reading the "3" out of "3 (three) calendar days" and
        # announcing that the document was missing clauses 1 and 2 was alarming
        # and wrong.
        tops: set[int] = set()
        for clause in clauses:
            if clause.number and clause.number[0].isdigit():
                tops.add(int(clause.number.split(".")[0]))

        if len(tops) >= MIN_NUMBERS_FOR_GAP_CHECK:
            missing = sorted(set(range(1, max(tops) + 1)) - tops)
            if missing:
                warnings.append(
                    ParseWarning(
                        code="clause_numbering_gap",
                        message=(
                            "The document's clause numbering skips "
                            f"{', '.join(str(n) for n in missing)}. Either the source "
                            "document omits those clauses, or they were not extracted."
                        ),
                    )
                )

        return warnings

    @staticmethod
    def _title(document_id: str, source: ParseSource, clauses: tuple[Clause, ...]) -> str:
        for clause in clauses[:5]:
            candidate = (clause.heading or clause.text.split("\n")[0]).strip()
            parts = [part.strip() for part in candidate.split(" / ") if part.strip()]
            meaningful = [p for p in parts if p.lower() not in UNINFORMATIVE_TITLES]
            if not meaningful:
                continue
            title = " / ".join(meaningful)
            if 6 <= len(title) <= 120:
                return title
        return source.filename or document_id

    @staticmethod
    def _metadata(full_text: str, source: ParseSource, kind: DocumentKind) -> DocumentMetadata:
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
            agreement_date=resolve(
                hint.agreement_date if hint else None, detect.detect_agreement_date
            ),
            language=hint.language if hint else "en",
            extra=dict(hint.extra) if hint else {},
        )


__all__ = ["DocxParser"]
