"""The scanned-document path.

A scanned agreement is the input most likely to produce a confidently wrong
answer, because a page with no embedded text extracts as empty and the document
still looks successfully parsed. These tests cover both outcomes that are
acceptable -- text recovered by OCR, or an explicit warning -- and assert that
the third outcome, silence, never happens.

The OCR tests skip when Tesseract is absent so the suite stays green on a
machine without it, but the degradation test runs everywhere.
"""

from __future__ import annotations

import fitz
import pytest

from app.config import Settings, get_settings
from app.core.contracts import ParseSource
from app.parsers.lite.ocr import OcrEngine
from app.parsers.lite.parser import LiteParser

SCANNED_TEXT_LINES = [
    "6. PREPAYMENT AND FORECLOSURE",
    "6.1 The Borrower may prepay the Loan after twelve months.",
    "6.2 A foreclosure charge of 4% shall apply on prepayment.",
]


def _scanned_pdf() -> bytes:
    """A PDF whose only content is an image of text -- no selectable text."""
    rendered = fitz.open()
    page = rendered.new_page(width=595, height=842)
    y = 90
    for line in SCANNED_TEXT_LINES:
        page.insert_text((60, y), line, fontsize=16, fontname="helv")
        y += 40
    pixmap = page.get_pixmap(dpi=200)
    rendered.close()

    scanned = fitz.open()
    image_page = scanned.new_page(width=595, height=842)
    image_page.insert_image(fitz.Rect(0, 0, 595, 842), stream=pixmap.tobytes("png"))
    data = scanned.tobytes()
    scanned.close()
    return data


@pytest.fixture(scope="module")
def scanned_bytes() -> bytes:
    return _scanned_pdf()


def _ocr_available() -> bool:
    settings = get_settings()
    return OcrEngine(enabled=True, tesseract_cmd=settings.tesseract_cmd).is_available()


needs_tesseract = pytest.mark.skipif(not _ocr_available(), reason="Tesseract not installed")


def test_the_fixture_really_has_no_embedded_text(scanned_bytes: bytes) -> None:
    """Guard on the test itself: if this PDF had text, the OCR test would
    pass without OCR ever running."""
    with fitz.open(stream=scanned_bytes, filetype="pdf") as document:
        assert document[0].get_text().strip() == ""


@needs_tesseract
def test_scanned_pages_are_read_by_ocr(scanned_bytes: bytes) -> None:
    settings = get_settings()
    parser = LiteParser(ocr_enabled=True, tesseract_cmd=settings.tesseract_cmd)
    parsed = parser.parse(ParseSource(content=scanned_bytes, filename="scanned.pdf"))

    assert parsed.clauses, "OCR should have recovered text from the scan"
    combined = " ".join(clause.text for clause in parsed.clauses).lower()
    assert "prepay" in combined
    assert "foreclosure" in combined


@needs_tesseract
def test_ocr_derived_clauses_are_flagged(scanned_bytes: bytes) -> None:
    """Citations into OCR text carry more uncertainty, so the provenance is
    recorded rather than presented as though it were extracted cleanly."""
    settings = get_settings()
    parser = LiteParser(ocr_enabled=True, tesseract_cmd=settings.tesseract_cmd)
    parsed = parser.parse(ParseSource(content=scanned_bytes, filename="scanned.pdf"))

    assert any(clause.is_ocr for clause in parsed.clauses)


def test_without_ocr_the_document_warns_instead_of_looking_empty(
    scanned_bytes: bytes, settings: Settings
) -> None:
    """The important case: no Tesseract must not mean a silently blank parse."""
    parser = LiteParser(ocr_enabled=False)
    parsed = parser.parse(ParseSource(content=scanned_bytes, filename="scanned.pdf"))

    codes = {warning.code for warning in parsed.warnings}
    assert "ocr_disabled" in codes
    assert any(warning.page == 1 for warning in parsed.warnings)
    assert not parsed.clauses


def test_a_missing_tesseract_binary_degrades_without_raising(scanned_bytes: bytes) -> None:
    parser = LiteParser(ocr_enabled=True, tesseract_cmd="C:/definitely/not/tesseract.exe")
    parsed = parser.parse(ParseSource(content=scanned_bytes, filename="scanned.pdf"))

    codes = {warning.code for warning in parsed.warnings}
    assert codes & {"ocr_unavailable", "ocr_failed"}


def test_a_text_native_pdf_does_not_invoke_ocr() -> None:
    """OCR is a fallback, not a default: rendering every page would be slow
    and would degrade text that extracted perfectly well."""
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((60, 90), "1.1 " + "The Loan shall be repaid in equated instalments. " * 3)
    data = document.tobytes()
    document.close()

    parser = LiteParser(ocr_enabled=True, tesseract_cmd="C:/definitely/not/tesseract.exe")
    parsed = parser.parse(ParseSource(content=data, filename="native.pdf"))

    assert parsed.clauses
    assert not any(clause.is_ocr for clause in parsed.clauses)
    assert not {w.code for w in parsed.warnings} & {"ocr_unavailable", "ocr_failed"}
