"""OCR fallback for scanned pages.

Scanned agreements are common, and a page that yields no embedded text is the
single most dangerous input this system can receive: silently treating it as
empty produces a document that looks parsed, indexes cleanly, and is missing
exactly the clause the borrower asked about.

So every path through this module ends in either extracted text or an explicit
warning. None of them ends in silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Below this many characters a page is treated as image-only rather than text.
# Scanned pages usually yield a handful of stray glyphs, not zero.
MIN_TEXT_CHARS_PER_PAGE = 40

# Rendering above the PDF's nominal 72 dpi materially improves Tesseract's
# accuracy on the small type used for fee schedules.
OCR_RENDER_DPI = 300


@dataclass(frozen=True)
class OcrOutcome:
    text: str
    used_ocr: bool
    warning_code: str | None = None
    warning_message: str | None = None


class OcrEngine:
    """Thin wrapper over Tesseract that never raises into the parser."""

    def __init__(self, *, enabled: bool = True, tesseract_cmd: str = "") -> None:
        self.enabled = enabled
        self._tesseract_cmd = tesseract_cmd
        self._available: bool | None = None

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available

        try:
            import pytesseract

            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
            pytesseract.get_tesseract_version()
            self._available = True
        except Exception as exc:
            logger.info("OCR unavailable: %s", exc)
            self._available = False

        return self._available

    def image_to_text(self, page, *, languages: str = "eng") -> OcrOutcome:
        """Render a PyMuPDF page and read it with Tesseract."""
        if not self.enabled:
            return OcrOutcome(
                text="",
                used_ocr=False,
                warning_code="ocr_disabled",
                warning_message=(
                    "This page contains no selectable text and OCR is disabled, "
                    "so its contents were not read."
                ),
            )

        if not self.is_available():
            return OcrOutcome(
                text="",
                used_ocr=False,
                warning_code="ocr_unavailable",
                warning_message=(
                    "This page appears to be scanned, but Tesseract is not installed or not "
                    "reachable, so its contents were not read. Answers will not cover this page."
                ),
            )

        try:
            import io

            import pytesseract
            from PIL import Image

            pixmap = page.get_pixmap(dpi=OCR_RENDER_DPI)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image, lang=languages)
        except Exception as exc:
            logger.warning("OCR failed on a page: %s", exc)
            return OcrOutcome(
                text="",
                used_ocr=False,
                warning_code="ocr_failed",
                warning_message=f"OCR failed on this page ({type(exc).__name__}); it was not read.",
            )

        if not text.strip():
            return OcrOutcome(
                text="",
                used_ocr=True,
                warning_code="ocr_empty",
                warning_message=(
                    "OCR ran on this page but recovered no readable text. "
                    "It may be blank, or the scan quality may be too low."
                ),
            )

        return OcrOutcome(text=text, used_ocr=True)
