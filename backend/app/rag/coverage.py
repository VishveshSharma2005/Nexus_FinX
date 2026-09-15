"""Coverage gaps: questions the corpus cannot answer for *this* loan's date.

The applicability filter already keeps the 2025 pre-payment Directions away
from a loan sanctioned before 1 January 2026 -- they do not govern it. But
excluding the wrong rule is only half of an honest answer. The Directions
repealed eight earlier circulars, and those still govern older loans; none of
them is in FinX's corpus. Without a notice, a borrower with a 2025 loan asking
"can they charge me for prepaying?" would get an answer drawn only from their
agreement, silently missing the regulation that actually applied.

This module turns the manifest's repeal records into that notice. It is
deterministic: the dates and the list of missing circulars come from the
manifest, never from a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.core.corpus import CorpusManifest


@dataclass(frozen=True)
class CoverageGap:
    """Regulation that governed this loan and is not in the corpus."""

    superseded_by: str
    superseded_by_title: str
    effective_from: date
    missing_circulars: tuple[str, ...]
    message: str


def find_gaps(
    question: str,
    *,
    loan_date: date | None,
    manifest: CorpusManifest,
) -> list[CoverageGap]:
    """Report repealed regulation that governed this loan and is not held.

    A gap exists when the question falls within the subject of a repeal, and the
    loan predates the repeal's effective date -- so the repealed circulars, not
    their replacement, are what applied to it.
    """
    if loan_date is None:
        return []

    lowered = question.lower()
    gaps: list[CoverageGap] = []

    for document in manifest.documents:
        repeal = document.repeals
        if repeal is None or loan_date >= repeal.effective_from:
            continue
        if not any(keyword in lowered for keyword in repeal.subject_keywords):
            continue

        gaps.append(
            CoverageGap(
                superseded_by=document.rbi_reference,
                superseded_by_title=document.title,
                effective_from=repeal.effective_from,
                missing_circulars=repeal.circulars,
                message=(
                    f"Your loan is dated {_long(loan_date)}, before "
                    f"{_long(repeal.effective_from)}. The RBI rules on this subject that "
                    f"applied to it are the {len(repeal.circulars)} earlier circulars that "
                    f"{document.rbi_reference} later repealed, and FinX does not hold them. "
                    f"FinX can explain what your agreement says, but cannot tell you from a "
                    f"primary source whether the regulation of the time permitted it. "
                    f"{document.rbi_reference} itself does not apply to your loan."
                ),
            )
        )

    return gaps


def _long(day: date) -> str:
    """'1 January 2026' -- strftime's unpadded day is not portable to Windows."""
    return f"{day.day} {day:%B %Y}"
