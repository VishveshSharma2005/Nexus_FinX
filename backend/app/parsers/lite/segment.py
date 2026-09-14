"""Clause segmentation.

Citation quality is bounded by segmentation quality: "clause 7.2 on page 4" is
only useful if clause 7.2 is actually one unit. Indian loan agreements mix
three shapes -- decimal-numbered clauses, capitalised section headings, and
unnumbered prose -- so segmentation tries them in order of how much structure
they carry, and records which one it fell back to.

The fallback level is reported rather than hidden. A document segmented by
paragraph produces weaker citations than one segmented by clause number, and
downstream code and the user are both entitled to know that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class SegmentationStrategy(StrEnum):
    """Which structure the segmenter actually found, best to worst."""

    CLAUSE_NUMBERING = "clause_numbering"
    HEADINGS = "headings"
    PARAGRAPHS = "paragraphs"
    PAGES = "pages"


@dataclass(frozen=True)
class Line:
    text: str
    page: int
    is_ocr: bool = False


@dataclass
class SegmentedClause:
    text: str
    page_start: int
    page_end: int
    number: str | None = None
    heading: str | None = None
    is_ocr: bool = False


@dataclass
class SegmentationResult:
    clauses: list[SegmentedClause]
    strategy: SegmentationStrategy
    notes: list[str] = field(default_factory=list)


# A numbered clause start: "7.", "7.2", "12.3.1", optionally followed by a
# title. The trailing \s+ before the body is what keeps "7.5% per annum" and
# "1,00,000" from being mistaken for clause numbers -- in those the character
# after the digits is not whitespace.
_NUMBERED = re.compile(r"^(?P<number>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?P<rest>[A-Za-z\"'(].*)$")

# "SCHEDULE II", "ANNEXURE A", "PART 3", "ARTICLE 5", "SECTION 2"
_STRUCTURAL = re.compile(
    r"^(?P<number>(?:SCHEDULE|ANNEXURE|ANNEX|APPENDIX|PART|SECTION|ARTICLE|CLAUSE)"
    r"(?:\s+[IVXLC0-9]+[A-Z]?)?)\b[:.\-\s]*(?P<rest>.*)$",
    re.IGNORECASE,
)

_MAX_HEADING_WORDS = 12


def _is_capitalised_heading(line: str) -> bool:
    """A short, mostly-uppercase line with no terminal full stop."""
    stripped = line.strip()
    if not (3 <= len(stripped) <= 90) or stripped.endswith("."):
        return False
    if len(stripped.split()) > _MAX_HEADING_WORDS:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= 0.85


@dataclass
class _Block:
    start: int
    number: str | None
    heading: str | None
    lines: list[Line]


def _find_starts(lines: list[Line]) -> tuple[list[tuple[int, str | None, str | None]], bool]:
    """Locate clause-start lines. Returns (starts, found_real_numbering)."""
    starts: list[tuple[int, str | None, str | None]] = []
    numbered_count = 0

    for i, line in enumerate(lines):
        text = line.text.strip()
        if not text:
            continue

        match = _NUMBERED.match(text)
        if match:
            # Only a genuine title becomes a heading. Taking any short line
            # would set a one-sentence clause's heading to its own text, which
            # then gets duplicated onto everything that follows it.
            rest = match.group("rest").strip()
            heading = rest if _looks_like_a_heading(text) else None
            starts.append((i, match.group("number"), heading))
            numbered_count += 1
            continue

        match = _STRUCTURAL.match(text)
        # Only when the line reads as a title. Without this, prose beginning
        # "Section 21 of the Banking Regulation Act..." would open a new clause
        # in the middle of a sentence.
        if match and (text.isupper() or len(text) <= 90):
            # The whole line is the title. Using only the part after the
            # keyword would turn "SCHEDULE OF CHARGES" into "OF CHARGES".
            starts.append((i, match.group("number").upper(), text))
            continue

        if _is_capitalised_heading(text):
            starts.append((i, None, text))

    return starts, numbered_count >= 3


def _blocks_from_starts(
    lines: list[Line], starts: list[tuple[int, str | None, str | None]]
) -> list[_Block]:
    blocks: list[_Block] = []
    for position, (index, number, heading) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        blocks.append(_Block(index, number, heading, lines[index:end]))
    return blocks


def _looks_like_a_heading(text: str) -> bool:
    """Whether a lone line is a section title rather than a one-line clause.

    This distinction decides whether the line is folded into the next clause as
    context or kept as a clause in its own right. Getting it wrong in the
    folding direction silently deletes content, so the test is deliberately
    strict: a line that ends in a full stop, or runs to sentence length, is
    treated as substance.
    """
    stripped = text.strip()
    if not stripped or stripped[-1] in ".;:,":
        return False

    numbered = _NUMBERED.match(stripped)
    structural = None if numbered else _STRUCTURAL.match(stripped)
    if numbered:
        remainder = numbered.group("rest")
    elif structural:
        remainder = structural.group("rest")
    else:
        remainder = stripped

    if len(remainder.split()) > _MAX_HEADING_WORDS:
        return False

    # "3.1 Annual rate of interest: 13.90% per annum" is a Key Facts Statement
    # row, not a section title. Folding it into the next clause as a heading
    # would delete the single most important number in the document, so a
    # label:value line and anything carrying a figure are treated as content.
    label, separator, value = remainder.partition(":")
    if separator and value.strip():
        return False
    return not any(character.isdigit() for character in remainder)


def _promote_bare_headings(blocks: list[_Block]) -> list[_Block]:
    """Fold a heading-only block into the clauses it introduces.

    A document numbered "7." / "7.1" / "7.2" would otherwise yield a clause
    whose entire content is the word "DEFAULT". Carrying it as context on its
    children is more useful than citing it alone.

    A short block that is *not* heading-shaped -- a one-line fee clause, say --
    is kept as its own clause. Consecutive headings accumulate rather than
    overwrite one another, because an overwritten heading is lost text.
    """
    result: list[_Block] = []
    pending: list[str] = []
    section: str | None = None

    for position, block in enumerate(blocks):
        first_line = block.lines[0].text.strip() if block.lines else ""
        body = " ".join(line.text.strip() for line in block.lines[1:]).strip()
        has_successor = position + 1 < len(blocks)

        if not body and has_successor and _looks_like_a_heading(first_line):
            pending.append(block.heading or first_line)
            continue

        if pending:
            section = " / ".join(pending)
            pending.clear()
        elif block.number and not block.number[0].isdigit():
            # "SCHEDULE", "ANNEXURE", "PART" open a new top-level section, so
            # the previous one stops applying. Without this a schedule would be
            # cited as though it sat under the last numbered clause's heading.
            section = None

        # The section title applies to every clause under it, not only the
        # first. Citing "clause 7.2" without "Default and penal charges" loses
        # the context that tells a borrower what they are reading.
        block.heading = " / ".join(part for part in (section, block.heading) if part) or None
        result.append(block)

    return result


def strip_running_headers(lines: list[Line], page_count: int) -> tuple[list[Line], list[str]]:
    """Remove repeated page headers and footers.

    Running chrome otherwise becomes a false clause boundary -- a capitalised
    banner at the top of page 2 reads exactly like a section heading -- and
    pollutes both citations and the document title.

    Three conditions must hold together before a line is dropped, because
    discarding real text is a worse outcome than keeping some chrome: the line
    is short, it appears on most pages, and it sits at the top or bottom of
    every page it appears on.
    """
    if page_count < 2:
        return lines, []

    # A short document has no majority to speak of, so require the line on
    # every page. Longer ones tolerate a title page that omits the header.
    threshold = page_count if page_count <= 3 else int(page_count * 0.6 + 0.5)
    by_page: dict[int, list[str]] = {}
    for line in lines:
        text = line.text.strip()
        if text:
            by_page.setdefault(line.page, []).append(text)

    occurrences: dict[str, set[int]] = {}
    edge_only: dict[str, bool] = {}
    for page, texts in by_page.items():
        for position, text in enumerate(texts):
            if len(text) > 120:
                continue
            at_edge = position < 2 or position >= len(texts) - 2
            occurrences.setdefault(text, set()).add(page)
            edge_only[text] = edge_only.get(text, True) and at_edge

    chrome = {
        text
        for text, pages in occurrences.items()
        if len(pages) >= threshold and edge_only.get(text, False)
    }
    if not chrome:
        return lines, []

    kept = [line for line in lines if line.text.strip() not in chrome]
    return kept, sorted(chrome)


def _to_clause(block: _Block) -> SegmentedClause | None:
    text = "\n".join(line.text for line in block.lines).strip()
    if not text:
        return None
    pages = [line.page for line in block.lines]
    return SegmentedClause(
        text=text,
        page_start=min(pages),
        page_end=max(pages),
        number=block.number,
        heading=block.heading,
        is_ocr=any(line.is_ocr for line in block.lines),
    )


def _segment_by_paragraphs(lines: list[Line], target_chars: int = 900) -> list[SegmentedClause]:
    """Group blank-line-separated paragraphs up to a target size."""
    clauses: list[SegmentedClause] = []
    current: list[Line] = []

    def flush() -> None:
        if not current:
            return
        text = "\n".join(line.text for line in current).strip()
        if text:
            pages = [line.page for line in current]
            clauses.append(
                SegmentedClause(
                    text=text,
                    page_start=min(pages),
                    page_end=max(pages),
                    is_ocr=any(line.is_ocr for line in current),
                )
            )
        current.clear()

    size = 0
    for line in lines:
        if not line.text.strip():
            if size >= target_chars:
                flush()
                size = 0
            continue
        current.append(line)
        size += len(line.text)

    flush()
    return clauses


def _segment_by_pages(lines: list[Line]) -> list[SegmentedClause]:
    clauses: list[SegmentedClause] = []
    for page in sorted({line.page for line in lines}):
        page_lines = [line for line in lines if line.page == page]
        text = "\n".join(line.text for line in page_lines).strip()
        if text:
            clauses.append(
                SegmentedClause(
                    text=text,
                    page_start=page,
                    page_end=page,
                    is_ocr=any(line.is_ocr for line in page_lines),
                )
            )
    return clauses


MIN_STRUCTURED_CLAUSES = 3


def segment(lines: list[Line]) -> SegmentationResult:
    """Split document lines into clauses, degrading explicitly."""
    notes: list[str] = []
    lines = [line for line in lines if line.text is not None]

    if not any(line.text.strip() for line in lines):
        return SegmentationResult([], SegmentationStrategy.PAGES, ["Document had no text at all."])

    starts, has_numbering = _find_starts(lines)

    if starts:
        blocks = _promote_bare_headings(_blocks_from_starts(lines, starts))

        # Text before the first detected start would otherwise be dropped.
        first = starts[0][0]
        if any(line.text.strip() for line in lines[:first]):
            preamble = _Block(0, None, "Preamble", lines[:first])
            blocks.insert(0, preamble)

        clauses = [clause for clause in (_to_clause(block) for block in blocks) if clause]
        if len(clauses) >= MIN_STRUCTURED_CLAUSES:
            strategy = (
                SegmentationStrategy.CLAUSE_NUMBERING
                if has_numbering
                else SegmentationStrategy.HEADINGS
            )
            if not has_numbering:
                notes.append(
                    "No decimal clause numbering found; segmented on capitalised headings. "
                    "Citations will name a heading rather than a clause number."
                )
            return SegmentationResult(clauses, strategy, notes)

    clauses = _segment_by_paragraphs(lines)
    if len(clauses) >= MIN_STRUCTURED_CLAUSES:
        notes.append(
            "No clause numbering or headings found; segmented on paragraphs. "
            "Citations will be page-level and less precise."
        )
        return SegmentationResult(clauses, SegmentationStrategy.PARAGRAPHS, notes)

    notes.append(
        "Document has no detectable internal structure; segmented one clause per page. "
        "Treat citations from this document as page references only."
    )
    return SegmentationResult(_segment_by_pages(lines), SegmentationStrategy.PAGES, notes)
