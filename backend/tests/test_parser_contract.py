"""The DocumentParser contract.

Every test here is written against the interface in ``app.core.contracts`` and
parameterised over every registered implementation. Nothing in this file names
``LiteParser`` except the registry lookup that enumerates implementations, so
these tests keep passing -- and keep meaning something -- when a production
parser replaces the hackathon one.

If you add a parser, you add nothing here. That is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz
import pytest

from app.config import Settings, get_settings
from app.core.contracts import DocumentKind, DocumentParser, ParsedDocument, ParseSource
from app.parsers.lite.text import normalise_for_hash
from app.parsers.registry import PARSERS, build_parser

SAMPLES_DIR = get_settings().sample_corpus_dir
AGREEMENT = SAMPLES_DIR / "personal_loan_agreement_v1.pdf"


@pytest.fixture(params=sorted(PARSERS), ids=sorted(PARSERS))
def parser(request, settings: Settings) -> DocumentParser:
    """Each registered implementation, in turn."""
    return build_parser(request.param, settings)


def _source(path: Path, **kwargs) -> ParseSource:
    return ParseSource(content=path.read_bytes(), filename=path.name, **kwargs)


@pytest.fixture(scope="session")
def agreement_bytes() -> bytes:
    if not AGREEMENT.is_file():
        pytest.skip(f"Run scripts/make_sample_documents.py first: {AGREEMENT} missing")
    return AGREEMENT.read_bytes()


@pytest.fixture
def parsed(parser: DocumentParser, agreement_bytes: bytes) -> ParsedDocument:
    return parser.parse(ParseSource(content=agreement_bytes, filename=AGREEMENT.name))


# --- Interface conformance -------------------------------------------------


def test_implements_the_protocol(parser: DocumentParser) -> None:
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser.name, str) and parser.name


def test_supports_pdf_and_rejects_unknown_formats(parser: DocumentParser) -> None:
    assert parser.supports(ParseSource(content=b"%PDF-1.7", filename="a.pdf"))
    assert not parser.supports(
        ParseSource(content=b"hello", filename="notes.txt", media_type="text/plain")
    )


# --- Result invariants -----------------------------------------------------


def test_returns_a_parsed_document(parsed: ParsedDocument) -> None:
    assert isinstance(parsed, ParsedDocument)
    assert parsed.page_count >= 1
    assert parsed.clauses


def test_ordinals_start_at_zero_and_have_no_gaps(parsed: ParsedDocument) -> None:
    assert [c.ordinal for c in parsed.clauses] == list(range(len(parsed.clauses)))


def test_clause_ids_are_unique(parsed: ParsedDocument) -> None:
    ids = [c.clause_id for c in parsed.clauses]
    assert len(set(ids)) == len(ids)


def test_page_ranges_are_within_the_document(parsed: ParsedDocument) -> None:
    for clause in parsed.clauses:
        assert 1 <= clause.page_start <= clause.page_end <= parsed.page_count


def test_pages_advance_monotonically_with_ordinal(parsed: ParsedDocument) -> None:
    """Clause order must follow reading order, or citations point backwards."""
    starts = [c.page_start for c in parsed.clauses]
    assert starts == sorted(starts)


def test_clause_text_is_never_empty(parsed: ParsedDocument) -> None:
    assert all(clause.text.strip() for clause in parsed.clauses)


def test_every_clause_carries_a_content_hash(parsed: ParsedDocument) -> None:
    for clause in parsed.clauses:
        assert clause.content_hash
        assert ":" in clause.content_hash, "hash should name its algorithm"


def test_identical_text_hashes_identically(parsed: ParsedDocument) -> None:
    """The invariant Phase 2's embedding reuse depends on."""
    by_hash: dict[str, str] = {}
    for clause in parsed.clauses:
        normalised = normalise_for_hash(clause.text)
        if clause.content_hash in by_hash:
            assert by_hash[clause.content_hash] == normalised
        by_hash[clause.content_hash] = normalised


# --- Determinism -----------------------------------------------------------


def test_parsing_is_deterministic(parser: DocumentParser, agreement_bytes: bytes) -> None:
    """Equal bytes must yield an equal document.

    ``parsed_at`` is excluded: it records when the parse ran, which is
    genuinely different between runs and is not part of the content.
    """
    first = parser.parse(ParseSource(content=agreement_bytes, filename="a.pdf"))
    second = parser.parse(ParseSource(content=agreement_bytes, filename="a.pdf"))
    assert first.model_dump(exclude={"parsed_at"}) == second.model_dump(exclude={"parsed_at"})


def test_document_id_is_derived_from_content_not_filename(
    parser: DocumentParser, agreement_bytes: bytes
) -> None:
    a = parser.parse(ParseSource(content=agreement_bytes, filename="one.pdf"))
    b = parser.parse(ParseSource(content=agreement_bytes, filename="two.pdf"))
    assert a.document_id == b.document_id


def test_version_is_carried_through(parser: DocumentParser, agreement_bytes: bytes) -> None:
    parsed = parser.parse(ParseSource(content=agreement_bytes, filename="a.pdf", version=3))
    assert parsed.version == 3
    assert all(str(3) in clause.clause_id for clause in parsed.clauses)


# --- No content may be lost ------------------------------------------------


def _sample_pdfs() -> list[Path]:
    return sorted(SAMPLES_DIR.glob("*.pdf")) if SAMPLES_DIR.is_dir() else []


@pytest.mark.parametrize(
    "sample",
    _sample_pdfs() or [pytest.param(None, marks=pytest.mark.skip(reason="no samples"))],
    ids=lambda p: p.name if p else "none",
)
def test_no_body_text_is_dropped(parser: DocumentParser, sample: Path) -> None:
    """Every substantive line of the PDF must survive into some clause.

    This is the invariant that matters most to a borrower: a parser that
    quietly discards the clause about foreclosure charges produces a system
    that is confidently, citably wrong. Running headers and footers are
    excluded deliberately and are reported as a warning when they are.

    Run across every sample because the documents fail differently -- prose
    agreements hide content in folded headings, and Key Facts Statements hide
    it in label:value rows that look like headings but carry the rate.
    """
    agreement_bytes = sample.read_bytes()
    parsed = parser.parse(ParseSource(content=agreement_bytes, filename=sample.name))

    # A section title is retained on the clauses it introduces rather than in
    # clause text, so headings count as retained content too.
    combined = normalise_for_hash(
        " ".join(f"{clause.heading or ''} {clause.text}" for clause in parsed.clauses)
    )

    removed_chrome = any(w.code == "running_headers_removed" for w in parsed.warnings)

    with fitz.open(stream=agreement_bytes, filetype="pdf") as document:
        source_lines = [
            line.strip()
            for page in document
            for line in page.get_text("text").split("\n")
            if len(line.strip()) > 25
        ]

    def retained(line: str) -> bool:
        normalised = normalise_for_hash(line)
        if normalised in combined:
            return True
        # The printed number of a section title lives in Clause.number, so
        # compare the title without its numeric prefix.
        stripped = re.sub(r"^\d{1,2}(?:\.\d{1,2})*[.)]?\s+", "", normalised)
        return stripped != normalised and stripped in combined

    missing = [
        line
        for line in source_lines
        if not retained(line) and not (removed_chrome and source_lines.count(line) > 1)
    ]
    assert not missing, f"Parser dropped {len(missing)} line(s), e.g. {missing[:3]}"


# --- Honest failure --------------------------------------------------------


def test_unreadable_input_warns_instead_of_raising(parser: DocumentParser) -> None:
    parsed = parser.parse(ParseSource(content=b"not a pdf at all", filename="broken.pdf"))
    assert parsed.warnings, "a failed parse must explain itself"
    assert parsed.clauses == ()


def test_a_page_without_text_produces_a_warning(parser: DocumentParser) -> None:
    """An image-only page must never be silently treated as blank."""
    document = fitz.open()
    document.new_page(width=595, height=842)  # deliberately empty
    blank = document.tobytes()
    document.close()

    parsed = parser.parse(ParseSource(content=blank, filename="scanned.pdf"))
    assert parsed.warnings, "an unreadable page must be reported, not ignored"


# --- Metadata is filter input, so abstaining beats guessing ----------------


def test_detected_metadata_is_correct_for_the_sample(parsed: ParsedDocument) -> None:
    assert parsed.kind is DocumentKind.LOAN_AGREEMENT
    assert parsed.metadata.loan_type is not None
    assert parsed.metadata.lender_class is not None


def test_metadata_hints_from_the_caller_win(parser: DocumentParser, agreement_bytes: bytes) -> None:
    from app.core.contracts import DocumentMetadata, LoanType

    parsed = parser.parse(
        ParseSource(
            content=agreement_bytes,
            filename="a.pdf",
            kind_hint=DocumentKind.SANCTION_LETTER,
            metadata_hint=DocumentMetadata(loan_type=LoanType.GOLD),
        )
    )
    assert parsed.kind is DocumentKind.SANCTION_LETTER
    assert parsed.metadata.loan_type is LoanType.GOLD
