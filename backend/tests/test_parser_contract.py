"""The DocumentParser contract.

Every test here is written against the interface in ``app.core.contracts`` and
parameterised over every registered implementation. Nothing in this file names
a concrete parser class, and nothing hard-codes which formats exist: a parser
is asked, through ``supports()``, which formats it handles, and is then tested
only against documents of those formats.

So adding a parser means adding a registry entry and nothing here. That is the
point of the interface, and it is why this file grew a second implementation
(Word documents) without gaining a single new test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.core.contracts import DocumentKind, DocumentParser, ParsedDocument, ParseSource
from app.parsers.lite.text import normalise_for_hash
from app.parsers.registry import PARSERS, build_parser

SAMPLES_DIR = get_settings().sample_corpus_dir

# Formats the suite knows how to construct. A parser is probed against these
# rather than being asked to declare them, so the mapping stays the only place
# format knowledge lives.
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _source(path: Path, **kwargs) -> ParseSource:
    return ParseSource(
        content=path.read_bytes(),
        filename=path.name,
        media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
        **kwargs,
    )


def _probe(extension: str) -> ParseSource:
    return ParseSource(
        content=b"probe",
        filename=f"probe{extension}",
        media_type=MEDIA_TYPES[extension],
    )


def formats_supported_by(parser: DocumentParser) -> list[str]:
    return [extension for extension in MEDIA_TYPES if parser.supports(_probe(extension))]


def _empty_document(extension: str) -> bytes:
    """A structurally valid document containing no extractable text."""
    if extension == ".pdf":
        import fitz

        document = fitz.open()
        document.new_page(width=595, height=842)
        data = document.tobytes()
        document.close()
        return data

    import io

    import docx

    buffer = io.BytesIO()
    docx.Document().save(buffer)
    return buffer.getvalue()


@pytest.fixture(params=sorted(PARSERS), ids=sorted(PARSERS))
def parser(request, settings: Settings) -> DocumentParser:
    """Each registered implementation, in turn."""
    return build_parser(request.param, settings)


@pytest.fixture
def samples(parser: DocumentParser) -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        pytest.skip(f"No samples directory at {SAMPLES_DIR}")
    supported = formats_supported_by(parser)
    found = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in supported)
    if not found:
        pytest.skip(f"No {supported} samples for parser {parser.name!r}")
    return found


@pytest.fixture
def sample(samples: list[Path]) -> Path:
    """One representative document: the longest, which exercises the most."""
    return max(samples, key=lambda p: p.stat().st_size)


@pytest.fixture
def parsed(parser: DocumentParser, sample: Path) -> ParsedDocument:
    return parser.parse(_source(sample))


# --- Interface conformance -------------------------------------------------


def test_implements_the_protocol(parser: DocumentParser) -> None:
    assert isinstance(parser, DocumentParser)
    assert isinstance(parser.name, str) and parser.name


def test_declares_at_least_one_format(parser: DocumentParser) -> None:
    assert formats_supported_by(parser), f"{parser.name} supports nothing the suite can build"


def test_rejects_formats_it_does_not_handle(parser: DocumentParser) -> None:
    assert not parser.supports(
        ParseSource(content=b"hello", filename="notes.txt", media_type="text/plain")
    )


def test_supports_does_not_depend_on_an_asserted_media_type(parser: DocumentParser) -> None:
    """A caller that does not know the type must not be able to mislead a parser.

    ParseSource used to default to application/pdf, so omitting the field
    silently asserted PDF -- and the PDF parser then claimed Word documents and
    reported them as corrupt. Filename evidence has to stand on its own.
    """
    for extension in formats_supported_by(parser):
        assert parser.supports(ParseSource(content=b"probe", filename=f"probe{extension}")), (
            f"{parser.name} relies on media_type alone for {extension}"
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


def test_parsing_is_deterministic(parser: DocumentParser, sample: Path) -> None:
    """Equal bytes must yield an equal document.

    ``parsed_at`` is excluded: it records when the parse ran, which is
    genuinely different between runs and is not part of the content.
    """
    first = parser.parse(_source(sample))
    second = parser.parse(_source(sample))
    assert first.model_dump(exclude={"parsed_at"}) == second.model_dump(exclude={"parsed_at"})


def test_document_id_is_derived_from_content_not_filename(
    parser: DocumentParser, sample: Path
) -> None:
    content = sample.read_bytes()
    media_type = MEDIA_TYPES[sample.suffix.lower()]
    a = parser.parse(
        ParseSource(content=content, filename=f"one{sample.suffix}", media_type=media_type)
    )
    b = parser.parse(
        ParseSource(content=content, filename=f"two{sample.suffix}", media_type=media_type)
    )
    assert a.document_id == b.document_id


def test_version_is_carried_through(parser: DocumentParser, sample: Path) -> None:
    parsed = parser.parse(_source(sample, version=3))
    assert parsed.version == 3
    assert all("v3" in clause.clause_id for clause in parsed.clauses)


# --- No content may be lost ------------------------------------------------


def test_no_body_text_is_dropped(parser: DocumentParser, samples: list[Path]) -> None:
    """Every substantive line of every supported sample survives into a clause.

    This is the invariant that matters most to a borrower: a parser that
    quietly discards the clause about foreclosure charges produces a system
    that is confidently, citably wrong.

    The documents fail differently, which is why all of them are checked:
    prose agreements hide content in folded headings, Key Facts Statements hide
    it in label:value rows that look like headings but carry the rate, and Word
    documents hide it in table cells.
    """
    for path in samples:
        parsed = parser.parse(_source(path))

        # A section title is retained on the clauses it introduces rather than
        # in clause text, so headings count as retained content too.
        combined = normalise_for_hash(
            " ".join(f"{clause.heading or ''} {clause.text}" for clause in parsed.clauses)
        )
        removed_chrome = any(w.code == "running_headers_removed" for w in parsed.warnings)
        source_lines = _substantive_lines(path)

        def retained(line: str, combined: str = combined) -> bool:
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
        assert not missing, f"{path.name}: dropped {len(missing)} line(s), e.g. {missing[:2]}"


def _substantive_lines(path: Path) -> list[str]:
    """Read the document independently of the parser under test."""
    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(path) as document:
            raw = [line for page in document for line in page.get_text("text").split("\n")]
    else:
        import docx

        document = docx.Document(str(path))
        raw = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                raw.extend(cell.text for cell in row.cells)

    return [line.strip() for line in raw if len(line.strip()) > 25]


# --- Honest failure --------------------------------------------------------


def test_unreadable_input_warns_instead_of_raising(parser: DocumentParser) -> None:
    extension = formats_supported_by(parser)[0]
    parsed = parser.parse(
        ParseSource(
            content=b"not a real document at all",
            filename=f"broken{extension}",
            media_type=MEDIA_TYPES[extension],
        )
    )
    assert parsed.warnings, "a failed parse must explain itself"
    assert parsed.clauses == ()


def test_a_document_without_text_produces_a_warning(parser: DocumentParser) -> None:
    """An empty or image-only document must never be silently treated as read."""
    extension = formats_supported_by(parser)[0]
    parsed = parser.parse(
        ParseSource(
            content=_empty_document(extension),
            filename=f"blank{extension}",
            media_type=MEDIA_TYPES[extension],
        )
    )
    assert parsed.warnings, "an unreadable document must be reported, not ignored"
    assert not parsed.clauses


# --- Metadata is filter input, so abstaining beats guessing ----------------


def test_detected_metadata_is_correct_for_the_sample(parsed: ParsedDocument) -> None:
    assert parsed.kind in {DocumentKind.LOAN_AGREEMENT, DocumentKind.KEY_FACTS_STATEMENT}
    assert parsed.metadata.loan_type is not None
    assert parsed.metadata.lender_class is not None


def test_metadata_hints_from_the_caller_win(parser: DocumentParser, sample: Path) -> None:
    from app.core.contracts import DocumentMetadata, LoanType

    parsed = parser.parse(
        _source(
            sample,
            kind_hint=DocumentKind.SANCTION_LETTER,
            metadata_hint=DocumentMetadata(loan_type=LoanType.GOLD),
        )
    )
    assert parsed.kind is DocumentKind.SANCTION_LETTER
    assert parsed.metadata.loan_type is LoanType.GOLD
