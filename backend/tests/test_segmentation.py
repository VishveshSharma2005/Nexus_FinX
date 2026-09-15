"""Unit tests for segmentation, detection and hashing.

The contract test proves the parser keeps its promises on real documents.
These pin the specific judgement calls inside it, including the ones that were
wrong first: every "was folded away" case below is a bug that silently deleted
a clause from a real sample before it was caught.
"""

from __future__ import annotations

from datetime import date

from app.core.contracts import BorrowerType, DocumentKind, LenderClass, LoanType
from app.core.text import clean_extracted_text, content_hash, normalise_for_hash
from app.parsers.lite.detect import (
    detect_agreement_date,
    detect_borrower_type,
    detect_kind,
    detect_lender_class,
    detect_loan_type,
)
from app.parsers.lite.segment import (
    Line,
    SegmentationStrategy,
    segment,
    strip_running_headers,
)


def lines(*texts: str, page: int = 1) -> list[Line]:
    return [Line(text=text, page=page) for text in texts]


# --- Segmentation ----------------------------------------------------------


def test_numbered_clauses_are_separated() -> None:
    result = segment(
        lines(
            "1. DEFINITIONS",
            "1.1 The Loan means the facility described in the Schedule.",
            "1.2 EMI means the equated monthly instalment payable each month.",
            "2. INTEREST",
            "2.1 Interest accrues at 14.5% per annum on a reducing balance.",
        )
    )
    assert result.strategy is SegmentationStrategy.CLAUSE_NUMBERING
    assert [clause.number for clause in result.clauses] == ["1.1", "1.2", "2.1"]


def test_a_bare_section_title_becomes_the_heading_of_its_children() -> None:
    result = segment(
        lines(
            "7. DEFAULT AND PENAL CHARGES",
            "7.1 Penal charges of 2% per month shall be levied on overdue amounts.",
            "7.2 The Lender may recall the Loan upon an event of default occurring.",
            "8. NOTICES",
            "8.1 Notices shall be sent to the address recorded in the Schedule.",
        )
    )
    assert result.clauses[0].heading == "DEFAULT AND PENAL CHARGES"
    assert result.clauses[1].heading == "DEFAULT AND PENAL CHARGES"
    assert result.clauses[2].heading == "NOTICES"


def test_a_single_line_clause_is_not_mistaken_for_a_heading() -> None:
    """Regression: clause 4.2 disappeared from the sample agreement.

    It fitted on one line and ended in a full stop, and the segmenter folded
    it away as though it were a section title.
    """
    result = segment(
        lines(
            "4. FEES AND CHARGES",
            "4.1 A processing fee of 2% of the sanctioned amount shall be deducted.",
            "4.2 Documentation and stamping charges of Rs. 2,500 shall be payable.",
            "4.3 A loan administration charge of Rs. 500 per annum shall be recovered.",
        )
    )
    numbers = [clause.number for clause in result.clauses]
    assert numbers == ["4.1", "4.2", "4.3"]
    assert any("2,500" in clause.text for clause in result.clauses)


def test_a_key_value_row_is_not_mistaken_for_a_heading() -> None:
    """Regression: the Key Facts Statement lost its interest rate.

    "3.1 Annual rate of interest: 13.90% per annum" has no trailing full stop
    and is short, so it read as a section title and was folded into the next
    clause -- deleting the number the whole KFS audit turns on.
    """
    result = segment(
        lines(
            "3. INTEREST RATE AND TYPE",
            "3.1 Annual rate of interest: 13.90% per annum",
            "3.2 Type of interest: Floating",
            "3.3 Reference benchmark and spread: External benchmark plus 5.15%",
        )
    )
    assert [clause.number for clause in result.clauses] == ["3.1", "3.2", "3.3"]
    assert "13.90%" in result.clauses[0].text


def test_consecutive_bare_headings_accumulate_rather_than_overwrite() -> None:
    """Two titles in a row must both survive; overwriting one loses text."""
    result = segment(
        lines(
            "PART A",
            "SCHEDULE OF CHARGES",
            "1.1 Foreclosure charge of 4% of principal outstanding shall apply.",
            "1.2 Documentation charges of Rs. 2,500 shall be payable on execution.",
            "1.3 A loan administration charge of Rs. 500 per annum shall be recovered.",
        )
    )
    heading = result.clauses[0].heading or ""
    assert "SCHEDULE OF CHARGES" in heading
    assert "PART" in heading


def test_prose_beginning_with_section_does_not_open_a_clause() -> None:
    result = segment(
        lines(
            "1. AUTHORITY",
            "1.1 These instructions are issued under Section 21 and Section 35A of the "
            "Banking Regulation Act, 1949, and are binding on all regulated entities "
            "to whom this circular is addressed by the Reserve Bank.",
            "2. SCOPE",
            "2.1 This applies to all retail loans sanctioned after the effective date.",
            "2.2 Section 45JA of the Reserve Bank of India Act, 1934 is also relied upon.",
            "3. COMMENCEMENT",
            "3.1 These instructions come into effect from January 1, 2024.",
        )
    )
    assert [clause.number for clause in result.clauses] == ["1.1", "2.1", "2.2", "3.1"]


def test_text_before_the_first_clause_is_kept_as_a_preamble() -> None:
    result = segment(
        lines(
            "THIS AGREEMENT is made on 10 February 2026 between the Lender and the Borrower.",
            "1. DEFINITIONS",
            "1.1 The Loan means the facility described in the Schedule hereto.",
            "1.2 EMI means the equated monthly instalment payable by the Borrower.",
        )
    )
    assert any("THIS AGREEMENT is made" in clause.text for clause in result.clauses)


def test_unstructured_text_falls_back_and_says_so() -> None:
    body = [f"This is an ordinary sentence of running prose, number {i}." for i in range(40)]
    padded: list[Line] = []
    for index, text in enumerate(body):
        padded.append(Line(text=text, page=1 + index // 10))
        if index % 4 == 3:
            padded.append(Line(text="", page=1 + index // 10))

    result = segment(padded)
    assert result.strategy in {SegmentationStrategy.PARAGRAPHS, SegmentationStrategy.PAGES}
    assert result.notes, "a degraded segmentation must explain itself"


def test_empty_document_reports_rather_than_crashes() -> None:
    result = segment(lines("", "   ", ""))
    assert result.clauses == []
    assert result.notes


# --- Running headers -------------------------------------------------------


def test_repeated_page_header_is_removed() -> None:
    page_lines: list[Line] = []
    for page in (1, 2, 3):
        page_lines.append(Line(text="MERIDIAN FINSERV - CONFIDENTIAL", page=page))
        page_lines.append(Line(text=f"Body content unique to page {page}.", page=page))

    kept, chrome = strip_running_headers(page_lines, page_count=3)
    assert chrome == ["MERIDIAN FINSERV - CONFIDENTIAL"]
    assert all("CONFIDENTIAL" not in line.text for line in kept)


def test_body_text_repeated_mid_page_is_not_removed() -> None:
    """Only edge-positioned repeats are chrome; repeated body text is content."""
    page_lines: list[Line] = []
    for page in (1, 2, 3):
        page_lines.append(Line(text=f"Header {page}", page=page))
        page_lines.append(Line(text="filler one", page=page))
        page_lines.append(Line(text="Subject to the terms of this Agreement.", page=page))
        page_lines.append(Line(text="filler two", page=page))
        page_lines.append(Line(text=f"Footer {page}", page=page))

    _, chrome = strip_running_headers(page_lines, page_count=3)
    assert "Subject to the terms of this Agreement." not in chrome


# --- Detection -------------------------------------------------------------


def test_borrower_type_is_not_inferred_from_the_lenders_constitution() -> None:
    """Regression: every retail loan was classified as a business loan.

    Detection scanned the whole document, and the lender is a private limited
    company in essentially every Indian loan agreement.
    """
    text = (
        "THIS PERSONAL LOAN AGREEMENT is made between Meridian Finserv Private Limited, "
        "a Non-Banking Financial Company (the Lender), AND Mr. Rakesh Bhavsar, an "
        "individual holding Aadhaar ending 4417 (the Borrower)."
    )
    assert detect_borrower_type(text) is BorrowerType.INDIVIDUAL


def test_business_borrower_is_still_detected() -> None:
    text = (
        "THIS FACILITY AGREEMENT is made between the Lender AND Sunrise Textiles Private "
        "Limited, a company holding Udyam registration, hereinafter the Borrower."
    )
    assert detect_borrower_type(text) is BorrowerType.BUSINESS


def test_ambiguous_borrower_evidence_abstains() -> None:
    assert detect_borrower_type("The Borrower shall repay the Loan in equated instalments.") is None


def test_rbi_circular_is_not_classified_as_a_key_facts_statement() -> None:
    """Regression: the penal-charges circular was read as a KFS.

    It requires lenders to disclose charges "in the Key Facts Statement",
    which is a reference to one, not evidence of being one.
    """
    text = (
        "RBI/2023-24/53 DoR.MCS.REC.28/01.01.001/2023-24 August 18, 2023 "
        "Fair Lending Practice - Penal Charges in Loan Accounts. The quantum and reason "
        "for penal charges shall be disclosed in the loan agreement and Key Facts "
        "Statement as applicable."
    )
    assert detect_kind(text) is DocumentKind.RBI_CIRCULAR


def test_a_loan_agreement_naming_the_regulator_is_still_an_agreement() -> None:
    text = (
        "PERSONAL LOAN AGREEMENT. This agreement is made between Meridian Finserv "
        "Private Limited, a company registered with the Reserve Bank of India, and the "
        "Borrower."
    )
    assert detect_kind(text) is DocumentKind.LOAN_AGREEMENT


def test_key_facts_statement_is_detected_from_its_title() -> None:
    assert detect_kind("KEY FACTS STATEMENT Part 1 - Interest rate and fees") is (
        DocumentKind.KEY_FACTS_STATEMENT
    )


def test_caller_hint_overrides_detection() -> None:
    assert detect_kind("KEY FACTS STATEMENT", hint=DocumentKind.SANCTION_LETTER) is (
        DocumentKind.SANCTION_LETTER
    )


def test_loan_type_abstains_when_two_families_appear() -> None:
    assert detect_loan_type("This personal loan and the home loan are both outstanding.") is None
    assert detect_loan_type("This personal loan is repayable in 48 months.") is LoanType.PERSONAL


def test_specific_lender_class_wins_over_the_generic_one() -> None:
    text = "Sanctioned by a Housing Finance Company registered with the Reserve Bank."
    assert detect_lender_class(text) is LenderClass.HOUSING_FINANCE_COMPANY


def test_agreement_date_is_read_from_the_opening() -> None:
    text = "THIS PERSONAL LOAN AGREEMENT is made on 10 February 2026 at Ahmedabad between"
    assert detect_agreement_date(text) == date(2026, 2, 10)


# --- Hashing ---------------------------------------------------------------


def test_reflowed_whitespace_hashes_identically() -> None:
    """Re-flowed line breaks must not count as a change, or every re-upload
    would pay to embed clauses that did not change."""
    a = "6.1 The Borrower may prepay the Loan\nat any time after twelve months."
    b = "6.1 The Borrower may prepay the Loan at any  time after twelve months."
    assert content_hash(a) == content_hash(b)


def test_a_changed_number_changes_the_hash() -> None:
    """The opposite failure, and the dangerous one: a v2 clause charging 4%
    must never reuse the embedding of a v1 clause charging 2%."""
    a = "6.1 A foreclosure charge of 2% of the principal outstanding shall apply."
    b = "6.1 A foreclosure charge of 4% of the principal outstanding shall apply."
    assert content_hash(a) != content_hash(b)


def test_case_is_preserved_by_normalisation() -> None:
    assert normalise_for_hash("Penal Charges") != normalise_for_hash("penal charges")


def test_typographic_quotes_and_dashes_are_normalised() -> None:
    assert content_hash('The "Loan" - as defined') == content_hash("The “Loan” – as defined")


def test_hyphenated_line_breaks_are_rejoined() -> None:
    assert "prepayment" in clean_extracted_text("The pre-\npayment charge applies.")
