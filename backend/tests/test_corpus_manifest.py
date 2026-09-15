"""The RBI manifest is filter input, so it is tested like code, not like docs."""

from __future__ import annotations

from datetime import date

import pytest

from app.config import get_settings
from app.core.contracts import LenderClass
from app.core.corpus import load_manifest

CORPUS_DIR = get_settings().rbi_corpus_dir


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(CORPUS_DIR)


def test_manifest_loads_and_every_file_exists(manifest) -> None:
    # load_manifest raises if a catalogued file is missing from disk.
    assert len(manifest.documents) >= 5


def test_every_document_was_verified_against_the_pdf_text(manifest) -> None:
    unverified = [d.id for d in manifest.documents if not d.verified_from_pdf]
    assert not unverified, f"Reference numbers assumed rather than read from the PDF: {unverified}"


def test_every_document_names_at_least_one_lender_class(manifest) -> None:
    """An empty lender_classes would mean 'binds everyone', which is never true
    of these circulars and would defeat the applicability filter."""
    for doc in manifest.documents:
        assert doc.applicability.lender_classes, f"{doc.id} has no lender class"


def test_lender_classes_are_known_vocabulary(manifest) -> None:
    known = set(LenderClass)
    for doc in manifest.documents:
        for lender_class in doc.applicability.lender_classes:
            assert lender_class in known, f"{doc.id}: unknown lender class {lender_class}"


def test_effective_from_is_never_before_the_issue_date(manifest) -> None:
    for doc in manifest.documents:
        effective_from = doc.applicability.effective_from
        if effective_from is not None:
            assert effective_from >= doc.issued_on, (
                f"{doc.id} binds loans before the circular was issued"
            )


# The three cases below are the ones a wrong answer would hurt a borrower most,
# so they are pinned explicitly rather than left to the generic checks above.


def test_prepayment_directions_do_not_govern_a_loan_sanctioned_before_2026(manifest) -> None:
    doc = manifest.by_id("rbi-2025-26-64-prepayment-charges-directions")
    assert doc.applicability.effective_from == date(2026, 1, 1)
    assert not doc.is_in_force_on(date(2025, 11, 30))
    assert doc.is_in_force_on(date(2026, 1, 1))


def test_penal_charges_binds_from_the_extended_date_not_the_printed_one(manifest) -> None:
    """The circular says January 1, 2024. A later circular moved it to April.

    Taking the date printed on a circular at face value would have FinX judge a
    loan against rules that were not yet in force on the day it was signed.
    """
    penal = manifest.by_id("rbi-2023-24-53-penal-charges")
    assert penal.issued_on == date(2023, 8, 18)
    assert penal.applicability.effective_from == date(2024, 4, 1)
    assert not penal.is_in_force_on(date(2024, 2, 1))
    assert penal.is_in_force_on(date(2024, 4, 1))

    extension = manifest.by_id(penal.amended_by)
    assert extension.amends == penal.id


def test_a_locally_rendered_file_says_so(manifest) -> None:
    """Provenance is recorded where a file is not the regulator's own PDF."""
    for doc in manifest.documents:
        path = CORPUS_DIR / doc.file
        if path.stat().st_size < 20_000:
            assert doc.source_capture, f"{doc.id} looks rendered but claims no provenance"


def test_the_corpus_holds_no_document_that_governs_nothing(manifest) -> None:
    """Regression: a UCB-only Master Circular sat in the corpus contributing
    244 chunks of co-operative-bank material to a corpus used for NBFC and
    bank loans. Retrieval filtered it out, but it was pure noise in the index."""
    ids = {doc.id for doc in manifest.documents}
    assert "rbi-2025-26-18-management-of-advances-ucb" not in ids


def test_nbfc_fair_practices_code_does_not_bind_banks(manifest) -> None:
    doc = manifest.by_id("rbi-2012-13-27-fair-practices-code-nbfc")
    assert LenderClass.SCHEDULED_COMMERCIAL_BANK not in doc.applicability.lender_classes
    assert LenderClass.NBFC in doc.applicability.lender_classes


def test_kfs_circular_binds_from_october_2024_not_its_april_issue_date(manifest) -> None:
    doc = manifest.by_id("rbi-2024-25-18-key-facts-statement")
    assert doc.issued_on == date(2024, 4, 15)
    assert doc.applicability.effective_from == date(2024, 10, 1)
    assert not doc.is_in_force_on(date(2024, 6, 1))


def test_every_recorded_source_url_was_verified(manifest) -> None:
    """Links are checked against the circular they claim to point at.

    A judge clicking a citation through to the wrong circular is worse than a
    citation with no link, so an unverified URL fails the build.
    """
    for doc in manifest.documents:
        if doc.source_url:
            assert doc.source_url_verified, f"{doc.id}: unverified source_url"
            assert doc.source_url.startswith("https://"), f"{doc.id}: insecure source_url"


def test_a_missing_source_url_explains_itself(manifest) -> None:
    for doc in manifest.documents:
        if not doc.source_url:
            assert doc.source_url_status, f"{doc.id}: silent missing source_url"


def test_most_of_the_corpus_is_clickable(manifest) -> None:
    linked = [doc for doc in manifest.documents if doc.source_url]
    assert len(linked) >= len(manifest.documents) - 1
