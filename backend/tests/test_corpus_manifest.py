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


def test_ucb_master_circular_is_scoped_to_cooperative_banks_only(manifest) -> None:
    """Catalogued by content, not by its misleading filename. If this widened,
    UCB-only paragraphs would surface for NBFC and bank loans."""
    doc = manifest.by_id("rbi-2025-26-18-management-of-advances-ucb")
    assert doc.applicability.lender_classes == (LenderClass.COOPERATIVE_BANK,)
    assert doc.filename_mismatch is not None


def test_nbfc_fair_practices_code_does_not_bind_banks(manifest) -> None:
    doc = manifest.by_id("rbi-2012-13-27-fair-practices-code-nbfc")
    assert LenderClass.SCHEDULED_COMMERCIAL_BANK not in doc.applicability.lender_classes
    assert LenderClass.NBFC in doc.applicability.lender_classes


def test_kfs_circular_binds_from_october_2024_not_its_april_issue_date(manifest) -> None:
    doc = manifest.by_id("rbi-2024-25-18-key-facts-statement")
    assert doc.issued_on == date(2024, 4, 15)
    assert doc.applicability.effective_from == date(2024, 10, 1)
    assert not doc.is_in_force_on(date(2024, 6, 1))
