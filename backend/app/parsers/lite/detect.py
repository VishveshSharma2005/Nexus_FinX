"""Conservative detection of document kind and loan metadata.

This metadata is what retrieval filters RBI applicability against, so a wrong
guess here is worse than no guess: mislabelling an NBFC loan as a bank loan
would surface circulars that do not govern it, and mislabelling the lender
class would hide ones that do.

Every function here therefore returns ``None`` when the evidence is weak, and
the retrieval layer treats ``None`` as "widen the candidate set" rather than
"exclude". Guessing is the failure mode; abstaining is not.
"""

from __future__ import annotations

import re
from datetime import date

from app.core.contracts import BorrowerType, DocumentKind, LenderClass, LoanType

# --- Document kind ---------------------------------------------------------

# An RBI reference number is unambiguous. Plain "Reserve Bank of India" is not:
# it appears in most loan agreements, which name the regulator the lender is
# registered with.
_RBI_REFERENCE = re.compile(r"\bRBI/\d{4}-\d{2}/\d+\b", re.IGNORECASE)
_RBI_TITLE = re.compile(r"\b(master circular|master direction|notification)\b", re.IGNORECASE)

# Only the title zone counts for these. A penal-charges circular requires the
# lender to disclose charges "in the Key Facts Statement", and a loan agreement
# refers to its own KFS -- neither makes the document one.
_TITLE_ZONE = 700

_TITLE_MARKERS: list[tuple[DocumentKind, tuple[str, ...]]] = [
    (
        DocumentKind.KEY_FACTS_STATEMENT,
        ("key facts statement", "key fact statement"),
    ),
    (
        DocumentKind.SANCTION_LETTER,
        ("sanction letter", "letter of sanction", "sanction advice"),
    ),
    (
        DocumentKind.LOAN_AGREEMENT,
        ("loan agreement", "loan cum hypothecation", "facility agreement", "credit agreement"),
    ),
]


def detect_kind(text: str, *, hint: DocumentKind | None = None) -> DocumentKind:
    """Classify the document, preferring the caller's hint when given."""
    if hint is not None and hint is not DocumentKind.UNKNOWN:
        return hint

    head = text[:3000]
    if _RBI_REFERENCE.search(head) or _RBI_TITLE.search(head[:_TITLE_ZONE]):
        return DocumentKind.RBI_CIRCULAR

    title_zone = head[:_TITLE_ZONE].lower()
    for kind, markers in _TITLE_MARKERS:
        if any(marker in title_zone for marker in markers):
            return kind

    lowered = head.lower()
    for kind, markers in _TITLE_MARKERS:
        if any(marker in lowered for marker in markers):
            return kind
    return DocumentKind.UNKNOWN


# --- Loan type -------------------------------------------------------------

_LOAN_TYPE_MARKERS: list[tuple[LoanType, tuple[str, ...]]] = [
    (LoanType.HOME, ("home loan", "housing loan", "mortgage loan")),
    (LoanType.VEHICLE, ("vehicle loan", "car loan", "auto loan", "two wheeler loan")),
    (LoanType.EDUCATION, ("education loan", "student loan")),
    (LoanType.GOLD, ("gold loan", "loan against gold", "gold jewellery")),
    (LoanType.MICROFINANCE, ("microfinance loan", "micro finance loan", "jlg loan")),
    (LoanType.CREDIT_CARD, ("credit card agreement", "cardholder agreement")),
    (LoanType.BUSINESS, ("business loan", "working capital", "msme loan", "term loan to msme")),
    (LoanType.PERSONAL, ("personal loan", "consumer loan")),
]


def detect_loan_type(text: str) -> LoanType | None:
    """Return a loan type only when exactly one family of markers appears.

    Ambiguity is reported as ``None``. A document mentioning both "home loan"
    and "personal loan" is not evidence of either.
    """
    lowered = text.lower()
    matched = {
        loan_type
        for loan_type, markers in _LOAN_TYPE_MARKERS
        if any(marker in lowered for marker in markers)
    }
    return matched.pop() if len(matched) == 1 else None


# --- Lender class ----------------------------------------------------------

_LENDER_MARKERS: list[tuple[LenderClass, tuple[str, ...]]] = [
    (LenderClass.HOUSING_FINANCE_COMPANY, ("housing finance company", "housing finance limited")),
    (LenderClass.NBFC_MFI, ("nbfc-mfi", "nbfc mfi", "micro finance institution")),
    (
        LenderClass.NBFC,
        ("non-banking financial company", "non banking financial company", "nbfc"),
    ),
    (LenderClass.SMALL_FINANCE_BANK, ("small finance bank",)),
    (LenderClass.REGIONAL_RURAL_BANK, ("regional rural bank", "gramin bank")),
    (LenderClass.COOPERATIVE_BANK, ("co-operative bank", "cooperative bank", "urban co-op")),
    (
        LenderClass.SCHEDULED_COMMERCIAL_BANK,
        ("scheduled commercial bank", "banking regulation act, 1949"),
    ),
]


def detect_lender_class(text: str) -> LenderClass | None:
    """Identify the regulated-entity class of the lender.

    Ordering matters: "Small Finance Bank" and "Housing Finance Company" are
    checked before the generic bank and NBFC markers, because the specific
    class determines which circulars bind.
    """
    lowered = text.lower()
    for lender_class, markers in _LENDER_MARKERS:
        if any(marker in lowered for marker in markers):
            return lender_class
    return None


# --- Borrower type ---------------------------------------------------------

_BUSINESS_MARKERS = (
    "private limited",
    "pvt. ltd",
    "pvt ltd",
    "llp",
    "partnership firm",
    "proprietorship",
    "msme",
    "udyam",
    "gstin",
)

_INDIVIDUAL_MARKERS = ("an individual", "aadhaar", "mr.", "ms.", "mrs.", "shri", "smt")

# How far either side of the word "Borrower" counts as describing the borrower.
_BORROWER_WINDOW = 180


def _borrower_context(text: str) -> str:
    """Text describing the borrower, as a lowercase blob.

    Scanning the whole document classifies almost every retail loan as a
    business loan, because the *lender* is a private limited company. Even a
    proximity window is not enough: the recital names both parties in one
    sentence, so "Private Limited" sits a few words from "Borrower".

    What actually separates them is that a recital reads
    "<lender> ... (the Lender) AND <borrower> ... (the Borrower)". So each
    window is clipped at the nearest mention of the lender, leaving only the
    words that describe the borrower.
    """
    # Party descriptions live in the recital; later mentions are obligations.
    lowered = text[:2500].lower()
    windows: list[str] = []
    start = 0

    while (found := lowered.find("borrower", start)) != -1:
        left = max(0, found - _BORROWER_WINDOW)
        preceding = lowered[left:found]
        if (cut := preceding.rfind("lender")) != -1:
            left += cut + len("lender")

        right = min(len(lowered), found + _BORROWER_WINDOW)
        following = lowered[found:right]
        if (cut := following.find("lender")) != -1:
            right = found + cut

        windows.append(lowered[left:right])
        start = found + len("borrower")
        if len(windows) >= 8:
            break

    return " ".join(windows)


def detect_borrower_type(text: str) -> BorrowerType | None:
    context = _borrower_context(text)
    if not context:
        return None

    is_business = any(marker in context for marker in _BUSINESS_MARKERS)
    is_individual = any(marker in context for marker in _INDIVIDUAL_MARKERS)

    if is_business and not is_individual:
        return BorrowerType.BUSINESS
    if is_individual and not is_business:
        return BorrowerType.INDIVIDUAL
    # Both or neither: the document is not clear enough to filter regulation on.
    return None


# --- Agreement date --------------------------------------------------------

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}

_DATE_PATTERNS = (
    re.compile(r"\b(\d{1,2})[stndrh]{0,2}\s+([A-Za-z]{3,9})[,\s]+(\d{4})\b"),
    re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2})[stndrh]{0,2}[,\s]+(\d{4})\b"),
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
)


def _coerce(day: str, month: str, year: str) -> date | None:
    month_number = _MONTHS.get(month.lower()[:9]) or _MONTHS.get(month.lower()[:3])
    if month_number is None:
        for name, number in _MONTHS.items():
            if name.startswith(month.lower()[:3]):
                month_number = number
                break
    if month_number is None:
        return None
    try:
        return date(int(year), month_number, int(day))
    except ValueError:
        return None


def detect_agreement_date(text: str) -> date | None:
    """Find the execution date, which decides which circulars were in force.

    Only the opening of the document is searched: later dates are repayment
    schedules and due dates, not the date the agreement was made.
    """
    head = text[:2500]

    for pattern in _DATE_PATTERNS[:2]:
        for match in pattern.finditer(head):
            first, second, year = match.groups()
            found = (
                _coerce(first, second, year) if first.isdigit() else _coerce(second, first, year)
            )
            if found:
                return found

    for match in _DATE_PATTERNS[2].finditer(head):
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            continue

    return None
