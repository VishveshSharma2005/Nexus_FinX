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

# Well-drafted Indian loan documents cite the regulation they comply with, by
# number and by date. Every detector here therefore has to tell a fact about
# this loan apart from a quotation of the rule the loan follows. Three separate
# misreadings came from missing that distinction: a housing loan typed as a
# personal loan because it quotes the circular on "EMI based Personal Loans";
# an amendment classified as an RBI circular because it cites RBI/2025-26/64;
# and a Key Facts Statement dated to the KFS circular it was issued under
# rather than to the day it was handed to the borrower.
_CITATION_CUE = re.compile(
    r"(circular|notification|master direction|master circular|directions,\s*\d{4}"
    r"|RBI/\d{4}-\d{2}|DOR\.|DoR\.|DNBS|DNBR)",
    re.IGNORECASE,
)

# How far back to look for a cue that the surrounding text is a citation.
_CITATION_LOOKBACK = 90


def _is_quoted_regulation(text: str, position: int) -> bool:
    """Whether the match at ``position`` sits inside a citation.

    Stops a document's compliance references being mistaken for statements
    about the loan itself.
    """
    window = text[max(0, position - _CITATION_LOOKBACK) : position]
    return bool(_CITATION_CUE.search(window))


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
    # A circular prints its own reference number at the very top. A loan
    # agreement that cites one does so in the body, so only the title zone
    # counts as evidence of being a circular.
    title_head = head[:_TITLE_ZONE]
    if _RBI_REFERENCE.search(title_head) or _RBI_TITLE.search(title_head):
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


# How far a family must lead the runner-up before it is treated as the answer
# rather than as ambiguity.
_DOMINANCE_RATIO = 3


def _score_loan_types(text: str) -> dict[LoanType, int]:
    lowered = text.lower()
    scores: dict[LoanType, int] = {}
    for loan_type, markers in _LOAN_TYPE_MARKERS:
        total = 0
        for marker in markers:
            start = 0
            while (found := lowered.find(marker, start)) != -1:
                if not _is_quoted_regulation(text, found):
                    total += 1
                start = found + len(marker)
        if total:
            scores[loan_type] = total
    return scores


def detect_loan_type(text: str) -> LoanType | None:
    """Identify the loan type, or abstain when the evidence is genuinely mixed.

    A first pass returned ``None`` for a housing loan agreement, because the
    agreement quotes the title of the RBI circular on "EMI based Personal
    Loans" in its rate-reset clause. A citation to a regulation is not evidence
    of what kind of loan this is, so a single passing mention must not cancel
    out a type named in the title and repeated throughout.

    Evidence is therefore weighed rather than merely counted as present:

    * a type named in the document's title zone wins outright, if it is the
      only one named there;
    * otherwise the most-mentioned family wins if it clearly leads;
    * otherwise abstain, which widens the candidate set rather than filtering
      governing regulation away.
    """
    title_zone = _score_loan_types(text[:_TITLE_ZONE])
    if len(title_zone) == 1:
        return next(iter(title_zone))

    scores = _score_loan_types(text)
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) == 1:
        return ranked[0][0]

    (best, best_score), (_, runner_up) = ranked[0], ranked[1]
    return best if best_score >= runner_up * _DOMINANCE_RATIO else None


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

_INDIVIDUAL_MARKERS = (
    "an individual",
    "individual borrower",
    "co-borrower",
    "aadhaar",
    "mr.",
    "ms.",
    "mrs.",
    "shri",
    "smt",
)

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
    # The whole document is scanned, not just the recital. A first pass
    # stopped at 2,500 characters and missed a home loan agreement that
    # identifies its borrower as "an individual borrower availing the Loan for
    # a purpose other than business" in the penal-charges clause, well past
    # that cut-off. Clipping at the lender is what keeps the widened scan from
    # reading the lender's constitution as the borrower's.
    lowered = text.lower()
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
        if len(windows) >= 40:
            break

    return " ".join(windows)


def detect_borrower_type(text: str) -> BorrowerType | None:
    context = _borrower_context(text)
    if not context:
        return None

    business = sum(context.count(marker) for marker in _BUSINESS_MARKERS)
    individual = sum(context.count(marker) for marker in _INDIVIDUAL_MARKERS)

    if not business and not individual:
        return None
    if business and individual:
        # Weigh them: a stray "GSTIN" in boilerplate should not outvote a
        # borrower described as an individual throughout. A close call stays
        # an abstention.
        if individual >= business * _DOMINANCE_RATIO:
            return BorrowerType.INDIVIDUAL
        if business >= individual * _DOMINANCE_RATIO:
            return BorrowerType.BUSINESS
        return None
    return BorrowerType.BUSINESS if business else BorrowerType.INDIVIDUAL


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


def _first_real_date(window: str) -> date | None:
    for pattern in _DATE_PATTERNS[:2]:
        for match in pattern.finditer(window):
            if _is_quoted_regulation(window, match.start()):
                continue
            first, second, year = match.groups()
            found = (
                _coerce(first, second, year) if first.isdigit() else _coerce(second, first, year)
            )
            if found:
                return found

    for match in _DATE_PATTERNS[2].finditer(window):
        if _is_quoted_regulation(window, match.start()):
            continue
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day))
        except ValueError:
            continue
    return None


def detect_agreement_date(text: str) -> date | None:
    """Find the execution date, which decides which circulars were in force.

    The opening of the document is searched first, because that is where an
    agreement states the date it was made and where the middle is only
    repayment schedules and due dates.

    The signature block is searched as a fallback. A Key Facts Statement
    carries no date in its header at all -- the only one it bears is in the
    borrower's acknowledgement at the foot -- so a head-only scan abstained on
    a document that plainly states its date.
    """
    found = _first_real_date(text[:2500])
    if found:
        return found
    return _first_real_date(text[-1200:]) if len(text) > 2500 else None
