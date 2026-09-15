"""Recognising a delta amendment, and reading what it says about its base.

A revised agreement comes in two shapes. A **restatement** reprints the whole
contract; a clause missing from it was removed. An **amendment** reprints only
what changed and says the rest stands; a clause missing from it is unchanged
and still binding.

Nothing else in the system can tell them apart, and the difference decides what
a version actually contains. So detection is deliberately conservative: an
amendment is only declared when the document says so in more than one way, and
the phrases that convinced it are recorded on the result so a person can check
the call rather than trust it.

Detection is a separate module from :mod:`app.parsers.lite.detect` because it
answers a structural question about how to assemble a version, not a
descriptive one about what the loan is.
"""

from __future__ import annotations

import re

from app.core.contracts import AmendmentTarget

# Each phrase is evidence that the document revises another and does not
# restate it. Drawn from how these documents are actually drafted: the
# operative sentence is almost always some form of "save as expressly revised
# herein, all other terms continue in full force".
_AMENDMENT_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "carry-forward clause",
        re.compile(r"save as (?:expressly |otherwise )?(?:revised|amended|varied)", re.I),
    ),
    ("carry-forward clause", re.compile(r"continue in full force", re.I)),
    (
        "carry-forward clause",
        re.compile(r"(?:remain|continue)s? (?:unchanged|in effect|to apply)", re.I),
    ),
    ("not restated", re.compile(r"not restated in this document", re.I)),
    ("names a base agreement", re.compile(r"\boriginal agreement\b", re.I)),
    ("revision summary", re.compile(r"summary of revisions?", re.I)),
    ("per-clause revision", re.compile(r"this revises clause\b", re.I)),
    ("version marker", re.compile(r"\brevised\s*[-–—]?\s*version\s*\d+", re.I)),
    ("stands revised", re.compile(r"stands? revised\b", re.I)),
)

# Two independent cues before the call is made. A single mention of "the
# Original Agreement" appears in plenty of restatements too.
MIN_EVIDENCE = 2

_EFFECTIVE_FROM = (
    re.compile(r"this revision effective\s*:?\s*(?P<date>[^\n|]{6,40})", re.I),
    re.compile(r"with effect from\s+(?P<date>[^\n,.;|]{6,40})", re.I),
    re.compile(r"effective\s+(?:from\s+)?(?P<date>\d{1,2}\s+\w+\s+\d{4})", re.I),
)

_BASE_DATED = (
    re.compile(r"original agreement dated\s*:?\s*(?P<date>[^\n|]{6,40})", re.I),
    re.compile(r"agreement dated\s+(?P<date>\d{1,2}\s+\w+\s+\d{4})", re.I),
)

_LOAN_ACCOUNT = re.compile(
    r"loan (?:account|a/c)(?:\s*no\.?|\s*number)?\s*:?\s*(?P<value>[A-Z0-9][A-Z0-9/\-]{5,40})",
    re.I,
)


def _first_date(text: str, patterns) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("date").strip()
    return None


def detect_amendment(text: str) -> AmendmentTarget | None:
    """Identify a delta amendment and read what it says about its base.

    Returns ``None`` for a document that merely mentions a prior version --
    absence of a positive finding means "treat this as a standalone version",
    which is the safe reading for a restatement.
    """
    from app.parsers.lite.detect import detect_agreement_date

    head = text[:6000]

    evidence: list[str] = []
    for label, pattern in _AMENDMENT_CUES:
        if pattern.search(head) and label not in evidence:
            evidence.append(label)

    if len(evidence) < MIN_EVIDENCE:
        return None

    effective_raw = _first_date(head, _EFFECTIVE_FROM)
    base_raw = _first_date(head, _BASE_DATED)
    account = _LOAN_ACCOUNT.search(head)

    return AmendmentTarget(
        loan_account_number=account.group("value").strip() if account else None,
        base_dated=detect_agreement_date(base_raw) if base_raw else None,
        effective_from=detect_agreement_date(effective_raw) if effective_raw else None,
        evidence=tuple(evidence),
    )
