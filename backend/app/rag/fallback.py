"""The fallback path: what FinX says when it has no source to answer from.

Three parts, every time, in this order:

1. **A notice** that no source was found. Stated first so nothing below it can
   be mistaken for a grounded answer.
2. **A general explanation** of the terms in the question -- what "lock-in" or
   "EMI" means in general -- and nothing more. No claim about what RBI requires
   and no claim about the borrower's document, because on this path FinX has
   no evidence for either.
3. **An offer of a human advisor** for anything specific.

The explanation comes from a separate prompt with no document context and no
RBI context injected -- there is nothing on this path the model could cite, so
nothing is given to it. Its output then passes a filter that removes any
sentence asserting a rule or describing the borrower's own loan. The filter is
needed even though the prompt forbids those sentences: a prompt is a request,
and a fluent model will reach for "RBI requires lenders to..." because it is
usually a helpful thing to say. Here it would be an unsourced regulatory claim.

This path never streams. A streamed draft reaches the screen before the filter
has run, so the only way to guarantee a filtered sentence is never shown is to
show nothing until filtering is done. The explanation is short enough that the
wait is not felt.

The risk and cost engine never runs here. No rupee figure is shown without a
parsed, matched clause behind it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.core.contracts import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

NOTICE = (
    "No source found. Nothing in your document or in FinX's RBI corpus answers this "
    "question closely enough to rely on, so FinX will not give you a specific answer."
)

# Kept free of regulatory claims too. An earlier draft named the RBI ombudsman
# scheme and its 30-day window -- accurate, but a claim about what RBI provides,
# made on the one path that has no source for any such claim.
ADVISOR = (
    "For a specific answer about your loan, speak to a qualified financial adviser or to "
    "your lender's grievance officer, and take your agreement and Key Facts Statement with you."
)

FALLBACK_SYSTEM = """You explain financial terms in plain English to a loan borrower.

You have NOT been given the person's loan document and you have NOT been given any regulation.
So you must not say anything about:
- what their agreement, contract, lender or loan says or does;
- what RBI, the law, or any regulation requires, allows or prohibits.

Only explain, in general terms, what the words in their question usually mean. Use phrases like
"generally" or "usually". Two to four short sentences. No numbers, rates or amounts. No advice.
If the question contains no financial term you can explain, say only that."""


# --- The claim filter --------------------------------------------------------
#
# Each pattern is a way of asserting something this path has no evidence for.
# Matching is on the whole sentence, and one match removes it.

_REGULATORY = re.compile(
    r"\b(rbi|reserve bank|circular|direction|regulat\w*|regulator|law|legal|illegal|lawful|"
    r"statut\w*|ombudsman rules?|guidelines?|mandat\w*|compliance|compliant)\b",
    re.I,
)
_DOCUMENT_SPECIFIC = re.compile(
    r"\b(your (agreement|contract|loan|lender|bank|kfs|key facts|document|emi|rate|clause|"
    r"schedule|tenure|policy)|this (agreement|contract|loan|clause|document)|"
    r"the agreement|the clause|clause \d|schedule [ivx\d]|in your case)\b",
    re.I,
)
_OBLIGATION = re.compile(
    r"\b(must|shall|required|requires?|not allowed|allowed to|permitted|prohibit\w*|"
    r"forbid\w*|entitled|cannot legally|is not legal|are obliged|obligated|have the right)\b",
    re.I,
)
_FIGURE = re.compile(r"\d")

_FILTERS = (
    ("regulatory claim", _REGULATORY),
    ("claim about the borrower's document", _DOCUMENT_SPECIFIC),
    ("statement of obligation or permission", _OBLIGATION),
    ("specific figure", _FIGURE),
)


@dataclass
class FilteredExplanation:
    text: str
    kept: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    flat = " ".join(line.strip(" -*•") for line in text.splitlines() if line.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def filter_claims(text: str) -> FilteredExplanation:
    """Remove every sentence that asserts a rule or describes the borrower's loan.

    Rejections are logged, so a developer can see the filter working rather
    than trust that it does.
    """
    result = FilteredExplanation(text="")
    for sentence in _sentences(text):
        reason = next((label for label, pattern in _FILTERS if pattern.search(sentence)), None)
        if reason:
            result.rejected.append((sentence, reason))
            logger.info("fallback filter rejected (%s): %s", reason, sentence)
        else:
            result.kept.append(sentence)
    result.text = " ".join(result.kept)
    return result


# --- A general glossary, for when no model is available ----------------------
#
# Used by the quoting provider, which has no passages to quote on this path.
# Every entry is written to pass the claim filter above: general, no figures,
# no rules, nothing about any particular loan.

GLOSSARY: dict[str, str] = {
    "lock-in": (
        "A lock-in period is generally a stretch of time at the start of a loan during which "
        "the borrower cannot repay it early."
    ),
    "prepay": (
        "Prepayment usually means paying back some or all of a loan before its scheduled end, "
        "which reduces the interest paid over the remaining term."
    ),
    "foreclos": (
        "Foreclosure, in Indian lending usage, generally means closing a loan entirely by "
        "repaying the whole outstanding amount early."
    ),
    "penal": (
        "Penal charges are usually fees a lender applies when a payment is late or a loan "
        "condition is not met."
    ),
    "emi": (
        "An EMI, or equated monthly instalment, is the fixed monthly payment that covers both "
        "interest and part of the principal."
    ),
    "apr": (
        "The annual percentage rate generally expresses the yearly cost of a loan including "
        "fees, which makes offers easier to compare than the interest rate alone."
    ),
    "key facts": (
        "A Key Facts Statement is a short summary a lender gives before signing, setting out "
        "the main costs and terms of a loan in a standard format."
    ),
    "floating": (
        "A floating interest rate moves up or down with a benchmark rate, so the EMI or the "
        "tenure can change over the life of the loan."
    ),
    "tenure": "The tenure of a loan is the length of time over which it is repaid.",
    "principal": (
        "The principal is the amount actually borrowed, as distinct from the interest charged "
        "on it."
    ),
    "processing fee": (
        "A processing fee is usually a one-time charge for assessing and setting up a loan."
    ),
    "insurance": (
        "Credit life insurance on a loan generally pays off the outstanding balance if the "
        "borrower dies, and its premium is sometimes added to the amount borrowed."
    ),
    "collateral": (
        "Collateral, or security, is an asset pledged against a loan that the lender can "
        "claim if the loan is not repaid."
    ),
    "mortgage": (
        "A mortgage is a charge over property, usually a home, given as security for a loan."
    ),
    "interest": (
        "Interest is the price of borrowing money, usually expressed as a yearly percentage "
        "of the amount owed."
    ),
    "tax": (
        "How loan repayments are treated for tax depends on personal circumstances and is "
        "outside what a loan document covers."
    ),
    "refinanc": (
        "Refinancing generally means taking a new loan, often from a different lender, to pay "
        "off an existing one on better terms."
    ),
    "co-borrower": (
        "A co-borrower shares responsibility for repaying a loan alongside the main borrower."
    ),
    "moratorium": (
        "A moratorium is a period during which repayments are paused or reduced, though "
        "interest may still accrue."
    ),
}


def glossary_explanation(question: str) -> str:
    """Define, in general terms, any known term the question contains."""
    lowered = question.lower()
    matched = [entry for key, entry in GLOSSARY.items() if key in lowered]
    if not matched:
        return (
            "FinX did not recognise a general loan term in this question that it could "
            "explain without a source."
        )
    # Stable order, no duplicates, at most three -- enough to orient, not a lecture.
    seen: list[str] = []
    for entry in matched:
        if entry not in seen:
            seen.append(entry)
    return " ".join(seen[:3])


# --- Assembling the three parts ----------------------------------------------


@dataclass
class FallbackAnswer:
    notice: str
    explanation: str
    advisor: str
    reason: str
    rejected: list[tuple[str, str]] = field(default_factory=list)
    source: str = ""


async def general_explanation(
    question: str,
    *,
    provider: LLMProvider | None,
    use_model: bool,
) -> tuple[str, list[tuple[str, str]], str]:
    """Explain the question's terms generally, then filter.

    Returns the filtered text, the rejected sentences with reasons, and which
    source produced it. The model is sent only the rules and the question --
    no document, no passages, no regulation.
    """
    if use_model and provider is not None:
        try:
            response = await provider.complete(
                [
                    LLMMessage(role="system", content=FALLBACK_SYSTEM),
                    LLMMessage(role="user", content=question),
                ],
                temperature=0.1,
                max_tokens=220,
            )
            filtered = filter_claims(response.text)
            if filtered.text:
                return filtered.text, filtered.rejected, provider.name
            logger.info("fallback: model output was entirely filtered; using the glossary")
            glossary = filter_claims(glossary_explanation(question))
            return glossary.text, filtered.rejected + glossary.rejected, "glossary"
        except Exception as exc:
            logger.warning("fallback explanation from %s failed: %s", provider.name, exc)

    filtered = filter_claims(glossary_explanation(question))
    return filtered.text, filtered.rejected, "glossary"


async def build_fallback(
    question: str,
    *,
    reason: str,
    provider: LLMProvider | None,
    use_model: bool,
) -> FallbackAnswer:
    explanation, rejected, source = await general_explanation(
        question, provider=provider, use_model=use_model
    )
    return FallbackAnswer(
        notice=NOTICE,
        explanation=explanation,
        advisor=ADVISOR,
        reason=reason,
        rejected=rejected,
        source=source,
    )
