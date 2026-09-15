"""The evidence gate: whether retrieval found enough to answer from at all.

Lane 2 ends here. Above the threshold, the question goes to Lane 3 and gets a
cited answer. Below it, FinX does not produce a confident answer -- it takes
the fallback path, which says so plainly, explains the terms in general, and
offers a human.

The signal is the raw cosine similarity of the best passage, never a score
normalised across the candidate set. Normalisation was tried first and made the
top hit of every query score about 1.0, including questions the corpus cannot
answer, which would have made this gate pass everything. Measured on the
goldset with the local embedding model, answerable questions land at 0.74-0.82
and unanswerable ones at 0.66-0.70; the threshold sits in that gap.

The gate is deliberately a single, inspectable comparison. A borrower who asks
"why didn't it answer?" deserves a reason that fits in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.core.contracts import SourceKind
from app.rag.retrieve import RetrievalResult

# Measured, not chosen. Answerable goldset questions: 0.74-0.82. Unanswerable:
# 0.66-0.70. Re-measure with `run_eval.py` if the embedding model changes --
# the numbers are properties of the model, not of the documents.
MIN_CONFIDENCE = 0.72


class GateOutcome(StrEnum):
    ANSWER = "answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOTHING_RETRIEVED = "nothing_retrieved"


@dataclass(frozen=True)
class GateDecision:
    outcome: GateOutcome
    confidence: float
    threshold: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.outcome is GateOutcome.ANSWER


def evaluate(retrieval: RetrievalResult, *, threshold: float = MIN_CONFIDENCE) -> GateDecision:
    """Decide whether the retrieved passages are strong enough to answer from."""
    if retrieval.is_empty:
        return GateDecision(
            GateOutcome.NOTHING_RETRIEVED,
            0.0,
            threshold,
            "No passage in your document or the RBI corpus matched the question.",
        )

    confidence = retrieval.confidence
    if confidence < threshold:
        return GateDecision(
            GateOutcome.INSUFFICIENT_EVIDENCE,
            confidence,
            threshold,
            f"The closest passage found matched at {confidence:.2f}, below the {threshold:.2f} "
            "needed to answer from it with confidence.",
        )

    # An answer about the borrower's loan needs the borrower's document in it.
    # Regulation alone can say what lenders may do; it cannot say what this
    # lender did.
    if not any(p.chunk.source_kind is SourceKind.USER_DOCUMENT for p in retrieval.passages):
        return GateDecision(
            GateOutcome.INSUFFICIENT_EVIDENCE,
            confidence,
            threshold,
            "Nothing in your own document addressed the question.",
        )

    return GateDecision(GateOutcome.ANSWER, confidence, threshold, "")
