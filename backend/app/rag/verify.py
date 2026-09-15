"""Citation binding: nothing reaches the borrower that a source does not support.

The model is asked to mark every sentence with the passage it came from. This
module checks that it did, and that the passage actually says what the sentence
claims. A sentence that cannot be traced is removed before the answer is shown.

Two failure modes, and they need different treatment:

* **Uncited.** The model wrote a sentence with no marker. It may be true, but
  nothing binds it to a source, so it cannot be shown as grounded.
* **Miscited.** The model wrote a marker pointing at a passage that does not
  support the sentence. This is the more dangerous one, because the citation
  makes it look verified.

Support is measured by overlap of meaningful terms, and deliberately not by
another model call. A verifier that can hallucinate is not a verifier, and one
that costs a round trip per sentence cannot run on every answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.contracts import Citation, RetrievedPassage

# A sentence must share at least this share of its meaningful terms with the
# passage it cites. Low enough that an explanatory sentence phrased in plainer
# words than the clause still passes; high enough that a marker attached to an
# unrelated passage does not.
MIN_SUPPORT = 0.28

# Sentences that make no factual claim need no citation.
_NO_CLAIM = re.compile(
    r"^\s*(?:based on|here is|here are|in summary|to summarise|in short|"
    r"each point|open the citations|the clauses? (?:retrieved|below)|"
    r"this means|note that)\b",
    re.I,
)

_CITATION = re.compile(r"\[(\d+)\]")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
_STOP = frozenset(
    """a an the and or of to in on for by with is are be as at from this that
    shall will may can your you their its it if any such other under""".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


@dataclass
class VerifiedSentence:
    text: str
    citations: tuple[Citation, ...]
    support: float
    kept: bool
    reason: str = ""


@dataclass
class VerifiedAnswer:
    """An answer and the audit trail behind it."""

    text: str
    citations: tuple[Citation, ...]
    sentences: list[VerifiedSentence] = field(default_factory=list)
    stripped: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def fully_grounded(self) -> bool:
        return not self.stripped


def split_sentences(text: str) -> list[str]:
    """Split on sentence ends and list items, keeping markers attached."""
    parts: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            parts.append("")
            continue
        # A bullet is one unit even when it contains several sentences: the
        # citation belongs to the bullet.
        if stripped.startswith(("- ", "* ", "• ")):
            parts.append(stripped)
            continue
        parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip())
    return parts


def verify(
    answer: str,
    passages: list[RetrievedPassage],
    *,
    min_support: float = MIN_SUPPORT,
) -> VerifiedAnswer:
    """Keep only what the retrieved passages support, and bind it to them.

    ``passages`` must be in the order they were numbered in the prompt: the
    model's ``[n]`` refers to the n-th, one-based.
    """
    by_number = {index + 1: passage for index, passage in enumerate(passages)}

    kept_lines: list[str] = []
    sentences: list[VerifiedSentence] = []
    stripped: list[str] = []
    used: list[Citation] = []

    for line in split_sentences(answer):
        if not line:
            kept_lines.append("")
            continue

        markers = [int(m) for m in _CITATION.findall(line)]
        bare = _CITATION.sub("", line).strip()

        # Framing and connective sentences assert nothing and are not citable.
        if _NO_CLAIM.match(bare) or len(_terms(bare)) < 3:
            kept_lines.append(line)
            sentences.append(VerifiedSentence(line, (), 1.0, True, "no factual claim"))
            continue

        if not markers:
            stripped.append(line)
            sentences.append(VerifiedSentence(line, (), 0.0, False, "no citation"))
            continue

        claim_terms = _terms(bare)
        best_support = 0.0
        supporting: list[Citation] = []

        for number in markers:
            passage = by_number.get(number)
            if passage is None:
                continue
            overlap = len(claim_terms & _terms(passage.chunk.text))
            support = overlap / len(claim_terms) if claim_terms else 0.0
            best_support = max(best_support, support)
            if support >= min_support:
                supporting.append(passage.chunk.citation)

        if not supporting:
            stripped.append(line)
            sentences.append(
                VerifiedSentence(
                    line,
                    (),
                    best_support,
                    False,
                    "cited passage does not support the claim"
                    if best_support > 0
                    else "citation points at nothing",
                )
            )
            continue

        kept_lines.append(line)
        sentences.append(VerifiedSentence(line, tuple(supporting), best_support, True))
        for citation in supporting:
            if citation not in used:
                used.append(citation)

    text = "\n".join(kept_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return VerifiedAnswer(
        text=text,
        citations=tuple(used),
        sentences=sentences,
        stripped=stripped,
    )
