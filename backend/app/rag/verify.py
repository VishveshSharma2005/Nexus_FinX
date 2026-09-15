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

# How far a full figure match lifts a sentence past the support threshold.
FIGURE_BONUS = 0.05

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
    shall will may can your you their its it if any such other under has have had
    yes not but also there here which what when would could does did was were
    been being into onto about than then them they these those our out""".split()
)


_SUFFIXES = ("ments", "ment", "ings", "ing", "ions", "ion", "ed", "es", "er", "ly", "s")

# Borrowers and models say "pay off", "switch", "move the loan"; agreements say
# "prepay", "foreclose", "refinance". These are the same act. Each group maps to
# one canonical stem so a faithful paraphrase is not mistaken for invention.
_SYNONYMS = {
    "prepa": "prepa",
    "forec": "prepa",
    "payof": "prepa",
    "repai": "prepa",
    "charg": "charg",
    "fee": "charg",
    "cost": "charg",
    "penal": "charg",
    "levi": "charg",
    "lende": "lende",
    "bank": "lende",
    "insti": "lende",
    "borro": "borro",
    "loan": "borro",
    "lock": "lock",
    "lockin": "lock",
}


def _stem(word: str) -> str:
    """Crude on purpose: strip a suffix, keep five characters, fold synonyms.

    Without it the verifier compared exact words, and removed a correct
    paraphrase -- "if you pay the loan off with money borrowed from another
    lender, you'd be charged 3% [2]" -- because the clause says "prepayment",
    "source of funds" and "attract a charge". A fluent model paraphrases in
    nearly every sentence, so exact matching would have deleted most of what a
    hosted model writes and kept only what it quoted.
    """
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    stem = word[:5]
    return _SYNONYMS.get(stem, stem)


def _terms(text: str) -> set[str]:
    return {
        _stem(w)
        for w in _WORD.findall(
            text.lower().replace("pay off", "payoff").replace("lock-in", "lockin")
        )
        if w not in _STOP and len(w) > 2
    }


# A figure: a rate, an amount, a count. Rupee markers and Indian digit grouping
# are stripped so "Rs. 1,90,000", "₹190000" and "1,90,000" compare equal.
_FIGURE = re.compile(r"(?:rs\.?\s*|₹\s*)?\d[\d,]*(?:\.\d+)?", re.I)

# A number straight after one of these names a place in a document, not a
# quantity. "Clause 11.3" is an address; "3%" is a claim.
_ADDRESS_WORD = re.compile(
    r"(?:clauses?|paragraphs?|para|sections?|schedule|annex(?:ure)?|part|item|"
    r"page|circular|no\.?)\s*$",
    re.I,
)


def figures(text: str) -> set[str]:
    """Every quantity stated in the text, normalised for comparison."""
    found: set[str] = set()
    for match in _FIGURE.finditer(text):
        if _ADDRESS_WORD.search(text[max(0, match.start() - 16) : match.start()]):
            continue
        digits = re.sub(r"[^\d.]", "", match.group(0)).strip(".")
        if not digits:
            continue
        if "." in digits:
            digits = digits.rstrip("0").rstrip(".")
        found.add(digits)
    return found


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


# Abbreviations whose full stop does not end a sentence. "Rs." matters most:
# splitting on it cut "The premium of Rs. 1,90,000 is financed [3]" in two,
# and the half carrying the figure then escaped the figure check entirely --
# in precisely the sentences where money appears.
_ABBREVIATIONS = re.compile(
    r"\b(Rs|No|Nos|Mr|Mrs|Ms|Dr|Smt|Shri|Cl|Para|viz|approx|p\.a|i\.e|e\.g|vs|etc|Ltd|Pvt)\.",
    re.I,
)
# A private-use character stands in for the protected full stop while splitting.
_PROTECT = chr(0xE000)


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
        protected = _ABBREVIATIONS.sub(lambda m: m.group(0)[:-1] + _PROTECT, stripped)
        # A sentence ends at . ! or ? followed by space and a capital or a
        # bracket -- not at a decimal point, and not before a lower-case word.
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])", protected):
            if sentence.strip():
                parts.append(sentence.strip().replace(_PROTECT, "."))
    return parts


def verify(
    answer: str,
    passages: list[RetrievedPassage],
    *,
    question: str = "",
    min_support: float = MIN_SUPPORT,
) -> VerifiedAnswer:
    """Keep only what the retrieved passages support, and bind it to them.

    ``passages`` must be in the order they were numbered in the prompt: the
    model's ``[n]`` refers to the n-th, one-based.

    Two checks per sentence, and both must pass:

    * **Support.** Enough of the sentence's meaningful terms appear in a passage
      it cites.
    * **Figures.** Every number in the sentence appears in a passage it cites,
      or in the borrower's own question. This is the check that matters most
      for a fluent model. Term overlap passes "the lock-in lasts 18 months [1]"
      against a clause saying twelve, because almost every word matches; the
      figure check does not. A model is also told never to convert or add up
      amounts, and this is what enforces it: "Rs 1.9 lakh" does not appear in a
      clause that says "Rs. 1,90,000", so it is removed rather than trusted.
    """
    by_number = {index + 1: passage for index, passage in enumerate(passages)}
    asked = figures(question)

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
        supporting_figures: set[str] = set()

        for number in markers:
            passage = by_number.get(number)
            if passage is None:
                continue
            overlap = len(claim_terms & _terms(passage.chunk.text))
            support = overlap / len(claim_terms) if claim_terms else 0.0

            # A sentence whose every figure appears in the cited clause is
            # anchored to it more firmly than word overlap can show: "3%" and
            # "principal" together identify clause 11.3 even when the rest is
            # paraphrased. Figures can only raise support, never lower it --
            # the separate figure check below is what catches a wrong number.
            claim_figures = figures(bare)
            if claim_figures and claim_figures <= figures(passage.chunk.text) and overlap >= 2:
                support = max(support, min_support + FIGURE_BONUS)

            best_support = max(best_support, support)
            if support >= min_support:
                supporting.append(passage.chunk.citation)
                supporting_figures |= figures(passage.chunk.text)

        unsupported_figures = figures(bare) - supporting_figures - asked
        if supporting and unsupported_figures:
            stripped.append(line)
            sentences.append(
                VerifiedSentence(
                    line,
                    (),
                    best_support,
                    False,
                    "states a figure its source does not contain: "
                    + ", ".join(sorted(unsupported_figures)),
                )
            )
            continue

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
