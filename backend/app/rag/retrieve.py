"""Lane 2: retrieve, filter on applicability, rank.

Three stages, in this order, and the order is the point.

1. **Filter.** Candidates are restricted to one document version, and RBI
   passages are restricted to those that actually govern this loan, inside the
   index query. A passage that does not apply is never scored, never reranked,
   and cannot reach the model's context by any later accident.

2. **Hybrid search.** Dense similarity alone ranks these documents poorly. The
   embedding model scores the top handful of clauses within about 0.02 of each
   other, because every clause of a loan agreement is written in the same
   register about the same loan. Lexical scoring breaks those ties on the terms
   that actually distinguish clauses -- a rate, a percentage, "lock-in",
   "foreclosure" -- which dense similarity smooths away. Neither alone is
   enough: dense finds the clause a question never names, lexical separates the
   near-identical clauses dense has bunched together.

3. **Rerank.** Signals that are cheap and deterministic, applied last: prefer
   the borrower's own document over general regulation when both match, prefer
   a clause that states an obligation over one that cross-references another,
   and demote near-duplicates so eight context slots hold eight distinct
   clauses rather than the same clause from three sources.

Everything here is deterministic. No model is called.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from app.core.contracts import (
    DocumentMetadata,
    EmbeddingProvider,
    RetrievalFilter,
    RetrievedPassage,
    SourceKind,
    VectorIndex,
)

# How far the lexical signal can move a result relative to the dense one.
# Dense decides what is roughly relevant; lexical decides the order within it.
#
# Both are min-max normalised across the candidate set before they are weighted,
# and they have to be. An earlier version weighted a raw cosine against a
# normalised BM25 score: because this embedding model compresses its scores into
# a band about 0.08 wide while BM25 spanned the full 0 to 1, lexical actually
# outweighed dense by roughly four to one and the stated weights were fiction.
# The visible symptom was two oversized tables from the penal-charges circular
# outranking the floating-rate circular on a question about rate resets, despite
# the latter having the best dense score of any candidate.
#
# Confidence is reported separately, from the raw cosine, precisely because
# normalisation destroys absolute meaning -- see RetrievalResult.confidence.
DENSE_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35

# A passage this similar to one already selected adds nothing to the context.
NEAR_DUPLICATE_THRESHOLD = 0.92

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")

_STOPWORDS = frozenset(
    """a an the and or of to in on for by with is are be as at from this that
    shall will may can what how do does i my me if it its their there any""".split()
)


@dataclass
class RetrievalResult:
    """What retrieval found, and enough of why to debug it."""

    passages: list[RetrievedPassage]
    query: str
    considered: int = 0
    filtered_out: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def best_score(self) -> float:
        """Top combined ranking score. Use it to order, not to trust."""
        return self.passages[0].score if self.passages else 0.0

    @property
    def confidence(self) -> float:
        """How well the best passage actually matches, on an absolute scale.

        This is the raw cosine similarity, deliberately never normalised across
        the candidate set. Min-max normalisation was the first implementation
        and it made every query's top hit score about 1.0, including questions
        the corpus cannot answer at all -- the number told you the ranking and
        nothing about whether anything was found. Measured on the goldset,
        answerable questions land at 0.74-0.82 here and unanswerable ones at
        0.66-0.70, which is the separation the evidence gate reads.
        """
        return max((p.dense_score or 0.0 for p in self.passages), default=0.0)

    @property
    def is_empty(self) -> bool:
        return not self.passages


def tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _bm25_scores(query: str, documents: list[str]) -> list[float]:
    """Okapi BM25 over the filtered candidate set.

    Computed across candidates rather than the whole corpus on purpose: the
    question is which of *these* passages best matches, and rare-term weighting
    computed over an already-narrow set is what separates the clause that says
    "lock-in" from the four around it that do not.
    """
    k1, b = 1.5, 0.75
    tokenised = [tokenise(doc) for doc in documents]
    lengths = [len(doc) for doc in tokenised]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    total = len(tokenised)

    frequencies = [Counter(doc) for doc in tokenised]
    document_frequency = Counter()
    for doc in tokenised:
        document_frequency.update(set(doc))

    scores = []
    query_terms = tokenise(query)
    for index, counts in enumerate(frequencies):
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            n = document_frequency[term]
            idf = math.log(1 + (total - n + 0.5) / (n + 0.5))
            tf = counts[term]
            norm = 1 - b + b * (lengths[index] / average if average else 1.0)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * norm)
        scores.append(score)
    return scores


def _normalise(values: list[float]) -> list[float]:
    """Min-max to [0, 1] so two differently-scaled signals can be summed."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


# Clauses that only point elsewhere are poor answers even when they match well.
_CROSS_REFERENCE_ONLY = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(?:refer|please refer|see|as (?:set out|specified|provided))\b",
    re.I,
)

# Above this a passage is a table or an annex, not a clause.
OVERSIZED_BLOCK_CHARS = 1200

# A clause that states a number is usually what a borrower asking about money
# actually wants.
_HAS_FIGURE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%|rs\.?\s*[\d,]+|\d+\s*(?:months?|days?|years?))", re.I
)


# How much a curated topic match can lift a circular.
TOPIC_BONUS = 0.12


def build_topic_index(manifest) -> dict[str, set[str]]:
    """Map each corpus document to the vocabulary of its declared topics.

    The manifest already says what each circular is *about*, in the words a
    borrower would use -- "no pre-payment charges on floating rate loans to
    individuals for non-business purposes". That is curated knowledge no
    similarity score has access to, and it settles exactly the cases where
    several circulars discuss the same subject and only one governs it.

    Without it, "are prepayment charges allowed on a floating rate home loan?"
    ranked the floating-rate reset circular first, because the question says
    "floating rate" and that circular is named after it. The circular that
    actually prohibits the charge sat at rank five.
    """
    return {
        document.id: set(tokenise(" ".join(document.topics)))
        for document in manifest.documents
        if document.topics
    }


def _topic_overlap(question_terms: set[str], topics: set[str]) -> float:
    if not question_terms or not topics:
        return 0.0
    shared = question_terms & topics
    return len(shared) / len(question_terms)


def _rerank_bonus(
    passage: RetrievedPassage,
    *,
    wants_figure: bool,
    wants_regulation: bool,
    question_terms: set[str] | None = None,
    topic_index: Mapping[str, set[str]] | None = None,
) -> float:
    bonus = 0.0
    text = passage.chunk.text
    is_document = passage.chunk.source_kind is SourceKind.USER_DOCUMENT

    # Which source answers the question depends on what was asked. "What does
    # my agreement charge?" is answered by the agreement; "is that charge
    # allowed?" can only be answered by the circular. Preferring the borrower's
    # document unconditionally put clause 11.3 above the Directions that
    # prohibit it -- the agreement cannot tell you it is wrong.
    if wants_regulation:
        bonus += 0.03 if is_document else 0.10
    else:
        bonus += 0.06 if is_document else 0.0

    if _CROSS_REFERENCE_ONLY.match(text):
        bonus -= 0.08

    # An unsegmented schedule or annex table matches many queries on term
    # overlap alone and cites poorly -- a whole page rather than a clause.
    if len(text) > OVERSIZED_BLOCK_CHARS:
        bonus -= 0.05

    if wants_figure and _HAS_FIGURE.search(text):
        bonus += 0.05

    # A clause with a printed number can be cited precisely; an unnumbered
    # block can only be cited by page.
    if passage.chunk.citation.clause_number:
        bonus += 0.02

    if topic_index and question_terms and not is_document:
        overlap = _topic_overlap(question_terms, topic_index.get(passage.chunk.document_id, set()))
        bonus += TOPIC_BONUS * overlap

    return bonus


_MONEY_QUESTION = re.compile(
    r"\b(charge|charges|fee|fees|cost|costs|rate|penalty|penal|percent|%|rs|rupees|"
    r"how much|amount|emi|premium|lock-?in|period|months?|years?)\b",
    re.I,
)


# "Is this allowed?" is a different question from "what does my contract say?",
# and it wants a different source. The agreement states what the lender wrote;
# the circular states what the lender was permitted to write, which is the only
# thing that can tell a borrower a clause should not be there at all.
_REGULATORY_QUESTION = re.compile(
    r"\b(allowed|permitted|permissible|legal|lawful|is it legal|rbi|reserve bank|"
    r"regulations?|rules?|compliant|complies|comply|entitled to charge|"
    r"are they allowed|supposed to)\b",
    re.I,
)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _drop_near_duplicates(
    passages: list[RetrievedPassage], limit: int
) -> tuple[list[RetrievedPassage], int]:
    """Keep the best of each group of near-identical passages.

    The corpus repeats itself: a penal-charges rule appears in the circular
    that issued it and again in the agreement that complies with it. Both are
    worth having; three paraphrases of one are not, and they crowd out the
    clause that would have answered the rest of the question.
    """
    kept: list[RetrievedPassage] = []
    seen: list[set[str]] = []
    dropped = 0

    for passage in passages:
        tokens = set(tokenise(passage.chunk.text))
        if any(_jaccard(tokens, other) >= NEAR_DUPLICATE_THRESHOLD for other in seen):
            dropped += 1
            continue
        kept.append(passage)
        seen.append(tokens)
        if len(kept) >= limit:
            break

    return kept, dropped


async def retrieve(
    question: str,
    *,
    index: VectorIndex,
    embedder: EmbeddingProvider,
    document_id: str | None = None,
    version: int | None = None,
    metadata: DocumentMetadata | None = None,
    as_of: date | None = None,
    source_kinds: tuple[SourceKind, ...] = (),
    topic_index: Mapping[str, set[str]] | None = None,
    top_k: int = 24,
    top_n: int = 8,
) -> RetrievalResult:
    """Find the passages that answer a question about one version of one loan.

    ``metadata`` and ``as_of`` come from the borrower's own parsed document and
    are what the RBI corpus is filtered against. Passing neither retrieves on
    similarity alone, which is a weaker and less honest answer.
    """
    filters = RetrievalFilter(
        document_id=document_id,
        version=version,
        source_kinds=source_kinds,
        metadata=metadata,
        as_of=as_of,
    )

    embedding = await embedder.embed_query(question)
    candidates = await index.search(embedding, query_text=question, filters=filters, top_k=top_k)

    notes: list[str] = []
    if not candidates:
        return RetrievalResult(passages=[], query=question, notes=["No passage passed the filter."])

    lexical = _normalise(_bm25_scores(question, [c.chunk.text for c in candidates]))
    dense = _normalise([c.score for c in candidates])
    wants_figure = bool(_MONEY_QUESTION.search(question))
    wants_regulation = bool(_REGULATORY_QUESTION.search(question))
    question_terms = set(tokenise(question))

    scored: list[RetrievedPassage] = []
    for passage, dense_rank, lexical_score in zip(candidates, dense, lexical, strict=True):
        combined = DENSE_WEIGHT * dense_rank + LEXICAL_WEIGHT * lexical_score
        combined += _rerank_bonus(
            passage,
            wants_figure=wants_figure,
            wants_regulation=wants_regulation,
            question_terms=question_terms,
            topic_index=topic_index,
        )
        scored.append(
            passage.model_copy(
                update={
                    "score": round(combined, 6),
                    # The raw cosine, kept unnormalised so the evidence gate has
                    # something absolute to read.
                    "dense_score": round(passage.score, 6),
                    "lexical_score": round(lexical_score, 6),
                    "rerank_score": round(combined, 6),
                }
            )
        )

    scored.sort(key=lambda p: p.score, reverse=True)
    passages, duplicates = _drop_near_duplicates(scored, top_n)
    if duplicates:
        notes.append(f"Set aside {duplicates} near-duplicate passage(s).")

    if version is not None and any(p.chunk.version != version for p in passages):
        raise AssertionError("version scoping leaked; this must never happen")

    return RetrievalResult(
        passages=passages,
        query=question,
        considered=len(candidates),
        filtered_out=duplicates,
        notes=notes,
    )


# Slots reserved for the borrower's own document, whatever the ranking says.
MIN_DOCUMENT_PASSAGES = 4


async def retrieve_for_answer(
    question: str,
    *,
    index: VectorIndex,
    embedder: EmbeddingProvider,
    document_id: str,
    version: int,
    metadata: DocumentMetadata | None = None,
    as_of: date | None = None,
    topic_index: Mapping[str, set[str]] | None = None,
    top_k: int = 32,
    top_n: int = 8,
) -> RetrievalResult:
    """Retrieve what is needed to answer one question about one loan.

    Two searches rather than one, because the two sources are scoped
    differently: the borrower's document is pinned to a single version, while
    the corpus is filtered on applicability and effective date. One filter
    cannot express both.

    The merge reserves slots for the borrower's own document and fills the rest
    by score. The asymmetry is deliberate. An answer that cites only what RBI
    requires, without the clause the borrower actually signed, is not an answer
    about their loan -- so the document is guaranteed a floor. Regulation is
    not, because forcing circulars into every answer pushed the borrower's own
    clauses out of questions that were never about regulation at all: reserving
    two corpus slots for "what did I pledge as security?" dropped clause 6.1
    entirely. When a question does concern what is permitted, the regulatory
    bonus in the ranking promotes the circulars on merit instead.
    """
    document = await retrieve(
        question,
        index=index,
        embedder=embedder,
        document_id=document_id,
        version=version,
        source_kinds=(SourceKind.USER_DOCUMENT,),
        top_k=top_k,
        top_n=top_n,
    )
    corpus = await retrieve(
        question,
        index=index,
        embedder=embedder,
        source_kinds=(SourceKind.RBI_CORPUS,),
        metadata=metadata,
        as_of=as_of,
        topic_index=topic_index,
        top_k=top_k,
        top_n=top_n,
    )

    reserved = min(MIN_DOCUMENT_PASSAGES, len(document.passages))
    chosen = list(document.passages[:reserved])

    remainder = sorted(
        document.passages[reserved:] + corpus.passages,
        key=lambda p: p.score,
        reverse=True,
    )
    chosen.extend(remainder[: max(0, top_n - len(chosen))])
    chosen.sort(key=lambda p: p.score, reverse=True)

    return RetrievalResult(
        passages=chosen,
        query=question,
        considered=document.considered + corpus.considered,
        filtered_out=document.filtered_out + corpus.filtered_out,
        notes=document.notes + corpus.notes,
    )
