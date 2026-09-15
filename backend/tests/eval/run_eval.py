"""Retrieval evaluation against the goldset.

Measures three things, kept separate because they fail for different reasons
and a single blended score would hide which:

* **Clause recall** -- did the clauses that actually answer the question come
  back at all? A miss here is a retrieval failure.
* **Corpus precision at rank 1** -- when a question is about what the
  regulator permits rather than what the contract says, did the governing
  circular rank first? This is the one dense similarity alone gets wrong,
  because every prepayment passage in the corpus looks alike to it.
* **Refusal accuracy** -- on questions this corpus cannot answer, did
  retrieval come back weak instead of forcing a confident match?

Run:
    python backend/tests/eval/run_eval.py
    python backend/tests/eval/run_eval.py --verbose
    python backend/tests/eval/run_eval.py --compare   # dense-only vs hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.config import get_settings  # noqa: E402
from app.core.contracts import DocumentMetadata, LenderClass, LoanType, SourceKind  # noqa: E402
from app.core.corpus import load_manifest  # noqa: E402
from app.core.deps import get_embedding_provider, get_vector_index  # noqa: E402
from app.rag.retrieve import (  # noqa: E402
    RetrievalResult,
    build_topic_index,
    retrieve,
    retrieve_for_answer,
)

GOLDSET = Path(__file__).with_name("goldset.jsonl")

# What the parser detects for the demo agreement. Passing it is what turns
# similarity search into applicability-filtered retrieval.
DEMO_METADATA = DocumentMetadata(
    loan_type=LoanType.HOME,
    lender_class=LenderClass.HOUSING_FINANCE_COMPANY,
)
# Sanctioned 10 January 2026, executed 14 January 2026.
from datetime import date  # noqa: E402

DEMO_AS_OF = date(2026, 1, 14)

# What each circular declares itself to be about, from the manifest.
TOPICS = build_topic_index(load_manifest(get_settings().rbi_corpus_dir))

# The evidence gate's own threshold, imported rather than restated so the
# harness measures the gate the app actually runs.
from app.rag.gate import MIN_CONFIDENCE as INSUFFICIENT_CONFIDENCE  # noqa: E402


@dataclass
class CaseResult:
    id: str
    clause_recall: float
    corpus_ok: bool
    rank1_ok: bool
    refusal_ok: bool
    forbidden_hit: str | None
    best_score: float
    top: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.clause_recall >= 1.0
            and self.corpus_ok
            and self.rank1_ok
            and self.refusal_ok
            and self.forbidden_hit is None
        )


def _mean(values) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in GOLDSET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _label(passage) -> str:
    citation = passage.chunk.citation
    where = citation.clause_number or f"p{citation.page}"
    origin = "doc" if passage.chunk.source_kind is SourceKind.USER_DOCUMENT else "rbi"
    return f"{origin}:{where}"


def score_case(case: dict, result: RetrievalResult) -> CaseResult:
    found_clauses = {
        p.chunk.citation.clause_number
        for p in result.passages
        if p.chunk.source_kind is SourceKind.USER_DOCUMENT and p.chunk.citation.clause_number
    }
    found_corpus = [
        p.chunk.document_id for p in result.passages if p.chunk.source_kind is SourceKind.RBI_CORPUS
    ]

    expected = set(case.get("expect_clauses", []))
    recall = len(expected & found_clauses) / len(expected) if expected else 1.0

    expected_corpus = set(case.get("expect_corpus", []))
    corpus_ok = expected_corpus <= set(found_corpus)

    rank1_ok = True
    if (wanted := case.get("require_corpus_rank")) is not None and result.passages:
        top_ids = [p.chunk.document_id for p in result.passages[:wanted]]
        rank1_ok = bool(expected_corpus & set(top_ids))

    refusal_ok = True
    if case.get("expect_insufficient"):
        refusal_ok = result.is_empty or result.confidence < INSUFFICIENT_CONFIDENCE

    forbidden = None
    for phrase in case.get("must_not_contain", []):
        if any(phrase.lower() in p.chunk.text.lower() for p in result.passages):
            forbidden = phrase
            break

    return CaseResult(
        id=case["id"],
        clause_recall=recall,
        corpus_ok=corpus_ok,
        rank1_ok=rank1_ok,
        refusal_ok=refusal_ok,
        forbidden_hit=forbidden,
        best_score=result.confidence,
        top=[_label(p) for p in result.passages[:4]],
    )


async def run_case(case: dict, *, index, embedder, dense_only: bool = False) -> RetrievalResult:
    from app.rag import retrieve as retrieve_module

    original = (retrieve_module.DENSE_WEIGHT, retrieve_module.LEXICAL_WEIGHT)
    if dense_only:
        retrieve_module.DENSE_WEIGHT, retrieve_module.LEXICAL_WEIGHT = 1.0, 0.0
    try:
        return (
            await retrieve(
                case["question"],
                index=index,
                embedder=embedder,
                document_id=None,  # the borrower's document AND the corpus
                version=None,
                metadata=DEMO_METADATA,
                as_of=DEMO_AS_OF,
                top_k=32,
                top_n=8,
            )
            if case.get("document_id") is None
            else await _scoped(case, index, embedder)
        )
    finally:
        retrieve_module.DENSE_WEIGHT, retrieve_module.LEXICAL_WEIGHT = original


async def _scoped(case: dict, index, embedder) -> RetrievalResult:
    """Exactly what the chat endpoint will call, so the eval measures the app."""
    return await retrieve_for_answer(
        case["question"],
        index=index,
        embedder=embedder,
        document_id=case["document_id"],
        version=case["version"],
        metadata=DEMO_METADATA,
        as_of=DEMO_AS_OF,
        topic_index=TOPICS,
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Dense-only vs hybrid.")
    args = parser.parse_args()

    settings = get_settings()
    index = get_vector_index(settings)
    embedder = get_embedding_provider(settings)
    cases = load_cases()

    modes = [("hybrid", False)] if not args.compare else [("dense-only", True), ("hybrid", False)]
    summaries = {}

    for label, dense_only in modes:
        results = [
            score_case(
                case, await run_case(case, index=index, embedder=embedder, dense_only=dense_only)
            )
            for case in cases
        ]
        summaries[label] = results

        print(f"\n=== {label} ===")
        for case, outcome in zip(cases, results, strict=True):
            mark = "PASS" if outcome.passed else "FAIL"
            print(
                f"  {mark}  {outcome.id:<26} recall={outcome.clause_recall:.0%} "
                f"score={outcome.best_score:.3f}  {', '.join(outcome.top)}"
            )
            if args.verbose or not outcome.passed:
                if outcome.clause_recall < 1.0:
                    print(f"        missing clauses: {case.get('expect_clauses')}")
                if not outcome.corpus_ok:
                    print(f"        missing circular: {case.get('expect_corpus')}")
                if not outcome.rank1_ok:
                    print("        governing circular did not rank first")
                if not outcome.refusal_ok:
                    print(f"        should have been weak, scored {outcome.best_score:.3f}")
                if outcome.forbidden_hit:
                    print(f"        leaked forbidden text: {outcome.forbidden_hit!r}")

        answerable = [
            r for r, c in zip(results, cases, strict=True) if not c.get("expect_insufficient")
        ]
        unanswerable = [
            r for r, c in zip(results, cases, strict=True) if c.get("expect_insufficient")
        ]
        print(
            f"  -> clause recall {_mean(r.clause_recall for r in answerable):.0%}"
            f" | circular found {sum(r.corpus_ok for r in answerable)}/{len(answerable)}"
            f" | refusals {sum(r.refusal_ok for r in unanswerable)}/{len(unanswerable)}"
            f" | passed {sum(r.passed for r in results)}/{len(results)}"
        )

    if args.compare:
        dense = sum(r.passed for r in summaries["dense-only"])
        hybrid = sum(r.passed for r in summaries["hybrid"])
        print(f"\ndense-only {dense}/{len(cases)}  ->  hybrid {hybrid}/{len(cases)}")

    failed = sum(not r.passed for r in summaries[modes[-1][0]])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
