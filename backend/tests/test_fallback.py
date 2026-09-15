"""The evidence gate, the fallback path, and coverage gaps.

The fallback is the one path where FinX has no source, so every test here is
about what it must not say.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.config import get_settings
from app.core.contracts import LLMMessage, LLMResponse, RetrievedPassage, SourceKind
from app.core.corpus import load_manifest
from app.rag.coverage import find_gaps
from app.rag.fallback import (
    ADVISOR,
    FALLBACK_SYSTEM,
    GLOSSARY,
    NOTICE,
    build_fallback,
    filter_claims,
)
from app.rag.gate import MIN_CONFIDENCE, GateOutcome, evaluate
from app.rag.retrieve import RetrievalResult
from tests.index_contract import make_chunk


def _result(*confidences: float, source: SourceKind = SourceKind.USER_DOCUMENT):
    passages = [
        RetrievedPassage(
            chunk=make_chunk("loan", 1, f"clause text {i}", ordinal=i, source_kind=source),
            score=value,
            dense_score=value,
        )
        for i, value in enumerate(confidences)
    ]
    return RetrievalResult(passages=passages, query="q")


# --- The gate -----------------------------------------------------------------


def test_a_strong_match_passes() -> None:
    assert evaluate(_result(0.81, 0.77)).passed


@pytest.mark.parametrize("confidence", [0.695, 0.663, 0.70])
def test_the_measured_unanswerable_band_is_refused(confidence) -> None:
    """The goldset's unanswerable questions scored 0.66-0.70 with this model."""
    decision = evaluate(_result(confidence))
    assert not decision.passed
    assert decision.outcome is GateOutcome.INSUFFICIENT_EVIDENCE


def test_the_threshold_sits_between_the_measured_bands() -> None:
    assert 0.70 < MIN_CONFIDENCE < 0.74


def test_nothing_retrieved_is_its_own_outcome() -> None:
    assert evaluate(RetrievalResult(passages=[], query="q")).outcome is (
        GateOutcome.NOTHING_RETRIEVED
    )


def test_regulation_alone_is_not_an_answer_about_the_loan() -> None:
    """A circular can say what lenders may do, not what this lender did."""
    decision = evaluate(_result(0.85, source=SourceKind.RBI_CORPUS))
    assert not decision.passed


def test_a_refusal_explains_itself() -> None:
    decision = evaluate(_result(0.69))
    assert "0.69" in decision.reason and "0.72" in decision.reason


# --- The claim filter ---------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "RBI does not allow lenders to charge prepayment penalties on floating rate loans.",
        "Your agreement has a lock-in in clause 11.2.",
        "Lenders must disclose every charge in the Key Facts Statement.",
        "This is prohibited under the current guidelines.",
        "The penalty is usually around two percent, or 2% a month.",
        "In your case the lender cannot legally refuse.",
    ],
)
def test_unsourced_claims_are_removed(sentence) -> None:
    filtered = filter_claims(sentence)
    assert filtered.text == ""
    assert filtered.rejected


@pytest.mark.parametrize(
    "sentence",
    [
        "A lock-in period generally means a time during which a loan cannot be repaid early.",
        "Prepayment usually reduces the total interest paid over the life of a loan.",
    ],
)
def test_general_definitions_survive(sentence) -> None:
    assert filter_claims(sentence).text == sentence


def test_every_glossary_entry_passes_its_own_filter() -> None:
    """The no-model explanation must never be emptied by the filter meant for models."""
    for key, entry in GLOSSARY.items():
        assert filter_claims(entry).text == entry, f"glossary entry {key!r} would be filtered"


def test_the_fixed_notice_and_advisor_make_no_regulatory_claim() -> None:
    from app.rag.fallback import _FIGURE, _REGULATORY

    for text in (NOTICE, ADVISOR):
        assert not _REGULATORY.search(text.replace("RBI corpus", "corpus"))
        assert not _FIGURE.search(text)


# --- The separate prompt path -------------------------------------------------


class _RecordingProvider:
    """Captures what the fallback path sends, and answers like a helpful model."""

    name = "recording"
    model = "recording"
    supported_languages = frozenset({"en"})

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.sent: list[LLMMessage] = []

    async def complete(self, messages, **_):
        self.sent = list(messages)
        return LLMResponse(text=self.reply, model=self.model)

    async def stream(self, messages, **_):  # pragma: no cover - fallback never streams
        raise AssertionError("the fallback path must not stream")
        yield ""


@pytest.mark.asyncio
async def test_the_fallback_prompt_carries_no_document_and_no_regulation() -> None:
    provider = _RecordingProvider("A lock-in period generally means an early-repayment bar.")
    await build_fallback("is there a lock-in?", reason="r", provider=provider, use_model=True)

    sent = "\n".join(message.content for message in provider.sent)
    assert provider.sent[0].content == FALLBACK_SYSTEM
    assert provider.sent[-1].content == "is there a lock-in?"
    for forbidden in ("Passages:", "[1]", "clause 11", "RBI/20", "document being discussed"):
        assert forbidden not in sent


@pytest.mark.asyncio
async def test_a_helpful_model_is_filtered_before_anything_is_shown() -> None:
    provider = _RecordingProvider(
        "A lock-in period generally means a time when a loan cannot be repaid early. "
        "RBI prohibits lock-ins on floating rate home loans. "
        "Your agreement sets a 12 month lock-in."
    )
    answer = await build_fallback("lock-in?", reason="r", provider=provider, use_model=True)
    assert "RBI" not in answer.explanation
    assert "12" not in answer.explanation
    assert "generally means" in answer.explanation
    assert len(answer.rejected) == 2


@pytest.mark.asyncio
async def test_all_three_parts_are_present_every_time() -> None:
    answer = await build_fallback("crypto shares", reason="r", provider=None, use_model=False)
    assert answer.notice == NOTICE
    assert answer.explanation
    assert answer.advisor == ADVISOR


@pytest.mark.asyncio
async def test_a_failing_model_still_yields_a_fallback() -> None:
    class Broken(_RecordingProvider):
        async def complete(self, messages, **_):
            raise RuntimeError("endpoint down")

    answer = await build_fallback(
        "what is an EMI?", reason="r", provider=Broken(""), use_model=True
    )
    assert answer.explanation
    assert answer.source == "glossary"


# --- Coverage gaps ------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(get_settings().rbi_corpus_dir)


def test_a_pre_2026_prepayment_question_reports_the_missing_regulation(manifest) -> None:
    """The 2025 Directions do not govern a 2025 loan, and what did is not held.

    Silence here would let an answer drawn only from the agreement read as
    though no rule applied.
    """
    gaps = find_gaps(
        "can they charge me for prepaying my loan?", loan_date=date(2025, 11, 18), manifest=manifest
    )
    assert len(gaps) == 1
    assert gaps[0].superseded_by == "RBI/2025-26/64"
    assert len(gaps[0].missing_circulars) == 8
    assert "does not apply to your loan" in gaps[0].message


def test_a_loan_after_the_directions_has_no_gap(manifest) -> None:
    assert not find_gaps("can I prepay?", loan_date=date(2026, 1, 14), manifest=manifest)


def test_an_unrelated_question_has_no_gap(manifest) -> None:
    assert not find_gaps(
        "what penal charges apply?", loan_date=date(2025, 11, 18), manifest=manifest
    )


def test_an_undated_loan_has_no_gap_rather_than_a_guessed_one(manifest) -> None:
    assert not find_gaps("can I prepay?", loan_date=None, manifest=manifest)


# --- End to end, through the API ---------------------------------------------


def test_an_unanswerable_question_takes_the_fallback_path(tmp_path) -> None:
    """Asserts the whole path through HTTP: no citations, three parts, done."""
    import json

    from fastapi.testclient import TestClient

    from app.config import IndexBackend, Settings
    from app.main import create_app

    settings = Settings(
        _env_file=None,
        env="test",
        index=IndexBackend.LOCAL,
        local_index_path=tmp_path / "api.sqlite3",
        embedding_provider="fake",
        evidence_min_confidence=0.99,  # force the gate shut with the fake embedder
    )
    agreement = settings.sample_corpus_dir / "home_loan_agreement_A.docx"
    if not agreement.is_file():
        pytest.skip("sample agreement not present")

    client = TestClient(create_app(settings))
    upload = client.post(
        "/documents",
        files={"file": (agreement.name, agreement.read_bytes(), "application/octet-stream")},
    )
    document_id = upload.json()["document_id"]

    with client.stream(
        "POST",
        "/chat",
        json={"question": "Can I pay in cryptocurrency?", "document_id": document_id},
    ) as response:
        events: dict[str, dict] = {}
        current = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                current = line[7:]
            elif line.startswith("data: ") and current:
                events.setdefault(current, json.loads(line[6:]))

    assert events["meta"]["gate"] == "insufficient_evidence"
    assert "fallback" in events
    assert "citations" not in events, "a fallback must not present sources as if it had answered"
    assert "delta" not in events, "the fallback path must not stream unfiltered text"
    assert {"notice", "explanation", "advisor"} <= set(events["fallback"])
