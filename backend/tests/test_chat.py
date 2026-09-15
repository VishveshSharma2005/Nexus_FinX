"""Citation binding, generation, and the chat endpoint.

The verifier is the last thing standing between a model's output and a
borrower, so most of this file is about what it refuses to pass.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import IndexBackend, Settings
from app.core.contracts import SourceKind
from app.main import create_app
from app.providers.embeddings import HashingEmbeddingProvider
from app.providers.llm import ExtractiveProvider
from app.rag.generate import build_messages, generate
from app.rag.index import LocalVectorIndex
from app.rag.retrieve import retrieve
from app.rag.verify import verify
from tests.index_contract import make_chunk

CLAUSES = [
    "11.2 The Borrower shall not prepay the Loan during the Lock-in Period of 12 months.",
    "9.1 Penal charges of 2% per month shall be levied on the overdue amount.",
    "7.1 The Borrower shall maintain a Property Insurance policy for the whole tenure.",
]


@pytest.fixture
async def passages(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.sqlite3")
    embedder = HashingEmbeddingProvider()
    chunks = [make_chunk("loan-1", 1, text, ordinal=i) for i, text in enumerate(CLAUSES)]
    await index.upsert(chunks, await embedder.embed_documents(CLAUSES))
    result = await retrieve(
        "lock-in period prepayment",
        index=index,
        embedder=embedder,
        document_id="loan-1",
        version=1,
        top_n=3,
    )
    index.close()
    return result.passages


# --- Verification -----------------------------------------------------------


def test_a_supported_sentence_is_kept(passages) -> None:
    answer = "The Borrower shall not prepay during the Lock-in Period of 12 months [1]."
    checked = verify(answer, passages)
    assert checked.text
    assert checked.citations
    assert checked.fully_grounded


def test_an_uncited_claim_is_removed(passages) -> None:
    """It may even be true. Nothing binds it to a source, so it cannot stand."""
    answer = "Your lender will waive the charge if you ask nicely."
    checked = verify(answer, passages)
    assert checked.text == ""
    assert checked.stripped


def test_a_claim_citing_an_unrelated_passage_is_removed(passages) -> None:
    """The dangerous case: a citation makes an invention look verified."""
    answer = "You may prepay at any time with no charge whatsoever [3]."
    checked = verify(answer, passages)
    assert "prepay at any time" not in checked.text
    assert checked.stripped


def test_a_citation_pointing_at_nothing_is_removed(passages) -> None:
    answer = "The agreement caps all charges at one rupee [99]."
    checked = verify(answer, passages)
    assert checked.text == ""


def test_framing_sentences_survive_without_a_citation(passages) -> None:
    answer = (
        "Based on the clauses retrieved from your documents:\n"
        "- Penal charges of 2% per month are levied on overdue amounts [2]."
    )
    checked = verify(answer, passages)
    assert "Based on the clauses" in checked.text
    assert "2%" in checked.text


def test_verification_never_adds_text(passages) -> None:
    """The stream shows the draft first, so the checked answer must be a subset."""
    answer = (
        "Penal charges of 2% per month are levied on the overdue amount [2].\n"
        "The lender may also seize your passport."
    )
    checked = verify(answer, passages)
    for line in checked.text.split("\n"):
        assert line in answer


def test_every_kept_sentence_reports_which_chunk_backs_it(passages) -> None:
    checked = verify("Penal charges of 2% per month are levied [2].", passages)
    kept = [s for s in checked.sentences if s.kept and s.citations]
    assert kept
    for sentence in kept:
        for citation in sentence.citations:
            assert citation.chunk_id


# --- Prompt construction ----------------------------------------------------


def test_the_prompt_carries_only_the_retrieved_clauses(passages) -> None:
    """Each turn sends the passages plus a short summary, never the document."""
    messages = build_messages("what is my lock-in?", passages, summary="Home loan, v1")
    system = messages[0].content
    assert "Home loan, v1" in system
    assert system.count("[1]") == 1
    assert len(system) < 6000


def test_passages_are_numbered_from_one(passages) -> None:
    system = build_messages("q", passages)[0].content
    for number in range(1, len(passages) + 1):
        assert f"[{number}]" in system


@pytest.mark.asyncio
async def test_the_default_provider_produces_a_grounded_answer(passages) -> None:
    result = await generate("is there a lock-in period?", passages, provider=ExtractiveProvider())
    assert result.answer.text
    assert result.answer.citations
    # The extractive provider quotes, so nothing should ever be stripped.
    assert result.answer.fully_grounded


# --- The endpoint -----------------------------------------------------------


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(
        _env_file=None,
        env="test",
        index=IndexBackend.LOCAL,
        local_index_path=tmp_path / "api.sqlite3",
        embedding_provider="fake",
    )
    return TestClient(create_app(settings))


def test_asking_about_an_unknown_document_is_a_404(client: TestClient) -> None:
    response = client.post("/chat", json={"question": "what is my rate?", "document_id": "nope"})
    assert response.status_code == 404


def test_uploading_an_unsupported_format_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


def test_uploading_an_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("a.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_the_stream_sends_citations_before_the_answer(tmp_path) -> None:
    """Sources arrive first, because the answer comes from them."""
    settings = Settings(
        _env_file=None,
        env="test",
        index=IndexBackend.LOCAL,
        local_index_path=tmp_path / "api.sqlite3",
        embedding_provider="fake",
    )
    app = create_app(settings)
    client = TestClient(app)

    samples = settings.sample_corpus_dir
    agreement = samples / "home_loan_agreement_A.docx"
    if not agreement.is_file():
        pytest.skip("sample agreement not present")

    upload = client.post(
        "/documents",
        files={
            "file": (
                agreement.name,
                agreement.read_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert upload.status_code == 200, upload.text
    document_id = upload.json()["document_id"]

    with client.stream(
        "POST",
        "/chat",
        json={
            "question": "Is there a lock-in period before I can prepay?",
            "document_id": document_id,
        },
    ) as response:
        assert response.status_code == 200
        events = []
        payloads = {}
        current = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                current = line[7:]
                events.append(current)
            elif line.startswith("data: ") and current:
                payloads.setdefault(current, line[6:])

    assert "citations" in events
    assert "verified" in events
    assert events.index("citations") < events.index("verified")

    citations = json.loads(payloads["citations"])
    assert citations
    assert all(c["chunk_id"] for c in citations)
    assert any(c["source"] == SourceKind.USER_DOCUMENT.value for c in citations)


# --- Fluent-model output: what a hosted model does that the quoter does not --

FLUENT_PASSAGES = [
    "11.2 The Borrower shall not be entitled to prepay the Loan, in whole or in part, during "
    "the Lock-in Period of 12 (twelve) months from the date of first disbursement.",
    "11.3 Any prepayment made after the Lock-in Period, where the source of funds for such "
    "prepayment is other than the Borrower's own verified savings or income, shall attract a "
    "prepayment charge of 3% (three percent) of the principal amount so prepaid.",
    "7.2 The Borrower shall additionally obtain a Loan Protection (Credit Life) Insurance "
    "policy. The single premium of Rs. 1,90,000 shall be financed into the Loan.",
]


@pytest.fixture
def fluent_passages():
    from app.core.contracts import RetrievedPassage

    return [
        RetrievedPassage(chunk=make_chunk("loan", 1, text, ordinal=i), score=0.8, dense_score=0.8)
        for i, text in enumerate(FLUENT_PASSAGES)
    ]


@pytest.mark.parametrize(
    "sentence",
    [
        "Yes, your agreement has a lock-in [1].",
        "You cannot prepay the loan at all during the first 12 months after disbursement [1].",
        "After that, if you pay the loan off using money borrowed from another lender, you "
        "would be charged 3% of the principal you prepay [2].",
        "The insurance premium of Rs. 1,90,000 is added to your loan amount [3].",
    ],
)
def test_a_faithful_paraphrase_is_kept(fluent_passages, sentence) -> None:
    """A fluent model paraphrases nearly every sentence.

    Exact word matching deleted the third of these -- a correct summary of
    clause 11.3 -- because the clause says "prepayment" and "source of funds"
    where the sentence says "pay off" and "money borrowed". A verifier that
    strips good paraphrase would gut every answer a hosted model wrote.
    """
    checked = verify(sentence, fluent_passages)
    assert checked.text, f"wrongly stripped: {checked.sentences[0].reason}"


@pytest.mark.parametrize(
    ("sentence", "reason"),
    [
        ("The lock-in lasts 18 months from disbursement [1].", "figure"),
        ("The insurance premium works out to roughly Rs 1.9 lakh [3].", "figure"),
        ("RBI rules also let you cancel the insurance at any time without penalty [3].", "support"),
        ("Most borrowers in this situation refinance with a public sector bank.", "citation"),
    ],
)
def test_a_fluent_invention_is_stripped(fluent_passages, sentence, reason) -> None:
    """The four ways a fluent model goes wrong that a quoter cannot.

    The first is the one term overlap alone lets through: "the lock-in lasts
    18 months [1]" shares nearly every word with a clause that says twelve.
    """
    checked = verify(sentence, fluent_passages)
    assert not checked.text
    assert reason in checked.sentences[0].reason or (
        reason == "support" and "does not support" in checked.sentences[0].reason
    )


def test_a_figure_the_borrower_asked_about_may_be_repeated(fluent_passages) -> None:
    """Echoing "month 13" from the question is not an invented figure."""
    sentence = "If you switch in month 13, the lock-in has ended but the 3% charge applies [2]."
    kept = verify(sentence, fluent_passages, question="what if I switch lenders in month 13?")
    stripped = verify(sentence, fluent_passages)
    assert kept.text
    assert not stripped.text


def test_rupee_abbreviations_do_not_split_a_sentence() -> None:
    """Splitting at "Rs." let the half carrying the figure escape the check."""
    from app.rag.verify import split_sentences

    parts = split_sentences("The premium of Rs. 1,90,000 is financed [3]. The rate is 8.35% [1].")
    assert parts == ["The premium of Rs. 1,90,000 is financed [3].", "The rate is 8.35% [1]."]


def test_figures_ignore_clause_addresses() -> None:
    from app.rag.verify import figures

    assert figures("Clause 11.3 charges 3% on Rs. 1,90,000") == {"3", "190000"}


# --- A hosted model that fails must degrade, never hang or crash ------------


class _FailingProvider:
    name = "broken"
    model = "broken-model"
    supported_languages = frozenset({"en"})

    async def complete(self, messages, **_):  # pragma: no cover - not used
        raise RuntimeError("the NVIDIA API key was rejected")

    async def stream(self, messages, **_):
        raise RuntimeError("the NVIDIA API key was rejected")
        yield ""  # pragma: no cover - makes this an async generator


class _HangingProvider(_FailingProvider):
    name = "hanging"

    async def stream(self, messages, **_):
        import asyncio

        await asyncio.sleep(30)
        yield "too late"


class _MidStreamFailure(_FailingProvider):
    name = "flaky"

    async def stream(self, messages, **_):
        yield "The lock-in lasts "
        raise RuntimeError("the NVIDIA service returned an error (502)")


async def _collect(provider, passages, **kwargs):
    from app.rag.generate import generate_stream

    return [
        event
        async for event in generate_stream(
            "is there a lock-in period?",
            passages,
            provider=provider,
            fallback=ExtractiveProvider(),
            **kwargs,
        )
    ]


@pytest.mark.asyncio
async def test_a_failing_model_falls_back_with_a_stated_reason(passages) -> None:
    events = await _collect(_FailingProvider(), passages)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "degraded"
    assert "rejected" in events[0][1]
    assert kinds[-1] == "verified"
    assert events[-1][1], "the fallback must still deliver an answer"


@pytest.mark.asyncio
async def test_a_silent_model_is_abandoned_after_the_first_token_timeout(passages) -> None:
    import time

    started = time.monotonic()
    events = await _collect(_HangingProvider(), passages, first_token_timeout=0.5)
    assert time.monotonic() - started < 5, "a slow endpoint must not hold the screen"
    assert events[0][0] == "degraded"
    assert "did not start answering" in events[0][1]
    assert events[-1][1]


@pytest.mark.asyncio
async def test_a_mid_stream_failure_discards_the_partial_draft(passages) -> None:
    """Half a sentence from the failed model must not survive into the answer."""
    events = await _collect(_MidStreamFailure(), passages)
    assert "degraded" in [kind for kind, _ in events]
    assert "The lock-in lasts" not in events[-1][1]


def test_a_retired_model_is_reported_as_retired() -> None:
    import httpx

    from app.providers.llm import _describe

    response = httpx.Response(410, request=httpx.Request("POST", "https://x"))
    reason = _describe(httpx.HTTPStatusError("gone", request=response.request, response=response))
    assert "retired" in reason
