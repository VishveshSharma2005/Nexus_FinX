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
