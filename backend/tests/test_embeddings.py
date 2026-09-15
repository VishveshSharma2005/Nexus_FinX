"""Embedding providers.

The interface conformance tests run anywhere. The semantic-quality test needs
the model weights on disk and skips without them, so a fresh clone still gets a
green suite -- but the claim that the default embedder is semantic is checked
rather than asserted in a docstring.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.core.contracts import EmbeddingProvider
from app.providers.embeddings import HashingEmbeddingProvider, LocalSemanticEmbeddingProvider

# Six clauses from the sample agreement, and questions a borrower would
# actually ask about them. The questions deliberately avoid the words the
# clauses use: "close my loan early" never says prepayment, "pledge as
# collateral" never says mortgage. That gap is the whole point of using a
# semantic model.
CLAUSES = {
    "prepayment": (
        "11.3 Any prepayment made after the Lock-in Period, where the source of funds is "
        "other than the Borrower's own funds, shall attract a prepayment charge of 3% of "
        "the principal prepaid."
    ),
    "penal": (
        "9.1 In the event of default in payment of any EMI, the Lender shall levy penal "
        "charges at 2% per month on the overdue amount."
    ),
    "insurance": (
        "7.1 The Borrower shall obtain and maintain, for the entire tenure of the Loan, a "
        "Property Insurance policy covering the Secured Property."
    ),
    "security": (
        "6.1 The Loan shall be secured by way of a registered mortgage over the Secured "
        "Property in favour of the Lender."
    ),
    "rate": (
        "3.1 The Loan carries interest at 8.35% per annum (floating), linked to the Repo "
        "Linked Lending Rate, computed on a daily reducing balance basis."
    ),
}

WORDLESS_QUESTIONS = {
    "what did I pledge as collateral?": "security",
    "how is my interest rate decided?": "rate",
    "what happens if I miss a monthly payment?": "penal",
}


async def _rank(provider: EmbeddingProvider, question: str) -> list[str]:
    keys = list(CLAUSES)
    vectors = await provider.embed_documents([CLAUSES[key] for key in keys])
    query = await provider.embed_query(question)

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    scored = sorted(zip(keys, vectors, strict=True), key=lambda kv: -cosine(query, kv[1]))
    return [key for key, _ in scored]


# --- Conformance, for every provider ---------------------------------------


@pytest.mark.parametrize(
    "provider",
    [HashingEmbeddingProvider(), LocalSemanticEmbeddingProvider()],
    ids=["hashing", "local"],
)
def test_implements_the_interface(provider) -> None:
    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension > 0
    assert provider.name and provider.model


def test_constructing_the_local_provider_does_not_load_the_model() -> None:
    """Importing and wiring must stay cheap: the API process, the CLI and the
    test suite all construct providers they may never use."""
    provider = LocalSemanticEmbeddingProvider()
    assert provider._model is None


@pytest.mark.asyncio
async def test_hashing_embeddings_are_deterministic() -> None:
    provider = HashingEmbeddingProvider()
    first = await provider.embed_query("penal charges on overdue EMI")
    second = await provider.embed_query("penal charges on overdue EMI")
    assert first == second


@pytest.mark.asyncio
async def test_embedding_an_empty_batch_returns_nothing() -> None:
    assert await HashingEmbeddingProvider().embed_documents([]) == []


# --- The reason the default is not the lexical one -------------------------


def _weights_present() -> bool:
    cache = get_settings().model_cache_dir
    return cache.is_dir() and any(cache.rglob("*.onnx"))


needs_weights = pytest.mark.skipif(
    not _weights_present(),
    reason="embedding model not downloaded; run scripts/build_index.py once",
)


@needs_weights
@pytest.mark.asyncio
async def test_the_default_embedder_matches_meaning_not_words() -> None:
    """A question that shares no vocabulary with the clause must still find it.

    The lexical fallback cannot do this, which is why it is not the default:
    with no shared terms its scores are tied and the ranking is arbitrary.
    """
    provider = LocalSemanticEmbeddingProvider(cache_dir=get_settings().model_cache_dir)
    for question, expected in WORDLESS_QUESTIONS.items():
        ranking = await _rank(provider, question)
        assert ranking[0] == expected, f"{question!r} ranked {ranking[:3]}, wanted {expected}"


@needs_weights
@pytest.mark.asyncio
async def test_the_local_provider_reports_its_real_dimension() -> None:
    provider = LocalSemanticEmbeddingProvider(cache_dir=get_settings().model_cache_dir)
    vector = await provider.embed_query("prepayment charges")
    assert len(vector) == provider.dimension, (
        "a configured dimension that disagrees with the model silently poisons the index"
    )
