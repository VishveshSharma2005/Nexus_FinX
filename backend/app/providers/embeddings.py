"""Embedding providers.

Two implementations behind :class:`app.core.contracts.EmbeddingProvider`. The
RAG layer imports neither; it receives one through dependency injection, which
is what keeps a vendor SDK out of retrieval.

The default is deterministic and offline. That is not only a convenience for
tests: an evaluation harness whose numbers move because a hosted model was
re-versioned is not measuring the retrieval changes you made, and a demo that
needs a working API key is a demo that can fail in the room.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.core.contracts import EmbeddingProvider

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")

# Words that appear in almost every clause of every loan agreement carry no
# discriminating signal, and letting them dominate makes every clause look
# alike to a lexical embedding.
_STOPWORDS = frozenset(
    """a an and or of to in on for by with the this that shall be is are as at from any
    such other under which it its their his her been being will may must not no
    borrower lender agreement clause loan""".split()
)


def _tokenise(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class HashingEmbeddingProvider:
    """A deterministic bag-of-words embedding. No network, no model weights.

    Words are hashed into a fixed number of buckets and weighted by frequency,
    then L2-normalised so cosine similarity is a dot product. This is a real
    lexical similarity -- two clauses about prepayment genuinely score higher
    against each other than against a clause about insurance -- so retrieval,
    the evidence gate and the evaluation harness can all be exercised, and
    their thresholds tuned, before any API key exists.

    It is not semantic. It will not match "foreclosure" to "prepayment" unless
    the words co-occur. Hybrid retrieval compensates in part, and swapping in
    the hosted provider is one environment variable, which is the point of the
    interface.
    """

    name = "hashing"

    def __init__(self, *, dimension: int = 512, model: str = "hashing-bow-v1") -> None:
        self.dimension = dimension
        self.model = model

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _tokenise(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            # Sign from a second slice, so unrelated words sharing a bucket
            # tend to cancel rather than reinforce.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        # Sub-linear term weighting, then L2 normalisation.
        vector = [math.copysign(math.log1p(abs(value)), value) for value in vector]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class NvidiaEmbeddingProvider:
    """Hosted embeddings over the NIM OpenAI-compatible endpoint."""

    name = "nvidia"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "nvidia/nv-embedqa-e5-v5",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        dimension: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Use FINX_EMBEDDING_PROVIDER=fake to run offline."
            )
        self.model = model
        self.dimension = dimension
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "input": texts,
                    "model": self.model,
                    "input_type": input_type,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            payload = response.json()

        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._post(texts, "passage") if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return (await self._post([text], "query"))[0]


def _assert_conformance() -> None:
    """Both implementations satisfy the interface they claim to."""
    assert isinstance(HashingEmbeddingProvider(), EmbeddingProvider)


_assert_conformance()
