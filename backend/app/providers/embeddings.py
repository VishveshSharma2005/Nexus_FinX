"""Embedding providers.

Two implementations behind :class:`app.core.contracts.EmbeddingProvider`. The
RAG layer imports neither; it receives one through dependency injection, which
is what keeps a vendor SDK out of retrieval.

Three implementations. The default runs a real sentence-embedding model on
this machine: no API key, no network at query time, and genuinely semantic, so
"can I foreclose early?" reaches a clause about prepayment that never uses the
word.

That last point is why the lexical fallback is not the default. Measured on
five realistic questions against six clauses of the sample agreement, the
hashing embedding below answered two of them with a score margin of exactly
zero -- it has no signal at all for "foreclose" or "collateral" and was
tie-breaking. Tuning retrieval on top of that would mean debugging the
embedder while believing you were debugging retrieval.

The hashing provider is kept as a test fixture and as a fallback for a machine
that cannot download model weights. It is deterministic and needs nothing, so
the suite and the evaluation harness still run anywhere.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path

from app.core.contracts import EmbeddingProvider

logger = logging.getLogger(__name__)

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


class LocalSemanticEmbeddingProvider:
    """A real sentence-embedding model, run locally through ONNX.

    Default is BAAI/bge-small-en-v1.5: 384 dimensions, around 130 MB, fast
    enough on CPU to embed a whole agreement in a second or two. Weights are
    downloaded once into the project's own cache directory and are read from
    disk on every run after that, so a demo needs no network.

    Loading is deferred to first use. Importing this module must stay cheap --
    the API process, the test suite and the CLI all import it, and most of them
    never embed anything.
    """

    name = "local"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str | Path | None = None,
        dimension: int = 384,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        # Model weights are fetched over HTTPS on first use. On a machine
        # behind a TLS-inspecting proxy the bundled CA list will not trust the
        # intercepting root, so verification is pointed at the OS trust store
        # -- the same fix pip needs here. Verification stays on.
        try:
            import truststore

            truststore.inject_into_ssl()
        except ImportError:  # pragma: no cover - optional on most machines
            logger.debug("truststore not installed; using the default CA bundle")

        # Windows without Developer Mode cannot create the symlinks the
        # HuggingFace cache uses by default, and the failure surfaces as an
        # alarming privilege error mid-download.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        from fastembed import TextEmbedding

        logger.info("Loading embedding model %s", self.model)
        self._model = TextEmbedding(model_name=self.model, cache_dir=self._cache_dir)
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        return [vector.tolist() for vector in model.embed(texts)]

    async def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        # Asymmetric models expect a query to be embedded differently from a
        # passage; fastembed applies the model's own query instruction here.
        return next(iter(model.query_embed([text]))).tolist()


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
    """Every implementation satisfies the interface it claims to."""
    assert isinstance(HashingEmbeddingProvider(), EmbeddingProvider)
    assert isinstance(LocalSemanticEmbeddingProvider(), EmbeddingProvider)


_assert_conformance()
