"""Dependency wiring.

This is the one place allowed to name concrete implementations. Routers and the
RAG layer receive interfaces from here, which is what keeps
``LiteParser`` and any vendor SDK out of the rest of the codebase.

Phase 0: the registry exists and resolves by configuration, but the concrete
implementations land in their own phases.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from app.core.contracts import DocumentParser, EmbeddingProvider, LLMProvider, VectorIndex


def get_document_parser(settings: Settings | None = None) -> DocumentParser:
    """Resolve the default parser.

    The import is local so that ``app.parsers`` stays out of every other
    module's import graph -- callers receive the interface, never the class.
    """
    from app.parsers.registry import DEFAULT_PARSER, build_parser

    settings = settings or get_settings()
    return build_parser(DEFAULT_PARSER, settings)


def get_parser_for(source, settings: Settings | None = None) -> DocumentParser:
    """Resolve whichever implementation handles this document's format."""
    from app.parsers.registry import select_parser

    return select_parser(source, settings or get_settings())


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Resolve the configured model provider.

    Falls back to the extractive provider when the hosted one is selected
    without a key. A demo that dies on a missing environment variable is worse
    than one that says which provider it is using, which /readyz reports.
    """
    from app.config import LLMProviderName
    from app.providers.llm import ExtractiveProvider, NemotronProvider

    settings = settings or get_settings()

    if settings.llm_provider is LLMProviderName.NEMOTRON:
        if not settings.nvidia_api_key:
            logger.warning(
                "FINX_LLM_PROVIDER=nemotron but NVIDIA_API_KEY is empty; "
                "using the extractive provider instead."
            )
            return ExtractiveProvider()
        return NemotronProvider(
            api_key=settings.nvidia_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    return ExtractiveProvider()


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Resolve the configured embedding provider."""
    from app.config import EmbeddingProviderName
    from app.providers.embeddings import (
        HashingEmbeddingProvider,
        LocalSemanticEmbeddingProvider,
        NvidiaEmbeddingProvider,
    )

    settings = settings or get_settings()

    if settings.embedding_provider is EmbeddingProviderName.NVIDIA:
        return NvidiaEmbeddingProvider(
            api_key=settings.nvidia_api_key,
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            dimension=settings.embedding_dim,
        )
    if settings.embedding_provider is EmbeddingProviderName.FAKE:
        return HashingEmbeddingProvider()
    return LocalSemanticEmbeddingProvider(
        model=settings.embedding_model,
        cache_dir=settings.model_cache_dir,
        dimension=settings.embedding_dim,
    )


@lru_cache(maxsize=4)
def _build_index(backend: str, location: str, dimension: int) -> VectorIndex:
    """One index instance per configuration, held for the process's lifetime.

    Opening a fresh SQLite connection on every request leaks handles and keeps
    the database file locked, which on Windows means it cannot be deleted or
    rebuilt while the API is running. Cached on the resolved values rather than
    on the Settings object, which is not hashable.
    """
    from app.config import IndexBackend
    from app.rag.index import LocalVectorIndex

    if backend == IndexBackend.PGVECTOR.value:
        from app.rag.pgvector_index import PgVectorIndex

        return PgVectorIndex(location, dimension=dimension)
    return LocalVectorIndex(location)


def get_vector_index(settings: Settings | None = None) -> VectorIndex:
    """Resolve the configured index backend.

    Both implementations satisfy the same interface, so retrieval is written
    once and the choice is an environment variable.
    """
    from app.config import IndexBackend

    settings = settings or get_settings()
    location = (
        settings.database_url
        if settings.index is IndexBackend.PGVECTOR
        else str(settings.local_index_path)
    )
    return _build_index(settings.index.value, location, settings.embedding_dim)
