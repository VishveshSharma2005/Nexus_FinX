"""Dependency wiring.

This is the one place allowed to name concrete implementations. Routers and the
RAG layer receive interfaces from here, which is what keeps
``LiteParser`` and any vendor SDK out of the rest of the codebase.

Phase 0: the registry exists and resolves by configuration, but the concrete
implementations land in their own phases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import Settings, get_settings

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
    """Resolve the configured model provider. Implemented in Phase 4."""
    settings = settings or get_settings()
    raise NotImplementedError("Provider registry lands in Phase 4")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Resolve the configured embedding provider."""
    from app.config import EmbeddingProviderName
    from app.providers.embeddings import HashingEmbeddingProvider, NvidiaEmbeddingProvider

    settings = settings or get_settings()

    if settings.embedding_provider is EmbeddingProviderName.NVIDIA:
        return NvidiaEmbeddingProvider(
            api_key=settings.nvidia_api_key,
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            dimension=settings.embedding_dim,
        )
    return HashingEmbeddingProvider()


def get_vector_index(settings: Settings | None = None) -> VectorIndex:
    """Resolve the configured index backend.

    Both implementations satisfy the same interface, so retrieval is written
    once and the choice is an environment variable.
    """
    from app.config import IndexBackend
    from app.rag.index import LocalVectorIndex

    settings = settings or get_settings()

    if settings.index is IndexBackend.PGVECTOR:
        from app.rag.pgvector_index import PgVectorIndex

        return PgVectorIndex(settings.database_url, dimension=settings.embedding_dim)
    return LocalVectorIndex(settings.local_index_path)
