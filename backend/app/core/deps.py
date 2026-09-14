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
    """Resolve the configured parser.

    The import is local so that ``app.parsers`` stays out of every other
    module's import graph -- callers receive the interface, never the class.
    """
    from app.parsers.registry import DEFAULT_PARSER, build_parser

    settings = settings or get_settings()
    return build_parser(DEFAULT_PARSER, settings)


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Resolve the configured model provider. Implemented in Phase 4."""
    settings = settings or get_settings()
    raise NotImplementedError("Provider registry lands in Phase 4")


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Resolve the configured embedding provider. Implemented in Phase 2."""
    settings = settings or get_settings()
    raise NotImplementedError("Embedding registry lands in Phase 2")


def get_vector_index(settings: Settings | None = None) -> VectorIndex:
    """Resolve pgvector or the local SQLite index. Implemented in Phase 2."""
    settings = settings or get_settings()
    raise NotImplementedError("Index registry lands in Phase 2")
