"""FinX API entrypoint.

Phase 0 exposes health and configuration-introspection endpoints only. Feature
routers are mounted in their own phases.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.routers import chat, documents

logger = logging.getLogger("finx")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="FinX API",
        version="0.1.0",
        summary="Loan agreement intelligence: explain, flag, quantify, compare.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents.router)
    app.include_router(chat.router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    def readyz() -> dict[str, object]:
        """Report which interchangeable backends this process resolved to.

        Useful in a live demo: it makes the swap-a-model / swap-an-index claim
        visible rather than asserted.
        """
        # Report what actually resolved, not what was requested. The two
        # differ when the hosted provider is selected without a key.
        from app.core.deps import get_embedding_provider, get_llm_provider

        llm = get_llm_provider(settings)
        embedder = get_embedding_provider(settings)
        return {
            "status": "ok",
            "env": settings.env,
            "index_backend": settings.index.value,
            "llm_provider": llm.name,
            "llm_model": llm.model,
            "llm_requested": settings.llm_provider.value,
            "embedding_provider": embedder.name,
            "embedding_model": embedder.model,
            "llm_credentials_present": bool(settings.nvidia_api_key),
        }

    return app


app = create_app()
