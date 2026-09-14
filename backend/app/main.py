"""FinX API entrypoint.

Phase 0 exposes health and configuration-introspection endpoints only. Feature
routers are mounted in their own phases.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings

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
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    def readyz() -> dict[str, object]:
        """Report which interchangeable backends this process resolved to.

        Useful in a live demo: it makes the swap-a-model / swap-an-index claim
        visible rather than asserted.
        """
        return {
            "status": "ok",
            "env": settings.env,
            "index_backend": settings.index.value,
            "llm_provider": settings.llm_provider.value,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider.value,
            "llm_credentials_present": bool(settings.nvidia_api_key),
        }

    return app


app = create_app()
