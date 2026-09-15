"""Cited chat over one version of one document, streamed.

Server-sent events rather than a single response, because retrieval plus
generation takes a few seconds and a borrower watching a blank box learns
nothing. The event order is deliberate:

    citations -> delta* -> verified -> done

Citations arrive first, so the sources are on screen before the answer is --
the point being that the answer comes from them, not the other way round. The
verified event carries the checked answer and replaces the streamed draft;
verification only ever removes sentences, so it is always a subset of what was
shown, and anything removed is reported rather than quietly dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.corpus import load_manifest
from app.core.deps import get_embedding_provider, get_llm_provider, get_vector_index
from app.db.documents import DocumentStore
from app.rag.generate import generate_stream
from app.rag.retrieve import build_topic_index, retrieve_for_answer
from app.rag.verify import verify
from app.routers.documents import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_TOPIC_INDEX: dict[str, set[str]] | None = None


def topic_index(settings: Settings) -> dict[str, set[str]]:
    """Curated topics from the RBI manifest, loaded once."""
    global _TOPIC_INDEX
    if _TOPIC_INDEX is None:
        try:
            _TOPIC_INDEX = build_topic_index(load_manifest(settings.rbi_corpus_dir))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("RBI manifest unavailable, topic ranking disabled: %s", exc)
            _TOPIC_INDEX = {}
    return _TOPIC_INDEX


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: str
    version: int | None = None


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _citation_payload(passage, number: int) -> dict:
    citation = passage.chunk.citation
    return {
        "n": number,
        "chunk_id": citation.chunk_id,
        "source": citation.source_kind.value,
        "document_id": citation.document_id,
        "version": citation.version,
        "clause_number": citation.clause_number,
        "page": citation.page,
        "title": citation.title,
        "circular_id": citation.circular_id,
        "source_url": citation.source_url,
        "text": passage.chunk.text,
        "score": round(passage.score, 4),
        "confidence": round(passage.dense_score or 0.0, 4),
    }


@router.post("")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: DocumentStore = Depends(get_store),
) -> StreamingResponse:
    document = (
        store.get(request.document_id, request.version)
        if request.version is not None
        else store.latest(request.document_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Upload a document before asking about it.")

    index = get_vector_index(settings)
    embedder = get_embedding_provider(settings)
    provider = get_llm_provider(settings)

    async def stream():
        try:
            retrieval = await retrieve_for_answer(
                request.question,
                index=index,
                embedder=embedder,
                document_id=document.document_id,
                version=document.version,
                metadata=document.metadata,
                as_of=document.as_of,
                topic_index=topic_index(settings),
                top_n=settings.rerank_top_n,
            )

            yield _sse(
                "meta",
                {
                    "document_id": document.document_id,
                    "version": document.version,
                    "title": document.title,
                    "confidence": round(retrieval.confidence, 4),
                    "considered": retrieval.considered,
                    "provider": provider.name,
                    "model": provider.model,
                },
            )

            if retrieval.is_empty:
                yield _sse(
                    "verified",
                    {
                        "text": (
                            "Nothing in this document matched your question closely enough "
                            "to answer from. Try naming a charge, a clause, or a term you "
                            "saw in the agreement."
                        ),
                        "stripped": [],
                    },
                )
                yield _sse("done", {"grounded": False})
                return

            citations = [
                _citation_payload(passage, number)
                for number, passage in enumerate(retrieval.passages, start=1)
            ]
            yield _sse("citations", citations)

            draft: list[str] = []
            async for kind, payload in generate_stream(
                request.question,
                retrieval.passages,
                provider=provider,
                summary=document.summary,
            ):
                if kind == "delta":
                    draft.append(payload)
                    yield _sse("delta", {"text": payload})
                    await asyncio.sleep(0)  # let the event reach the browser
                else:
                    checked = verify("".join(draft), retrieval.passages)
                    yield _sse(
                        "verified",
                        {
                            "text": checked.text,
                            "stripped": checked.stripped,
                            "fully_grounded": checked.fully_grounded,
                        },
                    )

            yield _sse("done", {"grounded": True})

        except Exception as exc:
            logger.exception("chat failed")
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
