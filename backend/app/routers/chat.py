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
from app.providers.llm import ExtractiveProvider
from app.rag.coverage import find_gaps
from app.rag.fallback import build_fallback
from app.rag.gate import evaluate
from app.rag.generate import generate_stream
from app.rag.retrieve import build_topic_index, retrieve_for_answer
from app.rag.verify import verify
from app.routers.documents import get_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_TOPIC_INDEX: dict[str, set[str]] | None = None
_MANIFEST = None


def manifest(settings: Settings):
    """The RBI manifest, loaded once. None if it cannot be read."""
    global _MANIFEST
    if _MANIFEST is None:
        try:
            _MANIFEST = load_manifest(settings.rbi_corpus_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("RBI manifest unavailable: %s", exc)
            _MANIFEST = False
    return _MANIFEST or None


def topic_index(settings: Settings) -> dict[str, set[str]]:
    """Curated topics from the RBI manifest, loaded once."""
    global _TOPIC_INDEX
    if _TOPIC_INDEX is None:
        loaded = manifest(settings)
        _TOPIC_INDEX = build_topic_index(loaded) if loaded else {}
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

            decision = evaluate(retrieval, threshold=settings.evidence_min_confidence)

            yield _sse(
                "meta",
                {
                    "document_id": document.document_id,
                    "version": document.version,
                    "title": document.title,
                    "confidence": round(retrieval.confidence, 4),
                    "threshold": decision.threshold,
                    "gate": decision.outcome.value,
                    "considered": retrieval.considered,
                    "provider": provider.name,
                    "model": provider.model,
                },
            )

            # --- Below the gate: no confident answer -----------------------
            # A separate path, and a short one. No citations, no risk engine,
            # no document or RBI context sent to the model. Nothing streams,
            # because a streamed draft would reach the screen before the claim
            # filter had run over it.
            if not decision.passed:
                fallback_answer = await build_fallback(
                    request.question,
                    reason=decision.reason,
                    provider=provider,
                    use_model=provider.name != ExtractiveProvider.name,
                )
                yield _sse(
                    "fallback",
                    {
                        "notice": fallback_answer.notice,
                        "reason": fallback_answer.reason,
                        "explanation": fallback_answer.explanation,
                        "advisor": fallback_answer.advisor,
                        "explanation_source": fallback_answer.source,
                        "filtered_out": len(fallback_answer.rejected),
                    },
                )
                yield _sse("done", {"grounded": False})
                return

            # --- Above the gate, but regulation of the time is not held ----
            loaded = manifest(settings)
            gaps = (
                find_gaps(request.question, loan_date=document.as_of, manifest=loaded)
                if loaded
                else []
            )
            if gaps:
                yield _sse(
                    "coverage",
                    [
                        {
                            "superseded_by": gap.superseded_by,
                            "effective_from": gap.effective_from.isoformat(),
                            "missing_circulars": list(gap.missing_circulars),
                            "message": gap.message,
                        }
                        for gap in gaps
                    ],
                )

            citations = [
                _citation_payload(passage, number)
                for number, passage in enumerate(retrieval.passages, start=1)
            ]
            yield _sse("citations", citations)

            # The quoting provider is the safety net: if the hosted model is
            # rejected, slow or down, the answer still arrives, visibly marked.
            fallback = None if provider.name == ExtractiveProvider.name else ExtractiveProvider()

            draft: list[str] = []
            async for kind, payload in generate_stream(
                request.question,
                retrieval.passages,
                provider=provider,
                summary=document.summary,
                fallback=fallback,
                first_token_timeout=settings.llm_first_token_timeout,
            ):
                if kind == "delta":
                    draft.append(payload)
                    yield _sse("delta", {"text": payload})
                    await asyncio.sleep(0)  # let the event reach the browser
                elif kind == "degraded":
                    draft = []
                    yield _sse(
                        "degraded",
                        {
                            "reason": payload,
                            "provider": ExtractiveProvider.name,
                            "message": (
                                f"The language model was unavailable: {payload}. This answer "
                                "was assembled directly from the quoted clauses instead."
                            ),
                        },
                    )
                else:
                    checked = verify("".join(draft), retrieval.passages, question=request.question)
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
