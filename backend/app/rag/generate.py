"""Lane 3: build the prompt, stream the answer, bind the citations.

What is sent to the model on each turn is the retrieved clauses -- typically
four to eight -- plus a short pinned summary of the document. Never the whole
agreement. The cost of a conversation therefore does not grow with the size of
the document, and a fifty-clause agreement costs the same per turn as a five-
clause one.

The prompt states the grounding rules, but the rules are not enforced by asking
nicely: :mod:`app.rag.verify` checks every sentence against the passage it
cites afterwards and removes what does not hold up.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.contracts import LLMMessage, LLMProvider, RetrievedPassage, SourceKind
from app.rag.verify import VerifiedAnswer, verify

logger = logging.getLogger(__name__)

SYSTEM_RULES = """You are FinX, explaining a borrower's own loan documents to them.

Rules you must follow exactly:
- Answer only from the numbered passages below. If they do not contain the
  answer, say so plainly.
- Put the passage number in square brackets at the end of every sentence that
  states a fact, like this [2]. A sentence without a bracket will be deleted
  before the borrower sees it.
- Never state a rupee amount, rate, or total that does not appear verbatim in a
  passage. Do not add up, convert, or estimate figures.
- Distinguish what the borrower's agreement says from what RBI requires. The
  agreement is what they signed; the circular is what the lender was permitted
  to write.
- Plain language. Short sentences. No legal jargon unless you immediately
  explain it.
- Do not give advice on what the borrower should do. Explain what the documents
  say."""


@dataclass
class GeneratedAnswer:
    answer: VerifiedAnswer
    passages: list[RetrievedPassage]
    prompt_passage_count: int


def _clause_label(citation) -> str:
    """Name a clause only when the number really is one.

    Segmentation reads a structural keyword as a clause number, so a revision
    summary table whose header row begins "Clause | Subject | ..." was cited as
    "clause CLAUSE". A page reference is less precise and not wrong.
    """
    number = citation.clause_number
    if number and number[0].isdigit():
        return f"clause {number}"
    return f"page {citation.page}"


def _label(passage: RetrievedPassage) -> str:
    citation = passage.chunk.citation
    where = _clause_label(citation)
    if passage.chunk.source_kind is SourceKind.RBI_CORPUS:
        return f"RBI {citation.circular_id or citation.document_id}, {where}"
    return f"Your agreement, {where}"


def build_messages(
    question: str,
    passages: list[RetrievedPassage],
    *,
    summary: str = "",
) -> list[LLMMessage]:
    """Assemble the turn: rules, pinned summary, numbered passages, question."""
    blocks = [SYSTEM_RULES]
    if summary:
        blocks.append("The document being discussed:\n" + summary)

    numbered = []
    for index, passage in enumerate(passages, start=1):
        numbered.append(f"[{index}] {_label(passage)}\n{passage.chunk.text.strip()}")
    blocks.append("Passages:\n\n" + "\n\n".join(numbered))

    return [
        LLMMessage(role="system", content="\n\n".join(blocks)),
        LLMMessage(role="user", content=question),
    ]


async def generate(
    question: str,
    passages: list[RetrievedPassage],
    *,
    provider: LLMProvider,
    summary: str = "",
) -> GeneratedAnswer:
    """Produce a verified answer in one shot."""
    messages = build_messages(question, passages, summary=summary)
    response = await provider.complete(messages)
    return GeneratedAnswer(
        answer=verify(response.text, passages),
        passages=passages,
        prompt_passage_count=len(passages),
    )


async def generate_stream(
    question: str,
    passages: list[RetrievedPassage],
    *,
    provider: LLMProvider,
    summary: str = "",
) -> AsyncIterator[tuple[str, str]]:
    """Stream an answer, then verify it.

    Yields ``("delta", text)`` as the model writes, then exactly one
    ``("verified", text)`` carrying the checked answer.

    The draft is streamed before it has been verified, because a borrower
    watching a blank box for eight seconds is a worse experience than watching
    text appear. The final event replaces it: verification can only remove
    sentences, never add them, so the verified answer is always a subset of
    what was shown. The client swaps the text when it arrives, and the
    difference is reported rather than hidden.
    """
    messages = build_messages(question, passages, summary=summary)

    draft: list[str] = []
    async for delta in provider.stream(messages):
        draft.append(delta)
        yield "delta", delta

    yield "verified", verify("".join(draft), passages).text
