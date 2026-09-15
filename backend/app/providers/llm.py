"""Language model providers.

Two implementations of :class:`app.core.contracts.LLMProvider`. The RAG layer
imports neither; it receives one by injection, which is what keeps a vendor SDK
out of retrieval and makes the model an environment variable.

The default needs no API key. That is a deliberate demo property rather than a
placeholder: a live demonstration that fails because a hosted endpoint is slow,
rate-limited or unreachable is a failed demonstration, and the grounding rules
this project is built on are not a property of any particular model.

Neither provider is ever asked for a number. Money is computed in ``app.risk``
and a model only narrates a figure Python has already produced.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from app.core.contracts import LLMMessage, LLMProvider, LLMResponse, LLMUsage

logger = logging.getLogger(__name__)

# Marker the generation prompt asks for, and the verifier checks.
CITATION = re.compile(r"\[(\d+)\]")

# A passage scoring below this share of the best one is padding, not an answer.
RELATIVE_FLOOR = 0.5


class ExtractiveProvider:
    """Composes an answer out of the retrieved passages, deterministically.

    Not a stub. It reads the question and the numbered context the RAG layer
    assembled, selects the passages that actually bear on the question, and
    writes a short answer that quotes them and cites each one. No network, no
    key, no variation between runs.

    What it deliberately cannot do is paraphrase, infer, or join two clauses
    into a conclusion neither states. That makes it a weaker writer than a
    hosted model and a strictly safer one: every sentence it produces is
    anchored to a passage by construction, so the citation verifier has nothing
    to strip. Set NVIDIA_API_KEY and FINX_LLM_PROVIDER=nemotron for fluency.
    """

    name = "extractive"
    model = "finx-extractive-v1"
    supported_languages = frozenset({"en"})

    # Sentences that are pure cross-reference make poor answers.
    _NOISE = re.compile(r"^\s*(?:refer|see|as set out|as specified)\b", re.I)

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        text = "".join([chunk async for chunk in self.stream(messages)])
        return LLMResponse(
            text=text,
            model=self.model,
            usage=LLMUsage(prompt_tokens=0, completion_tokens=0),
            finish_reason="stop",
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        question, passages = _parse_prompt(messages)
        answer = self._compose(question, passages)
        # Emitted in word groups so the interface streams like a model does.
        words = answer.split(" ")
        for start in range(0, len(words), 4):
            yield " ".join(words[start : start + 4]) + (" " if start + 4 < len(words) else "")

    def _compose(self, question: str, passages: list[tuple[int, str, str]]) -> str:
        if not passages:
            return "I could not find anything in your document that answers that."

        question_terms = _terms(question)
        scored: list[tuple[float, int, str, str]] = []
        for number, label, text in passages:
            best, score = _best_sentence(text, question_terms)
            if best and not self._NOISE.match(best):
                scored.append((score, number, label, best))

        scored.sort(key=lambda item: -item[0])
        if not scored or scored[0][0] <= 0:
            return "I could not find anything in your document that answers that."

        # Keep only passages that come close to the best match. Term overlap
        # alone lets a clause about overdraft renewal into an answer about a
        # lock-in period, because both mention prepayment and charges; a
        # relative floor drops those without needing a judgement call.
        floor = scored[0][0] * RELATIVE_FLOOR
        chosen = [item for item in scored if item[0] >= floor][:4]

        lines = ["Based on the clauses retrieved from your documents:", ""]
        for _score, number, label, sentence in chosen:
            lines.append(f"- {label}: {sentence} [{number}]")
        lines.append("")
        lines.append(
            "Each point above is quoted from the source shown in brackets. "
            "Open the citations panel to read the clause in full."
        )
        return "\n".join(lines)


class NemotronProvider:
    """NVIDIA NIM, over the OpenAI-compatible chat completions endpoint."""

    name = "nemotron"
    # Nemotron's instruction models are English-first. Phase 9 routes Hindi and
    # Gujarati elsewhere rather than assuming this one copes.
    supported_languages = frozenset({"en"})

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Leave FINX_LLM_PROVIDER=extractive to run "
                "without a key."
            )
        self.model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def _payload(self, messages, temperature, max_tokens, stream):
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._payload(messages, temperature, max_tokens, stream=False),
            )
            response.raise_for_status()
            body = response.json()

        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return LLMResponse(
            text=choice["message"]["content"],
            model=body.get("model", self.model),
            usage=LLMUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        import json

        import httpx

        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._payload(messages, temperature, max_tokens, stream=True),
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    return
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if content := delta.get("content"):
                    yield content


# --- Helpers shared by the extractive provider ------------------------------

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*%?")
_STOP = frozenset(
    """a an the and or of to in on for by with is are be as at from this that i my me
    what how do does can will shall if it its their there any would could""".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


def _best_sentence(text: str, question_terms: set[str]) -> tuple[str, float]:
    """The sentence of a passage that best answers the question.

    Quoting one sentence rather than a whole clause keeps the answer readable
    while leaving the citation pointing at the clause it came from.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text.replace("\n", " ")) if s.strip()]
    if not sentences:
        return "", 0.0

    best, best_score = sentences[0], 0.0
    for sentence in sentences:
        overlap = len(_terms(sentence) & question_terms)
        # Slight preference for a sentence carrying a figure, and against very
        # short fragments that read as headings.
        bonus = 0.5 if re.search(r"\d", sentence) else 0.0
        length_penalty = 0.5 if len(sentence) < 40 else 0.0
        score = overlap + bonus - length_penalty
        if score > best_score:
            best, best_score = sentence, score

    return best[:400], best_score


def _parse_prompt(messages: list[LLMMessage]) -> tuple[str, list[tuple[int, str, str]]]:
    """Recover the question and the numbered passages from the built prompt.

    The extractive provider is handed the same prompt a hosted model would be,
    so that swapping providers changes nothing upstream.
    """
    question = ""
    passages: list[tuple[int, str, str]] = []

    for message in messages:
        if message.role == "user":
            question = message.content

    context = "\n".join(m.content for m in messages if m.role == "system")
    for block in re.split(r"\n(?=\[\d+\])", context):
        header = re.match(r"\[(\d+)\]\s*(.*)", block.strip())
        if not header:
            continue
        number = int(header.group(1))
        remainder = block.strip()[header.end(1) + 1 :].strip()
        label, _, body = remainder.partition("\n")
        passages.append((number, label.strip(" -—"), body.strip() or label.strip()))

    return question, passages


def _assert_conformance() -> None:
    assert isinstance(ExtractiveProvider(), LLMProvider)


_assert_conformance()
