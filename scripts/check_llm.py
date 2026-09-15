"""Check the configured language model end to end, in about ten seconds.

Run after putting NVIDIA_API_KEY into .env:

    python scripts/check_llm.py

It asks the real model one grounded question over three real clauses and
reports: which provider resolved, how long the first token took, whether the
model leaked any reasoning, and what the citation verifier kept and removed.
The last part is the one that matters -- a fluent model is more likely than
the quoting fallback to add a claim its sources do not support.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.core.contracts import RetrievedPassage
from app.core.deps import get_llm_provider
from app.rag.generate import build_messages
from app.rag.verify import verify
from tests.index_contract import make_chunk

CLAUSES = [
    "11.2 The Borrower shall not be entitled to prepay the Loan, in whole or in part, during "
    "the Lock-in Period of 12 (twelve) months from the date of first disbursement.",
    "11.3 Any prepayment made after the Lock-in Period, where the source of funds for such "
    "prepayment is other than the Borrower's own verified savings or income, shall attract a "
    "prepayment charge of 3% (three percent) of the principal amount so prepaid.",
    "Reserve Bank of India (Pre-payment Charges on Loans) Directions, 2025, para 5(i): For all "
    "loans granted for purposes other than business to individuals, an RE shall not levy "
    "pre-payment charges on floating rate loans.",
]
QUESTION = "Is there a lock-in period, what would prepaying cost me, and is that allowed?"


async def main() -> int:
    settings = get_settings()
    provider = get_llm_provider(settings)
    print(f"requested  {settings.llm_provider.value}")
    print(f"resolved   {provider.name}  ({provider.model})")
    if provider.name != "nemotron":
        print(
            "\nNemotron is not active. Put NVIDIA_API_KEY in .env and set "
            "FINX_LLM_PROVIDER=nemotron, then run this again."
        )
        return 1

    passages = [
        RetrievedPassage(chunk=make_chunk("check", 1, text, ordinal=i), score=0.8, dense_score=0.8)
        for i, text in enumerate(CLAUSES)
    ]
    messages = build_messages(QUESTION, passages, summary="Home loan, individual borrower")

    started = time.monotonic()
    first_token = None
    parts: list[str] = []
    try:
        async for delta in provider.stream(messages):
            if first_token is None:
                first_token = time.monotonic() - started
            parts.append(delta)
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        print("In the app this degrades to the quoting provider with a visible notice.")
        return 1

    draft = "".join(parts)
    print(f"first token {first_token:.1f}s   total {time.monotonic() - started:.1f}s")
    print(f"reasoning leaked: {'<think>' in draft}")
    print("\n--- model draft ---\n" + draft.strip())

    checked = verify(draft, passages, question=QUESTION)
    print("\n--- verifier ---")
    for sentence in checked.sentences:
        mark = "KEEP " if sentence.kept else "STRIP"
        reason = "" if sentence.kept else f"   <- {sentence.reason}"
        print(f"  {mark} {sentence.text[:88]}{reason}")
    print(f"\nkept {sum(s.kept for s in checked.sentences)} / stripped {len(checked.stripped)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
