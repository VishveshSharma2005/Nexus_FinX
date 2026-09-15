# FinX — Loan Agreement Intelligence

**From 40 pages of fine print to one evidence-backed decision.**

Built for the Nexus Hackathon.

<p align="center">

[🚀 **Live Demo**](https://your-demo-link-here)   
[💻 **GitHub Repository**](https://github.com/VishveshSharma2005/Nexus_FinX)

</p>

## The problem

Millions of people sign loan agreements they never fully read — lock-ins, prepayment
penalties, penal charges, and bundled insurance buried in the fine print, some of it
non-compliant with current RBI rules. Calculators don't open the document. Generic AI
chatbots answer fluently, including when they're wrong. For a loan, a confident wrong
answer is worse than no answer.

## What FinX does

Upload your real loan agreement (PDF or Word) and ask questions in plain English.
FinX answers from your actual document — every sentence cited to a clause number, page,
or RBI circular — and **refuses to answer** when it doesn't have a real source, instead
of guessing.

> **Status:** Phases 0–5 complete and working end to end in a browser. 184 tests
> passing. Runs with no API key and no Docker; add an NVIDIA key for fluent answers.

## Try it — 5 minutes

```bash
git clone https://github.com/VishveshSharma2005/Nexus_FinX.git
cd Nexus_FinX
cp .env.example .env
```

```powershell
# Windows
.\make.ps1 setup   # deps, ~1 min
.\make.ps1 index   # builds the index, ~2 min
.\make.ps1 dev     # terminal 1 — API :8000
.\make.ps1 web     # terminal 2 — UI :3000
```

```bash
# macOS / Linux
make setup && make index
make dev   # terminal 1
make web   # terminal 2
```

Open **http://localhost:3000**:

1. Upload `corpus/samples/home_loan_agreement_A.docx` → 58 clauses parsed.
2. Ask *"Is there a lock-in period, and what would prepaying cost me?"* — cited
   answer streams in, sourced to Clause 11.2, 11.3, and the RBI 2025 Directions.
3. Upload `home_loan_agreement_A_v2.docx`, the revised loan, and ask again — the
   lock-in was deleted in the revision, and the answer correctly reflects that.
4. Ask something with no real answer — *"Can I pay my EMI in cryptocurrency?"* —
   and get an honest fallback: no-source notice, general explanation, human adviser.

Optional, for fluent generation instead of the built-in quoting mode:
```
NVIDIA_API_KEY=nvapi-...
FINX_LLM_PROVIDER=nemotron
```
If the key is missing or NVIDIA is unreachable, FinX detects it and falls back
automatically within 20 seconds — it never hangs or crashes.

## How it works

```
INGEST    Upload → Parse → Clause-split → Fingerprint → Embed → Index
RETRIEVE  Question → Hybrid search → Filter by applicability → Evidence gate
RESPOND   Generate from cited sources only → Verify every sentence → Stream
```

- **Swappable parsers and models.** `DocumentParser` and `LLMProvider` are
  interfaces, not hardcoded choices — parser and model can change without
  touching the rest of the system.
- **Applicability, not just similarity.** Every RBI passage is tagged with the
  loan type, lender class, and effective date it governs. A rule that doesn't
  apply to *this* loan is excluded before it's ever scored — it can't reach an
  answer.
- **Every sentence is verified.** Each generated sentence, and every number in
  it, is checked against its cited source before being shown. What fails is
  removed.

## What the engineering caught

- **Corpus verified by content, not filename.** One RBI file was misnamed — it
  was actually a co-operative-bank circular contributing 244 of 343 chunks to
  the index, and would have surfaced UCB-only rules on an NBFC loan.
- **Effective dates read from the text, not the header.** The 2025 Pre-payment
  Directions apply only to loans sanctioned on/after 1 Jan 2026 — earlier loans
  are correctly flagged as outside its coverage rather than misjudged by it.
- **Amendments resolved as their own type.** A revised agreement that only
  reprints changed clauses is handled explicitly — a clause is only "deleted"
  if the amendment says so, never inferred from absence. Re-uploading a revision
  reuses 85% of existing embeddings.
- **Retrieval measured, not assumed.** Hybrid search beats semantic-only 10/10
  vs 9/10 on a goldset built around the hardest cases, including two questions
  the corpus genuinely can't answer.
- **Citation verifier catches invented numbers** — e.g. an 18-month lock-in when
  the clause says twelve — while correctly keeping faithful paraphrases.

## Tech stack

Python 3.11 / FastAPI · PyMuPDF + Tesseract OCR (PDF) · python-docx (Word) ·
BGE-small embeddings, local via ONNX · SQLite + NumPy vector index · BM25 hybrid
search · NVIDIA Nemotron (optional) · Next.js 15 + React 19 + TypeScript ·
pytest, 184 passing tests

## What's built vs next

**Built:** document parsing (PDF + Word, OCR fallback), verified applicability-
tagged RBI corpus, clause-level versioning, hybrid retrieval with applicability
filtering, evidence-gated cited chat, honest fallback path, live Nemotron
integration with citation verification, working browser demo end to end.

**Future scope:** deterministic risk/cost calculator, Key Facts Statement audit
(automated agreement-vs-KFS mismatch detection), side-by-side loan comparison,
fuller UI design pass, voice + Hindi/Gujarati support, cross-encoder reranker,
larger-scale evaluation.

## Known limitations

- One retrieval ranking edge case: a question's own wording can currently
  outrank a more relevant regulation in source *order* (not correctness) —
  recorded in the goldset, fix is a reranker.
- pgvector is implemented but untested (no Postgres available in dev); the
  local SQLite index is the tested default.
- One superseded RBI circular has no source link recorded, by design, rather
  than a guessed one.

## Data handling

All documents in `corpus/samples/` are synthetic and marked as such. RBI
documents in `corpus/rbi/` are real public filings. No data leaves the machine
unless `NVIDIA_API_KEY` is set.

---

*FinX provides source-backed decision support. It is not regulated financial or
legal advice.*
