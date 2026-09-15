# FinX — Loan Agreement Intelligence

Millions of people sign loan agreements they cannot read. FinX reads a borrower's
own loan documents, explains them in plain language, checks every clause against a
versioned RBI corpus, and shows the source behind every sentence it produces.

FinX does five things and nothing else: **explain, flag, quantify, compare, audit.**
It is deliberately not a general "chat with a PDF".

> **Status:** Phases 0–4 of 11 complete. Cited chat works end to end in a browser.
> 142 tests passing. Runs with **no API key and no Docker**.

---

## Try it in five minutes

Needs Python 3.11+ and Node 20+. Nothing else — no API key, no database server.

```bash
git clone <this-repo>
cd FinX_model
cp .env.example .env
```

```powershell
# Windows
.\make.ps1 setup     # virtualenv + dependencies   (~1 min)
.\make.ps1 index     # RBI corpus into the index   (~2 min, downloads a 130 MB embedding model once)
.\make.ps1 dev       # terminal 1 — API on :8000
.\make.ps1 web       # terminal 2 — web on :3000
```

```bash
# macOS / Linux
make setup && make index
make dev    # terminal 1
make web    # terminal 2
```

Then open **<http://localhost:3000>** and:

1. **Upload** `corpus/samples/home_loan_agreement_A.docx` → *58 clauses parsed*.
2. **Ask** "Is there a lock-in period before I can prepay my loan, and what would it
   cost?" Sources appear first, then the answer streams in, citing clause 11.2
   (12-month lock-in), 11.3 (3% charge) and the RBI 2025 Directions that prohibit
   both for a loan like this one.
3. **Upload** `home_loan_agreement_A_v2.docx` → *60 clauses, 85% of embeddings
   reused*. Ask the same question. The answer changes to "…at any time and without
   any lock-in period", and the deleted clause cannot appear.

---

## The one design decision everything else follows from

A wrong answer about someone's loan is worse than no answer. FinX is built so a
confident answer is *structurally* impossible without a source behind it.

```
LANE 1 — INGEST (once per document)
  Documents → Parse → Chunk → Tag → Index

LANE 2 — RETRIEVE (per question)
  Question → Retrieve → Filter (applicability) → gate: "enough evidence?"
    ├─ yes → Lane 3
    └─ no  → FALLBACK

LANE 3 — RESPOND
  Risk & Cost Engine → Explain → Cited Answer (page, clause, RBI circular)
```

| Rule | How it is enforced |
|---|---|
| **Parsers are swappable** | `DocumentParser` is a Protocol with two implementations (PDF, Word). An AST-walking test fails the build if anything outside `app/parsers/` imports a concrete one. |
| **Models are swappable** | `LLMProvider` is a Protocol; `app/rag/` never imports a vendor SDK, asserted by test. Model choice is one environment variable. |
| **The model never computes money** | All rupee maths is deterministic Python in `app/risk/` (Phase 6). The model only narrates figures Python produced. |
| **Applicability, not just similarity** | Every RBI chunk carries loan type, borrower type, lender class and effective date. Non-governing passages are excluded *inside the index query*, so they are never scored, never reranked, and cannot reach the model. |
| **Every sentence binds to a chunk** | `app/rag/verify.py` checks each sentence against the passage it cites and deletes what it cannot trace. |

---

## What is worth looking at

**Corpus integrity.** Every RBI circular is catalogued from its own text, never its
filename. That caught a file named `penal_charges_extension_2023.pdf` that was in
fact the UCB Master Circular on Management of Advances — co-operative-bank material
governing no NBFC loan, contributing 244 of 343 corpus chunks. Every `source_url`
was fetched and checked against the circular it claims to point at; one was wrong.
See [`corpus/rbi/manifest.json`](corpus/rbi/manifest.json).

**Dates are read, not assumed.** The penal-charges circular says it takes effect
1 January 2024; a later circular moved that to 1 April 2024. The 2025 pre-payment
Directions repeal eight earlier circulars, but only from 1 January 2026, and the
repealed ones stay in force for loans before that. Taking a printed date at face
value would have FinX judge a loan against rules that were not yet in force when it
was signed.

**Amendments are a distinct document kind.** The sample v2 reprints 15 clauses and
carries about 45 forward. Reading it as a restatement would delete the borrower's
security, insurance and default clauses; reading a restatement as an amendment
would leave a removed lock-in apparently in force. Deletion is only ever read from
the amendment's own words, never inferred from absence.
See [`versioning.py`](backend/app/rag/versioning.py).

**Re-uploading a revision is cheap.** Clauses are content-hashed, embeddings are
keyed by hash, and chunks point at them — so v2 embeds 9 clauses and reuses 51.
"What changed between versions" is a set operation costing zero tokens.

**Retrieval is measured, not asserted.** Ten goldset cases chosen for what they
stress, including the same question asked against two versions with different
correct answers, and two questions the corpus genuinely cannot answer.

```
$ python backend/tests/eval/run_eval.py --compare
dense-only  9/10   ->   hybrid  10/10
clause recall 100% | circular found 8/8 | refusals 2/2
```

---

## Commands

| `make` | `.\make.ps1` | Does |
|---|---|---|
| `setup` | `setup` | Create `.venv`, install backend dependencies |
| `index` | `index` | Build the vector index from corpus + samples |
| `dev` | `dev` | Run the API on :8000 |
| `web` | `web` | Run the web UI on :3000 |
| `test` | `test` | Run the backend test suite |
| `lint` | `lint` | Ruff check + format check |
| `parse FILE=<f>` | `parse <f>` | Parse one document to a clause table |
| `search Q="…"` | `search "…"` | Query the index, printing scored and cited hits |

---

## Configuration

Everything resolves in [`config.py`](backend/app/config.py) and nowhere else, so
changing a model or a database is an environment change, not a code change.
`GET /readyz` reports what a running process **actually** resolved to.

| Variable | Purpose |
|---|---|
| `FINX_INDEX` | `local` (SQLite + numpy, default) or `pgvector` |
| `FINX_LLM_PROVIDER` | `extractive` (no key, default) or `nemotron` |
| `FINX_EMBEDDING_PROVIDER` | `local` (BGE-small on this machine, default), `fake`, or `nvidia` |
| `NVIDIA_API_KEY` | Only needed for the hosted provider |

The default embedder is a real sentence-embedding model run locally through ONNX:
semantic, so *"can I close my loan early?"* reaches a clause about prepayment that
never uses the word. A deterministic lexical fallback ships alongside it so the
test suite runs on a machine that cannot download weights.

The default LLM provider needs no key and is not a stub: it quotes the retrieved
passages and cites them. It cannot paraphrase or infer, which makes it a weaker
writer and a strictly safer one. Set `NVIDIA_API_KEY` for fluency.

---

## Build status

- [x] **Phase 0** — Scaffold, interfaces, docker-compose, RBI manifest
- [x] **Phase 1** — Parsers (PDF + Word), OCR fallback, contract test
- [x] **Phase 2** — Chunking, content-hash versioning, amendment resolution, vector index
- [x] **Phase 3** — Hybrid retrieval + applicability filter, 10/10 on the goldset
- [x] **Phase 4** — Cited chat, streamed, clickable in a browser
- [ ] **Phase 5** — Evidence gate + fallback path
- [ ] **Phase 6** — Risk & cost engine
- [ ] **Phase 7** — KFS-vs-agreement audit + offer comparison
- [ ] **Phase 8** — Full frontend
- [ ] **Phase 9** — Voice + Hindi/Gujarati
- [ ] **Phase 10** — Evaluation at scale
- [ ] **Phase 11** — Demo hardening

---

## Repository layout

```
backend/app/core/contracts.py   Interfaces + shared schemas — the architectural spine
backend/app/parsers/            PDF and Word parsers (imported nowhere else)
backend/app/providers/          Embedding and LLM providers (imported nowhere else)
backend/app/rag/                chunk, embed, index, retrieve, generate, verify, versioning
backend/app/risk/               Money maths — deterministic, unit tested (Phase 6)
backend/app/routers/            HTTP surface: upload, chat (SSE)
backend/tests/eval/             Goldset + accuracy harness
frontend/                       Next.js UI: upload + cited chat
corpus/rbi/                     Public RBI circulars + applicability manifest
corpus/samples/                 Demo loan documents (synthetic)
```

---

## Data handling

No real borrower data. Every document in `corpus/samples/` is synthetic and says so
on its face. Nothing is sent to a hosted model unless `NVIDIA_API_KEY` is set; the
default configuration makes no outbound calls at query time.

---

## Known limitations

Stated rather than hidden, because a demo that hides them is a worse demo.

- **Retrieval ranking on one question.** Asked whether prepayment charges are
  *allowed*, retrieval ranks the floating-rate circular above the 2025 Directions,
  because the question says "floating rate" and that circular is named after it.
  The Directions still reach the model's context. A cross-encoder reranker is the
  honest fix; the case is recorded in the goldset rather than tuned away.
- **pgvector is unverified.** The implementation matches the same interface, but
  its tests skip unless a Postgres is reachable and none has been.
  `FINX_INDEX=local` is the tested path.
- **One RBI source URL is unresolved.** The 2012 NBFC Fair Practices Code is
  superseded and no longer in RBI's index. No URL is recorded rather than one that
  is close but wrong.
- **Word pagination is derived.** A `.docx` has no fixed pages, so page numbers come
  from explicit page breaks and may differ from what Word displays. Clause numbers
  are the reliable citation, and the parser says so.

---

## Development notes

- Backend targets Python 3.11.
- The embedding model downloads once on the first `index` run and is cached in
  `data/models/`. Behind a TLS-inspecting proxy it verifies against the OS trust
  store rather than a bundled CA list.
- OCR for scanned pages needs Tesseract. Without it the parser emits a warning
  rather than silently returning empty text.
