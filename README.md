# FinX — Loan Agreement Intelligence

Millions of people sign loan agreements they cannot read. FinX reads a borrower's
own loan documents, explains them in plain language, checks every clause against a
versioned RBI corpus, quantifies the rupee impact, and compares offers honestly.

FinX does five things, and nothing else: **explain, flag, quantify, compare, audit.**
It is deliberately not a general "chat with a PDF".

> **Status:** Phases 0-2 of 11 complete — interfaces, verified RBI corpus, two parsers, and a versioned index.
> **No user interface yet.** Everything below runs from the command line; the browser UI is Phase 8.
> See [Build status](#build-status).

---

## The one design decision everything else follows from

A wrong answer about someone's loan is worse than no answer. So FinX is built so
that a confident answer is *structurally* impossible without a source behind it.

```
LANE 1 — INGEST (once per document)
  Documents → Parse → Chunk → Tag → Index

LANE 2 — RETRIEVE (per question)
  Question → Retrieve → Filter (applicability) → gate: "enough evidence?"
    ├─ yes → Lane 3
    └─ no  → FALLBACK

LANE 3 — RESPOND
  Risk & Cost Engine → Explain → Cited Answer (page, clause, RBI circular)

FALLBACK (retrieval found no supporting passage)
  "No source found" notice
    → general-purpose explanation only (term definitions; no claims about RBI
      rules or about the user's document)
    → offer of a human advisor
```

Five rules make that real:

| Rule | Mechanism |
|---|---|
| **Parsers are swappable** | `DocumentParser` is a Protocol in [contracts.py](backend/app/core/contracts.py). The hackathon parser is one implementation; nothing outside `app/parsers/` may import it. A test fails the build if anything does. |
| **Models are swappable** | `LLMProvider` is a Protocol. `app/rag/` never imports a vendor SDK — a test asserts this. Model choice is one environment variable. |
| **The model never computes money** | Every rupee figure — fees, true cost over tenure, EMI scenarios, KFS diffs — is deterministic, unit-tested Python in `app/risk/`. The model only explains a number Python already produced. |
| **Retrieval filters on applicability, not just similarity** | Every RBI chunk carries loan type, borrower type, lender class and effective date. Candidates are filtered against the uploaded document's own metadata *before* reranking, so a passage that doesn't govern this loan never reaches the model's context. |
| **Every sentence binds to a chunk** | Answers are verified claim-by-claim against their bound source; anything untraceable is stripped before it reaches the user. |

Re-uploading a revised agreement is cheap by construction: clauses are
content-hashed, unchanged clauses reuse their existing embedding and gain a
version pointer, and "what changed between v1 and v2" is a set operation over
hashes costing zero tokens. Retrieval is scoped to exactly one version, so a v1
clause can never leak into a v2 answer.

---

## Run it

### Option A — Docker (full stack, Postgres + pgvector)

```bash
cp .env.example .env
docker compose up --build
```

API on `http://localhost:8000`, web on `http://localhost:3000`.

### Option B — No Docker required

The vector index sits behind a `VectorIndex` interface with two implementations:
pgvector, and a local SQLite + numpy index. `FINX_INDEX=local` selects the latter,
so the whole system runs with no external services.

```bash
cp .env.example .env

# Windows
.\make.ps1 setup
.\make.ps1 dev

# macOS / Linux
make setup
make dev
```

Then:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/readyz   # shows which backends this process resolved to
```

### Commands

| `make` | `.\make.ps1` | Does |
|---|---|---|
| `setup` | `setup` | Create `.venv`, install backend dependencies |
| `dev` | `dev` | Run the API with reload |
| `web` | `web` | Run the Next.js frontend |
| `test` | `test` | Run the backend test suite |
| `lint` | `lint` | Ruff check + format check |
| `samples` | `samples` | Generate the synthetic demo documents |
| `parse FILE=<f>` | `parse <f>` | Parse one document to `ParsedDocument` JSON |
| `index` | `index` | Build the vector index from corpus + samples |
| `search Q="…"` | `search "…"` | Query the index and print scored, cited hits |
| `corpus` | `corpus` | Fetch the public RBI corpus |
| `seed` | `seed` | Build a demo-ready database (synthetic data only) |

---

## Configuration

Everything resolves in [config.py](backend/app/config.py) and nowhere else, so
changing a model or a database is an environment change, not a code change.

| Variable | Purpose |
|---|---|
| `FINX_INDEX` | `pgvector` or `local` |
| `FINX_LLM_PROVIDER` | `nemotron` or `fake` (deterministic, offline, used by tests) |
| `FINX_LLM_MODEL` | The single knob for model choice |
| `FINX_EMBEDDING_PROVIDER` | Embeddings are separable from generation, so retrieval can stay multilingual while generation routes by language |
| `FINX_EVIDENCE_MIN_SCORE`, `FINX_EVIDENCE_MIN_PASSAGES` | The evidence gate's thresholds |
| `NVIDIA_API_KEY` | Only needed for the hosted provider |

A `fake` provider ships for both LLM and embeddings, so the full pipeline, the
test suite and the evaluation harness run with no API key and no network.

---

## Data handling

No real user financial data, ever. Demo documents are synthetic. The synthetic-data
guarantee is enforced in `scripts/seed_demo.py` in code, not by convention.

---

## Build status

Vertical slices, tested at each gate.

- [x] **Phase 0** — Scaffold, interfaces, docker-compose, booting API, RBI manifest
- [x] **Phase 1** — Parsers (PDF + Word) → `ParsedDocument`, OCR fallback, contract test
- [x] **Phase 2** — Clause chunking, content-hash versioning, amendment resolution, vector index
- [ ] **Phase 3** — Hybrid retrieval + applicability filter
- [ ] **Phase 4** — Cited chat (minimum viable demo)
- [ ] **Phase 5** — Evidence gate + fallback path
- [ ] **Phase 6** — Risk & cost engine
- [ ] **Phase 7** — KFS-vs-agreement audit + offer comparison
- [ ] **Phase 8** — Frontend
- [ ] **Phase 9** — Voice + Hindi/Gujarati
- [ ] **Phase 10** — Evaluation: retrieval, citation and refusal accuracy
- [ ] **Phase 11** — Demo hardening

---

## Repository layout

```
backend/app/core/contracts.py   Interfaces + shared schemas — the architectural spine
backend/app/parsers/            Parser implementations (imported nowhere else)
backend/app/providers/          Model providers (imported nowhere else)
backend/app/rag/                chunk, embed, index, retrieve, gate, generate, fallback, verify
backend/app/risk/               All money maths — deterministic, unit tested
backend/app/routers/            HTTP surface
backend/tests/eval/             Goldset + accuracy harness
corpus/rbi/                     Public RBI documents + applicability manifest
corpus/samples/                 Synthetic demo agreements
```

---

## Development notes

- The backend targets Python 3.11.
- On a machine behind a TLS-inspecting proxy, `pip` may not trust the intercepting
  root CA. `.venv/pip.ini` sets `use-feature = truststore`, which verifies against
  the OS certificate store instead of pip's bundled one. Verification stays on.
- OCR (scanned pages) needs Tesseract. Without it the parser degrades explicitly
  and emits a warning rather than silently returning empty text.
