# FinX — Loan Agreement Intelligence

### Turn 40 pages of loan fine print into one evidence-backed decision.

**FinX** is an AI-powered loan agreement intelligence platform that lets users upload a real **PDF or Word loan agreement** and ask questions in plain English.

Instead of giving fluent but potentially unreliable answers, FinX grounds every response in the **actual loan agreement and applicable RBI regulations** — with clause/page-level citations — and **refuses to guess when evidence is missing**.

<p align="center">

[🚀 **Live Demo**](https://your-demo-link-here)   
[💻 **GitHub Repository**](https://github.com/VishveshSharma2005/Nexus_FinX)

</p>

> **Built for the Nexus Hackathon**
> **184 tests passing · End-to-end browser demo · No API key required**

---

## 🎯 The Problem

Loan agreements can contain dozens of pages of complex terms:

* 🔒 Lock-in periods
* 💰 Prepayment / foreclosure charges
* ⚠️ Penal charges
* 🛡️ Bundled insurance
* 📜 Regulatory conditions

Generic AI chatbots may answer confidently even when the document does not support the answer.

**For financial decisions, a confident wrong answer is worse than no answer.**

---

## 💡 The FinX Approach

FinX follows an **evidence-first architecture**:

```text
Upload Agreement
       ↓
Parse & Extract Clauses
       ↓
Hybrid Retrieval
       ↓
Applicability Filtering
       ↓
Evidence Gate
       ↓
Cited Answer
       ↓
Sentence & Number Verification
```

### What makes it different?

**📄 Document-grounded**
Answers are based on the user's actual agreement.

**⚖️ RBI-aware**
Relevant RBI rules are filtered by loan type, lender class and effective date.

**🔎 Clause-level citations**
Answers point back to specific clauses, pages or RBI sources.

**🛑 Refuses to guess**
If no reliable evidence exists, FinX provides an honest fallback instead of hallucinating.

**✅ Citation verification**
Generated sentences and numerical values are checked against their cited evidence before being displayed.

---

## 🚀 Try It Locally

### 1. Clone

```bash
git clone https://github.com/VishveshSharma2005/Nexus_FinX.git
cd Nexus_FinX
cp .env.example .env
```

### 2. Windows

```powershell
.\make.ps1 setup
.\make.ps1 index
```

Terminal 1:

```powershell
.\make.ps1 dev
```

Terminal 2:

```powershell
.\make.ps1 web
```

### 3. macOS / Linux

```bash
make setup
make index
```

Terminal 1:

```bash
make dev
```

Terminal 2:

```bash
make web
```

Then open:

**http://localhost:3000**

---

## 🧪 Quick Demo

Upload:

```text
corpus/samples/home_loan_agreement_A.docx
```

Ask:

> **"Is there a lock-in period, and what would prepaying cost me?"**

FinX retrieves the relevant clauses and RBI guidance and returns a cited answer.

Then upload:

```text
home_loan_agreement_A_v2.docx
```

The revised agreement removes the lock-in provision, and FinX correctly reflects the updated version.

### Honest fallback

Ask:

> **"Can I pay my EMI in cryptocurrency?"**

If the corpus contains no reliable answer, FinX does **not invent one**.

Instead, it provides:

```text
No reliable source found
        ↓
General explanation
        ↓
Human adviser recommended
```

---

## 🧠 Key Engineering Highlights

### Applicability-Aware Retrieval

RBI rules are not simply retrieved based on semantic similarity.

Each regulatory passage is tagged with:

* Loan type
* Lender class
* Effective date

Rules that do not apply to the current loan are filtered **before retrieval scoring**, reducing the chance of irrelevant regulations reaching the answer.

### Evidence Verification

FinX verifies:

* Generated claims
* Numerical values
* Citation-to-source consistency

For example, if a clause states **12 months** but the model generates **18 months**, the citation verifier catches the mismatch before the answer is shown.

### Version-Aware Agreements

Revised agreements are treated explicitly as amendments.

A clause is not considered deleted merely because it disappears from a revised document — deletion must be supported by the amendment itself.

Re-uploading a revision can reuse approximately **85% of existing embeddings**.

### Corpus Validation

The RBI corpus is validated by **document content rather than filenames**.

This caught a mislabeled regulatory document that would otherwise have introduced inappropriate UCB-specific rules into an NBFC loan workflow.

### Retrieval Evaluation

A goldset was created around difficult cases, including questions the corpus genuinely cannot answer.

```text
Hybrid Search       10/10
Semantic Search      9/10
```

---

## 🏗️ Architecture

```text
                ┌──────────────────┐
                │   Loan Agreement │
                │     PDF / DOCX   │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Parse + OCR      │
                │ Clause Extraction│
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Hybrid Retrieval │
                │ BM25 + Embeddings│
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Applicability    │
                │ Filtering        │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Evidence Gate    │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ LLM Generation   │
                │ + Verification   │
                └────────┬─────────┘
                         ↓
                ┌──────────────────┐
                │ Cited Answer      │
                └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer           | Technology                       |
| --------------- | -------------------------------- |
| Backend         | Python 3.11, FastAPI             |
| PDF Processing  | PyMuPDF, Tesseract OCR           |
| Word Processing | python-docx                      |
| Embeddings      | BGE-small, ONNX                  |
| Retrieval       | BM25 + Vector Search             |
| Database        | SQLite + NumPy                   |
| LLM             | NVIDIA Nemotron *(optional)*     |
| Frontend        | Next.js 15, React 19, TypeScript |
| Testing         | pytest                           |
| Tests           | **184 passing**                  |

---

## 🔐 Privacy & Data Handling

* Sample agreements are **synthetic**.
* RBI documents are public filings.
* Documents remain local by default.
* No external LLM API is required for the built-in quoting mode.
* Data leaves the machine only when an NVIDIA API key is configured.

```text
No NVIDIA API key
       ↓
Local processing
       ↓
No external LLM call
```

Optional Nemotron generation:

```env
NVIDIA_API_KEY=nvapi-...
FINX_LLM_PROVIDER=nemotron
```

If NVIDIA is unavailable, FinX automatically falls back instead of hanging or crashing.

---

## 📊 Current Status

### ✅ Built

* PDF + Word document parsing
* OCR fallback
* Clause-level extraction
* RBI corpus with applicability metadata
* Agreement versioning
* Hybrid retrieval
* Applicability filtering
* Evidence-gated answers
* Citation verification
* Honest no-source fallback
* Nemotron integration
* Browser-based end-to-end demo
* **184 passing tests**

### 🔮 Future Scope

* Deterministic risk & cost calculator
* Key Facts Statement (KFS) mismatch detection
* Side-by-side loan comparison
* Cross-encoder reranking
* Larger-scale evaluation
* Hindi / Gujarati support
* Voice interface
* Production-scale pgvector deployment

---

## ⚠️ Known Limitations

* One retrieval-ranking edge case remains where question wording can appear before a more relevant regulation in source ordering.
* pgvector implementation exists but the tested development environment currently uses SQLite.
* One superseded RBI circular intentionally has no source link rather than using an unverified URL.

---

## 👥 Why FinX?

FinX is designed around one principle:

> **Don't just answer the question. Prove the answer.**

For financial documents, **traceability, applicability and refusal to hallucinate** matter as much as language quality.

---

### ⚖️ Disclaimer

FinX provides **source-backed decision support** and is not regulated financial or legal advice.
