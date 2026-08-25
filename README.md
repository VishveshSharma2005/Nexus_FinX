# FinX — Loan Agreement Intelligence

**From 40 pages of fine print to one evidence-backed decision.**

Built for Nexus Hackathon · Problem Statement by EvoDart

---

## The Problem

Millions of Indians sign loan agreements every year without understanding the fine print — hidden charges, penalty clauses, lock-in periods, and terms that may not even comply with current RBI regulations. Existing tools (EMI calculators, aggregator sites, generic chatbots) each answer one narrow question. None of them read the actual document.

## What FinX Does

FinX is a RAG-based assistant that reads a borrower's own loan agreement, sanction letter, MITC, or Key Facts Statement (KFS) and:

- **Explains** clauses in plain language, in English, Hindi, or Gujarati — by text or voice
- **Flags** risky terms: lock-ins, penal charges, rate-reset mechanics, bundled insurance
- **Checks** every clause against a versioned, applicability-filtered RBI corpus — not just text similarity, but whether a rule actually applies to *this* loan type, date, and borrower
- **Quantifies** the rupee impact of a clause using a deterministic calculator, never a model guess
- **Compares** two loan offers on true cost, not headline rate
- **Audits** the Key Facts Statement against the agreement itself, flagging mismatches
- **Declines honestly** when evidence is insufficient — offering a general explanation and a human advisor, instead of guessing

## Why It's Different

| | Existing tools | FinX |
|---|---|---|
| Reads your actual document | ✗ | ✓ |
| Grounded in current RBI rules | ✗ | ✓ |
| Quantifies rupee impact | ✗ | ✓ (deterministic, not model-generated) |
| Compares offers on true cost | ✗ | ✓ |
| Refuses instead of hallucinating | — | ✓ |

## Architecture

```text
Documents → Parse → Chunk → Tag → Index

↓

Question → Retrieve → Filter (applicability) → Enough evidence?

├─ yes → Risk & Cost Engine → Explain → Cited Answer

└─ no  → General info (no RBI/document claims) → Human advisor
````

* **Retrieval layer** — hybrid (semantic + keyword) search over a clause-aware, versioned index
* **Decision layer** — all financial calculations (fees, true cost, EMI, KFS diff) run in deterministic Python; the LLM never computes a number or asserts a regulation applies
* **Parser and model choice** sit behind interfaces (`DocumentParser`, `LLMProvider`), so the hackathon build can be swapped for production-grade components without touching the RAG logic

## Tech Stack

Next.js · FastAPI · PostgreSQL + pgvector · PyMuPDF + OCR · Nemotron 3.5 Lightning (behind a provider interface)

## Status

**Current Phase: Parser + Retrieval + Initial RAG Pipeline**

The core document ingestion and retrieval pipeline is currently under development, with the initial focus on parsing loan documents into structured clauses and building the clause-aware retrieval layer. Cited chat, deterministic risk and cost analysis, KFS auditing, loan comparison, and voice capabilities will be implemented incrementally in the next phases.

## Build Plan

1. **Parser** → structured clauses
2. **Retrieval** → clause-aware index, hybrid search
3. **Cited chat** → answers over one agreement, every claim sourced
4. **Risk & ₹** → rules engine, fee normaliser, true cost
5. **KFS & compare** → document diff, two-offer comparison
6. **Voice & evaluation** → speech in/out, refusal audit

## Team

Vishvesh · Marhama

---

*This project provides source-backed decision support, not regulated financial or legal advice.*

```
```
