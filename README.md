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
