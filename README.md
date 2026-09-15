# FinX — Loan Agreement Intelligence

### From 40 pages of fine print to one evidence-backed decision.

FinX lets users upload a **loan agreement (PDF/DOCX)** and ask questions in plain English. It answers using the **actual agreement + applicable RBI regulations**, with citations — and **refuses to guess when evidence is missing**.

<p align="center">

🚀 **[Live Demo](YOUR_DEMO_LINK)** · 💻 **[GitHub](https://github.com/VishveshSharma2005/Nexus_FinX)**

</p>

## ✨ Key Features

* 📄 **PDF & DOCX** loan agreement analysis
* 🔎 **Hybrid search** — BM25 + embeddings
* ⚖️ **RBI-aware applicability filtering**
* 📌 **Clause/page-level citations**
* 🛡️ **Citation & number verification**
* 🛑 **No-source fallback instead of hallucination**
* 🔄 **Agreement version comparison**
* 🤖 Optional **NVIDIA Nemotron** generation

## 🧠 How It Works

```text
Upload → Parse → Clause Extraction → Hybrid Retrieval
       → RBI Applicability Filter → Evidence Gate
       → Verified Cited Answer
```

## 🛠️ Tech Stack

**FastAPI · Python · PyMuPDF · Tesseract · BGE-small · BM25 · SQLite/NumPy · NVIDIA Nemotron · Next.js · React · TypeScript**

## 🚀 Quick Start

```bash
git clone https://github.com/VishveshSharma2005/Nexus_FinX.git
cd Nexus_FinX
```

**Windows**

```powershell
.\make.ps1 setup
.\make.ps1 index
.\make.ps1 dev
```

**Web UI**

```powershell
.\make.ps1 web
```

Open **http://localhost:3000**

> **184 tests passing · End-to-end browser demo · No API key required**

## 🔮 Future Scope

Risk & cost calculator · KFS audit · Loan comparison · Reranking · Hindi/Gujarati · Voice

---

**Built for the Nexus Hackathon**

*FinX provides source-backed decision support, not financial or legal advice.*
