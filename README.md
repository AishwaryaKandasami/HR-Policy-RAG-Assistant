# HR Policy Q&A Assistant — Full Stack RAG Bot
===================================================

A production-grade Retrieval-Augmented Generation (RAG) system for HR policy documents. 
Built with a **FastAPI** backend (Python) and a **Next.js 14** frontend (TypeScript).

---

*   **Self-Correcting RAG Loop**: Uses an **LLM-as-a-Judge** (Gemini 2.0 Flash) to verify answer faithfulness and relevance, with automated query rewriting on failure.
*   **Persistent Cloud Vector Storage**: Uses **Qdrant Cloud (Free Tier)** to ensure user-uploaded documents persist across server restarts.
*   **Structured Metadata Parsing**: Converts PDF/DOCX to **Structured Markdown** and uses **Header-Aware Chunking** (via `MarkdownHeaderTextSplitter`) to preserve policy context.
*   **Hybrid Retrieval Engine**: Combines **Qdrant** (Dense Vector) and **BM25** (Sparse Keyword) search.
*   **Reciprocal Rank Fusion (RRF)**: Merges search results for high-precision matching.
*   **Cross-Encoder Reranking**: Locally scores the top-10 chunks with `ms-marco-MiniLM-L-6-v2`.
*   **Multi-LLM Native Router**: Toggle between **Groq (Llama 3.1/3.3)**, **Google (Gemini 2.0)**, and **OpenAI (GPT-4o)**.
*   **HR-Specific Guardrails**: 
    *   **PII Block**: Regex filters for Employee IDs (`EMP12345`), emails, and phones.
    *   **Escalation Triggers**: Sensitive topics (harassment, grievances) are auto-escalated to human HR.
    *   **Injection Protection**: Shields the system prompt from malicious users.
*   **Audit Logging**: Every interaction is recorded in a session-scoped CSV for HR oversight.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14+ (App Router), Tailwind CSS, Lucide icons |
| **Backend** | FastAPI (Python 3.11), Uvicorn |
| **Embeddings** | `all-MiniLM-L6-v2` (Local, 384 dimensions) |
| **Vector DB** | Qdrant Cloud (Free Tier, Persistent) |
| **Retrieval** | `rank-bm25`, `sentence-transformers` (Cross-Encoder) |
| **Hosting** | Hugging Face Spaces (Backend), Vercel (Frontend) |

---

## 📦 Local Setup

### 1. Backend (FastAPI)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Set up .env with QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY, and GOOGLE_API_KEY
uvicorn main:app --reload --port 8000
```

### 2. Frontend (Next.js)
```bash
cd frontend
npm install
npm run dev
# Dashboard available at http://localhost:3000
```

---

## 🚀 Deployment

### Backend (Hugging Face Spaces)
1. Create a new **Docker SDK** Space.
2. Add your `OPENAI_API_KEY` (and others) to the **Variables and Secrets** tab.
3. Push the `backend/` directory content to the Space's Git repo.

### Frontend (Vercel)
1. Push the `frontend/` directory to GitHub.
2. Connect the repo to Vercel.
3. Add `NEXT_PUBLIC_BACKEND_URL` pointing to your HF Space URL.

---

## 🚦 Verification
Run the automated test suite to verify all layers are working:
```bash
cd backend
python test_full_pipeline.py
```

---

## ⚖️ License & Privacy
*   **Demonstration Only**: Ensure no real PII is uploaded in public demo environments.
*   **UK Context**: Default system prompt is tuned for UK employment law standards.
