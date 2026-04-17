# HR Policy Q&A Assistant — Full Stack RAG Bot

A production-grade Retrieval-Augmented Generation (RAG) system for UK HR policy documents.  
Built with a **FastAPI** backend (Python) and a **Next.js 14** frontend (TypeScript).  
Deployed on **Hugging Face Spaces** (backend) + **Vercel** (frontend). **Total infrastructure cost: £0.**

---

## Features

- **Intent Router** — classifies greetings, thanks, meta-questions and help requests before any RAG work starts, so small-talk gets a consistent templated response instead of random HR-corpus noise.
- **Multi-turn Session Memory** — the frontend sends the last 6 conversation turns with every request; the LLM resolves pronouns and follow-up questions without server-side state.
- **SSE Streaming Responses** — answers stream token-by-token via Server-Sent Events, dropping time-to-first-token from ~5 s to ~300 ms.
- **Self-Correcting RAG Loop** — uses an **LLM-as-a-Judge** to verify answer faithfulness; automatically rewrites the search query and retries once on failure. Judge is skipped when retrieval confidence is already high (> 0.75) to save latency.
- **Hybrid Retrieval Engine** — combines Qdrant dense vector search with in-memory BM25 keyword search, fused via **Reciprocal Rank Fusion (RRF)**.
- **Cross-Encoder Reranking** — locally scores the top-10 fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the best top-3 chunks.
- **Document Versioning** — re-uploading an existing policy file automatically replaces the old version (old chunks deleted, new chunks ingested), preventing conflicting answers.
- **Deduplicated Citations** — overlapping chunks from the same page are collapsed to a single citation card in the UI.
- **Multi-Tenancy Groundwork** — every chunk, retrieval call, and document operation is scoped by `tenant_id` (default `"public_uk"`), ready for client-isolated document collections.
- **Per-IP Rate Limiting** — 30 queries/min and 10 uploads/min per IP via `slowapi`, with correct `X-Forwarded-For` handling behind HF Spaces / Cloudflare proxies.
- **Persistent Cloud Vector Storage** — Qdrant Cloud Free Tier persists documents across HF Spaces container restarts; BM25 index is pickle-cached for sub-second warm starts.
- **HR-Specific Guardrails** — PII blocking (employee IDs, emails, phones), prompt injection detection, sensitive-topic escalation to HR Business Partners, and out-of-scope filtering.
- **Structured Metadata Parsing** — PDF/DOCX converted to structured Markdown; `MarkdownHeaderTextSplitter` preserves section context in every chunk.
- **Multi-LLM Router** — toggle between Groq (Llama 3.1/3.3), Google Gemini 2.0 Flash, and OpenAI GPT-4o with no code change.
- **Confidence Scoring** — reranker score passed through sigmoid → shown as High / Medium / Low pill in the UI, with a "Verify with HR" nudge on Low.
- **Thumbs Up/Down Feedback** — captured per response with optional reason; stored in the audit log for continuous improvement.
- **Audit Logging** — every interaction (query, answer, sources, session, latency, feedback) recorded in a downloadable CSV.
- **CORS Restriction** — allowed origins set via `ALLOWED_ORIGINS` environment variable; falls back to `*` only in local dev.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14+ (App Router), Tailwind CSS, Lucide icons |
| **Backend** | FastAPI (Python 3.11+), Uvicorn |
| **Embeddings** | `all-MiniLM-L6-v2` local, 384-d (no API cost) |
| **Vector DB** | Qdrant Cloud Free Tier (persistent, cosine) |
| **Sparse Search** | `rank-bm25` BM25Okapi, in-memory + pickle cache |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **LLM Providers** | Groq (Llama 3.1 8B / 3.3 70B), Gemini 2.0 Flash, OpenAI |
| **Rate Limiting** | `slowapi` per IP |
| **Streaming** | FastAPI `StreamingResponse` (SSE) |
| **Hosting** | Hugging Face Spaces Docker (backend), Vercel (frontend) |

---

## Local Setup

### 1. Backend (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
OPENAI_API_KEY=your_openai_key          # optional
ALLOWED_ORIGINS=http://localhost:3000   # comma-separated; omit for * in dev
```

```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend (Next.js)

```bash
cd frontend
npm install
# create frontend/.env.local
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local
npm run dev
# App at http://localhost:3000
```

---

## Deployment

### Backend — Hugging Face Spaces

1. Create a **Docker SDK** Space.
2. Push the `backend/` directory to the Space repository.
3. Add secrets in **Variables and Secrets**:
   - `QDRANT_URL`, `QDRANT_API_KEY`
   - `GROQ_API_KEY`, `GOOGLE_API_KEY`
   - `ALLOWED_ORIGINS` → your Vercel domain(s), comma-separated, **no trailing slash**  
     e.g. `https://hr-policy-rag-assistant.vercel.app`

### Frontend — Vercel

1. Connect the `frontend/` directory (or monorepo root) to a Vercel project.
2. Set environment variable:
   - `NEXT_PUBLIC_BACKEND_URL` → your HF Space URL (e.g. `https://username-hr-bot.hf.space`)

---

## Verification

Run the automated test suite to verify all pipeline layers:

```bash
cd backend
python -m pytest test_full_pipeline.py -v
```

All 7 tests should pass:

| Test | Covers |
|---|---|
| `test_01_guardrails_pass` | Clean HR query passes filters |
| `test_02_guardrails_pii` | PII in query is blocked |
| `test_03_guardrails_escalation` | Sensitive query is escalated |
| `test_04_guardrails_injection` | Prompt injection is blocked |
| `test_05_audit_log` | Session + session_id logged to CSV |
| `test_06_hybrid_retrieval` | Dense + BM25 + RRF + rerank returns dict |
| `test_07_generator_router` | LLM router resolves correct provider |

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Full RAG pipeline with judge loop |
| `/query/stream` | POST | Streaming SSE variant (no judge — faster TTFT) |
| `/ingest` | POST | Upload one or more HR policy documents |
| `/docs-list` | GET | List ingested documents for a tenant |
| `/docs` | DELETE | Remove a document (and all its chunks) |
| `/feedback` | POST | Submit thumbs up/down for a query |
| `/logs` | GET | Download audit log CSV |
| `/health` | GET | Health check + doc count |
| `/onboarding-checklist` | GET | Onboarding checklist JSON |

### `/query` and `/query/stream` request body

```json
{
  "query":                "What is the maternity leave entitlement?",
  "llm_provider":         "groq_llama_70b",
  "provider_api_key":     null,
  "session_id":           "uuid-generated-client-side",
  "conversation_history": [
    { "role": "user",      "content": "How many days holiday do I get?" },
    { "role": "assistant", "content": "Full-time employees are entitled to 28 days..." }
  ],
  "tenant_id":            "public_uk"
}
```

---

## License & Privacy

- **Demonstration Only** — do not upload documents containing real employee personal data in public demo environments.
- **UK Context** — the default system prompt is tuned for UK employment law and ACAS standards.
