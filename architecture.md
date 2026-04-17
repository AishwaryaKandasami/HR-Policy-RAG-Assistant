# HR Policy Q&A Bot — Technical Architecture

**Product:** HR Policy Q&A Bot  
**Target users:** Employees (ask questions), HR Managers (upload docs), Prospects (demo)  
**Document sources:** User-uploaded HR policy PDFs/DOCX — demo uses ACAS/CIPD UK documents  
**Retrieval method:** Hybrid RAG — dense vector search (Qdrant) + sparse keyword search (BM25)  
**Last updated:** 2026-04-17

---

## 1. CONFIRMED STACK

| Layer | Technology | Cost |
|---|---|---|
| **Frontend** | Next.js 14 App Router → Vercel | Free |
| **Backend** | FastAPI + Uvicorn → Hugging Face Spaces (Docker) | Free |
| **Vector DB** | Qdrant Cloud Free Tier — persistent, cosine | Free |
| **Sparse Search** | `rank-bm25` BM25Okapi — in-memory + pickle cache | Free |
| **Embeddings** | `all-MiniLM-L6-v2` (384-d) — local CPU | Free |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` — local | Free |
| **LLM Router** | Groq / Gemini Flash / OpenAI — user supplies key | Free (client key) |
| **Rate Limiting** | `slowapi` — per-IP, X-Forwarded-For aware | Free |
| **Streaming** | FastAPI `StreamingResponse` SSE | Free |
| **Demo docs** | ACAS/CIPD PDFs — pre-loaded at startup | Free |
| **Total** | | **£0** |

---

## 2. SYSTEM DIAGRAM

### Online Serving Path

```
Employee / HR Manager / Prospect — browser
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│              LAYER 1 — FRONTEND (Vercel)                   │
│                                                            │
│  ┌────────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  Chat UI       │  │ File Upload │  │ LLM Selector    │ │
│  │  (SSE stream)  │  │  sidebar    │  │ + API key       │ │
│  │  + history[]   │  │             │  │                 │ │
│  └───────┬────────┘  └──────┬──────┘  └────────┬────────┘ │
└──────────┼─────────────────┼──────────────────┼───────────┘
           │  POST /query/stream            POST /ingest
           │  {query, history, session_id,
           │   tenant_id, provider_api_key}
           ▼
┌────────────────────────────────────────────────────────────┐
│      LAYER 2 — BACKEND (Hugging Face Spaces, Docker)       │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Rate Limiter (slowapi)                              │  │
│  │  30 req/min /query  ·  10 req/min /ingest            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                  │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Intent Router  (intent_router.py)                   │  │
│  │  greeting / thanks / farewell / meta / help          │  │
│  │            → short-circuit (canned response)         │  │
│  │  hr_question → continue ↓                            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ hr_question only                 │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  HR Guardrails  (hr_guardrails.py)                   │  │
│  │  INJECTION → block                                   │  │
│  │  PII       → block                                   │  │
│  │  SENSITIVE → escalate (no LLM call)                  │  │
│  │  PASS      → continue ↓                              │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │ PASS only                        │
│            ┌────────────▼──────────────┐                   │
│            │  Embed query              │                   │
│            │  all-MiniLM-L6-v2 (local) │                   │
│            └────────────┬──────────────┘                   │
│                         │ 384-d vector                     │
│            ┌────────────▼──────────────┐                   │
│            │  Hybrid Retrieve          │                   │
│            │  (scoped to tenant_id)    │                   │
│            │  Dense: Qdrant Cloud ANN  │                   │
│            │  Sparse: BM25Okapi        │                   │
│            │  Fuse: RRF (k=60)         │                   │
│            │  Rerank: Cross-Encoder    │                   │
│            │  → top-3 chunks           │                   │
│            └────────────┬──────────────┘                   │
│                         │                                  │
│            confidence > 0.75?                              │
│            ┌────────────┴──────────────┐                   │
│            │ YES: skip judge           │ NO: run judge     │
│            └────────────┬──────────────┘                   │
│                         │                                  │
│            ┌────────────▼──────────────┐                   │
│            │  LLM Generator            │                   │
│            │  + conversation_history   │                   │
│            │  (Groq / Gemini / OpenAI) │                   │
│            └────────────┬──────────────┘                   │
│                         │ SSE token stream                 │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │  Audit Log  (audit_log.py)                           │  │
│  │  session_id · query_id · answer · sources · latency  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
        │ SSE stream (tokens + meta frame)
        ▼
 Frontend renders answer progressively
 Confidence pill · Citations (deduped) · Thumbs up/down
```

### Offline Ingestion Pipeline

```
User uploads file (PDF / DOCX / TXT / MD)   OR   Demo docs pre-loaded at startup
        │
        │  Check: does filename already exist for this tenant_id?
        │  YES → delete old chunks from Qdrant + rebuild BM25  (doc versioning)
        │  NO  → proceed directly
        │
        ├── .pdf  → pdfplumber   → Unified Markdown
        ├── .docx → python-docx  → Unified Markdown
        └── .txt / .md → direct read
        │
        ▼
MarkdownHeaderTextSplitter (#, ##, ###)
  → ensures section context is preserved per chunk
        │
        ▼
RecursiveCharacterTextSplitter (800 chars / 100 overlap)
  → sub-splits oversized sections
        │
        ▼
Metadata per chunk:
  { doc_title, doc_type, department, section_heading, chunk_index,
    source_filename, tenant_id, ingested_at, page_number }
        │
        ├──► all-MiniLM-L6-v2 → embeddings (384-d, local)
        │         └──► Qdrant Cloud upsert (dense + full payload)
        │
        └──► BM25 corpus append → rebuild BM25Okapi index
                  └──► save pickle cache (bm25_cache.pkl)
```

---

## 3. LAYER BREAKDOWN

### Frontend (Next.js — Vercel)

| Area | Behaviour |
|---|---|
| Disclaimer banner | Orange bar: *"Proof of concept — do not upload documents containing real employee personal data."* Dismissible. |
| Left sidebar | File upload (drag-drop → `POST /ingest`), uploaded doc list, LLM selector (5 providers), API key input |
| Main chat area | Full-height SSE-streamed message thread. User messages right-aligned, bot messages left-aligned with streaming cursor. |
| Confidence pill | High (green) / Medium (amber) / Low (red) pill below each bot answer. Low shows "Verify with HR." |
| Citations | Collapsible "Citations (N)" section below each answer; deduplicated by filename + page + section. |
| Feedback buttons | Thumbs up / thumbs down per response; thumbs-down opens reason picker (Wrong answer / Incomplete / Not what I meant). |
| Session memory | `sessionId` generated once per tab via `crypto.randomUUID()`, stored in `sessionStorage`. Last 6 turns sent with every query. |
| Input bar | Text input + send button. Disabled while streaming. |
| Query log button | Fixed bottom-right → `GET /logs` downloads audit CSV. |

### Backend Endpoints (FastAPI — Hugging Face Spaces Docker)

| Endpoint | Method | Key inputs | Output |
|---|---|---|---|
| `/ingest` | POST | `files[]`, `tenant_id` (form) | `{ status, ingested[], total_chunks_added, docs_replaced }` |
| `/query` | POST | `QueryRequest` (see schema) | `QueryResponse` — full answer + metadata |
| `/query/stream` | POST | `QueryRequest` | SSE stream: `token` frames → `meta` frame → `[DONE]` |
| `/docs-list` | GET | `tenant_id` query param | `{ docs[], total, tenant_id }` |
| `/docs` | DELETE | `filename`, `tenant_id` query params | `{ status, removed_filename, chunks_removed }` |
| `/feedback` | POST | `{ query_id, rating, reason? }` | `{ status }` |
| `/logs` | GET | — | CSV file download |
| `/health` | GET | — | `{ status, docs_loaded }` |
| `/onboarding-checklist` | GET | — | JSON checklist |

### Hybrid Retrieval

| Step | Component | Detail |
|---|---|---|
| 1 | Dense search | Qdrant ANN cosine similarity, top-10, filtered by `tenant_id` |
| 2 | Sparse search | BM25Okapi keyword match, top-10, post-filtered to tenant corpus slice |
| 3 | RRF Fusion | Reciprocal Rank Fusion (k=60) → merged top-10 |
| 4 | Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-3 |
| 5 | Confidence | Sigmoid of top reranker score → 0–1 scalar |

**Why hybrid matters for UK HR:**

| Search type | Catches |
|---|---|
| Dense (semantic) | *"Time off for a new baby"* → maternity / paternity leave |
| Sparse (BM25) | *"SSP entitlement"*, *"TUPE regulations"*, *"IR35"*, *"ACAS Code"* |

### Intent Router

Runs before guardrails. Short-circuits non-HR queries with deterministic responses — same input always produces the same output.

| Intent bucket | Trigger | Action |
|---|---|---|
| `greeting` | "Hi", "Hello", "Hey" (anchored regex, exact-ish) | Canned welcome response, skip RAG |
| `thanks` | "Thanks", "Thank you", "Cheers" | Canned acknowledgement, skip RAG |
| `farewell` | "Bye", "Goodbye", "See ya" | Canned farewell, skip RAG |
| `meta` | "Who are you?", "What can you do?" (< 80 chars) | Bot description response, skip RAG |
| `help` | "Help me", "What can I ask?" (< 80 chars) | Example questions response, skip RAG |
| `hr_question` | Everything else | Full RAG pipeline |

### Self-Correction Loop

```
Attempt 1
  ├── retrieve(original_query, tenant_id)
  ├── generate_answer(chunks, history)
  ├── confidence > 0.75?
  │     YES → return (judge skipped)
  │     NO  → judge_answer()
  │             PASS → return
  │             FAIL → rewrite_query(original + history) → Attempt 2
Attempt 2
  ├── retrieve(rewritten_query, tenant_id)
  ├── generate_answer(chunks, history)
  └── return result (no further judge — fallback message if still bad)
```

### Session Memory (client-driven)

No server-side session state. The frontend sends the last **6 turns** (3 user + 3 assistant) in every request body. The generator builds messages as:

```
[system_prompt]
  + [conversation_history turns (normalised to user/assistant roles)]
  + [current user turn with CONTEXT block]
```

All three LLM providers (Groq, Gemini, OpenAI) consume multi-turn message arrays natively. Trade-off: no cross-device continuity — acceptable for MVP demo.

### LLM Router — 5 Provider Options

| Alias | Provider | Model | Notes |
|---|---|---|---|
| `groq_llama_8b` | Groq | Llama 3.1 8B | Free, fastest |
| `groq_llama_70b` | Groq | Llama 3.3 70B | Free, best open quality |
| `gemini_flash` | Google AI Studio | Gemini 2.0 Flash | Free, strong instruction-following |
| `openai_gpt35` | OpenAI | GPT-3.5 Turbo | Paid ~£0.001/query |
| `openai_gpt4o` | OpenAI | GPT-4o mini | Paid ~£0.003/query |

### Guardrails (hr_guardrails.py)

Priority order (first match wins):

```
1. INJECTION  → BLOCK  (system-prompt override attempts)
2. PII        → BLOCK  (employee IDs, emails, phone numbers)
3. SENSITIVE  → ESCALATE to HR Business Partner (no LLM call)
4. OUT_OF_SCOPE → fallback (not an HR topic)
5. PASS       → full RAG pipeline
```

---

## 4. DATA FLOW TRACES

### Trace 1: Multi-turn HR Query — *"And for part-timers?"* (after a maternity question)

```
Step  Component          Action                                    Result
────  ─────────────────  ────────────────────────────────────────  ──────────────────────────────
 1    Next.js UI         User types follow-up; sends last 6 turns  POST /query/stream
 2    Rate Limiter       Check per-IP quota (30/min)               PASS
 3    Intent Router      Not small-talk                            hr_question → continue
 4    HR Guardrails      No PII, no injection, not sensitive       PASS
 5    Local Embedding    Encode query → 384-d vector               query_vector
 6    Qdrant Cloud       ANN search filtered to tenant_id          top-10 semantic chunks
 7    BM25 Sparse        Keyword search scoped to tenant corpus     top-10 keyword chunks
 8    RRF Fusion         Merge + re-score                          top-10 fused
 9    Cross-Encoder      Rerank → top-3                            confidence_score
10    LLM Generator      System prompt + history + context + query Answer (streams via SSE)
11    Audit Log          Append row with session_id                logged
12    Next.js UI         Renders progressively; meta frame adds    Answer + confidence + citations
                         deduped citations and confidence pill
```

### Trace 2: Greeting — *"Hi"*

```
Step  Component          Action                                    Result
────  ─────────────────  ────────────────────────────────────────  ──────────────────────────────
 1    Next.js UI         User types "Hi"                           POST /query/stream
 2    Rate Limiter       Check per-IP quota                        PASS
 3    Intent Router      Matches GREETING_PATTERNS (anchored)      short_circuit=True
 4    —                  Canned response returned immediately      < 50 ms, no RAG
 5    Audit Log          Logged (0 ms retrieval latency)           logged
```

(Before the intent router, "Hi" would traverse embed → Qdrant → BM25 → rerank → 70B LLM → judge, taking 4–8 s and producing a different answer each time.)

### Trace 3: Re-upload of an Updated Policy

```
Step  Component          Action                                    Result
────  ─────────────────  ────────────────────────────────────────  ──────────────────────────────
 1    HR Manager         Uploads updated maternity_policy.pdf      POST /ingest
 2    Rate Limiter       10/min upload quota                       PASS
 3    check_doc_exists   filename + tenant_id already in Qdrant    True → delete old chunks
 4    delete_doc         Remove all old chunks from Qdrant + BM25  chunks_removed = N
 5    ingest_file        Parse → chunk → embed → upsert new doc    chunks_added = M
 6    Response           { replaced: true, chunks_replaced: N,    Old version gone; no duplicate
                          chunks_added: M }                        answers possible
```

### Trace 4: Sensitive Query — *"I want to file a harassment complaint"*

```
Step  Component          Action                                    Result
────  ─────────────────  ────────────────────────────────────────  ──────────────────────────────
 1    Next.js UI         User types sensitive query                POST /query/stream
 2    Rate Limiter       Check per-IP quota                        PASS
 3    Intent Router      Not small-talk                            hr_question → continue
 4    HR Guardrails      "harassment complaint" → ESCALATE         No embed, no LLM call
 5    Audit Log          escalated=true                            logged
 6    Next.js UI         Escalation card                           "Please contact your HR
                                                                   Business Partner directly."
```

---

## 5. DATA SCHEMAS

### QueryRequest (Frontend → Backend)

```json
{
  "query":                "What is the maternity leave entitlement?",
  "llm_provider":         "groq_llama_70b",
  "provider_api_key":     null,
  "openai_api_key":       null,
  "session_id":           "550e8400-e29b-41d4-a716-446655440000",
  "conversation_history": [
    { "role": "user",      "content": "How many days holiday?" },
    { "role": "assistant", "content": "Full-time employees get 28 days..." }
  ],
  "tenant_id": "public_uk"
}
```

### QueryResponse (Backend → Frontend, non-stream)

```json
{
  "answer":           "Employees are entitled to 52 weeks statutory maternity leave...",
  "sources":          [{ "doc_title": "...", "page_number": 3, "section_heading": "..." }],
  "llm_used":         "groq_llama_70b",
  "success":          true,
  "status":           "PASS",
  "confidence_score": 0.87,
  "confidence_label": "High",
  "latency_ms":       820,
  "query_id":         "uuid"
}
```

### SSE Stream Frames (`/query/stream`)

```
data: {"type": "token",  "content": "Employees "}
data: {"type": "token",  "content": "are entitled "}
...
data: {"type": "meta",   "query_id": "...", "sources": [...],
                         "confidence_label": "High", "confidence_score": 0.87,
                         "llm_used": "groq_llama_70b", "status": "PASS",
                         "latency_ms": 820}
data: [DONE]
```

### Qdrant Payload (HR chunk)

```json
{
  "chunk_text":       "Full-time employees are entitled to 28 days annual leave...",
  "doc_title":        "Holiday Entitlement Guide",
  "doc_type":         "guide",
  "department":       "All",
  "section_heading":  "2. Statutory Minimum Entitlement",
  "page_number":      3,
  "source_filename":  "gov_holiday_entitlement.pdf",
  "tenant_id":        "public_uk",
  "ingested_at":      "2026-04-17T10:00:00Z"
}
```

Payload indexes created: `source_filename` (KEYWORD), `tenant_id` (KEYWORD).

### Audit Log Row (CSV)

```
session_id, query_id, timestamp, query, answer_preview, doc_title, section,
page, llm_used, blocked, block_reason, escalated, latency_ms, rating, feedback_reason
```

---

## 6. COMPONENT OVERVIEW

| Component | File | Description |
|---|---|---|
| Intent Router | `intent_router.py` | Classifies small-talk (6 buckets); short-circuits before RAG |
| Document parser | `hr_doc_loader.py` | Extracts text from PDF, DOCX, TXT → structured Markdown |
| Ingestion pipeline | `hr_ingest.py` | Version-check, header-aware split, local embed, Qdrant + BM25 upsert |
| Embeddings | `all-MiniLM-L6-v2` | 384-d local dense vectors (no API cost) |
| Vector store | Qdrant Cloud | Persistent; payload-indexed on `source_filename` + `tenant_id` |
| Sparse search | `rank-bm25` | BM25Okapi in-memory; pickle-cached for fast restart |
| Retriever | `retriever.py` | Dense ANN + BM25 (both tenant-scoped) → RRF → cross-encoder → top-3 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local model, ~90 MB |
| LLM router | `generator.py` | Multi-turn-aware; routes to Groq / Gemini / OpenAI; streaming + non-streaming |
| Guardrails | `hr_guardrails.py` | Blocks PII + injection; escalates sensitive HR queries; OOS filter |
| System prompt | `hr_system_prompt.txt` | HR persona, UK employment context, citation format, refusal rules |
| Audit logger | `audit_log.py` | In-memory + CSV; session_id + query_id + feedback columns |
| API server | `main.py` | FastAPI — all endpoints, CORS from env var, rate limiter, SSE streaming |
| Frontend | `page.tsx` + `api.ts` | SSE consumer, sessionId, conversation history, confidence pill, feedback |

---

## 7. FILE STRUCTURE

```
HR_BOT/
│
├── backend/                         ← Python FastAPI (→ Hugging Face Spaces Docker)
│   ├── main.py                      FastAPI app — all endpoints, rate limiting, CORS
│   ├── intent_router.py             Pre-RAG intent classifier (greetings, meta, help)
│   ├── hr_guardrails.py             PII block, injection block, escalation, OOS filter
│   ├── hr_doc_loader.py             PDF/DOCX/TXT parser → structured Markdown
│   ├── hr_ingest.py                 Chunk, embed, Qdrant upsert, BM25 rebuild, versioning
│   ├── retriever.py                 Hybrid retrieve: dense + BM25 + RRF + rerank
│   ├── generator.py                 Multi-LLM router, streaming, conversation history
│   ├── audit_log.py                 Session CSV logger (session_id, feedback)
│   ├── hr_system_prompt.txt         HR persona + citation rules
│   ├── onboarding_checklist.json    Onboarding checklist data
│   ├── bm25_cache.pkl               Auto-generated pickle cache (gitignored)
│   ├── test_full_pipeline.py        7-test automated suite
│   ├── Dockerfile                   HF Spaces Docker container
│   └── requirements.txt             Python dependencies
│
├── frontend/                        ← Next.js 14 App Router (→ Vercel)
│   ├── src/app/
│   │   ├── page.tsx                 Main chat page — SSE, sessionId, history
│   │   └── layout.tsx               Root layout + disclaimer banner
│   ├── src/app/components/
│   │   ├── ChatThread.tsx           Message list + streaming cursor + skeleton suppression
│   │   ├── Sidebar.tsx              Upload + LLM selector + API key
│   │   └── ...
│   ├── src/lib/api.ts               HTTP + SSE client (queryStream async generator)
│   └── package.json
│
├── demo_docs/                       ← ACAS/CIPD sample HR documents (pre-loaded)
│   └── *.pdf / *.txt
│
├── README.md                        Setup + deployment guide
├── architecture.md                  This file
└── constraints.md                   Free-tier constraints rationale
```

---

## 8. DEPENDENCIES

```text
# API server
fastapi>=0.115.0
uvicorn>=0.32.0
python-multipart>=0.0.9

# Rate limiting
slowapi>=0.1.9

# Document parsing
pdfplumber>=0.11.6
python-docx>=1.1.0

# Text splitting
langchain-text-splitters>=0.3.0

# Embeddings + reranker
sentence-transformers>=3.4.1

# Vector database
qdrant-client>=1.13.3

# Sparse search
rank-bm25>=0.2.2

# LLM clients
openai>=1.63.2
groq>=0.18.0
google-generativeai>=0.8.0

# Utilities
python-dotenv>=1.0.1
numpy>=2.1.0,<3.0.0
tiktoken>=0.8.0
```

---

## 9. IMPLEMENTED IMPROVEMENTS

The items below are fully shipped (all on branch `claude/zen-almeida`):

| # | Improvement | Commit | Impact |
|---|---|---|---|
| 1 | **Intent Router** (`intent_router.py`) | `feat: Fix A+B` | "Hi" now returns deterministic response in < 50 ms vs. 4–8 s inconsistent before |
| 2 | **Client-driven Session Memory** | `feat: Fix A+B` | Multi-turn follow-ups ("and for part-timers?") resolve correctly via conversation history |
| 3 | **Citation Deduplication** | `fix: dedup + CORS + judge-skip` | Duplicate citation cards eliminated; deduped by `(filename, page, section)` |
| 4 | **CORS Restriction via env var** | `fix: dedup + CORS + judge-skip` | `ALLOWED_ORIGINS` set per-environment; no accidental wildcard in production |
| 5 | **Judge-skip on high confidence** | `fix: dedup + CORS + judge-skip` | Saves one LLM round-trip (~40% of latency) when reranker score > 0.75 |
| 6 | **SSE Streaming** (`/query/stream`) | `feat: SSE streaming` | TTFT drops from ~5 s to ~300 ms; streaming cursor in UI |
| 7 | **Document Versioning** | `fix: document versioning` | Re-uploads replace old chunks; no conflicting answers after policy updates |
| 8 | **Per-IP Rate Limiting** | `feat: per-IP rate limiting` | 30 req/min on `/query`, 10 req/min on `/ingest`; demo quota protected |
| 9 | **Multi-Tenancy Groundwork** | `feat: multi-tenancy groundwork` | `tenant_id` scopes all ingest, retrieve, list, delete operations |

---

## 10. PLANNED IMPROVEMENTS (Tier 2/3)

| # | Improvement | Effort | Impact |
|---|---|---|---|
| 10 | **Structured logging** — replace `print()` with `logging` + JSON formatter | Low | Observable in HF Spaces logs; shippable to Grafana Cloud free |
| 11 | **Semantic chunking for tables** — keep Markdown tables intact (one chunk per table) | Medium | Stops UK SSP-rate and holiday-accrual tables being split mid-row |
| 12 | **Answer cache** — SHA-256 keyed by `(tenant_id, query_norm, doc_set_version)` | Medium | Cuts Groq calls ~30–60% on repeated-question traffic |
| 13 | **HyDE query expansion** — generate a hypothetical answer with Groq 8B, embed that | Medium | Often outperforms raw-query embedding on policy corpora |
| 14 | **BGE-small embedding upgrade** — `BAAI/bge-small-en-v1.5` (same 384-d, better recall) | Medium | Drop-in swap; re-ingest required |
| 15 | **Evaluation harness** — RAGAS or DIY 30 Q/A pairs, runs in CI via Groq free | High | Catches retrieval or prompt regressions automatically |

---

## 11. RISKS & MITIGATIONS

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Stale policies** — bot serves outdated rules | `ingested_at` in every chunk; document versioning deletes old chunks on re-upload |
| 2 | **Hallucination** — LLM invents leave counts | LLM-as-Judge post-generation check; temperature=0.0; Low confidence pill nudges user |
| 3 | **Escalation gap** — sensitive query answered by LLM | Hard pre-filter block in guardrails; no embedding or LLM call for escalation triggers |
| 4 | **Employee PII in query** | PII regex pre-filter in `hr_guardrails.py`; blocks before embedding or LLM |
| 5 | **Multi-doc conflicts** — two policies say different things | `tenant_id` + `department` filter; LLM instructed to cite which policy applies |
| 6 | **Confidential doc uploaded** | Disclaimer banner; raw upload unlinked immediately after parse |
| 7 | **HF Spaces cold start** | BM25 pickle cache (sub-second); cross-encoder lazy-loaded on first query |
| 8 | **Demo quota exhaustion** | Per-IP rate limiting (slowapi, 30/min); Groq free tier is 14,400 req/day |
| 9 | **CORS misconfiguration** | `ALLOWED_ORIGINS` env var; wildcard only in local dev |
| 10 | **Tenant data bleed** | All Qdrant queries filtered by `tenant_id`; BM25 corpus post-filtered to tenant slice |
