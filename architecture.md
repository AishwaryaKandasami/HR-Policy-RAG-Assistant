# HR Policy Q&A Bot — Technical Architecture (MVP)

**Product:** HR Policy Q&A Bot  
**Target users:** Employees (ask questions), HR Managers (upload docs), Prospects (demo)  
**Document sources:** User-uploaded HR policy PDFs/DOCX — demo uses 8 free ACAS/CIPD UK documents  
**Retrieval method:** Hybrid RAG — dense vector search (Qdrant) + sparse keyword search (BM25)

---

## 1. CONFIRMED STACK

| Layer | Technology | Cost |
|---|---|---|
| **Frontend** | Next.js → Vercel | Free |
| **Backend** | FastAPI → Hugging Face Spaces (Docker) | Free |
| **Vector DB** | Qdrant `:memory:` — session-based | Free |
| **Sparse Search** | rank-bm25 — in-memory | Free |
| **Embeddings** | OpenAI `text-embedding-3-small` (768-d) | Client API key |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free (local model) |
| **LLM** | Groq / Gemini / OpenAI — client provides API key | Free (client key) |
| **Demo docs** | 8 ACAS/CIPD PDFs — pre-loaded at startup | Free |
| **Total** | | **£0** |

---

## 2. SYSTEM DIAGRAM

### Online Serving Path

```
Anyone in the world (employee / HR manager / prospect)
        │  opens browser
        ▼
┌─────────────────────────────────────────────────────────┐
│             LAYER 1 — FRONTEND (Vercel)                 │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Chat UI     │  │ File Upload  │  │ LLM Selector │  │
│  │  (thread)    │  │  sidebar     │  │ + API key    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
└─────────┼─────────────────┼─────────────────┼───────────┘
          │           HTTP (REST)              │
          ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────┐
│       LAYER 2 — BACKEND (Hugging Face Spaces, Docker)   │
│                                                         │
│   POST /ingest      POST /query      GET /logs          │
│        │                 │               │              │
│        ▼                 ▼               ▼              │
│   ┌─────────┐    ┌──────────────┐  ┌──────────┐       │
│   │  Doc    │    │  HR          │  │  Audit   │       │
│   │  Parser │    │  Guardrails  │  │  Log CSV │       │
│   │+Chunker │    │  (pre-filter)│  └──────────┘       │
│   └────┬────┘    └──────┬───────┘                     │
│        │                │ PASS                         │
│        ▼                ▼                              │
│   ┌─────────┐    ┌──────────────────────────────────┐  │
│   │Embedder │    │       HYBRID RETRIEVAL           │  │
│   │(OpenAI) │    │                                  │  │
│   └────┬────┘    │  ┌─────────────┐ ┌────────────┐  │  │
│        │         │  │ Dense ANN   │ │ BM25       │  │  │
│        ▼         │  │ (Qdrant)    │ │ (rank-bm25)│  │  │
│   ┌─────────┐    │  └──────┬──────┘ └─────┬──────┘  │  │
│   │ Qdrant  │◄───┘         │              │          │  │
│   │:memory: │         RRF Score Fusion    │          │  │
│   └─────────┘              └──────┬───────┘          │  │
│                                   │                   │  │
│                     Cross-Encoder Reranker            │  │
│                                   │ top-3 chunks      │  │
│                                   ▼                   │  │
│                         ┌─────────────────┐           │  │
│                         │   LLM ROUTER    │           │  │
│                         │ Groq / Gemini / │           │  │
│                         │ OpenAI          │           │  │
│                         └────────┬────────┘           │  │
│                                  │ answer + citation   │  │
└──────────────────────────────────┼─────────────────────┘
                                   ▼
                    Frontend renders answer card
                    + collapsible source citation
```

### Offline Ingestion Pipeline

```
User uploads file (PDF / DOCX / TXT)   OR   Demo docs pre-loaded at startup
        │
        ├── .pdf  → pdfplumber   → text per page
        ├── .docx → python-docx  → paragraphs + headings
        └── .txt / .md → direct read
        │
        ▼
RecursiveCharacterTextSplitter
  chunk_size=600 chars, overlap=80 chars
        │
        ▼
Metadata per chunk:
  { doc_title, doc_type, department, section_heading, page_number,
    source_filename, ingested_at }
        │
        ├──► OpenAI text-embedding-3-small → Qdrant :memory: (dense)
        └──► rank-bm25 index update (sparse, in-memory)
```

---

## 3. LAYER BREAKDOWN

### Frontend (Next.js — Vercel)

| Area | Content & Behaviour |
|---|---|
| Disclaimer banner | Orange bar: *"Proof of concept — do not upload documents containing real employee personal data."* Dismissible. |
| Left sidebar | (1) File upload — drag-drop, calls `POST /ingest`, shows uploaded doc list with ✅. (2) LLM selector — dropdown with 5 providers. (3) API key input — password field, sent with each `/query` call. |
| Main chat area | Full-height message thread. User messages right-aligned. Bot messages left-aligned with avatar. Source citation in collapsible expander below each answer. |
| Input bar | Text input + send button. Disabled while awaiting response. Shows typing indicator. |
| Query log button | Fixed bottom-right. Calls `GET /logs` → downloads CSV. |

### Backend (FastAPI — Hugging Face Spaces Docker)

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/ingest` | POST | `multipart/form-data` — one or more files | `{ status, chunks_added, docs[] }` |
| `/query` | POST | `{ query, llm_provider, api_key, department? }` | `{ answer, source, doc_title, page, blocked, escalated }` |
| `/logs` | GET | — | CSV file download |
| `/docs-list` | GET | — | `[{ filename, chunk_count, ingested_at }]` |
| `/docs` | DELETE | `{ filename }` | `{ status }` |

### Hybrid Retrieval

| Step | Component | Detail |
|---|---|---|
| 1 | Dense search | Qdrant ANN cosine similarity, top-10 |
| 2 | Sparse search | BM25 keyword match, top-10 |
| 3 | Fusion | Reciprocal Rank Fusion (RRF) → top-10 merged |
| 4 | Rerank | cross-encoder/ms-marco-MiniLM-L-6-v2 → top-3 |

**Why hybrid matters for HR:**

| Search type | Catches |
|---|---|
| Dense (semantic) | *"Time off for a new baby"* → maternity/paternity leave |
| Sparse (BM25) | *"SSP entitlement"*, *"TUPE regulations"*, *"IR35"*, *"ACAS Code"* |

### LLM Router — 5 Provider Options

| Provider | Model | Notes |
|---|---|---|
| Groq | Llama 3.1 8B | Free, fastest — best for demo |
| Groq | Llama 3.3 70B | Free, better quality |
| Google AI Studio | Gemini Flash 2.0 | Free, strong instruction-following |
| OpenAI | GPT-3.5 Turbo | Paid, ~£0.001/query |
| OpenAI | GPT-4o mini | Paid, best quality, ~£0.003/query |

### Guardrails

Priority order:

```
1. INJECTION       → block
2. PII             → block
3. SENSITIVE       → escalate to HR team  (no LLM call)
4. OUT_OF_SCOPE    → fallback
5. FACTUAL         → retrieve ✅
```

**Sensitive escalation triggers** (no LLM call — return HR contact card):
```python
ESCALATION_TRIGGERS = [
    "harassment", "discrimination", "bullying", "complaint",
    "grievance", "hostile work environment", "retaliation",
    "unfair dismissal", "wrongful termination", "misconduct report",
    "ethics violation", "am i being fired", "will i be fired",
    "sue", "legal action", "tribunal",
]
```

**In-scope HR topics:**
```python
HR_TOPICS = [
    "annual leave", "holiday", "sick leave", "sickness absence",
    "maternity", "paternity", "flexible working", "wfh", "remote work",
    "notice period", "redundancy", "dismissal", "disciplinary",
    "grievance procedure", "expense", "payroll", "salary", "ssp",
    "tupe", "ir35", "probation", "performance review", "appraisal",
    "code of conduct", "dress code", "onboarding", "offboarding",
]
```

---

## 4. DATA FLOW TRACES

### Trace 1: Factual HR Query — *"How many days annual leave am I entitled to?"*

```
Step  Component             Action                              Result
────  ────────────────────  ──────────────────────────────────  ─────────────────────────────────
 1    Next.js UI            User types question                 POST /query
 2    HR Guardrails         No PII, no injection, not sensitive PASS → retrieve
 3    OpenAI Embedding      Encode query → 768-d vector         query_vector: float[768]
 4    Qdrant Dense          ANN search — top-10 chunks          semantic matches
 5    BM25 Sparse           Keyword search — top-10 chunks      "holiday", "annual leave" matches
 6    RRF Fusion            Merge + re-score both lists         top-10 fused
 7    Cross-Encoder         Rerank → top-3                      best: "28 days statutory minimum..."
 8    LLM Router            HR system prompt + context + query  "Full-time employees are entitled
                                                                to 28 days annual leave including
                                                                bank holidays. (Holiday Entitlement
                                                                Guide, Section 2)"
 9    Post-Filter           Grounding check ✓                   PASS
10    Audit Log             Append row to session CSV           logged
11    Next.js UI            Render answer card + source link    Employee sees cited answer
```

### Trace 2: Sensitive Query — *"I want to file a harassment complaint"*

```
Step  Component             Action                              Result
────  ────────────────────  ──────────────────────────────────  ─────────────────────────────────
 1    Next.js UI            User types query                    POST /query
 2    HR Guardrails         "harassment complaint" matches      ESCALATE
                            ESCALATION_TRIGGERS                 (no embedding, no LLM call)
 3    Audit Log             Log as escalated=true               logged
 4    Next.js UI            Render escalation card              "For harassment or grievance
                                                                matters, please contact your HR
                                                                Business Partner directly.
                                                                This chatbot cannot process
                                                                complaint submissions."
```

---

## 5. DATA SCHEMAS

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
  "ingested_at":      "2026-04-02T20:00:00Z"
}
```

### Query Response (Backend → Frontend)

```json
{
  "answer":           "Full-time employees are entitled to 28 days annual leave...",
  "doc_title":        "Holiday Entitlement Guide",
  "section_heading":  "2. Statutory Minimum Entitlement",
  "page_number":      3,
  "source_filename":  "gov_holiday_entitlement.pdf",
  "grounded":         true,
  "blocked":          false,
  "escalated":        false,
  "llm_used":         "groq_llama_8b",
  "latency_ms":       450
}
```

### Audit Log Row (CSV)

```
timestamp, query, answer_preview, doc_title, section, page, llm_used, blocked, block_reason, escalated, latency_ms
```

---

## 6. COMPONENT OVERVIEW

| Component | File | Description |
|---|---|---|
| Document parser | `hr_doc_loader.py` | Extracts text from PDF, DOCX, TXT. Attaches metadata per chunk. |
| Ingestion pipeline | `hr_ingest.py` | Chunks text, embeds with OpenAI, upserts to Qdrant and BM25 index. |
| Embeddings | OpenAI `text-embedding-3-small` | 768-d dense vectors for semantic search. |
| Vector store | Qdrant `:memory:` | Holds embedded HR doc chunks for the session. Collection: `hr_docs`. |
| Sparse search | `rank-bm25` in-memory index | Keyword match for exact HR terms (SSP, TUPE, IR35, ACAS). |
| Retriever | `retriever.py` | Dense ANN + BM25 → RRF fusion → cross-encoder rerank → top-3. |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local model, scores top-10 fused chunks to return best top-3. |
| LLM router | `generator.py` | Routes to Groq / Gemini / OpenAI based on user's dropdown selection. |
| Guardrails | `hr_guardrails.py` | Blocks PII and injection; escalates sensitive HR queries; filters out-of-scope. |
| System prompt | `hr_system_prompt.txt` | HR persona, UK employment context, citation format, refusal rules. |
| Audit logger | `audit_log.py` | Appends every query and response to a session-scoped CSV. |
| API server | `main.py` | FastAPI — exposes all endpoints, handles CORS, pre-loads demo docs at startup. |
| Frontend | Next.js (Vercel) | Chat UI, file upload sidebar, LLM selector, API key input, query log button. |

---

## 7. FILE STRUCTURE

```
HR_BOT/
│
├── backend/                          ← Python FastAPI (→ Hugging Face Spaces)
│   ├── main.py                       🆕 FastAPI app — all endpoints
│   ├── hr_doc_loader.py              🆕 PDF/DOCX/TXT parser + metadata
│   ├── hr_ingest.py                  — chunking, embedding, Qdrant + BM25 upsert
│   ├── retriever.py                  — hybrid BM25 + dense ANN + RRF + reranker
│   ├── generator.py                  — multi-LLM router (Groq / Gemini / OpenAI)
│   ├── hr_guardrails.py              — HR rules, escalation triggers, PII detection
│   ├── hr_system_prompt.txt          — HR persona, UK context, citation rules
│   ├── audit_log.py                  — session-scoped CSV logger
│   ├── Dockerfile                    — HF Spaces Docker container definition
│   ├── requirements.txt              — all Python dependencies
│   └── .env.example
│
├── frontend/                         ← Next.js (→ Vercel)
│   ├── app/
│   │   ├── page.tsx                  🆕 Main chat page
│   │   └── layout.tsx                🆕 Root layout + disclaimer banner
│   ├── components/
│   │   ├── ChatThread.tsx            🆕 Message list
│   │   ├── MessageBubble.tsx         🆕 User/bot bubble + source expander
│   │   ├── Sidebar.tsx               🆕 Upload + LLM selector + API key
│   │   ├── FileUploader.tsx          🆕 Drag-drop → POST /ingest
│   │   ├── LLMSelector.tsx           🆕 5-provider dropdown
│   │   ├── DisclaimerBanner.tsx      🆕 Orange dismissible bar
│   │   └── QueryLogButton.tsx        🆕 Fixed → GET /logs CSV
│   ├── lib/api.ts                    🆕 HTTP client for backend calls
│   └── package.json
│
└── demo_docs/                        ← 8 ACAS/CIPD sample HR documents
    ├── acas_disciplinary_procedure.pdf
    ├── acas_grievance_policy.pdf
    ├── acas_sickness_absence.pdf
    ├── acas_flexible_working.pdf
    ├── acas_redundancy_procedure.pdf
    ├── gov_holiday_entitlement.pdf
    ├── cipd_bullying_harassment.pdf
    └── cipd_staff_handbook.pdf
```

---

## 8. DEPENDENCIES

```text
# New
fastapi>=0.115.0          # API server
uvicorn>=0.32.0           # ASGI server
python-docx>=1.1.0        # DOCX parsing
rank-bm25>=0.2.2          # BM25 sparse search
google-generativeai>=0.8  # Gemini Flash support
python-multipart>=0.0.9   # File upload handling

# Core dependencies
openai>=1.63.2
groq>=0.18.0
qdrant-client>=1.13.3
sentence-transformers>=3.4.1
langchain-text-splitters>=0.3.0
pdfplumber>=0.11.6
python-dotenv>=1.0.1
```

---

## 9. BUILD PHASES

### Phase 1 — Backend Core
- [ ] FastAPI project setup with CORS
- [ ] `hr_doc_loader.py` — PDF + DOCX + TXT parser
- [ ] `hr_ingest.py` — chunk, embed, Qdrant + BM25 upsert
- [ ] `POST /ingest`, `GET /docs-list`, `DELETE /docs` endpoints
- [ ] Pre-load 3 ACAS demo docs at startup, test ingestion

### Phase 2 — Hybrid Retrieval + Generation
- [ ] BM25 index + RRF fusion in `retriever.py`
- [ ] Multi-LLM router in `generator.py`
- [ ] `hr_system_prompt.txt`
- [ ] `POST /query` endpoint — full pipeline
- [ ] Test with Groq Llama 3.1 8B against demo docs

### Phase 3 — Guardrails + Audit Log
- [ ] `hr_guardrails.py` — escalation + PII + injection + OOS
- [ ] `audit_log.py` — session CSV logger
- [ ] `GET /logs` endpoint
- [ ] Unit test all guardrail cases

### Phase 4 — Docker + HF Spaces Deploy
- [ ] Write `Dockerfile` for FastAPI app
- [ ] Create Hugging Face Space (Docker SDK)
- [ ] Push backend, verify public URL is live
- [ ] Test all endpoints from external client

### Phase 5 — Next.js Frontend
- [ ] Scaffold Next.js + Tailwind CSS
- [ ] Build all components (sidebar, chat, disclaimer, log button)
- [ ] Wire to backend via `lib/api.ts`
- [ ] Deploy to Vercel, set backend URL env var

### Phase 6 — Integration Test + Demo Prep
- [ ] Load all 8 ACAS/CIPD docs
- [ ] Run 20+ HR query tests (all 6 spec example interactions)
- [ ] Verify escalation triggers on sensitive queries
- [ ] Test all 5 LLM providers
- [ ] Final smoke test end-to-end

---

## 10. RISKS & MITIGATIONS

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Stale policies** — bot serves outdated rules | `effective_date` in every chunk; UI shows "Policy as of [date]" |
| 2 | **Hallucination** — LLM invents leave counts or policy numbers | Post-generation grounding check; temperature=0.0 |
| 3 | **Escalation gap** — sensitive query answered by LLM | Hard pre-filter block; no LLM call made for escalation triggers |
| 4 | **Employee PII in query** | PII pre-filter in `hr_guardrails.py`; blocks before embedding or LLM call |
| 5 | **Multi-doc conflicts** — two policies say different things | `department` filter; LLM instructed to cite which policy applies |
| 6 | **Confidential doc uploaded** | Disclaimer banner; MVP instructs no real employee data |
| 7 | **HF Spaces cold start** | Cross-encoder + demo docs pre-loaded at startup; ~30s first load |
