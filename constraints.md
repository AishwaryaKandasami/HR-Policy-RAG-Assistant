# HR Policy Q&A Bot — Hosting Constraints & Decision Log

## Context

This document records the hosting constraints for the MVP and explains why each option was
evaluated and ruled out, leaving Hugging Face Spaces as the only viable zero-cost choice.

### Hard Constraints
- **Budget: £0** — no spend permitted at MVP stage
- **Global access** — anyone (employee, HR manager, prospect) must be able to open a URL,
  upload their own document, paste their own API key, and use the chatbot independently
- **No always-on requirement** — the bot is activated on demand; it does not need to serve
  traffic 24/7, but it must be reliably accessible when turned on
- **Session-based document handling** — documents are uploaded per session and held in memory;
  no persistent storage is required between sessions
- **ML model RAM requirement** — the cross-encoder reranker
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and sentence-transformers require approximately
  300–400 MB RAM on top of FastAPI and the Qdrant in-memory store. Total backend RAM usage
  is estimated at 500–700 MB under load

---

## Options Evaluated & Ruled Out

### ❌ Option 1 — Railway (Free Tier)

**What it offers:** Managed Python hosting, no sleep on free tier, simple deploy via CLI.

**Why it was ruled out:**

| Concern | Detail |
|---|---|
| **Cost** | Railway's free tier provides $5/month of credit. At ~$0.000463/vCPU-second, a continuously running FastAPI process exhausts this in roughly 3–4 days of uptime. The service would stop mid-month. |
| **Not truly free** | Any sustained usage beyond the credit window requires a paid plan (~$5/month minimum). Budget constraint is £0. |
| **No workaround** | Reducing instance size to stay within credit doesn't solve the problem — it just delays when the credits run out. |

**Verdict:** Ruled out. Not free beyond a few days of runtime.

---

### ❌ Option 2 — Vercel (Backend)

**What it offers:** Serverless functions, global CDN, tight Next.js integration.

**Why it was ruled out:**

| Concern | Detail |
|---|---|
| **Serverless function timeout** | Vercel free tier enforces a **10-second execution limit** on serverless functions. The RAG pipeline (embed → hybrid search → rerank → LLM call) routinely takes 3–8 seconds. Document ingestion with pdfplumber on a multi-page PDF can exceed 10 seconds easily. |
| **No persistent process** | Vercel functions are stateless and short-lived. The cross-encoder reranker and Qdrant in-memory store must be initialised on every request — adding 5–10 seconds of cold-start overhead **per query**. |
| **No in-memory state** | Qdrant `:memory:` and the BM25 index cannot persist across serverless invocations. Every query would require re-ingesting all documents, making the product non-functional. |
| **RAM cap** | Vercel free tier limits function RAM to 1024 MB, but the initialisation overhead makes this impractical regardless. |
| **Conclusion** | Vercel is the correct choice for the **frontend only** (Next.js). It cannot run the Python ML backend. |

**Verdict:** Ruled out for backend. Used for frontend only.

---

### ❌ Option 3 — Streamlit Community Cloud

**What it offers:** Free Python app hosting, tight Streamlit framework integration, 1 GB RAM.

**Why it was ruled out:**

| Concern | Detail |
|---|---|
| **UI quality** | Streamlit's component model is widget-driven and opinionated. It cannot replicate the conversational chat interface, collapsible source citations, drag-and-drop file upload, or fixed query-log button specified in the product brief. The MVP must visually impress prospects and clients during demos. |
| **No decoupled API** | Streamlit is a monolithic app framework — it cannot serve as a REST API backend for a separate Next.js frontend. Choosing Streamlit would require abandoning the Next.js frontend entirely. |
| **Not a product** | A Streamlit app looks like an internal tool or a data science prototype. The HR Policy Q&A Bot is positioned as a client-facing product and must look like one. |
| **Session handling** | Streamlit's session state model is less predictable for concurrent users — each browser tab is a session, but there is no clean multi-user isolation without significant workarounds. |

**Verdict:** Ruled out. Streamlit cannot deliver the required UI quality or serve as an API backend.

---

### ❌ Option 4 — Run Locally (localhost + ngrok)

**What it offers:** Zero cost, full hardware resources, no RAM constraints, easiest to develop against.

**Why it was ruled out for production use:**

| Concern | Detail |
|---|---|
| **Not globally accessible** | The chatbot only works while the developer's laptop is running and the `uvicorn` process is active. Any employee or prospect navigating to the Vercel URL when the laptop is off or the server is stopped will get a connection error. |
| **ngrok free tier URL changes** | The ngrok free tier assigns a new random URL on every tunnel start (e.g. `https://a1b2c3.ngrok.io`). The Vercel frontend's backend URL environment variable would need to be updated and redeployed every session — impractical for client-facing use. |
| **Only for controlled demos** | Running locally is only viable if the developer is physically present at their machine, actively running the server, and sharing their screen. This cannot scale to letting clients or employees use the chatbot independently. |
| **Not a product** | A product that only works when one person's laptop is open is not a shippable MVP. |

**Verdict:** Ruled out for product use. Appropriate for local development only.

---

### ❌ Option 5 — Render (Free Tier)

**What it offers:** Managed Python hosting, free tier with a stable URL.

**Why it was ruled out:**

| Concern | Detail |
|---|---|
| **Sleep after 15 minutes** | Render free tier spins down any service that has not received a request in 15 minutes. Cold start after sleep takes 30–60 seconds — acceptable in isolation, but the ML model initialisation (sentence-transformers download + load) adds another 20–40 seconds, making the total first-response wait 60–100 seconds. This is damaging during a live client demo. |
| **RAM cap: 512 MB** | Render free tier caps RAM at **512 MB**. The cross-encoder model alone uses ~300 MB. FastAPI, pdfplumber, qdrant-client, and rank-bm25 together add another 200–300 MB. The backend consistently exceeds 512 MB and would be killed by an OOM error during a demo. |
| **No workaround without downgrade** | Removing the cross-encoder reranker to fit within 512 MB would degrade retrieval quality — a core feature of the hybrid RAG pipeline. |

**Verdict:** Ruled out. RAM limit is incompatible with the ML stack; cold start is too slow for live demos.

---

## ✅ Chosen Option — Hugging Face Spaces (Docker)

**Why it is the only option that satisfies all constraints:**

| Requirement | How HF Spaces meets it |
|---|---|
| **£0 cost** | Public Spaces are permanently free with no credit limits or expiry |
| **Global access** | Fixed, stable public URL (e.g. `https://username-hr-bot.hf.space`) accessible from any browser worldwide |
| **RAM** | Free tier provides **2 vCPUs and 16 GB RAM** — the full ML stack (FastAPI + sentence-transformers + qdrant-client + rank-bm25 + pdfplumber) runs comfortably within this |
| **No sleep** | HF Spaces does not spin down — the service is available whenever a user visits the URL |
| **Docker support** | Runs any Docker container, giving full control over the Python environment and dependencies |
| **Session-based memory** | Qdrant `:memory:` and BM25 in-memory index work correctly — no persistent storage needed |
| **Pre-loadable demo docs** | 8 ACAS/CIPD PDFs can be bundled into the Docker image and pre-ingested at startup |
| **Stable Vercel integration** | The HF Spaces URL is fixed and can be set as a permanent environment variable in Vercel |

**Trade-off acknowledged:**

| Trade-off | Impact |
|---|---|
| **Space is public** | Anyone can view the HF Space page. Mitigated by the MVP constraint that no real employee personal data is uploaded (demo uses public ACAS/CIPD documents only). |
| **Shared infrastructure** | Free tier runs on shared hardware; performance may vary under heavy concurrent load. Acceptable for MVP with low expected concurrency. |

---

## Summary Decision Table

| Option | Free? | Global Access? | Sufficient RAM? | No Cold-Start Issues? | Verdict |
|---|---|---|---|---|---|
| Railway | ❌ ($5 credit only) | ✅ | ✅ | ✅ | ❌ Ruled out |
| Vercel (backend) | ✅ | ✅ | ⚠️ | ❌ (10s timeout, stateless) | ❌ Ruled out |
| Streamlit Cloud | ✅ | ✅ | ✅ | ⚠️ | ❌ UI unacceptable |
| Local + ngrok | ✅ | ❌ | ✅ | ✅ | ❌ Not a product |
| Render | ✅ | ✅ | ❌ (512 MB) | ❌ (60–100s cold start) | ❌ Ruled out |
| **Hugging Face Spaces** | ✅ | ✅ | ✅ (16 GB) | ✅ | ✅ **Chosen** |
