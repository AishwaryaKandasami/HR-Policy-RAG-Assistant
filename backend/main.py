"""
main.py — HR Policy Q&A Bot — FastAPI Backend
==============================================
Endpoints:
  POST   /ingest      Upload and ingest HR policy documents
  GET    /docs-list   List all documents currently in the session store
  DELETE /docs        Remove a document from the store
  GET    /health      Health check

  POST   /query       Ask a question  [Phase 2 — stub for now]
  GET    /logs        Download audit log CSV  [Phase 3 — stub for now]

Demo docs (ACAS/CIPD PDFs in demo_docs/) are pre-loaded at startup
if OPENAI_API_KEY is present in the environment.
"""

import os
import pathlib
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from audit_log import get_log_file_path, log_interaction
from generator import generate_answer
from hr_guardrails import classify_query
from hr_ingest import delete_doc, get_ingested_docs, get_openai, ingest_file
from retriever import retrieve


# ── Data Models ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the maternity leave entitlement?")
    llm_provider: str = Field("groq_llama_8b", description="Model alias (groq_llama_8b, gemini_flash, etc.)")
    # Keys are now optional; server will use its own secrets if these are None
    openai_api_key: Optional[str] = Field(None, description="Optional override for embedding")
    provider_api_key: Optional[str] = Field(None, description="Optional override for LLM provider")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    llm_used: str
    success: bool
    status: str = "PASS"   # PASS, BLOCK, ESCALATE
    latency_ms: Optional[float] = None


# ── Startup: pre-load demo docs ────────────────────────────────────

DEMO_DOCS_DIR = pathlib.Path(__file__).parent.parent / "demo_docs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: ingest all PDFs from demo_docs/ if OPENAI_API_KEY is set.
    This gives every new session an immediate knowledge base to query against.
    """
    if OPENAI_API_KEY and DEMO_DOCS_DIR.exists():
        demo_files = sorted(DEMO_DOCS_DIR.glob("*"))
        eligible = [f for f in demo_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if eligible:
            print(f"\n📂 Pre-loading {len(eligible)} demo HR document(s)...")
            for doc_path in eligible:
                try:
                    result = ingest_file(str(doc_path), api_key=OPENAI_API_KEY)
                    print(
                        f"  ✅ {result['doc_title']}"
                        f" — {result['chunks_added']} chunks"
                    )
                except Exception as e:
                    print(f"  ⚠  Could not pre-load {doc_path.name}: {e}")
            print()
        else:
            print("ℹ  demo_docs/ exists but contains no supported files.")
    else:
        if not OPENAI_API_KEY:
            print("ℹ  OPENAI_API_KEY not set — skipping demo doc pre-load.")
        if not DEMO_DOCS_DIR.exists():
            print(f"ℹ  demo_docs/ directory not found at {DEMO_DOCS_DIR}")

    yield  # app runs here


# ── FastAPI app ────────────────────────────────────────────────────

app = FastAPI(
    title="HR Policy Q&A Bot",
    description=(
        "RAG-powered backend for UK HR policy documents. "
        "Upload policy PDFs/DOCX, then ask natural language questions. "
        "Every answer includes a source citation from the uploaded documents."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict to Vercel domain in v2.0
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /ingest ───────────────────────────────────────────────────

@app.post("/ingest", summary="Upload and ingest HR policy documents")
async def ingest_documents(
    files: list[UploadFile] = File(..., description="One or more policy documents (PDF, DOCX, TXT, MD)"),
    api_key: Optional[str] = Form(None, description="Optional OpenAI API key override"),
):
    """
    Upload one or more HR policy documents.
    Each file is parsed, chunked (600 chars), embedded via OpenAI,
    and added to the session's Qdrant + BM25 store.

    Returns a summary of what was ingested and any errors.
    """
    results = []
    errors = []

    for upload in files:
        filename = upload.filename or "upload"
        suffix = pathlib.Path(filename).suffix.lower()

        # Reject unsupported types early
        if suffix not in SUPPORTED_EXTENSIONS:
            errors.append({
                "filename": filename,
                "error": (
                    f"Unsupported file type '{suffix}'. "
                    f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            })
            continue

        # Write to a secure temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = ingest_file(tmp_path, api_key=api_key)
            result["filename"] = filename  # restore original name
            results.append(result)
        except EnvironmentError as e:
            # Bad API key or missing env var — surface clearly
            raise HTTPException(status_code=401, detail=str(e))
        except ValueError as e:
            errors.append({"filename": filename, "error": str(e)})
        except Exception as e:
            errors.append({"filename": filename, "error": f"Ingestion failed: {e}"})
        finally:
            pathlib.Path(tmp_path).unlink(missing_ok=True)

    total_chunks = sum(r.get("chunks_added", 0) for r in results)
    return JSONResponse(
        status_code=200,
        content={
            "status":             "ok" if not errors else "partial",
            "ingested":           results,
            "errors":             errors,
            "total_chunks_added": total_chunks,
        },
    )


# ── GET /docs-list ─────────────────────────────────────────────────

@app.get("/docs-list", summary="List documents in the session store")
async def list_documents():
    """
    Return all documents currently ingested in this session's vector store,
    with their chunk counts and ingestion timestamps.
    """
    docs = get_ingested_docs()
    return JSONResponse({
        "docs":  docs,
        "total": len(docs),
    })


# ── DELETE /docs ───────────────────────────────────────────────────

@app.delete("/docs", summary="Remove a document from the session store")
async def remove_document(filename: str):
    """
    Remove all chunks for the specified filename from both
    the Qdrant vector store and the BM25 index.
    """
    try:
        result = delete_doc(filename)
        return JSONResponse({"status": "ok", **result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /health ────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
async def health():
    """
    Lightweight endpoint for container health checks (Hugging Face Spaces)
    and frontend connectivity verification.
    """
    docs = get_ingested_docs()
    return {
        "status":      "healthy",
        "service":     "HR Policy Q&A Bot",
        "version":     "0.1.0",
        "docs_loaded": len(docs),
    }


# ── POST /query ────────────────────────────────────────────────────

@app.post("/query", summary="Ask an HR policy question", response_model=QueryResponse)
async def query_hr_bot(request: QueryRequest):
    """
    Full RAG pipeline with safety filters and auditing.
    """
    import time
    start_time = time.perf_counter()

    # 1. Guardrails (Safety Check)
    guard = classify_query(request.query)
    
    if guard["status"] != "PASS":
        latency = (time.perf_counter() - start_time) * 1000
        log_interaction(
            query=request.query,
            answer=guard["message"],
            blocked=(guard["status"] == "BLOCK"),
            block_reason=guard["reason"],
            escalated=(guard["status"] == "ESCALATE"),
            latency_ms=latency
        )
        return QueryResponse(
            answer=guard["message"],
            sources=[],
            llm_used="none",
            success=False,
            status=guard["status"],
            latency_ms=latency
        )

    # 2. Embed Query (OpenAI)
    try:
        openai_client = get_openai(api_key=request.openai_api_key)
        embed_resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[request.query],
            dimensions=768
        )
        query_vector = embed_resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Embedding error: {str(e)}")

    # 3. Hybrid Retrieve (Qdrant + BM25 + RRF + Rerank)
    try:
        retrieved_chunks = retrieve(
            query_text=request.query,
            query_vector=query_vector,
            top_k=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

    # 4. Generate Answer (Selected LLM)
    gen_result = generate_answer(
        query=request.query,
        retrieved_chunks=retrieved_chunks,
        model_alias=request.llm_provider,
        api_key=request.provider_api_key or request.openai_api_key
    )

    latency = (time.perf_counter() - start_time) * 1000

    # 5. Log Result
    log_interaction(
        query=request.query,
        answer=gen_result.get("answer", ""),
        sources=retrieved_chunks,
        llm_used=gen_result.get("model_used", "none"),
        blocked=False,
        escalated=False,
        latency_ms=latency
    )

    if not gen_result.get("success"):
        return QueryResponse(
            answer=gen_result.get("answer", "Unknown generation error."),
            sources=[],
            llm_used="error",
            success=False,
            status="PASS",
            latency_ms=latency
        )

    return QueryResponse(
        answer=gen_result["answer"],
        sources=[c["metadata"] for c in retrieved_chunks],
        llm_used=gen_result["model_used"],
        success=True,
        status="PASS",
        latency_ms=latency
    )


# ── GET /logs ──────────────────────────────────────────────────────

@app.get("/logs", summary="Download query audit log CSV")
async def get_audit_logs():
    """
    Returns the session_audit.csv file for download.
    Provides transparency for HR managers into bot performance.
    """
    log_path = get_log_file_path()
    if not os.path.exists(log_path):
        return JSONResponse(
            status_code=404,
            content={"detail": "Audit log not yet created. Perform a query first."}
        )
    return FileResponse(
        path=log_path,
        filename="hr_bot_audit_log.csv",
        media_type="text/csv"
    )
