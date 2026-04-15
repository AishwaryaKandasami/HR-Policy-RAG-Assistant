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

import json
import os
import pathlib
import tempfile
import uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from audit_log import get_log_file_path, log_interaction, log_feedback
from generator import generate_answer
from hr_guardrails import classify_query
from hr_ingest import delete_doc, get_embed_model, get_ingested_docs, get_openai, get_qdrant, ingest_file
from intent_router import classify_intent
from retriever import retrieve


# ── Data Models ───────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message text")


class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the maternity leave entitlement?")
    llm_provider: str = Field("groq_llama_8b", description="Model alias (groq_llama_8b, gemini_flash, etc.)")
    # Keys are now optional; server will use its own secrets if these are None
    openai_api_key: Optional[str] = Field(None, description="Optional override for embedding")
    provider_api_key: Optional[str] = Field(None, description="Optional override for LLM provider")
    # Session memory (client-driven — frontend sends last N turns per request)
    session_id: Optional[str] = Field(None, description="Client-generated UUID for audit correlation")
    conversation_history: Optional[list[ConversationTurn]] = Field(
        None, description="Prior turns in this conversation (last ~6), for multi-turn coherence"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    llm_used: str
    success: bool
    status: str = "PASS"   # PASS, BLOCK, ESCALATE
    confidence_score: float = 0.0
    confidence_label: str = "N/A"
    latency_ms: Optional[float] = None
    query_id: str = "none"


class FeedbackRequest(BaseModel):
    query_id: str
    rating: str = Field(..., pattern="^(up|down)$")
    reason: Optional[str] = Field(None, description="One of: Wrong answer, Incomplete, Not what I meant")


# ── Startup: pre-load demo docs ────────────────────────────────────

# Support both local dev (../demo_docs) and Docker/HF Space (./demo_docs)
DEMO_DOCS_DIR = pathlib.Path(__file__).parent / "demo_docs"
if not DEMO_DOCS_DIR.exists():
    DEMO_DOCS_DIR = pathlib.Path(__file__).parent.parent / "demo_docs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_DIMS = 384  # Match all-MiniLM-L6-v2

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup: 
    1. Sync the BM25 index from Qdrant Cloud (persistence).
    2. Ingest demo PDFs if this is a fresh setup.
    """
    from hr_ingest import sync_bm25_from_cloud
    
    # 1. Persistence Sync
    try:
        sync_bm25_from_cloud()
    except Exception as e:
        print(f"⚠️  Cloud sync failed: {e}. Starting with empty session.")

    # 2. Pre-load demo docs ONLY if collection is empty
    if DEMO_DOCS_DIR.exists():
        from hr_ingest import check_doc_exists
        # We check if there's *any* document in the store. 
        # get_ingested_docs() returns a list of all docs.
        existing_docs = get_ingested_docs()
        
        if not existing_docs:
            demo_files = sorted(DEMO_DOCS_DIR.glob("*"))
            eligible = [f for f in demo_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]
            if eligible:
                print(f"\n📂 Collection empty. Pre-loading {len(eligible)} demo documents...")
                for doc_path in eligible:
                    try:
                        result = ingest_file(str(doc_path), original_filename=doc_path.name)
                        print(f"  ✅ {result['doc_title']} — {result['chunks_added']} chunks")
                    except Exception as e:
                        print(f"  ⚠  Could not pre-load {doc_path.name}: {e}")
                print()
        else:
            print(f"ℹ  Collection already contains {len(existing_docs)} docs. Skipping demo pre-load.")
    else:
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

# ALLOWED_ORIGINS: comma-separated list set in HF Spaces → Variables & Secrets.
# e.g. "https://your-app.vercel.app,https://your-app-git-main.vercel.app"
# Falls back to "*" when unset (local dev only).
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
            result = ingest_file(tmp_path, api_key=api_key, original_filename=filename)
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
    query_id = str(uuid.uuid4())

    # 1a. Intent router — short-circuit small-talk before any RAG work
    intent = classify_intent(request.query)
    if intent["short_circuit"]:
        latency = (time.perf_counter() - start_time) * 1000
        log_interaction(
            query_id=query_id,
            query=request.query,
            answer=intent["response"],
            blocked=False,
            escalated=False,
            latency_ms=latency,
            session_id=request.session_id,
        )
        return QueryResponse(
            answer=intent["response"],
            sources=[],
            llm_used="none",
            success=True,
            status="PASS",
            confidence_score=1.0,
            confidence_label="N/A",
            latency_ms=latency,
            query_id=query_id,
        )

    # 1b. Guardrails (Safety Check)
    guard = classify_query(request.query)

    if guard["status"] != "PASS":
        latency = (time.perf_counter() - start_time) * 1000
        log_interaction(
            query_id=query_id,
            query=request.query,
            answer=guard["message"],
            blocked=(guard["status"] == "BLOCK"),
            block_reason=guard["reason"],
            escalated=(guard["status"] == "ESCALATE"),
            latency_ms=latency,
            session_id=request.session_id,
        )
        return QueryResponse(
            answer=guard["message"],
            sources=[],
            llm_used="none",
            success=False,
            status=guard["status"],
            confidence_score=0.0,
            confidence_label="N/A",
            latency_ms=latency,
            query_id=query_id
        )

    # ── Self-Correction Loop (Max 2 Attempts) ──────────────────────
    from generator import judge_answer, rewrite_query
    
    current_human_query = request.query
    current_search_query = request.query
    attempts_data = []
    final_gen_result = None
    final_retrieved_chunks = []
    final_confidence_score = 0.0

    for i in range(1, 3):
        print(f"🔍 Attempt {i} | Search Query: '{current_search_query}'")
        
        # 2. Embed Search Query
        try:
            embed_model = get_embed_model()
            query_vector = embed_model.encode(current_search_query).tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Embedding error: {str(e)}")

        # 3. Hybrid Retrieve
        try:
            retrieval_data = retrieve(
                query_text=current_search_query,
                query_vector=query_vector,
                top_k=3
            )
            retrieved_chunks = retrieval_data["chunks"]
            confidence_score = retrieval_data["confidence_score"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

        if not retrieved_chunks:
            print(f"  ⚠️ No chunks found for '{current_search_query}'")
            if i == 1:
                history = [t.model_dump() for t in request.conversation_history] if request.conversation_history else []
                current_search_query = rewrite_query(current_human_query, conversation_history=history)
                continue
            else:
                final_gen_result = {"answer": "I'm sorry, I couldn't find any documents related to your question.", "model_used": "none"}
                break

        # 4. Generate Answer
        history = [t.model_dump() for t in request.conversation_history] if request.conversation_history else []
        gen_result = generate_answer(
            query=current_human_query,
            retrieved_chunks=retrieved_chunks,
            model_alias=request.llm_provider,
            api_key=request.provider_api_key or request.openai_api_key,
            confidence_score=confidence_score,
            conversation_history=history,
        )
        
        # 5. Judge Answer — skip when retrieval confidence is already high
        # (reranker score > 0.75 means the top chunk is a strong match;
        # running a second LLM call adds latency with negligible safety gain)
        if confidence_score > 0.75:
            print(f"  ⚡ High confidence ({confidence_score:.2f}) — skipping judge.")
            is_pass, judge_reason = True, "Skipped: high retrieval confidence"
        else:
            is_pass, judge_reason = judge_answer(current_human_query, gen_result["answer"], retrieved_chunks)
        
        # Update trackers for possible final output
        final_gen_result = gen_result
        final_retrieved_chunks = retrieved_chunks
        final_confidence_score = confidence_score

        if is_pass:
            print(f"  ✅ Attempt {i} PASSED judgment.")
            break
        else:
            print(f"  ❌ Attempt {i} FAILED judgment: {judge_reason}")
            if i == 1:
                # Expand query for more context
                history = [t.model_dump() for t in request.conversation_history] if request.conversation_history else []
                current_search_query = rewrite_query(current_human_query, conversation_history=history)
            else:
                # Final refusal logic
                final_gen_result["answer"] = (
                    "I found some related information, but I couldn't verify it with enough "
                    "certainty to provide a reliable HR answer. Please contact HR directly. "
                    f"(Reason: {judge_reason})"
                )
                break

    latency = (time.perf_counter() - start_time) * 1000

    # 6. Log Result (using the original user query)
    log_interaction(
        query_id=query_id,
        query=request.query,
        answer=final_gen_result.get("answer", ""),
        sources=final_retrieved_chunks,
        llm_used=final_gen_result.get("model_used", "none"),
        blocked=False,
        escalated=False,
        latency_ms=latency,
        session_id=request.session_id,
    )

    # 7. Determine Confidence Label
    if final_confidence_score > 0.7:
        confidence_label = "High"
    elif final_confidence_score >= 0.4:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Action 3: GDPR Confidence Override
    # Reranker often scores DPA 2018 low for "GDPR" queries.
    q_lower = request.query.lower()
    a_lower = final_gen_result["answer"].lower()
    if "gdpr" in q_lower and "data protection act" in a_lower:
        print("💡 GDPR Confidence Override: Recalibrating to Medium.")
        confidence_label = "Medium"

    # Deduplicate sources: same filename + page + section can appear multiple
    # times because overlapping chunks from the same page all score highly.
    seen_sources: set[tuple] = set()
    unique_sources: list[dict] = []
    for chunk in final_retrieved_chunks:
        md = chunk["metadata"]
        key = (
            md.get("source_filename", ""),
            md.get("page_number", ""),
            md.get("section_heading", ""),
        )
        if key not in seen_sources:
            seen_sources.add(key)
            unique_sources.append(md)

    return QueryResponse(
        answer=final_gen_result["answer"],
        sources=unique_sources,
        llm_used=final_gen_result.get("model_used", "none"),
        success=True,
        status="PASS",
        confidence_score=final_confidence_score,
        confidence_label=confidence_label,
        latency_ms=latency,
        query_id=query_id
    )


@app.post("/feedback", summary="Submit feedback for a previous query")
async def submit_feedback(request: FeedbackRequest):
    """
    Associate a thumbs-up (up) or thumbs-down (down) with a query ID.
    Optional reasons for thumbs-down: 'Wrong answer', 'Incomplete', 'Not what I meant'.
    """
    success = log_feedback(
        query_id=request.query_id, 
        rating=request.rating, 
        reason=request.reason
    )
    
    if not success:
        raise HTTPException(
            status_code=404, 
            detail=f"Query ID {request.query_id} not found in this session."
        )
        
    return {"status": "ok", "message": "Feedback recorded."}


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

# ── GET /onboarding-checklist ──────────────────────────────────────

@app.get("/onboarding-checklist", summary="Get onboarding checklist")
async def get_onboarding_checklist():
    """
    Returns the onboarding checklist tasks grouped by phase.
    """
    checklist_path = pathlib.Path(__file__).parent / "onboarding_checklist.json"
    if not checklist_path.exists():
        raise HTTPException(status_code=404, detail="Checklist not found.")
    
    with open(checklist_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return JSONResponse(content=data)
