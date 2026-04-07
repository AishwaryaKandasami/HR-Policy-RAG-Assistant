"""
hr_ingest.py — HR RAG Ingestion Pipeline
==========================================
Loads HR documents, chunks them, embeds with OpenAI text-embedding-3-small,
and upserts to:
  1. Qdrant in-memory collection  (dense vector search)
  2. BM25Okapi in-memory index    (sparse keyword search)

Both stores are module-level singletons shared across the FastAPI app lifetime.
The BM25 index is rebuilt from scratch whenever a document is added or deleted.
"""

import os
import pathlib
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from hr_doc_loader import load_document_to_markdown, _infer_doc_title, _infer_doc_type

# ── Config ────────────────────────────────────────────────────────
COLLECTION_NAME = "hr_docs"
EMBED_MODEL     = "all-MiniLM-L6-v2"
EMBED_DIMS      = 384
CHUNK_SIZE      = 800   # slightly larger for context
CHUNK_OVERLAP   = 100
BATCH_SIZE      = 32    # Local batch size

# ── Module-level singletons ────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_embed_model: Optional[SentenceTransformer] = None
_openai_client: Optional[OpenAI] = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: list[dict] = []   # [{text, metadata}] — parallel to BM25 tokenised docs


# ── Initialisation helpers ─────────────────────────────────────────

def get_qdrant() -> QdrantClient:
    """Return (and lazily initialise) the Qdrant Cloud or in-memory client."""
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if url and api_key:
            print(f"📡 Connecting to Qdrant Cloud at {url}...")
            _qdrant_client = QdrantClient(url=url, api_key=api_key)
        else:
            print("💡 QDRANT_URL/API_KEY not found. Falling back to in-memory mode (session-scoped).")
            _qdrant_client = QdrantClient(":memory:")

        # Ensure collection exists
        try:
            _qdrant_client.get_collection(COLLECTION_NAME)
        except Exception:
            print(f"🛠️ Creating collection '{COLLECTION_NAME}'...")
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIMS, distance=Distance.COSINE),
            )
    return _qdrant_client


def get_embed_model() -> SentenceTransformer:
    """Return (and lazily initialise) the local SentenceTransformer model."""
    global _embed_model
    if _embed_model is None:
        print(f"📥 Loading local embedding model ({EMBED_MODEL})...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_openai(api_key: str | None = None) -> OpenAI:
    """Return (and lazily initialise) the OpenAI client (for LLM only)."""
    global _openai_client
    if _openai_client is None:
        if not api_key:
            load_dotenv(pathlib.Path(__file__).parent.parent / ".env")
            api_key = os.getenv("OPENAI_API_KEY")
        # Allow running without OpenAI key if only using Groq/Gemini
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_bm25() -> tuple[Optional[BM25Okapi], list[dict]]:
    """Return the current BM25 index and its parallel corpus list."""
    return _bm25_index, _bm25_corpus


def sync_bm25_from_cloud() -> int:
    """
    On startup: scroll through all points in Qdrant Cloud to 
    populate the in-memory BM25 index. This ensures persistence without 
    storing raw local files.
    """
    global _bm25_corpus
    qdrant = get_qdrant()
    
    print("🔄 Syncing BM25 index from Qdrant Cloud...")
    
    # Scroll through all points
    all_chunks = []
    offset = None
    
    while True:
        points, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset
        )
        
        for p in points:
            txt = p.payload.get("chunk_text") or p.payload.get("text", "")
            all_chunks.append({
                "text": txt,
                "metadata": {k: v for k, v in p.payload.items() if k not in ["chunk_text", "text"]}
            })
            
        if offset is None:
            break
            
    if all_chunks:
        _rebuild_bm25(all_chunks)
        print(f"  ✅ Synced {len(all_chunks)} chunks into BM25 index.")
    else:
        print("  ℹ️ No documents found in cloud storage.")
        
    return len(all_chunks)


def check_doc_exists(filename: str) -> bool:
    """
    Check if any chunks for this filename already exist in Qdrant.
    Used to skip re-ingestion of demo docs on every restart.
    """
    qdrant = get_qdrant()
    # Scroll with a limit of 1 to see if at least one point exists
    res, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_filename", match=MatchValue(value=filename))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False
    )
    return len(res) > 0


# ── Core pipeline functions ────────────────────────────────────────

def chunk_markdown(md_text: str, metadata_base: dict) -> list[dict]:
    """
    Split a structured Markdown string into chunks based on headers,
    then sub-split long sections by character count.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # 1. Split by headers
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = md_splitter.split_text(md_text)
    
    # 2. Sub-split long sections
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    final_chunks = []
    for sec in sections:
        # Merge section headers into a single section heading string
        section_heading = " > ".join([
            sec.metadata.get(h[1], "") for h in headers_to_split_on if h[1] in sec.metadata
        ]) or "General"
        
        texts = char_splitter.split_text(sec.page_content)
        for i, text in enumerate(texts):
            final_chunks.append({
                "text": text,
                "metadata": {
                    **metadata_base,
                    "section_heading": section_heading,
                    "chunk_index": i
                }
            })
    return final_chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of strings locally using all-MiniLM-L6-v2.
    """
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    return [vec.tolist() for vec in embeddings]


def _rebuild_bm25(corpus: list[dict]) -> None:
    """
    (Re)build the BM25 index from a full corpus list.
    Called after every ingest or delete operation.
    """
    global _bm25_index, _bm25_corpus
    _bm25_corpus = corpus
    if corpus:
        tokenised = [doc["text"].lower().split() for doc in corpus]
        _bm25_index = BM25Okapi(tokenised)
    else:
        _bm25_index = None


# ── Public ingestion API ───────────────────────────────────────────

def ingest_file(file_path: str, api_key: str | None = None, original_filename: str | None = None) -> dict:
    """
    Full ingestion pipeline for a single file:
      1. Parse (PDF / DOCX / TXT)
      2. Chunk (RecursiveCharacterTextSplitter, 600 chars)
      3. Embed (OpenAI text-embedding-3-small)
      4. Upsert to Qdrant (dense vectors + payload)
      5. Update BM25 index (sparse keyword index)

    Returns:
      {
        "filename":     str,
        "doc_title":    str,
        "chunks_added": int,
      }
    """
    # 0. Check for duplicates
    display_filename = original_filename if original_filename else pathlib.Path(file_path).name
    if check_doc_exists(display_filename):
        print(f"  ⏭️  Skipping {display_filename} (already in cloud store).")
        return {
            "filename":     display_filename,
            "doc_title":    "Already Ingested",
            "chunks_added": 0,
            "skipped":      True
        }

    # 1. Parse to Markdown
    md_text = load_document_to_markdown(file_path)
    if not md_text or len(md_text.strip()) < 20:
        return {
            "filename":     display_filename,
            "doc_title":    "Empty",
            "chunks_added": 0,
            "warning":      "No text could be extracted.",
        }

    # 2. Resolve metadata early to avoid scope conflicts
    display_title = _infer_doc_title(display_filename)
    display_type  = _infer_doc_type(display_filename)

    metadata_base = {
        "doc_title":       display_title,
        "doc_type":        display_type,
        "department":      "All",
        "ingested_at":     datetime.now(timezone.utc).isoformat(),
        "source_filename": display_filename
    }

    # 3. Chunk (Header-Aware)
    chunks = chunk_markdown(md_text, metadata_base)
    if not chunks:
        return {
            "filename":     display_filename,
            "doc_title":    display_title,
            "chunks_added": 0,
            "warning":      "No usable chunks produced.",
        }

    # 3. Embed (Locally)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # 4. Upsert to Qdrant
    qdrant = get_qdrant()
    points = []
    
    for c in chunks:
        c["metadata"]["source_filename"] = display_filename
        c["metadata"]["doc_title"]       = display_title
        
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            continue  # skip any failed embeddings
        m = chunk["metadata"]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chunk_text":      chunk.get("text", ""),
                    "doc_title":       m.get("doc_title", "Unknown"),
                    "doc_type":        m.get("doc_type", "document"),
                    "department":      m.get("department", "All"),
                    "section_heading": m.get("section_heading", "General"),
                    "page_number":     m.get("page_number", 1),
                    "source_filename": display_filename,
                    "ingested_at":     m.get("ingested_at", ""),
                },
            )
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    # 5. Update BM25 index
    new_corpus_entries = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
    updated_corpus = list(_bm25_corpus) + new_corpus_entries
    _rebuild_bm25(updated_corpus)

    return {
        "filename":     display_filename,
        "doc_title":    chunks[0]["metadata"]["doc_title"],
        "chunks_added": len(points),
    }


def get_ingested_docs() -> list[dict]:
    """
    Return a list of unique documents currently in the vector store,
    with chunk counts and ingestion timestamps.
    """
    qdrant = get_qdrant()
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[str, dict] = {}
    for point in results:
        fn = point.payload.get("source_filename", "unknown")
        if fn not in seen:
            seen[fn] = {
                "source_filename": fn,
                "doc_title":       point.payload.get("doc_title", fn),
                "doc_type":        point.payload.get("doc_type", "document"),
                "ingested_at":     point.payload.get("ingested_at", ""),
                "chunk_count":     0,
            }
        seen[fn]["chunk_count"] += 1

    return list(seen.values())


def delete_doc(filename: str) -> dict:
    """
    Remove all chunks for a given filename from Qdrant and rebuild BM25.

    Returns:
      { "removed_filename": str, "chunks_removed": int }
    """
    qdrant = get_qdrant()

    # Count before delete
    before, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_filename", match=MatchValue(value=filename))]
        ),
        limit=10_000,
        with_payload=False,
        with_vectors=False,
    )
    chunks_removed = len(before)

    # Delete from Qdrant
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source_filename", match=MatchValue(value=filename))]
        ),
    )

    # Rebuild BM25 without the deleted doc
    remaining = [
        d for d in _bm25_corpus
        if d["metadata"].get("source_filename") != filename
    ]
    _rebuild_bm25(remaining)

    return {
        "removed_filename": filename,
        "chunks_removed":   chunks_removed,
    }
