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
import time
import uuid
from typing import Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from hr_doc_loader import load_document

# ── Config ────────────────────────────────────────────────────────
COLLECTION_NAME = "hr_docs"
EMBED_MODEL     = "text-embedding-3-small"
EMBED_DIMS      = 768
CHUNK_SIZE      = 600   # characters
CHUNK_OVERLAP   = 80
BATCH_SIZE      = 20    # OpenAI embedding batch size

# ── Module-level singletons ────────────────────────────────────────
_qdrant_client: Optional[QdrantClient] = None
_openai_client: Optional[OpenAI] = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_corpus: list[dict] = []   # [{text, metadata}] — parallel to BM25 tokenised docs


# ── Initialisation helpers ─────────────────────────────────────────

def get_qdrant() -> QdrantClient:
    """Return (and lazily initialise) the Qdrant in-memory client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
        _qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIMS, distance=Distance.COSINE),
        )
    return _qdrant_client


def get_openai(api_key: str | None = None) -> OpenAI:
    """Return (and lazily initialise) the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if not api_key:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Pass it via the api_key parameter or set it in .env."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_bm25() -> tuple[Optional[BM25Okapi], list[dict]]:
    """Return the current BM25 index and its parallel corpus list."""
    return _bm25_index, _bm25_corpus


# ── Core pipeline functions ────────────────────────────────────────

def chunk_records(records: list[dict]) -> list[dict]:
    """
    Split page/section records into overlapping chunks.
    Metadata is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for rec in records:
        texts = splitter.split_text(rec["page_content"])
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "metadata": {
                    **rec["metadata"],
                    "chunk_index": i,
                },
            })
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Batch-embed a list of strings using OpenAI text-embedding-3-small.
    Respects rate limits with a brief pause between batches.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIMS,
        )
        # Preserve API ordering
        batch_embeds: list[Optional[list[float]]] = [None] * len(batch)
        for item in response.data:
            batch_embeds[item.index] = item.embedding
        all_embeddings.extend(batch_embeds)

        if i + BATCH_SIZE < len(texts):
            time.sleep(0.3)  # brief pause to stay within rate limits

    return all_embeddings


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

def ingest_file(file_path: str, api_key: str | None = None) -> dict:
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
    # 1. Parse
    records = load_document(file_path)
    if not records:
        return {
            "filename":     file_path,
            "doc_title":    "Unknown",
            "chunks_added": 0,
            "warning":      "No text could be extracted from this file.",
        }

    # 2. Chunk
    chunks = chunk_records(records)
    if not chunks:
        return {
            "filename":     file_path,
            "doc_title":    records[0]["metadata"]["doc_title"],
            "chunks_added": 0,
            "warning":      "File produced no usable chunks after splitting.",
        }

    # 3. Embed
    client = get_openai(api_key)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(client, texts)

    # 4. Upsert to Qdrant
    qdrant = get_qdrant()
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            continue  # skip any failed embeddings
        m = chunk["metadata"]
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "chunk_text":      chunk["text"],
                    "doc_title":       m["doc_title"],
                    "doc_type":        m["doc_type"],
                    "department":      m["department"],
                    "section_heading": m["section_heading"],
                    "page_number":     m["page_number"],
                    "source_filename": m["source_filename"],
                    "ingested_at":     m["ingested_at"],
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
        "filename":     chunks[0]["metadata"]["source_filename"],
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
