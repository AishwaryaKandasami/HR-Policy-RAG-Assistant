"""
retriever.py — RAG Retrieval Layer
====================================
Purpose : Embed a user query, search the Qdrant vector store for
          the most relevant chunks, optionally filter by scheme or
          topic, and re-rank using a cross-encoder.

Inputs  : User query string, optional scheme/topic filters
Outputs : List of ranked result dicts ready for the generator
Env vars: OPENAI_API_KEY
"""

import json
import os
import pathlib

from dotenv import load_dotenv
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
from sentence_transformers import CrossEncoder

# ── Config ────────────────────────────────────────────────────────
COLLECTION_NAME   = "mf_faq"
EMBED_MODEL       = "text-embedding-3-small"
EMBED_DIMS        = 768
VECTOR_STORE_DIR  = "vector_store"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singletons (initialised lazily)
_openai_client: OpenAI | None = None
_qdrant_client: QdrantClient | None = None
_reranker: CrossEncoder | None = None


# ── Initialisation ────────────────────────────────────────────────

def _get_openai(api_key: str | None = None) -> OpenAI:
    """Return (and cache) an OpenAI client."""
    global _openai_client
    if _openai_client is None:
        if not api_key:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_qdrant() -> QdrantClient:
    """Return (and cache) a Qdrant client, rebuilding from the
    persisted JSON fallback file if needed."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    fallback_path = pathlib.Path(VECTOR_STORE_DIR) / f"{COLLECTION_NAME}.json"
    snapshot_path = pathlib.Path(VECTOR_STORE_DIR) / f"{COLLECTION_NAME}.snapshot"

    # Prefer snapshot if it exists
    if snapshot_path.exists():
        _qdrant_client = QdrantClient(":memory:")
        # Attempt to restore from snapshot
        try:
            _qdrant_client.recover_snapshot(
                collection_name=COLLECTION_NAME,
                location=str(snapshot_path),
            )
            return _qdrant_client
        except Exception:
            pass  # fall through to JSON fallback

    # Fallback: rebuild from JSON
    if fallback_path.exists():
        _qdrant_client = _rebuild_from_json(fallback_path)
        return _qdrant_client

    raise FileNotFoundError(
        f"No vector store found. Run ingest.py first to create "
        f"{snapshot_path} or {fallback_path}."
    )


def _rebuild_from_json(json_path: pathlib.Path) -> QdrantClient:
    """Rebuild the Qdrant in-memory collection from the JSON fallback."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIMS, distance=Distance.COSINE),
    )

    points = []
    for item in data:
        points.append(
            PointStruct(
                id=item["id"],
                vector=item["embedding"],
                payload={
                    "chunk_text":       item["text"],
                    "source_url":       item["metadata"]["source_url"],
                    "scheme":           item["metadata"]["scheme"],
                    "topic":            item["metadata"]["topic"],
                    "plan":             item["metadata"]["plan"],
                    "verbatim_snippet": item["metadata"]["verbatim_snippet"],
                    "date_fetched":     item["metadata"]["date_fetched"],
                },
            )
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def _get_reranker() -> CrossEncoder:
    """Return (and cache) the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


# ── Core retrieval function ───────────────────────────────────────

def retrieve(
    query: str,
    scheme_filter: str | None = None,
    topic_filter: str | None = None,
    top_k: int = 5,
    api_key: str | None = None,
) -> list[dict]:
    """Retrieve the most relevant knowledge-base chunks for a query.

    Args:
        query:         Natural language question from the user.
        scheme_filter: Optional — restrict to a specific scheme.
                       Allowed: "SBI Large Cap", "SBI Flexi Cap",
                       "SBI ELSS", "General".
        topic_filter:  Optional — restrict to a specific topic.
                       Allowed: expense_ratio, exit_load, min_sip,
                       lock_in, riskometer, benchmark,
                       statement_download, scheme_category.
        top_k:         Number of ANN results to fetch before reranking.
        api_key:       Optional OpenAI API key.

    Returns:
        List of dicts (max 3 after reranking), each containing:
        chunk_text, source_url, scheme, topic, date_fetched,
        rerank_score.
    """
    openai_client = _get_openai(api_key)
    qdrant_client = _get_qdrant()
    reranker      = _get_reranker()

    # ── Step 1: Embed the query ──────────────────────────────────
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        dimensions=EMBED_DIMS,
    )
    query_vector = response.data[0].embedding

    # ── Step 2 & 3: Build payload filters ────────────────────────
    conditions = []
    if scheme_filter:
        conditions.append(
            FieldCondition(key="scheme", match=MatchValue(value=scheme_filter))
        )
    if topic_filter:
        conditions.append(
            FieldCondition(key="topic", match=MatchValue(value=topic_filter))
        )

    search_filter = Filter(must=conditions) if conditions else None

    # ── Step 4: ANN search (qdrant-client ≥1.12 API) ───────────────
    query_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    results = query_response.points
    if not results:
        return []

    # ── Step 5: Re-rank with cross-encoder ───────────────────────
    # Prepare (query, passage) pairs for the reranker
    pairs = [(query, hit.payload["chunk_text"]) for hit in results]
    rerank_scores = reranker.predict(pairs)

    # Combine results with rerank scores and sort descending
    scored = []
    for hit, score in zip(results, rerank_scores):
        scored.append({
            "chunk_text":    hit.payload["chunk_text"],
            "source_url":    hit.payload["source_url"],
            "scheme":        hit.payload["scheme"],
            "topic":         hit.payload["topic"],
            "date_fetched":  hit.payload["date_fetched"],
            "rerank_score":  float(score),
        })

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Return top 3 after reranking
    return scored[:3]
