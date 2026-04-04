"""
retriever.py — HR Hybrid RAG Retriever
=======================================
Combines:
  1. Dense vector search (Qdrant)
  2. Sparse keyword search (BM25)
  3. Reciprocal Rank Fusion (RRF, k=60)
  4. Cross-Encoder Reranking (MiniLM-L6)

Ensures that technical HR terms (SSP, TUPE) are caught by keywords,
while broad topics are caught by semantic similarity.
"""

import os
from typing import Optional

from sentence_transformers import CrossEncoder
import torch

from hr_ingest import COLLECTION_NAME, get_bm25, get_qdrant

# ── Config ────────────────────────────────────────────────────────
RRF_K = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FUSE_TOP_K = 10   # Candidates before reranking

_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    """Lazy-load the reranker model into memory."""
    global _reranker
    if _reranker is None:
        # This will download the ~90MB model on first call if not cached.
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _rrf_score(ranks: list[int]) -> float:
    """
    Calculate the Reciprocal Rank Fusion score for a document.
    ranks: list of ranks (1-based) across different search systems.
    """
    return sum(1.0 / (RRF_K + r) for r in ranks)


def retrieve(query_text: str, query_vector: list[float], top_k: int = 3) -> dict:
    """
    Main retrieval entry point.
    1. Search Dense (Qdrant)
    2. Search Sparse (BM25)
    3. Fuse (RRF)
    4. Rerank (Cross-Encoder)
    """
    qdrant = get_qdrant()
    bm25_index, bm25_corpus = get_bm25()

    if not bm25_corpus:
         # store is empty
        return {"chunks": [], "confidence_score": 0.0}

    # ── 1. Dense Search (top-10) ──────────────────────────────────
    # Note: Using query_points for better compatibility with Python 3.14+
    search_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=FUSE_TOP_K,
        with_payload=True
    )
    dense_results = search_result.points
    # Map by text to avoid duplicates in fusion
    dense_map = {res.payload["chunk_text"]: (idx + 1) for idx, res in enumerate(dense_results)}
    payload_map = {res.payload["chunk_text"]: res.payload for res in dense_results}

    # ── 2. Sparse Search (top-10) ──────────────────────────────────
    sparse_map = {}
    if bm25_index:
        tokenized_query = query_text.lower().split()
        scores = bm25_index.get_scores(tokenized_query)
        # Get indices of top scores
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:FUSE_TOP_K]

        for rank_idx, doc_idx in enumerate(top_indices):
            # Only count if score is > 0 (bm25 often gives 0 for non-matches)
            if scores[doc_idx] > 0:
                doc_text = bm25_corpus[doc_idx]["text"]
                sparse_map[doc_text] = rank_idx + 1
                # Ensure we have the payload for BM25-only matches
                if doc_text not in payload_map:
                    # BM25 corpus was built from chunks, lets store full metadata in it too
                    payload_map[doc_text] = bm25_corpus[doc_idx]["metadata"]
                    # Map 'text' back to 'chunk_text' for schema consistency
                    payload_map[doc_text]["chunk_text"] = doc_text

    # ── 3. RRF Fusion ──────────────────────────────────────────────
    unique_texts = set(dense_map.keys()) | set(sparse_map.keys())
    fused_results = []

    for text in unique_texts:
        ranks = []
        if text in dense_map: ranks.append(dense_map[text])
        if text in sparse_map: ranks.append(sparse_map[text])

        score = _rrf_score(ranks)
        fused_results.append({
            "text": text,
            "rrf_score": score,
            "payload": payload_map[text]
        })

    # Sort and take top candidates
    fused_results = sorted(fused_results, key=lambda x: x["rrf_score"], reverse=True)[:FUSE_TOP_K]

    if not fused_results:
        return {"chunks": [], "confidence_score": 0.0}

    # ── 4. Cross-Encoder Reranking ──────────────────────────────────
    reranker = _get_reranker()
    candidate_texts = [res["text"] for res in fused_results]

    # Model scores (query, chunk) pairs
    scores = reranker.predict([(query_text, t) for t in candidate_texts])

    # Combine and sort by model score
    for i, score in enumerate(scores):
        fused_results[i]["rerank_score"] = float(score)

    final_results = sorted(fused_results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    # ── 5. Confidence Score ─────────────────────────────────────────
    # The top chunk's rerank score is our raw confidence logit.
    # We pass it through a sigmoid to get a 0-1 probability.
    top_score = final_results[0]["rerank_score"] if final_results else 0.0
    confidence_score = torch.sigmoid(torch.tensor(top_score)).item()

    # Format for downstream
    return {
        "chunks": [
            {
                "text": res["text"],
                "metadata": res["payload"],
                "score": res["rerank_score"]
            }
            for res in final_results
        ],
        "confidence_score": confidence_score
    }
