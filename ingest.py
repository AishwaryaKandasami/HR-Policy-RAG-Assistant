"""
ingest.py — RAG Ingestion Pipeline
====================================
Purpose : Load extracted_facts.json, chunk each fact, embed with
          OpenAI text-embedding-3-small (768 dims), and store in
          a Qdrant in-memory collection.  Snapshot the collection
          to disk so the Streamlit app can reload without re-embedding.

Inputs  : extracted_facts.json (in repo root)
Outputs : vector_store/  directory with Qdrant snapshot files
Env vars: OPENAI_API_KEY
"""

import json
import os
import time
import pathlib

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────
FACTS_PATH       = "extracted_facts.json"
COLLECTION_NAME  = "mf_faq"
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIMS       = 768
CHUNK_SIZE       = 400      # characters (RecursiveCharacterTextSplitter default unit)
CHUNK_OVERLAP    = 50
BATCH_SIZE       = 20       # embed in batches to stay within rate limits
VECTOR_STORE_DIR = "vector_store"

# ── Helpers ───────────────────────────────────────────────────────

def load_facts(path: str) -> list[dict]:
    """Load extracted_facts.json and flatten into per-fact records."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for doc in data:
        source_url   = doc["source_url"]
        scheme       = doc["scheme"]
        date_fetched = doc["date_fetched"]
        fetch_method = doc.get("fetch_method", "automated")

        for fact in doc["facts"]:
            records.append({
                "page_content": fact["fact"],
                "metadata": {
                    "source_url":       source_url,
                    "scheme":           scheme,
                    "topic":            fact["topic"],
                    "plan":             fact.get("plan", "N/A"),
                    "verbatim_snippet": fact.get("verbatim_snippet", ""),
                    "date_fetched":     date_fetched,
                    "fetch_method":     fetch_method,
                },
            })
    return records


def chunk_records(records: list[dict]) -> list[dict]:
    """Split long facts into smaller chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,          # character-level splitting
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for rec in records:
        texts = splitter.split_text(rec["page_content"])
        for text in texts:
            chunks.append({
                "text": text,
                "metadata": rec["metadata"],  # metadata stays with every chunk
            })
    return chunks


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Batch-embed texts using OpenAI text-embedding-3-small."""
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIMS,
        )
        # Preserve ordering returned by the API
        batch_embeds = [None] * len(batch)
        for item in response.data:
            batch_embeds[item.index] = item.embedding
        all_embeddings.extend(batch_embeds)

        # Brief pause between batches to respect rate limits
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.5)

    return all_embeddings


def build_collection(chunks: list[dict], embeddings: list[list[float]]) -> QdrantClient:
    """Create an in-memory Qdrant collection and upsert all points."""
    client = QdrantClient(":memory:")  # pure in-memory — no server needed

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBED_DIMS,
            distance=Distance.COSINE,
        ),
    )

    points = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "chunk_text":       chunk["text"],
                    "source_url":       chunk["metadata"]["source_url"],
                    "scheme":           chunk["metadata"]["scheme"],
                    "topic":            chunk["metadata"]["topic"],
                    "plan":             chunk["metadata"]["plan"],
                    "verbatim_snippet": chunk["metadata"]["verbatim_snippet"],
                    "date_fetched":     chunk["metadata"]["date_fetched"],
                },
            )
        )

    # Upsert in one batch (small count — fits in memory)
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return client


def persist_collection(client: QdrantClient, output_dir: str) -> str:
    """Snapshot the in-memory collection to disk for later reloading.

    Qdrant's in-memory client supports creating snapshots to a local
    directory.  The Streamlit app will re-load from this snapshot on
    each cold start, avoiding re-embedding.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Export the entire collection as a snapshot file
    snapshot_info = client.create_snapshot(collection_name=COLLECTION_NAME)
    snapshot_name = snapshot_info.name

    # The snapshot is stored in a temp location by Qdrant; copy it
    # to our output directory
    snapshot_path = client.get_snapshot(
        collection_name=COLLECTION_NAME,
        snapshot_name=snapshot_name,
    )

    # If get_snapshot returns bytes-like, write them out
    dest = out / f"{COLLECTION_NAME}.snapshot"
    if isinstance(snapshot_path, bytes):
        dest.write_bytes(snapshot_path)
    else:
        # Some versions return a path string
        import shutil
        shutil.copy2(str(snapshot_path), str(dest))

    return str(dest)


# ── Main ──────────────────────────────────────────────────────────

def main():
    load_dotenv()  # load .env file if present

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    openai_client = OpenAI(api_key=api_key)

    # Step 1 — Load facts
    print("Loading facts from", FACTS_PATH, "...")
    records = load_facts(FACTS_PATH)
    print(f"  → {len(records)} fact records loaded")

    # Step 2 — Chunk
    print("Chunking ...")
    chunks = chunk_records(records)
    print(f"  → {len(chunks)} chunks created")

    # Step 3 — Embed
    print(f"Embedding with {EMBED_MODEL} (batch size={BATCH_SIZE}) ...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(openai_client, texts)
    print(f"  → {len(embeddings)} embeddings generated")

    # Step 4 — Store in Qdrant
    print("Building Qdrant in-memory collection ...")
    qdrant = build_collection(chunks, embeddings)
    count = qdrant.count(collection_name=COLLECTION_NAME).count
    print(f"  → {count} points in collection '{COLLECTION_NAME}'")

    # Step 5 — Persist snapshot
    print("Persisting to disk ...")
    try:
        dest = persist_collection(qdrant, VECTOR_STORE_DIR)
        print(f"  → Snapshot saved: {dest}")
    except Exception as e:
        # Fallback: save raw data as JSON for Streamlit reload
        print(f"  ⚠ Snapshot failed ({e}). Saving raw vectors as JSON fallback ...")
        fallback_path = save_fallback_json(chunks, embeddings, VECTOR_STORE_DIR)
        print(f"  → Fallback saved: {fallback_path}")

    # Step 6 — Summary
    print("\n" + "=" * 50)
    print(f"Total facts loaded:    {len(records)}")
    print(f"Total chunks created:  {len(chunks)}")
    print(f"Collection:            {COLLECTION_NAME}")
    print(f"Embedding model:       {EMBED_MODEL}")
    print(f"Output:                {VECTOR_STORE_DIR}/")
    print("=" * 50)


def save_fallback_json(chunks: list[dict], embeddings: list[list[float]], output_dir: str) -> str:
    """Save chunks + embeddings as a JSON file for fallback reloading.

    If Qdrant snapshots are unavailable (e.g. in-memory-only mode
    without snapshot support), this JSON file can be loaded at
    Streamlit startup to rebuild the collection.
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{COLLECTION_NAME}.json"

    data = []
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        data.append({
            "id": idx,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": emb,
        })

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return str(dest)


if __name__ == "__main__":
    main()
