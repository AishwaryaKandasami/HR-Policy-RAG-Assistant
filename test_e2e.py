"""
End-to-end test: "What is the exit load on SBI Large Cap?"
Runs: guardrails -> retriever -> generator
"""
import json
from guardrails import classify_query
from retriever import retrieve
from generator import generate

QUERY = "What is the exit load on SBI Large Cap?"

print("=" * 60)
print(f"  QUERY: {QUERY}")
print("=" * 60)

# Step 1 — Guardrails
print("\n--- Step 1: Guardrails Classification ---")
classification = classify_query(QUERY)
print(json.dumps(classification, indent=2))

if classification["action"] != "retrieve":
    print(f"\nQuery blocked by guardrails: {classification['action']}")
    print(f"   Reason: {classification['reason']}")
    raise SystemExit(0)

print("  Query passed guardrails -> proceeding to retrieval")

# Step 2 — Retriever
print("\n--- Step 2: Retrieval (embed + Qdrant + rerank) ---")
chunks = retrieve(QUERY)
print(f"  Retrieved {len(chunks)} chunks:")
for i, c in enumerate(chunks, 1):
    print(f"    [{i}] score={c['rerank_score']:.4f}  scheme={c['scheme']}")
    print(f"        topic={c['topic']}  source={c['source_url'][:60]}...")
    print(f"        text={c['chunk_text'][:120]}...")

# Step 3 — Generator
print("\n--- Step 3: Generation (Groq / Llama 3.1 8B) ---")
result = generate(QUERY, chunks)

print(f"\n{'=' * 60}")
print(f"  FINAL ANSWER:")
print(f"{'=' * 60}")
print(result["answer"])
print(f"\n  source_label: {result['source_label']}")
print(f"  intent:       {result['intent']}")
print("=" * 60)
