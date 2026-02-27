import json
from guardrails import classify_query
from retriever import retrieve
from generator import generate

QUERIES = [
    "1. What is the expense ratio of SBI Large Cap Fund?",
    "2. What is the exit load on SBI Flexi Cap Fund?",
    "3. What is the minimum SIP for SBI ELSS Tax Saver Fund?",
    "4. What is the lock-in period for ELSS?",
    "5. What is the riskometer level of SBI Flexi Cap Fund?",
    "6. What is the benchmark of SBI ELSS Tax Saver Fund?",
    "7. How do I download my capital gains statement from CAMS?",
    "8. Should I invest in SBI Large Cap or Flexi Cap?",
    "9. What is the expense ratio of HDFC Mid Cap Fund?",
    "10. Which SBI fund is safer?"
]

def evaluate():
    for q_line in QUERIES:
        q_num, query = q_line.split(". ", 1)
        print("=" * 80)
        print(f"QUERY {q_num}: {query}")
        print("=" * 80)

        # Step 1 — Guardrails
        classification = classify_query(query)
        print("\nStep 1 — Guardrails")
        print(f"  Intent: {classification.get('intent', 'N/A')}")
        print(f"  Action: {classification.get('action', 'N/A')}")
        print(f"  Reason: {classification.get('reason', 'N/A')}")

        action = classification.get('action')
        
        if action == "retrieve":
            # Step 2 — Retrieval
            print("\nStep 2 — Retrieval")
            chunks = retrieve(query)
            for i, c in enumerate(chunks[:3], 1):
                text_preview = c['chunk_text'][:100].replace('\n', ' ')
                print(f"  Chunk {i}: topic=[{c['topic']}] scheme=[{c['scheme']}] score=[{c['rerank_score']:.4f}]")
                print(f"           text=[{text_preview}]")

            # Step 3 — Generation
            print("\nStep 3 — Generation")
            result = generate(query, chunks)
            print(f"  Answer: {result['answer']}")
            print(f"  Source: {result.get('source_label', 'N/A')}")
            # we don't have last updated natively but let's check chunks
            
            # Step 4 — Score
            print("\nStep 4 — Score (0-2 per dimension, max 10)")
            print("  Factual accuracy:     [0/1/2]")
            print("  Citation correct:     [0/1/2]")
            print("  Within 3 sentences:   [0/1/2]")
            print("  Refusal if needed:    [0/1/2]")
            print("  No PII reflected:     [0/1/2]")
            print("  Query score:          [x/10]")
        else:
            print("\nStep 2 — Retrieval (skip if action is not retrieve)")
            print("\nStep 3 — Generation (skip if action is not retrieve)")
            print("\nStep 4 — Score (0-2 per dimension, max 10)")
            print("  Factual accuracy:     [N/A]")
            print("  Citation correct:     [N/A]")
            print("  Within 3 sentences:   [N/A]")
            print("  Refusal if needed:    [0/1/2]")
            print("  No PII reflected:     [0/1/2]")
            print("  Query score:          [x/10]")

        print("\n\n")

if __name__ == '__main__':
    evaluate()
