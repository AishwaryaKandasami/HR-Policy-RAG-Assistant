import os
import sys
import pathlib
from dotenv import load_dotenv

# Add current dir to path
sys.path.append(os.path.dirname(__file__))

# Load .env
env_path = pathlib.Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from hr_ingest import get_qdrant, sync_bm25_from_cloud, ingest_file
from generator import judge_answer, rewrite_query, generate_answer

def verify_setup():
    print("🚀 Starting Cloud RAG Verification...\n")

    # 1. Test Qdrant Cloud Connection
    print("1️⃣ Testing Qdrant Cloud Connection...")
    try:
        client = get_qdrant()
        collections = client.get_collections()
        print(f"   ✅ Connected! Collections found: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"   ❌ Qdrant Connection Failed: {e}")
        return

    # 2. Test Judge (Gemini)
    print("\n2️⃣ Testing LLM-as-a-Judge (Gemini)...")
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key")
    if not gemini_key:
        print("   ⚠️  Gemini API key missing in .env")
    else:
        try:
            # Mock a judgment
            is_pass, reason = judge_answer(
                query="What is the holiday entitlement?",
                answer="Employees get 28 days.",
                retrieved_chunks=[{"text": "Employees get 28 days.", "metadata": {"source": "manual.pdf"}}]
            )
            print(f"   ✅ Judge responded: {'PASS' if is_pass else 'FAIL'} ({reason})")
        except Exception as e:
            print(f"   ❌ Judge Test Failed: {e}")

    # 3. Test Rewriter (Groq)
    print("\n3️⃣ Testing Query Rewriter (Groq)...")
    if not os.getenv("GROQ_API_KEY"):
        print("   ⚠️  GROQ_API_KEY missing in .env")
    else:
        try:
            rewritten = rewrite_query("maternity leave?")
            print(f"   ✅ Rewriter responded: '{rewritten}'")
        except Exception as e:
            print(f"   ❌ Rewriter Test Failed: {e}")

    # 4. Test Startup Sync
    print("\n4️⃣ Testing BM25 Startup Sync...")
    try:
        count = sync_bm25_from_cloud()
        print(f"   ✅ Sync successful: {count} chunks loaded into memory.")
    except Exception as e:
        print(f"   ❌ Sync Failed: {e}")

    print("\n✨ Verification Complete!")

if __name__ == "__main__":
    verify_setup()
