"""
generator.py — HR Multi-LLM Router
====================================
Routes RAG queries to the user-selected LLM provider:
  1. Groq (Llama 3.1/3.3) — fastest, best for demo
  2. Google AI Studio (Gemini 2.0 Flash) — strong reasoning
  3. OpenAI (GPT-3.5/4o-mini) — benchmark standard

Includes:
  - System prompt loading
  - Context window formatting
  - Citation enforcement
"""

import os
from typing import Optional

import google.generativeai as genai
from groq import Groq
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "hr_system_prompt.txt")

# Standard model aliases from the UI to provider-specific IDs
MODEL_MAP = {
    "groq_llama_8b":   {"provider": "groq",   "id": "llama-3.1-8b-instant"},
    "groq_llama_70b":  {"provider": "groq",   "id": "llama-3.3-70b-versatile"},
    "gemini_flash":    {"provider": "gemini", "id": "gemini-2.0-flash"},
    "openai_gpt35":    {"provider": "openai", "id": "gpt-3.5-turbo"},
    "openai_gpt4o":    {"provider": "openai", "id": "gpt-4o-mini"},
}


def _load_system_prompt() -> str:
    """Read the HR specialist persona and citation rules."""
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "You are a professional HR assistant. Use provided context and cite sources."


def _format_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieval chunks into a readable numbered block for the LLM."""
    if not retrieved_chunks:
        return "No relevant HR policies found in current documents."

    lines = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        md = chunk.get("metadata", {})
        header = (
            f"--- [Doc {idx}: {md.get('doc_title', 'Unknown')} | "
            f"Section: {md.get('section_heading', 'General')} | "
            f"Page: {md.get('page_number', 'N/A')}] ---"
        )
        lines.append(header)
        lines.append(chunk["text"])
        lines.append("")  # separator
    return "\n".join(lines)


# ── Provider Clients ───────────────────────────────────────────────

def _call_groq(model_id: str, api_key: str, system_msg: str, user_msg: str) -> str:
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        model=model_id,
        temperature=0.1,  # Keep it grounded
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def _call_openai(model_id: str, api_key: str, system_msg: str, user_msg: str) -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_gemini(model_id: str, api_key: str, system_msg: str, user_msg: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system_msg
    )
    response = model.generate_content(user_msg)
    return response.text


# ── Public API ─────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model_alias: str = "groq_llama_8b",
    api_key: Optional[str] = None,
    confidence_score: float = 0.0
) -> dict:
    """
    Main entry point for generating an HR policy answer.
    """
    if not api_key:
        # Check environment as fallback (useful for pre-load and tests)
        env_keys = {
            "groq":   os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GOOGLE_API_KEY"),
        }
        # Match alias to provider then find key
        provider_name = MODEL_MAP.get(model_alias, {}).get("provider")
        api_key = env_keys.get(provider_name)

    if not api_key:
        return {
            "answer": "Error: Missing API key for the selected provider.",
            "success": False
        }

    # 1. Format inputs
    system_msg = _load_system_prompt()
    context_block = _format_context(retrieved_chunks)
    user_msg = (
        f"CONTEXT FROM HR DOCUMENTS:\n{context_block}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"INSTRUCTIONS: Answer the user's question using the provided context only. "
        f"Cite your sources precisely in [Source: Title, Section, Page] format."
    )

    # 2. Route to provider
    config = MODEL_MAP.get(model_alias, MODEL_MAP["groq_llama_8b"])
    provider = config["provider"]
    model_id = config["id"]

    try:
        if provider == "groq":
            answer = _call_groq(model_id, api_key, system_msg, user_msg)
        elif provider == "openai":
            answer = _call_openai(model_id, api_key, system_msg, user_msg)
        elif provider == "gemini":
            answer = _call_gemini(model_id, api_key, system_msg, user_msg)
        else:
            return {"answer": f"Unknown provider: {provider}", "success": False}

        return {
            "answer": answer.strip(),
            "model_used": model_id,
            "success": True,
            "confidence_score": confidence_score
        }
    except Exception as e:
        return {
            "answer": f"Generation Error ({provider}): {str(e)}",
            "success": False
        }
# ── Evaluation & Refinement ────────────────────────────────────────

def judge_answer(query: str, answer: str, retrieved_chunks: list[dict], model_alias: str = "groq_llama_70b") -> tuple[bool, str]:
    """
    Acts as the "LLM-as-a-Judge". 
    Evaluates faithfulness and relevance before returning to the user.
    """
    # 1. Load judge prompt
    judge_prompt_path = os.path.join(os.path.dirname(__file__), "judge_system_prompt.txt")
    try:
        with open(judge_prompt_path, "r", encoding="utf-8") as f:
            system_msg = f.read()
    except FileNotFoundError:
        return True, "Judge prompt not found; skipping check."

    # 2. Format context for judgment
    context_msg = _format_context(retrieved_chunks)
    user_msg = (
        f"USER QUERY: {query}\n\n"
        f"GENERATED ANSWER: {answer}\n\n"
        f"SOURCE CONTEXT:\n{context_msg}"
    )

    # 3. Call judge (Gemini preferred)
    def _get_env_key(keys: list[str]) -> Optional[str]:
        for k in keys:
            val = os.getenv(k)
            if val: return val.strip()
        return None

    api_key_gemini = _get_env_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "gemini_api_key"])
    api_key_openai = _get_env_key(["OPENAI_API_KEY", "openai_api_key"])
    api_key_groq   = _get_env_key(["GROQ_API_KEY", "groq_api_key"])

    env_keys = {
        "gemini": api_key_gemini,
        "openai": api_key_openai,
        "groq":   api_key_groq,
    }
    
    # Try to find a key for the judge
    judge_provider = MODEL_MAP.get(model_alias, {}).get("provider", "gemini")
    api_key = env_keys.get(judge_provider)
    
    if not api_key:
        return True, "Missing Judge API key; skipping check."

    try:
        if judge_provider == "gemini":
            raw_eval = _call_gemini(MODEL_MAP[model_alias]["id"], api_key, system_msg, user_msg)
        elif judge_provider == "groq":
             raw_eval = _call_groq(MODEL_MAP[model_alias]["id"], api_key, system_msg, user_msg)
        else:
             raw_eval = _call_openai(MODEL_MAP[model_alias]["id"], api_key, system_msg, user_msg)

        # Parse result (looking for [RESULT] or just PASS/FAIL in the first few lines)
        raw_upper = raw_eval.upper()
        # Find [RESULT] or search for the words PASS/FAIL
        is_pass = "PASS" in raw_upper and "FAIL" not in raw_upper
        
        reason = "Passed verification."
        if not is_pass:
            # Extract reason if possible
            if "[REASON]" in raw_eval:
                reason = raw_eval.split("[REASON]")[1].strip()
            else:
                reason = "Potential hallucination or out-of-context info detected."
        
        return is_pass, reason
    except Exception as e:
        return True, f"Critique script error: {str(e)}"


def rewrite_query(original_query: str, model_alias: str = "groq_llama_8b") -> str:
    """
    Uses an LLM to expand/rewrite the user's query for better retrieval.
    """
    rewriter_prompt_path = os.path.join(os.path.dirname(__file__), "rewriter_system_prompt.txt")
    try:
        with open(rewriter_prompt_path, "r", encoding="utf-8") as f:
            system_msg = f.read()
    except FileNotFoundError:
        return original_query

    # Call LLM (Groq preferred for speed)
    def _get_env_key(keys: list[str]) -> Optional[str]:
        for k in keys:
            val = os.getenv(k)
            if val: return val.strip()
        return None

    api_key_gemini = _get_env_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "gemini_api_key"])
    api_key_groq   = _get_env_key(["GROQ_API_KEY", "groq_api_key"])

    env_keys = {
        "groq":   api_key_groq,
        "gemini": api_key_gemini,
    }
    provider = MODEL_MAP.get(model_alias, {}).get("provider", "groq")
    api_key = env_keys.get(provider)
    
    if not api_key:
        return original_query

    try:
        if provider == "groq":
            rewritten = _call_groq(MODEL_MAP[model_alias]["id"], api_key, system_msg, original_query)
        else:
            rewritten = _call_gemini(MODEL_MAP[model_alias]["id"], api_key, system_msg, original_query)
        
        return rewritten.strip()
    except Exception:
        return original_query
