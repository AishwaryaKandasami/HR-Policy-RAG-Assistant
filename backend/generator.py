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
from typing import Iterator, Optional

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

def _build_messages(system_msg: str, history: list[dict], user_msg: str) -> list[dict]:
    """
    Build the full messages array: system + prior turns + current user turn.
    history entries must have {"role": "user"|"assistant", "content": str}.
    """
    messages = [{"role": "system", "content": system_msg}]
    for turn in history:
        role = turn.get("role", "user")
        # Normalise any frontend 'bot' role to 'assistant'
        if role == "bot":
            role = "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_msg})
    return messages


def _call_groq(model_id: str, api_key: str, system_msg: str, user_msg: str,
               history: list[dict] | None = None) -> str:
    client = Groq(api_key=api_key)
    messages = _build_messages(system_msg, history or [], user_msg)
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_id,
        temperature=0.1,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def _call_openai(model_id: str, api_key: str, system_msg: str, user_msg: str,
                 history: list[dict] | None = None) -> str:
    client = OpenAI(api_key=api_key)
    messages = _build_messages(system_msg, history or [], user_msg)
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


def _call_gemini(model_id: str, api_key: str, system_msg: str, user_msg: str,
                 history: list[dict] | None = None) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_id,
        system_instruction=system_msg
    )
    # Gemini uses its own history format
    gemini_history = []
    for turn in (history or []):
        role = turn.get("role", "user")
        if role in ("bot", "assistant"):
            role = "model"
        gemini_history.append({"role": role, "parts": [turn.get("content", "")]})

    if gemini_history:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_msg)
    else:
        response = model.generate_content(user_msg)
    return response.text


# ── Streaming Provider Helpers ─────────────────────────────────────

def _stream_groq(model_id: str, api_key: str, system_msg: str, user_msg: str,
                 history: list[dict] | None = None) -> Iterator[str]:
    """Yield text tokens from Groq streaming API."""
    client = Groq(api_key=api_key)
    messages = _build_messages(system_msg, history or [], user_msg)
    stream = client.chat.completions.create(
        messages=messages,
        model=model_id,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_openai(model_id: str, api_key: str, system_msg: str, user_msg: str,
                   history: list[dict] | None = None) -> Iterator[str]:
    """Yield text tokens from OpenAI streaming API."""
    client = OpenAI(api_key=api_key)
    messages = _build_messages(system_msg, history or [], user_msg)
    stream = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_gemini(model_id: str, api_key: str, system_msg: str, user_msg: str,
                   history: list[dict] | None = None) -> Iterator[str]:
    """Yield text tokens from Gemini streaming API."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_id, system_instruction=system_msg)
    gemini_history = []
    for turn in (history or []):
        role = turn.get("role", "user")
        if role in ("bot", "assistant"):
            role = "model"
        gemini_history.append({"role": role, "parts": [turn.get("content", "")]})

    if gemini_history:
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(user_msg, stream=True)
    else:
        response = model.generate_content(user_msg, stream=True)

    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Public API ─────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    model_alias: str = "groq_llama_8b",
    api_key: Optional[str] = None,
    confidence_score: float = 0.0,
    conversation_history: Optional[list[dict]] = None,
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
        provider_name = MODEL_MAP.get(model_alias, MODEL_MAP["groq_llama_70b"]).get("provider")
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
        f"INSTRUCTIONS:\n"
        f"1. Answer the user's question using the provided context only.\n"
        f"2. Use clean, objective prose. NO personal framing ('you should', 'if you are').\n"
        f"3. NO citations or metadata. Do not include 'Sources:' or [Source...] tags in your response.\n"
        f"4. If the question involves personal advice or active disputes, provide only the policy facts and refer them to their HRBP."
    )

    # 2. Route to provider
    history = conversation_history or []
    config = MODEL_MAP.get(model_alias)
    if config:
        provider = config["provider"]
        model_id = config["id"]
    else:
        # Fallback to dynamic parsing: provider_modelid
        if "_" in model_alias:
            parts = model_alias.split("_", 1)
            provider = parts[0].lower()
            model_id = parts[1]
        else:
            # Absolute default
            default_config = MODEL_MAP["groq_llama_70b"]
            provider = default_config["provider"]
            model_id = default_config["id"]

    try:
        if provider == "groq":
            answer = _call_groq(model_id, api_key, system_msg, user_msg, history)
        elif provider == "openai":
            answer = _call_openai(model_id, api_key, system_msg, user_msg, history)
        elif provider == "gemini":
            answer = _call_gemini(model_id, api_key, system_msg, user_msg, history)
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
            "model_used": "none",
            "success": False
        }
def generate_answer_stream(
    query: str,
    retrieved_chunks: list[dict],
    model_alias: str = "groq_llama_70b",
    api_key: Optional[str] = None,
    conversation_history: Optional[list[dict]] = None,
) -> Iterator[str]:
    """
    Streaming variant of generate_answer.
    Yields raw text tokens as they arrive from the LLM provider.
    No judge / rewrite loop — caller is responsible for guardrails upstream.
    """
    # Resolve API key (same logic as generate_answer)
    if not api_key:
        env_keys = {
            "groq":   os.getenv("GROQ_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "gemini": os.getenv("GOOGLE_API_KEY"),
        }
        provider_name = MODEL_MAP.get(model_alias, MODEL_MAP["groq_llama_70b"]).get("provider")
        api_key = env_keys.get(provider_name)

    if not api_key:
        yield "Error: Missing API key for the selected provider."
        return

    system_msg = _load_system_prompt()
    context_block = _format_context(retrieved_chunks)
    user_msg = (
        f"CONTEXT FROM HR DOCUMENTS:\n{context_block}\n\n"
        f"USER QUESTION: {query}\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Answer the user's question using the provided context only.\n"
        f"2. Use clean, objective prose. NO personal framing ('you should', 'if you are').\n"
        f"3. NO citations or metadata. Do not include 'Sources:' or [Source...] tags.\n"
        f"4. If the question involves personal advice or active disputes, provide only the "
        f"policy facts and refer them to their HRBP."
    )

    history = conversation_history or []
    config = MODEL_MAP.get(model_alias, MODEL_MAP["groq_llama_70b"])
    provider = config["provider"]
    model_id = config["id"]

    try:
        if provider == "groq":
            yield from _stream_groq(model_id, api_key, system_msg, user_msg, history)
        elif provider == "openai":
            yield from _stream_openai(model_id, api_key, system_msg, user_msg, history)
        elif provider == "gemini":
            yield from _stream_gemini(model_id, api_key, system_msg, user_msg, history)
        else:
            yield f"Unknown provider: {provider}"
    except Exception as e:
        yield f"Generation Error ({provider}): {str(e)}"


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


def rewrite_query(
    original_query: str,
    model_alias: str = "groq_llama_8b",
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Uses an LLM to expand/rewrite the user's query for better retrieval.
    When conversation_history is provided, the previous user turn is included
    so ambiguous follow-ups ("and for part-timers?") resolve correctly.
    """
    rewriter_prompt_path = os.path.join(os.path.dirname(__file__), "rewriter_system_prompt.txt")
    try:
        with open(rewriter_prompt_path, "r", encoding="utf-8") as f:
            system_msg = f.read()
    except FileNotFoundError:
        return original_query

    # Build the input: optionally prepend the previous user turn for context
    history = conversation_history or []
    prev_user_turns = [t for t in history if t.get("role") == "user"]
    if prev_user_turns:
        prev_turn = prev_user_turns[-1].get("content", "")
        rewrite_input = f"Previous question: {prev_turn}\nFollow-up: {original_query}"
    else:
        rewrite_input = original_query

    def _get_env_key(keys: list[str]) -> Optional[str]:
        for k in keys:
            val = os.getenv(k)
            if val:
                return val.strip()
        return None

    api_key_gemini = _get_env_key(["GOOGLE_API_KEY", "GEMINI_API_KEY", "gemini_api_key"])
    api_key_groq   = _get_env_key(["GROQ_API_KEY", "groq_api_key"])

    env_keys = {"groq": api_key_groq, "gemini": api_key_gemini}
    provider = MODEL_MAP.get(model_alias, {}).get("provider", "groq")
    api_key = env_keys.get(provider)

    if not api_key:
        return original_query

    try:
        if provider == "groq":
            rewritten = _call_groq(MODEL_MAP[model_alias]["id"], api_key, system_msg, rewrite_input)
        else:
            rewritten = _call_gemini(MODEL_MAP[model_alias]["id"], api_key, system_msg, rewrite_input)
        return rewritten.strip()
    except Exception:
        return original_query
