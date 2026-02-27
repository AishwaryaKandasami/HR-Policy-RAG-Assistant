"""
generator.py — RAG Generation Layer (Groq + Llama 3.1 8B)
===========================================================
Purpose : Given a user query and retrieved context chunks, call
          the Llama 3.1 8B model via Groq to produce a factual,
          sourced answer.

Inputs  : User query + list of retrieved chunk dicts from retriever.py
Outputs : Dict with answer, source_url, source_label, date_fetched, intent
Env vars: GROQ_API_KEY

Patches applied:
  P1 — Context delimiters: wraps context in <context> tags
  P4 — Programmatic source/date: strips LLM lines, appends from metadata
"""

import os
import re
import pathlib

from dotenv import load_dotenv
from groq import Groq

# ── Config ────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.1-8b-instant"
MAX_CONTEXT_CHARS = 3200     # ≈ 800 tokens at ~4 chars/token
TEMPERATURE       = 0.0      # factual — no creativity
MAX_TOKENS        = 200
TOP_P             = 1.0

# ── System prompt — loaded from system_prompt.txt ─────────────────
_PROMPT_PATH = pathlib.Path(__file__).parent / "system_prompt.txt"

def _load_system_prompt() -> str:
    """Read system_prompt.txt once; fall back to inline if missing."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    # Minimal inline fallback (should never be reached in prod)
    return (
        "You are a factual assistant for SBI Mutual Fund schemes. "
        "Answer using ONLY facts inside the <context> tags. "
        "Maximum 3 sentences. Plain prose only."
    )

SYSTEM_PROMPT = _load_system_prompt()

# Fallback answer used when no context is available
NO_CONTEXT_ANSWER = (
    "I don't have verified information on this. "
    "Please visit sbimf.com or amfiindia.com for the most current details."
)

# Module-level singleton
_groq_client: Groq | None = None


# ── Initialisation ────────────────────────────────────────────────

def _get_groq(api_key: str | None = None) -> Groq:
    """Return (and cache) a Groq client."""
    global _groq_client
    if _groq_client is None:
        if not api_key:
            load_dotenv()
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Core generation function ─────────────────────────────────────

def generate(query: str, retrieved_chunks: list[dict], api_key: str | None = None) -> dict:
    """Generate a factual answer using Groq (Llama 3.1 8B).

    Args:
        query:            The user's natural language question.
        retrieved_chunks: List of dicts from retriever.retrieve(),
                          each with chunk_text, source_url, scheme,
                          topic, date_fetched, rerank_score.

    Returns:
        Dict with keys: answer, source_url, source_label,
        date_fetched, intent.
    """
    # ── Handle empty context ─────────────────────────────────────
    if not retrieved_chunks:
        return {
            "answer":       NO_CONTEXT_ANSWER,
            "source_url":   "",
            "source_label": "",
            "date_fetched": "",
            "intent":       "factual",
        }

    # ── Build context string ─────────────────────────────────────
    # Sort by rerank_score descending so best chunk is first
    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda c: c.get("rerank_score", 0),
        reverse=True,
    )

    best_chunk = sorted_chunks[0]
    best_source_url  = best_chunk["source_url"]
    best_date        = best_chunk["date_fetched"]

    # Concatenate chunk texts, truncating to stay within context budget
    context_parts: list[str] = []
    total_len = 0
    for chunk in sorted_chunks:
        text = chunk["chunk_text"]
        if total_len + len(text) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_len
            if remaining > 50:
                context_parts.append(text[:remaining] + "…")
            break
        context_parts.append(text)
        total_len += len(text)

    context_body = "\n\n".join(context_parts)

    # ── Build user prompt — Patch 1: wrap in <context> tags ──────
    user_prompt = (
        f"<context>\n{context_body}\n</context>\n\n"
        f"Question: {query}"
    )

    # ── Call Groq API ────────────────────────────────────────────
    client = _get_groq(api_key)

    chat_response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )

    raw_answer = chat_response.choices[0].message.content.strip()

    # ── Patch 4: strip any Source / Last updated lines the LLM
    #    may have generated, then append programmatically ──────────
    clean_answer = _strip_source_lines(raw_answer)

    final_answer = (
        f"{clean_answer}\n\n"
        f"Source: {best_source_url}\n"
        f"Last updated from sources: {best_date}"
    )

    # ── Derive a readable source label ───────────────────────────
    source_label = _make_source_label(best_source_url)

    return {
        "answer":       final_answer,
        "source_url":   best_source_url,
        "source_label": source_label,
        "date_fetched": best_date,
        "intent":       "factual",
    }


# ── Patch 4 helper ───────────────────────────────────────────────

_SOURCE_LINE_RE = re.compile(
    r"\n*(?:Source|Last updated)[^\n]*",
    re.IGNORECASE,
)

def _strip_source_lines(text: str) -> str:
    """Remove any Source: or Last updated: lines the LLM generated."""
    cleaned = _SOURCE_LINE_RE.sub("", text).strip()
    return cleaned if cleaned else text.strip()


def _make_source_label(url: str) -> str:
    """Create a human-readable label from a source URL."""
    if "sbimf.com" in url:
        if "scheme-details" in url or "sbimf-scheme-details" in url:
            return "SBI MF — Scheme Overview"
        if "kim" in url.lower():
            return "SBI MF — Key Information Memorandum"
        if "sid" in url.lower():
            return "SBI MF — Scheme Information Document"
        if "factsheet" in url.lower():
            return "SBI MF — Factsheet"
        return "SBI Mutual Fund"
    if "amfiindia.com" in url:
        return "AMFI India"
    if "sebi.gov.in" in url:
        return "SEBI Circular"
    if "camsonline.com" in url:
        return "CAMS Online"
    return url
