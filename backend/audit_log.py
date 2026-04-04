"""
audit_log.py — HR Bot Session Audit Logger
===========================================
Stores every query attempt and allows for feedback updates (thumbs up/down).
HR managers can download the session audit as a CSV.
"""

import csv
import datetime
import os
from typing import List, Optional

# ── Config ──────────────────────────────────────────────────────────
LOG_FILE = "session_audit.csv"

# Columns to log
HEADERS = [
    "query_id", "timestamp", "query", "answer_preview", "doc_title", "section", 
    "page", "llm_used", "blocked", "block_reason", "escalated", 
    "latency_ms", "rating", "feedback_reason"
]

# ── Session State (In-Memory) ───────────────────────────────────────
# We keep an in-memory list so we can update rows with feedback (UUID-based).
SESSION_LOG: List[dict] = []


# ── Public API ──────────────────────────────────────────────────────

def log_interaction(
    query_id: str,
    query: str,
    answer: str = "",
    sources: list[dict] = [],
    llm_used: str = "none",
    blocked: bool = False,
    block_reason: Optional[str] = None,
    escalated: bool = False,
    latency_ms: float = 0.0
):
    """
    Records a single query-answer interaction into the session store.
    """
    # Extract first source metadata if present
    doc_title = sources[0].get("doc_title", "N/A") if sources else "N/A"
    section   = sources[0].get("section_heading", "N/A") if sources else "N/A"
    page      = sources[0].get("page_number", "N/A") if sources else "N/A"

    # Truncate answer for log preview
    answer_preview = (answer[:100] + "...") if len(answer) > 100 else answer

    row = {
        "query_id":       query_id,
        "timestamp":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query":          query,
        "answer_preview": answer_preview.replace("\n", " "),
        "doc_title":      doc_title,
        "section":        section,
        "page":           page,
        "llm_used":       llm_used,
        "blocked":        str(blocked).lower(),
        "block_reason":   block_reason or "",
        "escalated":      str(escalated).lower(),
        "latency_ms":     f"{latency_ms:.2f}",
        "rating":         "",
        "feedback_reason": ""
    }
    
    SESSION_LOG.append(row)


def log_feedback(query_id: str, rating: str, reason: Optional[str] = None):
    """
    Finds a previously logged interaction by query_id and updates its feedback.
    """
    for row in SESSION_LOG:
        if row["query_id"] == query_id:
            row["rating"] = rating
            row["feedback_reason"] = reason or ""
            return True
    return False


def get_log_file_path() -> str:
    """
    Generates a fresh CSV from the in-memory session log and 
    returns its absolute path.
    """
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(SESSION_LOG)
        
    return os.path.abspath(LOG_FILE)
