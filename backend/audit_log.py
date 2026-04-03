"""
audit_log.py — HR Bot Session Audit Logger
===========================================
Appends every query attempt (and its result) to a session-level CSV.
Ensures HR managers can audit bot interactions for policy accuracy.
"""

import csv
import datetime
import os
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────
LOG_FILE = "session_audit.csv"

# Columns to log
HEADERS = [
    "timestamp", "query", "answer_preview", "doc_title", "section", 
    "page", "llm_used", "blocked", "block_reason", "escalated", "latency_ms"
]


def _initialize_log_file():
    """Create the CSV with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()


# ── Public API ──────────────────────────────────────────────────────

def log_interaction(
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
    Records a single query-answer interaction into the session CSV.
    """
    _initialize_log_file()
    
    # Extract first source metadata if present
    doc_title = sources[0].get("doc_title", "N/A") if sources else "N/A"
    section   = sources[0].get("section_heading", "N/A") if sources else "N/A"
    page      = sources[0].get("page_number", "N/A") if sources else "N/A"

    # Truncate answer for log preview
    answer_preview = (answer[:100] + "...") if len(answer) > 100 else answer

    row = {
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
        "latency_ms":     f"{latency_ms:.2f}"
    }

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)


def get_log_file_path() -> str:
    """Return the absolute path to the session log file."""
    return os.path.abspath(LOG_FILE)
