"""
hr_guardrails.py — HR Safety & Escalation Layer
================================================
Pre-filter for all user queries. Logic:
  1. Detect PI (Employee IDs, Emails, Phones)
  2. Detect Prompt Injection
  3. Detect Sensitive HR Issues (Escalate to Human)
  4. Filter Out-of-Scope (Non-HR topics)
"""

import re
from typing import TypedDict, Optional

# ── Pattern Config ──────────────────────────────────────────────────

# regex for EMP followed by 5 digits
EMP_ID_PATTERN = re.compile(r"EMP\d{5}", re.IGNORECASE)
EMAIL_PATTERN  = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
PHONE_PATTERN  = re.compile(r"\b(?:\+44|0)[\d\s\-]{9,}\b")

INJECTION_KEYWORDS = [
    "ignore previous instructions", "system prompt", "dan mode", 
    "jailbreak", "forget your persona", "disregard all rules",
    "developer mode", "override instructions"
]

# Triggers for Human Escalation (Bypasses LLM)
ESCALATION_KEYWORDS = [
    "harassment", "bullying", "discrimination", 
    "hostile work environment", "misconduct",
    "report an employee", "sue", "legal action", "tribunal",
    "am i being fired", "unfair dismissal", "gross misconduct",
    "charge for", "sue for", "litigation"
]

# Out-of-Scope Topics
OOS_KEYWORDS = [
    "bitcoin", "crypto", "investment", "stock market", "mortgage",
    "medical advice", "diagnosis", "symptoms", "prescription",
    "password reset", "it support", "laptop broken", 
    "office key", "door access", "printer", "sale", "product"
]


# ── Data Models ─────────────────────────────────────────────────────

class GuardrailResult(TypedDict):
    status: str          # "PASS", "BLOCK", "ESCALATE"
    reason: str
    message: Optional[str]


# ── Internal Helpers ────────────────────────────────────────────────

def _check_pii(query: str) -> Optional[str]:
    if EMP_ID_PATTERN.search(query): return "Employee ID detected."
    if EMAIL_PATTERN.search(query):  return "Email address detected."
    if PHONE_PATTERN.search(query):  return "Phone number detected."
    return None


def _check_injection(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in INJECTION_KEYWORDS)


def _check_escalation(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ESCALATION_KEYWORDS)


def _check_out_of_scope(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in OOS_KEYWORDS)


# ── Public API ──────────────────────────────────────────────────────

def classify_query(query: str) -> GuardrailResult:
    """
    Classifies a query based on safety and relevance rules in priority order.
    """
    # 1. Prompt Injection (High Risk)
    if _check_injection(query):
        return {
            "status": "BLOCK",
            "reason": "Injection Attempt",
            "message": "I'm sorry, I cannot fulfill this request as it contains unauthorized instructions."
        }

    # 2. PII (Privacy Risk)
    if pii_reason := _check_pii(query):
        return {
            "status": "BLOCK",
            "reason": "PII Detected",
            "message": f"I'm sorry, for privacy reasons you must not include personal data like {pii_reason.lower()}"
        }

    # 3. Sensitive Escalation (Human HR required)
    if _check_escalation(query):
        return {
            "status": "ESCALATE",
            "reason": "Sensitive Topic",
            "message": (
                "For matters involving harassment, grievances, or legal action, "
                "this chatbot cannot provide specific advice. Please contact your "
                "HR Business Partner directly to open a formal case."
            )
        }

    # 4. Out of Scope (Relevance)
    if _check_out_of_scope(query):
        return {
            "status": "BLOCK",
            "reason": "Out of Scope",
            "message": (
                "This bot only handles HR policy questions. For personal finance, "
                "IT support, or medical advice, please visit the relevant internal portals."
            )
        }

    # 5. Default: Pass to Retrieval
    return {
        "status": "PASS",
        "reason": "Factual query",
        "message": None
    }
