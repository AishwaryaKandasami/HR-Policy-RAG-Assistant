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
    "developer mode", "override instructions", "act as", "roleplay",
    "you are now", "translator"
]
INJECTION_TAG_PATTERN = re.compile(r"\[(?:SYSTEM|USER|INSTRUCTION|ASSISTANT|ADMIN)\]", re.IGNORECASE)

# Keywords that indicate a query is about a personal situation rather than general policy.
PERSONAL_INTENT_KEYWORDS = [
    "my manager", "my boss", "what should i do", "what do i do", 
    "i have received", "i am being", "i was told", "my situation", 
    "my case", "my warning", "my dismissal", "advise me", "help me with my"
]

# Sensitive HR topics that might require human escalation if framed personally.
SENSITIVE_TOPICS = [
    "harassment", "bullying", "discrimination", "misconduct",
    "hostile work environment", "sue", "legal action",
    "disciplinary", "grievance", "dismissal", "fired", "litigation"
]

# Tribunal Contexts
TRIBUNAL_PERSONAL = [
    "my tribunal", "i am going to tribunal", "taking to tribunal", 
    "i want to sue", "i am suing", "facing a tribunal", "facing tribunal", 
    "at tribunal", "tribunal case"
]

# Factual Openers (Safety Net)
FACTUAL_OPENERS = [
    "according to", "based on", "under the", "per the", 
    "in the context of", "as per", "as stated", "does the",
    "is there a", "is there any", "can an employer", "can a company",
    "what happens when", "what happens if an employer"
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
    if any(k in q for k in INJECTION_KEYWORDS):
        return True
    if INJECTION_TAG_PATTERN.search(query):
        return True
    return False


def _check_personal_situational(query: str) -> bool:
    """
    Escalate ONLY when there is an explicit personal intent signal.
    Sensitive topic alone is NOT sufficient — it must be combined
    with a first-person or advice-seeking marker.
    """
    q = query.strip().lower()

    # 1. Direct personal intent markers — always escalate
    if any(k in q for k in PERSONAL_INTENT_KEYWORDS):
        return True

    # 2. Tribunal in personal context — always escalate
    if any(k in q for k in TRIBUNAL_PERSONAL):
        return True

    # 3. First-person pronouns combined with sensitive topic
    FIRST_PERSON = ["i am", "i was", "i've", "i have", "i need advice", "i want to"]
    has_first_person = any(k in q for k in FIRST_PERSON)
    has_sensitive_topic = any(k in q for k in SENSITIVE_TOPICS)

    if has_first_person and has_sensitive_topic:
        return True

    # 4. Factual openers safety net — always indicates a policy question
    if any(q.startswith(opener) or opener in q for opener in FACTUAL_OPENERS):
        return False

    return False


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

    # 3. Sensitive Escalation / Personal Situation (Human HR required)
    if _check_personal_situational(query):
        return {
            "status": "ESCALATE",
            "reason": "Personal/Sensitive Topic",
            "message": (
                "For specific situations regarding your individual circumstances, "
                "active grievances, or your manager, this chatbot cannot provide advice. "
                "Please contact your HR Business Partner directly to open a formal case."
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
