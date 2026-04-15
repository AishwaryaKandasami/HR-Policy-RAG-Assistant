"""
intent_router.py — Pre-RAG Intent Classification
=================================================
Short-circuits greetings, thanks, meta-questions, and other non-HR small-talk
so they never enter the embed → retrieve → rerank → LLM → judge pipeline.

Ordering: runs AFTER guardrails (PII/injection/OOS/escalation) and BEFORE
embedding. A query reaches retrieval only if intent == "hr_question".

Rationale: a greeting like "Hi" has no useful embedding against an HR corpus.
Letting it through produces noisy retrieval and an unpredictable LLM deflection
every time — the root cause of inconsistent "Hi" replies.
"""

import re
from typing import TypedDict, Optional


# ── Intent Patterns ────────────────────────────────────────────────

# Exact-match or near-exact greetings (short utterances only)
GREETING_PATTERNS = [
    r"^\s*hi[\s!.]*$",
    r"^\s*hii+[\s!.]*$",
    r"^\s*hello[\s!.]*$",
    r"^\s*hey[\s!.]*$",
    r"^\s*hey\s+there[\s!.]*$",
    r"^\s*good\s+(morning|afternoon|evening)[\s!.]*$",
    r"^\s*greetings[\s!.]*$",
    r"^\s*howdy[\s!.]*$",
    r"^\s*yo[\s!.]*$",
]

THANKS_PATTERNS = [
    r"^\s*thanks?[\s!.]*$",
    r"^\s*thank\s+you[\s!.]*$",
    r"^\s*ty[\s!.]*$",
    r"^\s*thx[\s!.]*$",
    r"^\s*cheers[\s!.]*$",
    r"^\s*appreciated?[\s!.]*$",
]

FAREWELL_PATTERNS = [
    r"^\s*bye[\s!.]*$",
    r"^\s*goodbye[\s!.]*$",
    r"^\s*see\s+(ya|you)[\s!.]*$",
    r"^\s*later[\s!.]*$",
]

# Meta-questions about the bot itself
META_KEYWORDS = [
    "who are you", "who built you", "who made you", "what are you",
    "what can you do", "what do you do", "how do you work",
    "are you an ai", "are you a bot", "are you human", "are you chatgpt",
    "your name", "what's your name", "whats your name",
]

# "Help" / capability queries that aren't HR questions
HELP_KEYWORDS = [
    "help me", "what can i ask", "what should i ask", "give me examples",
    "how can you help", "what topics",
]


# Compile once
_GREETING_RE = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]
_THANKS_RE   = [re.compile(p, re.IGNORECASE) for p in THANKS_PATTERNS]
_FAREWELL_RE = [re.compile(p, re.IGNORECASE) for p in FAREWELL_PATTERNS]


# ── Templated Responses ────────────────────────────────────────────
# Deterministic — same input always produces the same output.

GREETING_RESPONSE = (
    "Hello! I'm your UK HR policy assistant. I can answer questions about "
    "policies like annual leave, maternity, sickness absence, and similar topics "
    "based on the documents loaded in this workspace. What would you like to know?"
)

THANKS_RESPONSE = (
    "You're welcome. Let me know if you have any other HR policy questions."
)

FAREWELL_RESPONSE = (
    "Goodbye. Feel free to come back any time with more HR policy questions."
)

META_RESPONSE = (
    "I'm a UK HR policy Q&A assistant. I answer questions grounded in the HR "
    "policy documents loaded in this workspace (for example: annual leave, "
    "maternity, sickness absence). I don't handle personal grievances, payroll "
    "specifics, or legal advice — for those, please contact your HR Business Partner."
)

HELP_RESPONSE = (
    "You can ask me about UK HR policies loaded in this workspace. Example questions:\n"
    "• What is the maternity leave entitlement?\n"
    "• How many days of annual leave am I entitled to?\n"
    "• What is the sickness absence reporting process?\n"
    "• What are the rules on statutory sick pay?\n\n"
    "I answer using the ingested policy documents only, with source citations."
)


# ── Data Model ─────────────────────────────────────────────────────

class IntentResult(TypedDict):
    intent: str                   # "greeting", "thanks", "farewell", "meta", "help", "hr_question"
    response: Optional[str]       # Canned response if short-circuiting; None otherwise
    short_circuit: bool           # True → skip RAG pipeline entirely


# ── Public API ─────────────────────────────────────────────────────

def classify_intent(query: str) -> IntentResult:
    """
    Classify a user query into a small-talk bucket or default to hr_question.

    Only very-high-confidence small-talk short-circuits. Anything ambiguous
    defaults to hr_question so the RAG pipeline gets a chance.
    """
    if not query or not query.strip():
        return {
            "intent": "greeting",
            "response": GREETING_RESPONSE,
            "short_circuit": True,
        }

    q = query.strip()
    q_lower = q.lower()

    # 1. Greetings (exact-ish match only — avoid false-positives on "hello, what is...")
    if any(r.match(q) for r in _GREETING_RE):
        return {"intent": "greeting", "response": GREETING_RESPONSE, "short_circuit": True}

    # 2. Thanks
    if any(r.match(q) for r in _THANKS_RE):
        return {"intent": "thanks", "response": THANKS_RESPONSE, "short_circuit": True}

    # 3. Farewell
    if any(r.match(q) for r in _FAREWELL_RE):
        return {"intent": "farewell", "response": FAREWELL_RESPONSE, "short_circuit": True}

    # 4. Meta (about the bot itself) — only short-circuit if the query is
    #    dominantly about the bot, not just containing a keyword in a longer
    #    HR question.
    if len(q) < 80 and any(k in q_lower for k in META_KEYWORDS):
        return {"intent": "meta", "response": META_RESPONSE, "short_circuit": True}

    # 5. Help / capability
    if len(q) < 80 and any(k in q_lower for k in HELP_KEYWORDS):
        return {"intent": "help", "response": HELP_RESPONSE, "short_circuit": True}

    # 6. Default → real HR question
    return {"intent": "hr_question", "response": None, "short_circuit": False}
