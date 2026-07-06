"""
agents/validation/guardrails.py
================================
GuardrailsAgent — pre-flight safety checks run BEFORE any specialist agent.

Checks (in order):
  1. Input length          — reject absurdly long prompts
  2. Prompt injection      — detect attempts to override system prompts
  3. PII in the query      — mask and flag before any LLM sees it
  4. Department access     — HR users cannot query marketing financial data
  5. Harmful content       — block requests for PII harvesting, social engineering, etc.

Any check returning a "BLOCK" flag will stop the graph at the guardrails node
(the _route_after_guardrails function in graph.py checks for "BLOCK").
"WARN" flags are logged but do not block execution.
"""

from __future__ import annotations
import re
from agents.base import AgentState


# ── PII detection (same patterns as HR agent — applied to ALL queries) ─────────

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),                    "SSN"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  "EMAIL"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),                    "PHONE"),
    (re.compile(r"\bEMP[-#]?\d{4,8}\b", re.I),                             "EMPLOYEE_ID"),
    (re.compile(r"\$\d[\d,]+\b"),                                           "SALARY_AMOUNT"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"),                                "CREDIT_CARD"),
]


def _detect_pii(text: str) -> list[str]:
    return [label for pattern, label in _PII_PATTERNS if pattern.search(text)]


# ── Prompt injection detection ─────────────────────────────────────────────────

_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your system prompt",
    "forget your instructions",
    "you are now",
    "act as if you are",
    "pretend you are",
    "new system prompt",
    "override your rules",
    "jailbreak",
    "do anything now",
    "dan mode",
]


def _detect_injection(text: str) -> list[str]:
    t_lower = text.lower()
    return [p for p in _INJECTION_PHRASES if p in t_lower]


# ── Harmful content patterns ───────────────────────────────────────────────────

_HARMFUL_PATTERNS = [
    (re.compile(r"\b(harvest|scrape|dump|exfiltrate)\s+(customer|user|employee)\s+(data|pii|info)", re.I),
     "BLOCK: data exfiltration request"),
    (re.compile(r"\b(social\s+engineer|phish|spear\s+phish)\b", re.I),
     "BLOCK: social engineering request"),
    (re.compile(r"\b(bypass|circumvent)\s+(compliance|legal|gdpr|ccpa|tcpa)\b", re.I),
     "BLOCK: compliance bypass request"),
    (re.compile(r"\b(delete|drop|truncate)\s+(table|database|all records)\b", re.I),
     "BLOCK: destructive database operation"),
]


def _detect_harmful(text: str) -> list[str]:
    return [msg for pattern, msg in _HARMFUL_PATTERNS if pattern.search(text)]


# ── Department access control ──────────────────────────────────────────────────

# Marketing users must not be able to query HR personal data.
_HR_RESTRICTED_TERMS = [
    "salary of", "compensation of", "pay stub",
    "disciplinary record", "performance improvement plan for",
    "who was fired", "termination record",
]


def _check_access(query: str, department: str, routed_agent: str) -> list[str]:
    flags = []
    q_lower = query.lower()

    # Non-HR users asking for individual employee HR records
    if department != "hr" and routed_agent == "hr_docqa":
        for term in _HR_RESTRICTED_TERMS:
            if term in q_lower:
                flags.append(f"BLOCK: '{term}' requires HR department access")
                break

    return flags


# ── GuardrailsAgent ────────────────────────────────────────────────────────────

_MAX_QUERY_CHARS = 4_000   # refuse absurdly long prompts


class GuardrailsAgent:
    """LangGraph node — always the first node in the graph."""

    async def invoke(self, state: AgentState) -> dict:
        query      = state.get("user_query", "")
        department = state.get("department", "marketing")
        agent      = state.get("routed_agent", "")

        flags: list[str] = []

        # 1. Input length
        if len(query) > _MAX_QUERY_CHARS:
            flags.append(f"BLOCK: query too long ({len(query)} chars, max {_MAX_QUERY_CHARS})")

        # 2. Prompt injection
        injections = _detect_injection(query)
        if injections:
            flags.append(f"BLOCK: prompt injection detected — '{injections[0]}'")

        # 3. PII in query
        pii_types = _detect_pii(query)
        for p in pii_types:
            flags.append(f"WARN: PII type '{p}' detected in query — will be masked")

        # 4. Harmful content
        harmful = _detect_harmful(query)
        flags.extend(harmful)

        # 5. Department access
        access_flags = _check_access(query, department, agent)
        flags.extend(access_flags)

        error = None
        block_flags = [f for f in flags if f.startswith("BLOCK")]
        if block_flags:
            error = block_flags[0]

        return {
            "guardrail_flags": flags,
            "error":           error,
        }
