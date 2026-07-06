"""
agents/validation/report_validator.py
======================================
ReportValidatorAgent — post-flight fidelity gate.

Computes a fidelity score (0.0–1.0) against three dimensions:
  1. Structural completeness  (30%) — intent-aware: checks expected fields per intent,
                                       not just per agent
  2. Numeric grounding        (40%) — faithfulness: cross-checks response numbers
                                       against actual tool outputs, not just presence
  3. Citation coverage        (30%) — HR/financial answers cite a source

If score >= FIDELITY_THRESHOLD (0.92), route to response_builder.
Otherwise, route back to planner for one retry.

The retry logic is enforced in graph.py via _route_after_validation.
"""

from __future__ import annotations
import json
import re
from typing import Any

from agents.base import AgentState
from shared_config import FIDELITY_THRESHOLD, EVIDENCE_SUFFICIENT, EVIDENCE_PARTIAL, INDUSTRY_BENCHMARKS


# ── Intent-aware structural completeness ─────────────────────────────────────
#
# Each intent maps to a list of keyword groups. Each group is an OR — any one
# term present means the group passes. Score = fraction of groups that pass.
# Falls back to agent-level groups when intent is not specifically mapped.

_INTENT_KEYWORDS: dict[str, list[list[str]]] = {
    # ── Data Analyst intents ──────────────────────────────────────────────────
    "take_rate": [
        ["take rate", "adoption rate", "penetration"],
        ["passings", "homes passed"],
        ["subscribers", "subs", "active"],
        ["%", "percent"],
    ],
    "campaign_performance": [
        ["roas", "return on ad spend"],
        ["cac", "cost per acquisition", "cost to acquire"],
        ["impressions", "clicks", "leads"],
        ["campaign"],
    ],
    "funnel_metrics": [
        ["funnel", "pipeline"],
        ["drop", "drop-off", "leak"],
        ["stage", "step", "conversion"],
        ["rate", "%"],
    ],
    "cohort_analysis": [
        ["cohort"],
        ["retention", "churn"],
        ["month", "period"],
        ["%", "rate"],
    ],
    "anomaly_detection": [
        ["anomaly", "spike", "drop", "unusual"],
        ["z-score", "standard deviation", "outlier"],
        ["%", "delta", "change"],
    ],
    # ── Financial Planner intents ─────────────────────────────────────────────
    "roi_npv": [
        ["npv", "net present value"],
        ["irr", "internal rate of return"],
        ["payback"],
        ["capex", "capital expenditure"],
        ["recommendation", "build", "marginal"],
    ],
    "calix_demographics": [
        ["calix", "demographic"],
        ["income", "household"],
        ["fiber propensity", "propensity score"],
        ["zip", "zip code"],
    ],
    "copper_to_fiber": [
        ["copper", "fiber"],
        ["take rate", "switch rate"],
        ["passings", "homes"],
        ["npv", "irr", "payback", "recommendation"],
    ],
    "capex_planning": [
        ["capex", "capital"],
        ["cost per passing", "per home"],
        ["total cost", "budget"],
    ],
    "fiber_propensity": [
        ["propensity", "likelihood"],
        ["score", "predicted"],
        ["zip", "region"],
    ],
    # ── Strategist intents ────────────────────────────────────────────────────
    "social_content": [
        ["post", "copy", "content"],
        ["platform", "linkedin", "meta", "instagram", "x", "twitter", "youtube"],
        ["hashtag", "#"],
    ],
    "content_calendar": [
        ["calendar", "schedule", "week"],
        ["platform"],
        ["post", "content"],
        ["date", "monday", "tuesday", "wednesday", "thursday", "friday"],
    ],
    "competitor_content": [
        ["competitor", "competing"],
        ["content", "post", "strategy"],
        ["gap", "opportunity", "differentiat"],
    ],
    "brand_strategy": [
        ["brand", "positioning", "voice"],
        ["audience", "target"],
        ["message", "pillar", "theme"],
    ],
    # ── HR intents ────────────────────────────────────────────────────────────
    "policy_lookup": [
        ["policy", "section", "clause"],
        ["version", "effective date"],
        ["contact", "hr", "human resources", "manager"],
    ],
    "benefits": [
        ["benefit", "coverage", "plan"],
        ["eligible", "eligibility"],
        ["enroll", "enrollment", "open enrollment"],
    ],
    "leave_policy": [
        ["leave", "pto", "vacation", "sick"],
        ["days", "hours", "weeks"],
        ["request", "approval", "submit"],
    ],
    "remote_work": [
        ["remote", "work from home", "wfh", "flexible"],
        ["equipment", "reimburs", "stipend"],
        ["policy", "approval", "eligible"],
    ],
}

# Agent-level fallback when intent isn't specifically mapped
_AGENT_KEYWORDS: dict[str, list[list[str]]] = {
    "data_analyst":      [["take rate", "campaign", "funnel", "cohort", "anomaly"], ["%"], ["subscribers", "passings"]],
    "financial_planner": [["npv", "irr", "payback", "capex", "propensity"], ["$", "cost"], ["recommendation"]],
    "strategist":        [["content", "post", "copy"], ["platform", "linkedin", "meta"], ["hashtag", "#"]],
    "hr_docqa":          [["policy", "section", "version", "effective"], ["hr", "human resources", "contact"]],
}


def _structural_score(text: str, agent: str, intent: str) -> tuple[float, list[str]]:
    """
    Intent-aware structural completeness check.
    Uses the intent-specific keyword groups when available, falls back to agent-level.
    Returns (score 0.0–1.0, list of missing group descriptions).
    """
    if not text:
        return 0.0, ["empty response"]

    text_lower = text.lower()
    groups = _INTENT_KEYWORDS.get(intent) or _AGENT_KEYWORDS.get(agent, [])
    if not groups:
        return 1.0, []

    passed, missing = 0, []
    for group in groups:
        if any(kw in text_lower for kw in group):
            passed += 1
        else:
            missing.append(f"missing: {group[0]!r}")

    return round(passed / len(groups), 3), missing


# ── Numeric faithfulness (grounding against tool outputs) ─────────────────────

_PERCENTAGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_DOLLAR_RE     = re.compile(r"\$[\d,]+(?:\.\d{2})?[KMB]?", re.I)
_PLAIN_NUM_RE  = re.compile(r"\b\d{4,}\b")   # large numbers (passings, subscribers)

# Tolerance: a number in the response is "faithful" if it's within this fraction
# of a number found in the tool outputs. 5% covers rounding and formatting.
_FAITHFULNESS_TOLERANCE = 0.05


def _extract_numbers(text: str) -> list[float]:
    """Pull all numeric values from a text string."""
    nums: list[float] = []
    for m in _PERCENTAGE_RE.findall(text):
        nums.append(float(m))
    for raw in _DOLLAR_RE.findall(text):
        # Strip $, commas, and K/M/B suffixes
        clean = re.sub(r"[$,]", "", raw)
        multiplier = 1
        if clean.upper().endswith("K"):
            multiplier, clean = 1_000, clean[:-1]
        elif clean.upper().endswith("M"):
            multiplier, clean = 1_000_000, clean[:-1]
        elif clean.upper().endswith("B"):
            multiplier, clean = 1_000_000_000, clean[:-1]
        try:
            nums.append(float(clean) * multiplier)
        except ValueError:
            pass
    for m in _PLAIN_NUM_RE.findall(text):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def _numbers_from_tool_results(tool_results: list[dict]) -> list[float]:
    """Extract all numeric values from the accumulated tool result payloads."""
    nums: list[float] = []
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, bytes):
            content = content.decode()
        nums.extend(_extract_numbers(str(content)))
    return nums


def _is_faithful(response_num: float, tool_nums: list[float]) -> bool:
    """True if response_num is within tolerance of any number in tool_nums."""
    if not tool_nums:
        return True   # no tool outputs to compare against — can't penalise
    for tn in tool_nums:
        if tn == 0:
            if abs(response_num) < 1e-6:
                return True
            continue
        if abs(response_num - tn) / abs(tn) <= _FAITHFULNESS_TOLERANCE:
            return True
    return False


def _numeric_grounding_score(
    text: str,
    agent: str,
    tool_results: list[dict],
) -> tuple[float, list[str]]:
    """
    For analyst/financial agents: cross-checks numbers in the response against
    what the tools actually returned. A number is faithful if it matches any
    tool output value within FAITHFULNESS_TOLERANCE.

    Scoring:
      - No numbers in response when numbers are expected → 0.3
      - Numbers present but none match tool outputs      → 0.5 (possible hallucination)
      - Numbers present, some match tool outputs         → 0.85
      - All numbers match tool outputs                   → 1.0
      - Implausible range detected (e.g. take rate >100%) → capped at 0.7

    For strategist/hr: presence-only check (unchanged).
    """
    notes: list[str] = []

    if agent in ("data_analyst", "financial_planner"):
        response_nums = _extract_numbers(text)
        tool_nums     = _numbers_from_tool_results(tool_results)

        if not response_nums:
            return 0.3, ["no numeric data found in analyst/financial response"]

        # Plausibility: percentages should be 0–100
        for n in _extract_numbers(text):
            if "%" in text and n > 100:
                notes.append(f"implausible percentage: {n}%")

        # If no tool results were collected (agent called no tools), we can
        # only do a presence check — don't penalise for missing faithfulness data.
        if not tool_nums:
            notes.append("no tool outputs to cross-check — presence-only check applied")
            return (0.85, notes) if not notes or all("implausible" not in n for n in notes) else (0.70, notes)

        faithful     = [n for n in response_nums if _is_faithful(n, tool_nums)]
        unfaithful   = [n for n in response_nums if not _is_faithful(n, tool_nums)]

        if unfaithful:
            notes.append(
                f"{len(unfaithful)} number(s) in response not found in tool outputs "
                f"(possible hallucination): {unfaithful[:5]}"
            )

        if not faithful:
            return 0.50, notes + ["no response numbers match tool output values"]

        faithfulness_ratio = len(faithful) / len(response_nums)
        if faithfulness_ratio >= 1.0 and not notes:
            return 1.0, []
        if faithfulness_ratio >= 0.75:
            return 0.85, notes
        return 0.65, notes

    if agent == "strategist":
        platforms    = ["linkedin", "meta", "instagram", "facebook", "x", "twitter", "youtube"]
        has_platform = any(p in text.lower() for p in platforms)
        has_hashtag  = "#" in text
        if not has_platform and not has_hashtag:
            return 0.70, ["no platform or hashtag found in strategist response"]
        return 1.0, []

    # HR — text-heavy by nature, no numeric check
    return 1.0, []


# ── Citation coverage ─────────────────────────────────────────────────────────

_CITATION_PATTERNS = [
    re.compile(r"(section\s+\d[\d.]*)", re.I),
    re.compile(r"(policy\s+[A-Z0-9-]+)", re.I),
    re.compile(r"(version\s+\d[\d.]*)", re.I),
    re.compile(r"(effective\s+\d{4})", re.I),
    re.compile(r"(POL-\d{4}-[A-Z]+-\d+)", re.I),
    re.compile(r"(source[s]?:)", re.I),
]


def _citation_score(text: str, agent: str) -> tuple[float, list[str]]:
    """
    HR and Financial agents must cite sources. Strategist/Analyst are scored leniently.
    """
    if agent == "hr_docqa":
        has_citation = any(p.search(text) for p in _CITATION_PATTERNS)
        if not has_citation:
            return 0.50, ["HR response must cite policy name, section, version, and effective date"]
        return 1.0, []

    if agent == "financial_planner":
        has_calix = "calix" in text.lower() or "data vintage" in text.lower()
        has_model  = "model" in text.lower() or "assumption" in text.lower()
        if not has_calix:
            return 0.80, ["financial response should reference Calix data vintage"]
        if not has_model:
            return 0.90, ["financial response should state model assumptions"]
        return 1.0, []

    # Analyst and strategist: citations are good but not required for fidelity
    return 1.0, []


# ── Composite scorer ──────────────────────────────────────────────────────────

def _compute_fidelity(
    text:         str,
    agent:        str,
    intent:       str,
    tool_results: list[dict],
    state:        AgentState,
) -> tuple[float, list[str]]:
    """
    Composite fidelity = 0.30 × structural + 0.40 × numeric + 0.30 × citation

    structural: intent-aware — checks expected fields for the specific intent,
                not just the broad agent category
    numeric:    faithfulness — cross-checks response numbers against tool outputs
    citation:   unchanged — HR/financial must cite sources
    """
    struct_score,  struct_notes = _structural_score(text, agent, intent)
    numeric_score, num_notes    = _numeric_grounding_score(text, agent, tool_results)
    citation_score, cit_notes   = _citation_score(text, agent)

    composite = round(
        struct_score  * 0.30 +
        numeric_score * 0.40 +
        citation_score * 0.30,
        3,
    )

    all_notes = struct_notes + num_notes + cit_notes
    return composite, all_notes


def _evidence_coverage(fidelity: float) -> str:
    if fidelity >= EVIDENCE_SUFFICIENT:  return "full"
    if fidelity >= EVIDENCE_PARTIAL:     return "partial"
    return "insufficient"


def _confidence(text: str, agent: str, tool_results: list[dict]) -> float:
    """
    Confidence derived primarily from tool usage — agents that called tools and
    got real data back are inherently more trustworthy than those that didn't.

    Tiers:
      - Called tools and got non-error results : base 0.80
      - Called tools but all errored           : base 0.55
      - Called no tools (analyst/financial)    : base 0.45  (should have used tools)
      - Called no tools (strategist/hr)        : base 0.65  (expected for generative)

    Bonus: +0.05 for length > 500 chars, +0.05 for length > 1000 chars,
           +0.05 for markdown structure (## or **)
    """
    successful_calls = [
        t for t in tool_results
        if "error" not in str(t.get("content", "")).lower()[:50]
    ]
    total_calls = len(tool_results)

    if total_calls == 0:
        base = 0.45 if agent in ("data_analyst", "financial_planner") else 0.65
    elif successful_calls:
        base = 0.80
    else:
        base = 0.55

    if len(text) > 500:  base += 0.05
    if len(text) > 1000: base += 0.05
    if "**" in text or "##" in text: base += 0.05

    return round(min(base, 1.0), 2)


# ── ReportValidatorAgent ──────────────────────────────────────────────────────

class ReportValidatorAgent:
    """LangGraph node — runs after each specialist agent."""

    async def invoke(self, state: AgentState) -> dict:
        agent  = state.get("routed_agent", "")
        intent = state.get("intent", "")

        output_map: dict[str, Any] = {
            "data_analyst":      state.get("analyst_output"),
            "financial_planner": state.get("planner_output"),
            "strategist":        state.get("strategist_output"),
            "hr_docqa":          state.get("hr_output"),
        }
        output = output_map.get(agent) or {}

        response_text = (
            output.get("response")
            or output.get("answer")
            or str(output)
        )

        # Extract tool results from the specialist output for faithfulness check
        tool_results: list[dict] = output.get("tool_results") or []

        fidelity, notes = _compute_fidelity(response_text, agent, intent, tool_results, state)
        confidence      = _confidence(response_text, agent, tool_results)
        evidence        = _evidence_coverage(fidelity)

        # Add a note if below threshold
        if fidelity < FIDELITY_THRESHOLD:
            retry = state.get("retry_count", 0)
            notes.append(
                f"fidelity {fidelity:.2%} < threshold {FIDELITY_THRESHOLD:.2%}; "
                f"{'retrying' if retry == 0 else 'passing through (retry exhausted)'}"
            )

        return {
            "fidelity_score":    fidelity,
            "output_confidence": confidence,
            "evidence_coverage": evidence,
            "validation_notes":  notes,
            "retry_count":       state.get("retry_count", 0) + (1 if fidelity < FIDELITY_THRESHOLD else 0),
        }
