"""
agents/base.py
==============
AgentState TypedDict — the single shared state that flows through the LangGraph graph.
initial_state() — factory that creates a clean state for each new request.
"""

from __future__ import annotations
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────────────
    session_id:   str
    user_id:      str
    user_query:   str
    department:   str          # "marketing" | "hr" | "finance"

    # ── Routing ───────────────────────────────────────────────────────────────
    routed_agent: str          # "data_analyst" | "financial_planner" | "strategist" | "hr_docqa"
    intent:       str          # e.g. "take_rate", "roi_npv", "social_content", "policy_lookup"

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrail_flags: list[str]
    error:           Optional[str]

    # ── Conversation context ──────────────────────────────────────────────────
    context_window:    list[dict]        # recent Q&A turns from SQLite
    entity_memory:     dict[str, Any]   # extracted entities (regions, ZIP codes, campaign IDs)
    retrieved_context: list[dict]       # RAG results passed into specialist agents

    # ── Specialist outputs (exactly one populated per turn) ───────────────────
    analyst_output:   Optional[dict]
    planner_output:   Optional[dict]
    strategist_output:Optional[dict]
    hr_output:        Optional[dict]

    # ── Validation ────────────────────────────────────────────────────────────
    fidelity_score:   float    # 0.0–1.0; must be >= 0.92 to pass fidelity gate
    output_confidence:float    # agent self-assessed confidence 0.0–1.0
    evidence_coverage:str      # "full" | "partial" | "insufficient"
    validation_notes: list[str]

    # ── Retry tracking ────────────────────────────────────────────────────────
    completed_agents: list[str]   # which agents have run this turn
    retry_count:      int

    # ── Final output ──────────────────────────────────────────────────────────
    final_response: str
    updated_at:     str


def initial_state(
    session_id:     str,
    user_id:        str,
    user_query:     str,
    department:     str = "marketing",
    context_window: list[dict] | None = None,
) -> AgentState:
    """
    Creates a clean AgentState for a new request.
    All optional fields start as empty/zero so downstream nodes can safely .get() them.
    """
    return AgentState(
        session_id        = session_id,
        user_id           = user_id,
        user_query        = user_query,
        department        = department,
        routed_agent      = "",
        intent            = "",
        guardrail_flags   = [],
        error             = None,
        context_window    = context_window or [],
        entity_memory     = {},
        retrieved_context = [],
        analyst_output    = None,
        planner_output    = None,
        strategist_output = None,
        hr_output         = None,
        fidelity_score    = 0.0,
        output_confidence = 0.0,
        evidence_coverage = "unknown",
        validation_notes  = [],
        completed_agents  = [],
        retry_count       = 0,
        final_response    = "",
        updated_at        = "",
    )
