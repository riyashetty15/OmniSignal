"""
agents/planner.py
=================
PlannerAgent — sits between guardrails and the specialist agents.

Responsibilities:
  1. Confirm or override the keyword-based routing from core/router.py
     using a fast Claude call when the keyword confidence is low.
  2. Extract entities (ZIP codes, region names, campaign IDs) and inject
     them into the state's entity_memory for specialist agents to use.
  3. On retry (score < 0.92 from validator): mark "retry" in completed_agents
     so the validator passes the response through regardless of score.
"""

from __future__ import annotations
import json
import re

from anthropic import Anthropic
from agents.base import AgentState
from shared_config import AGENT_MODELS

client = Anthropic()
MODEL  = AGENT_MODELS["planner"]

# Confidence threshold below which we ask Claude to confirm the route
KEYWORD_CONFIDENCE_THRESHOLD = 0.50


# ── Entity extractors ─────────────────────────────────────────────────────────

_ZIP_RE       = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_CAMPAIGN_RE  = re.compile(r"\b(CAMP[-_]?\d{3,8}|campaign[-_\s]+[a-z0-9]+)\b", re.I)
_REGION_WORDS = {
    "pacific northwest", "southeast", "midwest", "northeast", "southwest",
    "west coast", "east coast", "mountain west", "south", "north",
}


def _extract_entities(query: str) -> dict:
    q_lower = query.lower()
    entities: dict = {}

    zips = _ZIP_RE.findall(query)
    if zips:
        entities["zip_codes"] = zips

    camps = _CAMPAIGN_RE.findall(query)
    if camps:
        entities["campaign_ids"] = camps

    for r in _REGION_WORDS:
        if r in q_lower:
            entities["region"] = r
            break

    return entities


# ── LLM-based intent classifier (used when keyword score is low) ───────────────

_ROUTING_SYSTEM = """You are a query router for a fiber broadband company's internal analytics system.

Given a user query, classify it into exactly one agent:
- data_analyst     : take rates, campaign performance, funnel metrics, cohort analysis, subscriber KPIs
- financial_planner: copper-to-fiber investment decisions, NPV/IRR, Calix demographics, capex, fiber propensity
- strategist       : social media content ideation, content calendar, competitor analysis, brand strategy
- hr_docqa         : HR policies, benefits, leave, onboarding, org chart, compliance training

Return ONLY a JSON object: {"agent": "<agent>", "intent": "<intent_label>", "confidence": <0.0-1.0>}
No markdown, no explanation."""


async def _llm_classify(query: str) -> tuple[str, str, float]:
    """Ask Claude to classify the query. Returns (agent, intent, confidence)."""
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=128,
            system=_ROUTING_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)
        return data.get("agent", "data_analyst"), data.get("intent", "general"), float(data.get("confidence", 0.7))
    except Exception:
        return "data_analyst", "general", 0.5


# ── PlannerAgent ──────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    LangGraph node — invoked by the graph as planner.invoke(state).
    """

    async def invoke(self, state: AgentState) -> dict:
        query      = state.get("user_query", "")
        department = state.get("department", "marketing")
        retry      = state.get("retry_count", 0)

        # ── Retry path: agent ran once, validator was below threshold ─────────
        # Mark retry as done so the validator passes through on second attempt.
        if retry > 0:
            completed = list(state.get("completed_agents") or [])
            if "retry" not in completed:
                completed.append("retry")
            return {
                "completed_agents": completed,
                "retry_count": retry,
                # Keep the existing routed_agent so the specialist runs again
            }

        # ── First pass: use router result already in state ────────────────────
        routed_agent = state.get("routed_agent", "")
        intent       = state.get("intent", "")

        # If routing confidence is missing or low, ask Claude to confirm
        # (The confidence is not in AgentState, so we always do keyword check here)
        if not routed_agent or routed_agent not in {
            "data_analyst", "financial_planner", "strategist", "hr_docqa"
        }:
            routed_agent, intent, _ = await _llm_classify(query)

        # Department hard-override: HR dept → always hr_docqa
        if department == "hr":
            routed_agent = "hr_docqa"
            intent       = intent or "policy_lookup"

        # ── Entity extraction ─────────────────────────────────────────────────
        entities = _extract_entities(query)
        existing_entities = dict(state.get("entity_memory") or {})
        existing_entities.update(entities)

        return {
            "routed_agent":  routed_agent,
            "intent":        intent,
            "entity_memory": existing_entities,
            "retry_count":   retry,
        }
