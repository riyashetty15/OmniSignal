"""
agents/graph.py
===============
LangGraph StateGraph — wires all agents into the execution graph.
Handles routing, sequential/parallel execution, fidelity gating, and retry.
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agents.base import AgentState
from agents.planner import PlannerAgent
from agents.specialist.data_analyst import DataAnalystAgent
from agents.specialist.financial_planner import FinancialPlannerAgent
from agents.specialist.strategist import StrategistAgent
from agents.hr.hr_docqa import HRDocQAAgent
from agents.validation.report_validator import ReportValidatorAgent
from agents.validation.guardrails import GuardrailsAgent


# ─────────────────────────────────────────────────────────────────────────────
#  Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(sqlite_db: str = "data/context_memory.db") -> StateGraph:
    """
    Constructs the full LangGraph execution graph.

    Flow:
      guardrails_pre → planner → [specialist] → validator → response_builder
                                                    ↓ (fidelity < 0.92)
                                               planner (retry once)
    """
    # Instantiate agents
    guardrails  = GuardrailsAgent()
    planner     = PlannerAgent()
    analyst     = DataAnalystAgent()
    fin_planner = FinancialPlannerAgent()
    strategist  = StrategistAgent()
    hr          = HRDocQAAgent()
    validator   = ReportValidatorAgent()

    wf = StateGraph(AgentState)

    # Add all nodes
    wf.add_node("guardrails",         guardrails.invoke)
    wf.add_node("planner",            planner.invoke)
    wf.add_node("data_analyst",       analyst.invoke)
    wf.add_node("financial_planner",  fin_planner.invoke)
    wf.add_node("strategist",         strategist.invoke)
    wf.add_node("hr_docqa",           hr.invoke)
    wf.add_node("validator",          validator.invoke)
    wf.add_node("response_builder",   response_builder_node)

    # Entry point
    wf.set_entry_point("guardrails")

    # guardrails → planner OR blocked end
    wf.add_conditional_edges("guardrails", _route_after_guardrails, {
        "planner":  "planner",
        "end":       END,
    })

    # planner → specialist
    wf.add_conditional_edges("planner", _route_from_planner, {
        "data_analyst":      "data_analyst",
        "financial_planner": "financial_planner",
        "strategist":        "strategist",
        "hr_docqa":          "hr_docqa",
        "validator":         "validator",
    })

    # each specialist → validator
    for node in ["data_analyst", "financial_planner", "strategist", "hr_docqa"]:
        wf.add_edge(node, "validator")

    # validator → response_builder OR retry planner
    wf.add_conditional_edges("validator", _route_after_validation, {
        "response_builder": "response_builder",
        "planner":           "planner",
    })

    wf.add_edge("response_builder", END)

    return wf


async def create_compiled_graph(sqlite_db: str = "data/context_memory.db"):
    """Returns the compiled graph with SQLite checkpointing (context memory)."""
    wf     = build_graph(sqlite_db)
    memory = AsyncSqliteSaver.from_conn_string(sqlite_db)
    return wf.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────────────────────
#  Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_guardrails(state: AgentState) -> Literal["planner", "end"]:
    if state.get("error") and "BLOCKED" in state.get("error", ""):
        return "end"
    if state.get("guardrail_flags") and any(
        "BLOCK" in f for f in state["guardrail_flags"]
    ):
        return "end"
    return "planner"


def _route_from_planner(state: AgentState) -> str:
    agent = state.get("routed_agent", "data_analyst")
    valid = {"data_analyst", "financial_planner", "strategist", "hr_docqa"}
    return agent if agent in valid else "data_analyst"


def _route_after_validation(
    state: AgentState,
) -> Literal["response_builder", "planner"]:
    score         = state.get("fidelity_score", 0.0)
    retry_done    = "retry" in state.get("completed_agents", [])
    if score >= 0.92 or retry_done:
        return "response_builder"
    return "planner"   # retry once if below threshold


# ─────────────────────────────────────────────────────────────────────────────
#  Response builder node
# ─────────────────────────────────────────────────────────────────────────────

async def response_builder_node(state: AgentState) -> AgentState:
    """
    Final node: assembles the specialist output into a clean response.
    Attaches fidelity score, confidence, and validation notes.
    """
    from datetime import datetime

    agent     = state.get("routed_agent", "unknown")
    output_map = {
        "data_analyst":      state.get("analyst_output"),
        "financial_planner": state.get("planner_output"),
        "strategist":        state.get("strategist_output"),
        "hr_docqa":          state.get("hr_output"),
    }
    specialist_output = output_map.get(agent, {}) or {}

    # Extract the response text from whatever the specialist produced
    response_text = (
        specialist_output.get("response")
        or specialist_output.get("answer")
        or str(specialist_output)
    )

    fidelity = state.get("fidelity_score", 0.0)
    notes    = state.get("validation_notes", [])

    # Append fidelity footer
    footer = f"\n\n---\n*Fidelity: {fidelity:.1%} | Agent: {agent} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*"
    if notes:
        footer += "\n*Validation notes: " + "; ".join(notes) + "*"

    return {
        **state,
        "final_response": response_text + footer,
        "updated_at":     datetime.utcnow().isoformat(),
    }