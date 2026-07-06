"""
services/decision_engine.py
============================
DecisionEngine — cross-agent delegation and compound query orchestration.

Enables agents to call each other's tools without duplicating implementations:
  - Data Analyst  needs Calix demographics  → delegate to Financial Planner tools
  - Strategist    needs current take rate   → delegate to Data Analyst tools
  - Financial     needs campaign context    → delegate to Data Analyst tools

Also handles compound queries that require multiple agents:
  "What's the take rate in 98101 and what content should we post about it?"
  → Data Analyst for take rate, then Strategist for content, results merged.
"""

from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import Any, Callable


# ── Delegation registry ────────────────────────────────────────────────────────

@dataclass
class DelegationResult:
    source_agent: str
    target_agent: str
    tool_name:    str
    inputs:       dict
    result:       Any
    success:      bool
    error:        str | None = None


# Maps tool names to their owning agent module so other agents can borrow them.
# Populated lazily on first use to avoid circular imports at module load time.
_TOOL_REGISTRY: dict[str, tuple[str, Callable]] = {}


def _build_registry() -> None:
    """Lazily imports all agent tool functions and registers them by name."""
    if _TOOL_REGISTRY:
        return

    try:
        from agents.specialist.data_analyst import (
            take_rate_calculator, campaign_performance_fetcher,
            funnel_metrics_builder, anomaly_detector, benchmark_comparator,
        )
        for fn in [take_rate_calculator, campaign_performance_fetcher,
                   funnel_metrics_builder, anomaly_detector, benchmark_comparator]:
            _TOOL_REGISTRY[fn.__name__] = ("data_analyst", fn)
    except Exception:
        pass

    try:
        from agents.specialist.financial_planner import (
            calix_demographics_api, copper_takerate_fetcher,
            fiber_propensity_model, capex_estimator, roi_npv_calculator,
        )
        for fn in [calix_demographics_api, copper_takerate_fetcher,
                   fiber_propensity_model, capex_estimator, roi_npv_calculator]:
            _TOOL_REGISTRY[fn.__name__] = ("financial_planner", fn)
    except Exception:
        pass

    try:
        from agents.specialist.strategist import (
            social_trend_fetcher, competitor_content_analyzer,
            hashtag_optimizer, ftc_compliance_checker,
        )
        for fn in [social_trend_fetcher, competitor_content_analyzer,
                   hashtag_optimizer, ftc_compliance_checker]:
            _TOOL_REGISTRY[fn.__name__] = ("strategist", fn)
    except Exception:
        pass


def build_delegated_tools(caller_agent: str, tool_names: list[str]) -> dict[str, Callable]:
    """
    Factory that generates async delegation wrappers for cross-agent tools.
    Returns a dict ready to be spread into a specialist's TOOL_FUNCTIONS.

    Usage in any specialist:
        TOOL_FUNCTIONS = {
            "my_own_tool": my_own_tool,
            **build_delegated_tools("strategist", ["take_rate_calculator"]),
        }

    The corresponding TOOL_DEFINITIONS entries (with context-specific descriptions
    telling Claude when to call each tool) should still be written manually in the
    specialist's TOOL_DEFINITIONS list.
    """
    tool_functions: dict[str, Callable] = {}
    for tool_name in tool_names:
        async def _wrapper(_tn: str = tool_name, _ca: str = caller_agent, **kwargs):
            result = await delegate_tool(tool_name=_tn, inputs=kwargs, caller_agent=_ca)
            if not result.success:
                return {"error": result.error}
            return result.result
        tool_functions[tool_name] = _wrapper
    return tool_functions


async def delegate_tool(
    tool_name:    str,
    inputs:       dict,
    caller_agent: str = "unknown",
) -> DelegationResult:
    """
    Calls a tool owned by another agent on behalf of the caller.
    Used for cross-agent data sharing without duplicating tool implementations.

    Example:
        result = await delegate_tool(
            tool_name    = "calix_demographics_api",
            inputs       = {"zip_codes": ["98101", "98102"]},
            caller_agent = "data_analyst",
        )
    """
    _build_registry()

    entry = _TOOL_REGISTRY.get(tool_name)
    if not entry:
        return DelegationResult(
            source_agent = caller_agent,
            target_agent = "unknown",
            tool_name    = tool_name,
            inputs       = inputs,
            result       = None,
            success      = False,
            error        = f"Tool '{tool_name}' not found in registry. Available: {list(_TOOL_REGISTRY)}",
        )

    target_agent, fn = entry
    try:
        result = await fn(**inputs) if inspect.iscoroutinefunction(fn) else fn(**inputs)
        if isinstance(result, list):
            result = [r.__dict__ if hasattr(r, "__dataclass_fields__") else r for r in result]
        elif hasattr(result, "__dataclass_fields__"):
            result = result.__dict__

        return DelegationResult(
            source_agent = caller_agent,
            target_agent = target_agent,
            tool_name    = tool_name,
            inputs       = inputs,
            result       = result,
            success      = True,
        )
    except Exception as exc:
        return DelegationResult(
            source_agent = caller_agent,
            target_agent = target_agent,
            tool_name    = tool_name,
            inputs       = inputs,
            result       = None,
            success      = False,
            error        = str(exc),
        )


# ── Compound query orchestration ───────────────────────────────────────────────

@dataclass
class CompoundQueryPlan:
    """Describes a multi-agent compound query."""
    steps:       list[dict]   # [{agent, query}]
    merge_mode:  str          # "sequential" | "parallel"
    description: str


@dataclass
class CompoundQueryResult:
    steps_completed: int
    results:         dict[str, Any]   # {agent: output}
    merged_response: str


def plan_compound_query(query: str) -> CompoundQueryPlan | None:
    """
    Detects whether a query spans multiple agents and builds an execution plan.
    Returns None if the query maps to a single agent.

    Compound triggers:
      - "take rate AND content/strategy"  → Data Analyst + Strategist
      - "financial AND social"            → Financial Planner + Strategist
      - "fiber build AND campaign"        → Financial Planner + Data Analyst
    """
    q = query.lower()

    has_analytics  = any(w in q for w in ["take rate", "campaign", "funnel", "cac", "roas"])
    has_financial  = any(w in q for w in ["npv", "copper", "fiber build", "capex", "calix"])
    has_strategy   = any(w in q for w in ["content", "post", "social", "hashtag", "strategy"])

    agents_needed = []
    if has_analytics:  agents_needed.append("data_analyst")
    if has_financial:  agents_needed.append("financial_planner")
    if has_strategy:   agents_needed.append("strategist")

    if len(agents_needed) < 2:
        return None   # single-agent query, normal routing applies

    # Build sequential plan: each step gets the previous step's result as context
    steps = [{"agent": agent, "query": query} for agent in agents_needed]

    return CompoundQueryPlan(
        steps       = steps,
        merge_mode  = "sequential",
        description = f"Compound query requiring: {', '.join(agents_needed)}",
    )


async def execute_compound_query(
    plan:    CompoundQueryPlan,
    context: list[dict] | None = None,
) -> CompoundQueryResult:
    """
    Executes a compound query plan by running each agent in sequence.
    Each agent's output is injected into the next agent's context.
    """
    results: dict[str, Any] = {}
    running_context = list(context or [])

    for step in plan.steps:
        agent_name = step["agent"]
        query      = step["query"]

        try:
            agent = _get_agent_instance(agent_name)
            if agent is None:
                results[agent_name] = {"error": f"Agent '{agent_name}' could not be loaded"}
                continue

            output = await agent.run(query=query, context=running_context)
            results[agent_name] = output

            # Inject this agent's response into running context for next agent
            response_text = output.get("response", str(output))
            running_context.append({"role": "assistant", "content": f"[{agent_name}]: {response_text[:500]}"})

        except Exception as exc:
            results[agent_name] = {"error": str(exc)}

    # Build merged response: concatenate with agent headers
    merged_parts = []
    for agent_name, output in results.items():
        response = output.get("response", str(output)) if isinstance(output, dict) else str(output)
        merged_parts.append(f"**{agent_name.replace('_', ' ').title()}**\n{response}")

    return CompoundQueryResult(
        steps_completed = len(results),
        results         = results,
        merged_response = "\n\n---\n\n".join(merged_parts),
    )


def _get_agent_instance(agent_name: str):
    """Returns a fresh agent instance by name. Returns None on import failure."""
    try:
        if agent_name == "data_analyst":
            from agents.specialist.data_analyst import DataAnalystAgent
            return DataAnalystAgent()
        if agent_name == "financial_planner":
            from agents.specialist.financial_planner import FinancialPlannerAgent
            return FinancialPlannerAgent()
        if agent_name == "strategist":
            from agents.specialist.strategist import StrategistAgent
            return StrategistAgent()
        if agent_name == "hr_docqa":
            from agents.hr.hr_docqa import HRDocQAAgent
            return HRDocQAAgent()
    except Exception:
        return None
    return None
