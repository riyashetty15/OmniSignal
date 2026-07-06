"""
Data Analyst Agent
Answers any internal analytics question — take rates, campaign performance,
funnel metrics, cohort analysis, anomaly detection.

All numeric calculations are DETERMINISTIC Python — never LLM-estimated.
The LLM interprets, narrates, and structures the results only.
"""

from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from anthropic import Anthropic
from shared_config import (
    AGENT_MODELS, INDUSTRY_BENCHMARKS, AZURE_SQL_SCHEMA
)

client = Anthropic()
MODEL  = AGENT_MODELS["data_analyst"]

# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TakeRateResult:
    rate: float
    passings: int
    active_subs: int
    region: str
    period: str
    delta_pp: Optional[float] = None      # vs prior period
    vs_industry: Optional[float] = None   # vs national benchmark
    industry_avg: float = INDUSTRY_BENCHMARKS["fiber_take_rate_national"]

    def to_summary(self) -> str:
        s = f"Take rate: {self.rate:.1%} ({self.active_subs:,} subs / {self.passings:,} passings)"
        if self.delta_pp is not None:
            direction = "+" if self.delta_pp >= 0 else ""
            s += f" | {direction}{self.delta_pp:.1%}pp vs prior period"
        s += f" | Industry avg: {self.industry_avg:.1%}"
        return s


@dataclass
class FunnelResult:
    stages: list[str]
    counts: list[int]
    conversion_rates: list[float]
    drop_offs: list[float]
    biggest_leak: str
    campaign_id: str


@dataclass
class CampaignMetrics:
    campaign_id: str
    channel: str
    impressions: int
    clicks: int
    leads: int
    installs: int
    spend: float
    revenue: float
    roas: float
    cac: float
    cpl: float
    date_range: tuple[str, str]


@dataclass
class CohortRow:
    cohort_month: str
    month_0_subs: int
    retention_by_month: list[float]   # retention % at each month


@dataclass
class Anomaly:
    date: str
    metric: str
    value: float
    z_score: float
    direction: str   # "spike" | "drop"


# ─────────────────────────────────────────────────────────────────────────────
#  Deterministic Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

async def sql_query_engine(
    nl_query: str,
    filters: dict = None,
    db_client=None,
) -> dict:
    """
    Translates natural language to SQL, validates read-only safety,
    executes against Azure SQL DW, returns structured results.

    SAFETY: All generated SQL is validated as read-only before execution.
    Any INSERT/UPDATE/DELETE/DROP is blocked.
    """
    filters = filters or {}

    # In production: call Azure OpenAI or Claude to generate SQL
    # with the DW schema injected as context
    sql = await _nl_to_sql(nl_query, filters)

    # Safety gate — block any mutating statements
    _validate_read_only(sql)

    if db_client:
        result = await db_client.execute(sql, params=filters)
        return {
            "rows": result.rows,
            "columns": result.columns,
            "row_count": len(result.rows),
            "sql_used": sql,
            "execution_ms": result.latency_ms,
        }
    # Mock for development
    return {"rows": [], "columns": [], "row_count": 0, "sql_used": sql}


async def _nl_to_sql(nl_query: str, filters: dict) -> str:
    """Generate SQL from natural language using Claude with schema context."""
    response = client.messages.create(
        model=MODEL, max_tokens=512,
        system=f"""You are a SQL generator for the {AZURE_SQL_SCHEMA} marketing data warehouse.
Schema tables: fiber_metrics(region, zip, period, passings, active_subs, take_rate),
campaign_performance(campaign_id, channel, date, impressions, clicks, leads, installs, spend, revenue),
funnel_events(campaign_id, stage, date, count), copper_subscribers(zip, subs, passings, arpu, tenure_months).
Generate read-only SELECT SQL only. Return SQL and nothing else.""",
        messages=[{"role": "user", "content": f"Query: {nl_query}\nFilters: {json.dumps(filters)}"}]
    )
    return response.content[0].text.strip()


def _validate_read_only(sql: str):
    """Block any mutating SQL operations."""
    dangerous = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE)\b",
        re.IGNORECASE
    )
    if dangerous.search(sql):
        raise ValueError(f"Mutating SQL blocked: {sql[:100]}")


def take_rate_calculator(
    region: str,
    period: str,
    compare_period: str = None,
    db_client=None,
) -> TakeRateResult:
    """
    DETERMINISTIC: take_rate = active_subs / total_passings
    Never estimated by LLM. Pulled directly from subscriber DB.
    """
    # In production: query Azure SQL
    # passings = db_client.scalar("SELECT SUM(passings) FROM fiber_metrics WHERE region=%s AND period=%s", region, period)
    # active_subs = db_client.scalar("SELECT SUM(active_subs) FROM fiber_metrics WHERE region=%s AND period=%s", region, period)

    # Mock values for scaffold
    passings    = 45_000
    active_subs = 10_800
    rate        = active_subs / passings

    delta_pp = None
    if compare_period:
        prior_subs = 9_765  # mock
        delta_pp   = rate - (prior_subs / passings)

    return TakeRateResult(
        rate=rate, passings=passings, active_subs=active_subs,
        region=region, period=period,
        delta_pp=delta_pp,
        vs_industry=rate - INDUSTRY_BENCHMARKS["fiber_take_rate_national"],
    )


async def campaign_performance_fetcher(
    campaign_id: str,
    metrics: list[str],
    date_range: tuple[str, str],
    group_by: list[str] = None,
    db_client=None,
) -> list[CampaignMetrics]:
    """
    Pulls campaign KPIs from marketing DW.
    Calculates derived metrics (ROAS, CAC, CPL) deterministically.
    """
    group_by = group_by or ["channel"]

    # In production: query Azure Synapse
    # rows = await db_client.execute(
    #     "SELECT channel, SUM(impressions), SUM(clicks), SUM(installs), SUM(spend), SUM(revenue) "
    #     "FROM campaign_performance WHERE campaign_id=? AND date BETWEEN ? AND ? GROUP BY channel",
    #     campaign_id, date_range[0], date_range[1]
    # )

    # Mock scaffold
    mock_row = {
        "campaign_id": campaign_id, "channel": "paid_search",
        "impressions": 500_000, "clicks": 12_500,
        "leads": 2_800, "installs": 420,
        "spend": 84_000, "revenue": 336_000,
    }
    spend, installs, revenue = mock_row["spend"], mock_row["installs"], mock_row["revenue"]

    return [CampaignMetrics(
        **mock_row,
        roas=round(revenue / spend, 2),
        cac=round(spend / installs, 2),
        cpl=round(spend / mock_row["leads"], 2),
        date_range=date_range,
    )]


def funnel_metrics_builder(
    campaign_id: str,
    stages: list[str] = None,
    db_client=None,
) -> FunnelResult:
    """
    Builds full marketing funnel: impression → click → lead → install → active_sub.
    Calculates conversion rates and drop-off at every stage deterministically.
    """
    stages = stages or ["impression","click","lead","install","active_sub"]

    # Mock counts — in production from DW
    counts = [500_000, 12_500, 2_800, 420, 378]

    rates     = [round(counts[i] / counts[i-1], 4) for i in range(1, len(counts))]
    drop_offs = [round(1 - r, 4) for r in rates]

    stage_pairs = list(zip(stages[:-1], drop_offs))
    biggest_leak = stage_pairs[drop_offs.index(max(drop_offs))][0]

    return FunnelResult(
        stages=stages, counts=counts,
        conversion_rates=rates, drop_offs=drop_offs,
        biggest_leak=biggest_leak, campaign_id=campaign_id,
    )


async def cohort_analyzer(
    cohort_start: str,
    cohort_end: str,
    segment: str = "all",
    metric: str = "retention",
    db_client=None,
) -> list[CohortRow]:
    """
    Segments subscribers by install cohort month.
    Analyzes: retention curves, ARPU evolution, churn patterns per cohort.
    """
    # In production: query subscriber DB grouped by install month
    # cohorts = await db_client.query(
    #     "SELECT DATE_TRUNC('month', install_date) as cohort, ..."
    # )

    # Mock scaffold: 3 cohorts
    return [
        CohortRow("2024-Q1", 850, [1.0, 0.96, 0.93, 0.91, 0.89, 0.87]),
        CohortRow("2024-Q2", 920, [1.0, 0.95, 0.92, 0.90, 0.88]),
        CohortRow("2024-Q3", 780, [1.0, 0.97, 0.94, 0.92]),
    ]


def anomaly_detector(
    metric: str,
    region: str,
    lookback_days: int = 30,
    z_threshold: float = 2.5,
    db_client=None,
) -> list[Anomaly]:
    """
    Z-score based anomaly detection on any metric time series.
    Flags values > z_threshold standard deviations from rolling mean.
    Never uses LLM to identify anomalies — pure statistics.
    """
    # In production: pull time series from DW
    # series = pd.Series(db_client.get_series(metric, region, lookback_days))

    # Mock scaffold
    series = pd.Series({
        "2024-10-01": 0.238, "2024-10-02": 0.241, "2024-10-03": 0.189,  # anomaly
        "2024-10-04": 0.240, "2024-10-05": 0.242, "2024-10-06": 0.239,
    })

    mean = series.mean()
    std  = series.std()

    return [
        Anomaly(
            date=str(date), metric=metric, value=float(val),
            z_score=round((val - mean) / std, 2),
            direction="spike" if val > mean else "drop",
        )
        for date, val in series.items()
        if abs((val - mean) / std) > z_threshold
    ]


def benchmark_comparator(
    metric: str,
    actual_value: float,
) -> dict:
    """Compares an actual metric against industry benchmarks."""
    benchmark = INDUSTRY_BENCHMARKS.get(metric)
    if not benchmark:
        return {"benchmark_found": False, "metric": metric}

    delta    = actual_value - benchmark
    delta_pct = delta / benchmark

    return {
        "metric":       metric,
        "actual":       round(actual_value, 4),
        "benchmark":    round(benchmark, 4),
        "delta":        round(delta, 4),
        "delta_pct":    round(delta_pct, 4),
        "above_benchmark": delta >= 0,
        "interpretation": (
            f"{'Above' if delta >= 0 else 'Below'} industry benchmark "
            f"by {abs(delta_pct):.1%}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the internal Data Analyst for a fiber broadband company.

Your role:
1. Translate business questions into precise SQL queries using the sql_query_engine tool
2. Calculate metrics using deterministic tools (take_rate_calculator, funnel_metrics_builder, etc.)
3. Interpret results in plain business language — don't just report numbers, explain them
4. Always show the formula or SQL used — never hide the calculation
5. Flag anomalies. Compare against industry benchmarks when available
6. If data is missing or the question can't be answered, say exactly what data is needed

Output rules:
- Round all metrics to 2 decimal places
- Always specify time period and geographic scope
- For take rate: state passings, subs, and rate — all three
- For campaigns: always include ROAS alongside absolute numbers
- Use tables when presenting multi-row comparisons
- End every response with a "Data freshness" note (when was this data last updated)

CRITICAL: You NEVER guess or estimate numbers. All calculations come from tools.
"""

TOOL_DEFINITIONS = [
    {
        "name": "sql_query_engine",
        "description": "Translates a natural language question into SQL and executes it against the marketing data warehouse. Use for any question that needs raw data retrieval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nl_query": {"type": "string", "description": "The natural language question to convert to SQL"},
                "filters": {"type": "object", "description": "Optional filter overrides: {region, period, channel, campaign_id}"}
            },
            "required": ["nl_query"]
        }
    },
    {
        "name": "take_rate_calculator",
        "description": "Calculates fiber take rate (active_subs / passings) for a region and period. Optionally compares to a prior period. Always deterministic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string"},
                "period": {"type": "string", "description": "e.g. 'Q3-2024', '2024-10', 'YTD-2024'"},
                "compare_period": {"type": "string", "description": "Optional prior period to delta against"}
            },
            "required": ["region", "period"]
        }
    },
    {
        "name": "campaign_performance_fetcher",
        "description": "Pulls campaign KPIs: impressions, clicks, leads, installs, ROAS, CAC, CPL by channel. Use for any campaign analytics question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}, "description": "e.g. ['roas','cac','installs','cpl']"},
                "date_range": {"type": "array", "items": {"type": "string"}, "description": "['YYYY-MM-DD', 'YYYY-MM-DD']"}
            },
            "required": ["campaign_id", "date_range"]
        }
    },
    {
        "name": "funnel_metrics_builder",
        "description": "Builds full marketing funnel with conversion rates and drop-off at each stage. Identifies the biggest leak.",
        "input_schema": {
            "type": "object",
            "properties": {
                "campaign_id": {"type": "string"},
                "stages": {"type": "array", "items": {"type": "string"}, "description": "Optional custom stages. Defaults to impression→click→lead→install→active_sub"}
            },
            "required": ["campaign_id"]
        }
    },
    {
        "name": "cohort_analyzer",
        "description": "Segments subscribers by install cohort and analyzes retention, ARPU, and churn over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cohort_start": {"type": "string"},
                "cohort_end": {"type": "string"},
                "metric": {"type": "string", "enum": ["retention","arpu","churn"]},
                "segment": {"type": "string", "description": "all | residential | business"}
            },
            "required": ["cohort_start", "cohort_end"]
        }
    },
    {
        "name": "anomaly_detector",
        "description": "Detects statistical anomalies (z-score > 2.5σ) in any metric time series. Use when user asks about unusual patterns or drops.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "region": {"type": "string"},
                "lookback_days": {"type": "integer", "default": 30}
            },
            "required": ["metric", "region"]
        }
    },
    {
        "name": "benchmark_comparator",
        "description": "Compares an actual metric value against the industry benchmark. Use after calculating any metric to add context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "e.g. 'fiber_take_rate_national', 'roas_typical', 'cac_residential'"},
                "actual_value": {"type": "number"}
            },
            "required": ["metric", "actual_value"]
        }
    },
    {
        "name": "calix_demographics_api",
        "description": "Fetches Calix ZIP-level demographic data (median income, broadband spend, household size, fiber propensity score). Delegated from financial_planner. Use when analytics need geographic demographic context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_codes": {"type": "array", "items": {"type": "string"}, "description": "List of ZIP codes to fetch demographics for"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of demographic fields to return"}
            },
            "required": ["zip_codes"]
        }
    },
]

from services.decision_engine import build_delegated_tools

TOOL_FUNCTIONS = {
    "sql_query_engine":             sql_query_engine,
    "take_rate_calculator":         lambda **kw: take_rate_calculator(**kw),
    "campaign_performance_fetcher": campaign_performance_fetcher,
    "funnel_metrics_builder":       lambda **kw: funnel_metrics_builder(**kw),
    "cohort_analyzer":              cohort_analyzer,
    "anomaly_detector":             lambda **kw: anomaly_detector(**kw),
    "benchmark_comparator":         lambda **kw: benchmark_comparator(**kw),
    **build_delegated_tools("data_analyst", ["calix_demographics_api"]),
}


class DataAnalystAgent:
    """
    Data Analyst agent with agentic tool loop.
    Runs tools deterministically, interprets with Claude.
    """

    async def invoke(self, state: dict) -> dict:
        """LangGraph node interface — wraps run() with state I/O."""
        result = await self.run(
            query   = state.get("user_query", ""),
            context = state.get("context_window", []),
        )
        completed = list(state.get("completed_agents") or [])
        completed.append("data_analyst")
        return {"analyst_output": result, "completed_agents": completed}

    async def run(self, query: str, context: list[dict] = None) -> dict:
        context = context or []
        messages = context[-6:] + [{"role": "user", "content": query}]
        all_tool_results: list[dict] = []   # accumulated across all loop iterations

        while True:
            response = client.messages.create(
                model=MODEL, max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            # If Claude stopped without tool calls → final answer
            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")), ""
                )
                return {
                    "response":     final_text,
                    "agent":        "data_analyst",
                    "tool_calls":   [t["tool_name"] for t in all_tool_results],
                    "tool_results": all_tool_results,
                }

            # Execute all tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_fn = TOOL_FUNCTIONS.get(block.name)
                if not tool_fn:
                    result = {"error": f"Unknown tool: {block.name}"}
                else:
                    try:
                        if asyncio.iscoroutinefunction(tool_fn):
                            result = await tool_fn(**block.input)
                        else:
                            result = tool_fn(**block.input)
                        # Convert dataclass to dict if needed
                        if hasattr(result, "__dataclass_fields__"):
                            result = result.__dict__
                    except Exception as e:
                        result = {"error": str(e)}

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "tool_name":   block.name,
                    "content":     json.dumps(result, default=str),
                })

            all_tool_results.extend(tool_results)
            messages.append({"role": "user", "content": tool_results})