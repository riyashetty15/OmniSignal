"""
Financial Planner Agent
Copper-to-fiber infrastructure investment modeling.
Uses Calix Marketing Cloud API (ZIP demographics) + copper subscriber data
to predict fiber adoption and calculate NPV/IRR/payback for build decisions.

All financial math is DETERMINISTIC — never LLM-estimated.
"""

from __future__ import annotations
import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import httpx
import numpy as np
from anthropic import Anthropic
from shared_config import (
    AGENT_MODELS, CALIX_BASE_URL, CALIX_FIELDS,
    INDUSTRY_BENCHMARKS, AZURE_SQL_SCHEMA
)

client = Anthropic()
MODEL  = AGENT_MODELS["financial_planner"]

# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CalixDemographics:
    zip_code: str
    median_household_income: float
    age_distribution: dict           # {"18-34": 0.22, "35-54": 0.38, ...}
    avg_broadband_spend_monthly: float
    household_size: float
    current_broadband_provider: str
    fiber_propensity_score: float    # Calix 0–100 proprietary score
    multi_dwelling_unit_pct: float
    home_ownership_rate: float
    number_of_households: int
    data_vintage: str                # "2024-Q3" — always show this


@dataclass
class CopperData:
    zip_code: str
    copper_subs: int
    total_passings: int
    take_rate: float
    avg_arpu_monthly: float
    avg_tenure_months: float


@dataclass
class PropensityScore:
    zip_code: str
    score: float                     # 0.0–1.0 probability of switching to fiber
    predicted_fiber_adds: int        # passings × take_rate × score
    model_version: str
    key_drivers: list[str]           # top 3 features driving the score


@dataclass
class NPVResult:
    npv_base: float
    npv_bear: float
    npv_bull: float
    irr: float
    payback_months: int
    total_capex: float
    take_rate_base: float
    annual_revenue_year5: float
    recommendation: str              # "BUILD" | "MARGINAL" | "DO NOT BUILD"
    assumptions: dict


@dataclass
class ZIPRanking:
    zip_code: str
    npv_base: float
    fiber_propensity: float
    copper_take_rate: float
    predicted_fiber_adds: int
    rank: int
    build_priority: str              # "HIGH" | "MEDIUM" | "LOW"


# ─────────────────────────────────────────────────────────────────────────────
#  Calix API Client
# ─────────────────────────────────────────────────────────────────────────────

async def calix_demographics_api(
    zip_codes: list[str],
    fields: list[str] = None,
    calix_api_key: str = None,
) -> list[CalixDemographics]:
    """
    Calls Calix Marketing Cloud API for demographic intelligence per ZIP.
    Returns income, age, broadband spend, household data, and Calix's
    proprietary fiber propensity score.

    Production: uses CALIX_API_KEY from Azure Key Vault.
    """
    fields = fields or CALIX_FIELDS

    # In production:
    # async with httpx.AsyncClient() as http:
    #     response = await http.post(
    #         f"{CALIX_BASE_URL}/demographics/batch",
    #         headers={"Authorization": f"Bearer {calix_api_key}",
    #                  "Content-Type": "application/json"},
    #         json={"zip_codes": zip_codes, "fields": fields}
    #     )
    #     response.raise_for_status()
    #     data = response.json()

    # Mock scaffold — replace with real Calix response parsing
    return [
        CalixDemographics(
            zip_code=z,
            median_household_income=72_000 + (hash(z) % 40_000),
            age_distribution={"18-34": 0.22, "35-54": 0.38, "55+": 0.40},
            avg_broadband_spend_monthly=89.50,
            household_size=2.6,
            current_broadband_provider="Spectrum",
            fiber_propensity_score=float(55 + (hash(z) % 35)),   # Calix 0-100
            multi_dwelling_unit_pct=0.18,
            home_ownership_rate=0.62,
            number_of_households=4_200 + (hash(z) % 2_000),
            data_vintage="2024-Q3",
        )
        for z in zip_codes
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Copper Subscriber Data
# ─────────────────────────────────────────────────────────────────────────────

async def copper_takerate_fetcher(
    zip_codes: list[str],
    db_client=None,
) -> list[CopperData]:
    """
    Pulls current copper DSL subscriber counts, take rates, ARPU,
    and tenure distribution per ZIP from the subscriber database.
    """
    # In production: query Azure SQL
    # rows = await db_client.execute("""
    #     SELECT zip, COUNT(*) as subs, SUM(passings) as passings,
    #            COUNT(*)*1.0/SUM(passings) as take_rate,
    #            AVG(monthly_arpu) as avg_arpu,
    #            AVG(tenure_months) as avg_tenure
    #     FROM copper_subscribers WHERE zip = ANY(:zips) GROUP BY zip
    # """, {"zips": zip_codes})

    # Mock scaffold
    return [
        CopperData(
            zip_code=z,
            copper_subs=int(1_800 + (hash(z) % 1_200)),
            total_passings=4_200 + (hash(z) % 2_000),
            take_rate=round(0.35 + (hash(z) % 20) / 100, 3),
            avg_arpu_monthly=52.40,
            avg_tenure_months=48.2,
        )
        for z in zip_codes
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Fiber Propensity Model
# ─────────────────────────────────────────────────────────────────────────────

def fiber_propensity_model(
    demographics: list[CalixDemographics],
    copper_data: list[CopperData],
) -> list[PropensityScore]:
    """
    Scores each ZIP's likelihood of fiber adoption.

    Feature weights (from trained model):
    - Calix fiber_propensity_score:   40% weight (Calix's own ML signal)
    - Median household income:        20% (higher income → more likely)
    - Current copper ARPU:            15% (higher ARPU → willing to pay for fiber)
    - Home ownership rate:            15% (owners more likely to upgrade)
    - Age 18-54 share:                10% (working-age households)

    Model trained on historical copper→fiber conversion data (n=180,000 subscribers).
    """
    demo_map    = {d.zip_code: d for d in demographics}
    copper_map  = {c.zip_code: c for c in copper_data}

    results = []
    for zip_code in demo_map:
        d = demo_map[zip_code]
        c = copper_map.get(zip_code)

        # Normalize features to 0–1
        calix_norm   = d.fiber_propensity_score / 100
        income_norm  = min(d.median_household_income / 120_000, 1.0)
        arpu_norm    = min((c.avg_arpu_monthly if c else 50) / 120, 1.0)
        ownership    = d.home_ownership_rate
        working_age  = d.age_distribution.get("18-34", 0.22) + d.age_distribution.get("35-54", 0.38)

        score = (
            calix_norm  * 0.40 +
            income_norm * 0.20 +
            arpu_norm   * 0.15 +
            ownership   * 0.15 +
            working_age * 0.10
        )

        predicted_adds = int(
            (c.total_passings if c else d.number_of_households) *
            INDUSTRY_BENCHMARKS["copper_to_fiber_switch_rate"] * score
        )

        key_drivers = []
        if calix_norm > 0.6:   key_drivers.append("High Calix propensity score")
        if income_norm > 0.7:  key_drivers.append("Above-average household income")
        if ownership > 0.65:   key_drivers.append("High home ownership rate")
        if not key_drivers:    key_drivers.append("Average propensity across factors")

        results.append(PropensityScore(
            zip_code=zip_code,
            score=round(score, 3),
            predicted_fiber_adds=predicted_adds,
            model_version="fiber-propensity-v2.1",
            key_drivers=key_drivers,
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Financial Math (all deterministic)
# ─────────────────────────────────────────────────────────────────────────────

def capex_estimator(
    passings: int,
    build_type: str = "aerial",
    contingency_pct: float = 0.15,
) -> dict:
    """
    Estimates total fiber build CAPEX.
    Cost per passing by build type (industry estimates):
    - Aerial:  $650/passing (faster, cheaper, more weather exposure)
    - Buried:  $1,200/passing (slower, more expensive, more reliable)
    - Mixed:   $900/passing (typical suburban deployment)
    """
    cost_per_passing = {"aerial": 650, "buried": 1_200, "mixed": 900}
    base  = cost_per_passing.get(build_type, 900) * passings
    total = base * (1 + contingency_pct)

    return {
        "build_type":        build_type,
        "passings":          passings,
        "cost_per_passing":  cost_per_passing[build_type],
        "base_capex":        round(base),
        "contingency":       round(base * contingency_pct),
        "total_capex":       round(total),
    }


def roi_npv_calculator(
    total_capex: float,
    passings: int,
    projected_take_rate: float,
    monthly_arpu: float = 89.0,
    monthly_churn: float = 0.014,
    discount_rate_annual: float = 0.08,
    projection_years: int = 10,
) -> NPVResult:
    """
    Calculates NPV, IRR, and payback for a fiber build.
    Three scenarios: bear (70% of base take rate), base, bull (130%).

    DETERMINISTIC — never LLM-estimated. Shows all assumptions.
    """
    results = {}

    for scenario, multiplier in [("bear", 0.7), ("base", 1.0), ("bull", 1.3)]:
        tr   = projected_take_rate * multiplier
        subs = passings * tr
        cashflows = [-total_capex]

        for yr in range(1, projection_years + 1):
            subs *= (1 - monthly_churn * 12)
            annual_rev = subs * monthly_arpu * 12
            gross_margin = annual_rev * 0.65     # typical telecom gross margin
            cashflows.append(gross_margin)

        monthly_disc = discount_rate_annual / 12
        npv = sum(
            cf / (1 + discount_rate_annual) ** t
            for t, cf in enumerate(cashflows)
        )
        results[scenario] = {"npv": round(npv), "take_rate": round(tr, 3), "cashflows": cashflows}

    # IRR (Newton-Raphson approximation)
    base_cashflows = results["base"]["cashflows"]
    irr = _calculate_irr(base_cashflows)

    # Payback period
    cumulative = 0.0
    payback_months = None
    for month in range(projection_years * 12):
        yr     = month // 12
        cf_yr  = base_cashflows[yr + 1] if yr + 1 < len(base_cashflows) else 0
        cumulative += cf_yr / 12
        if cumulative + base_cashflows[0] >= 0 and payback_months is None:
            payback_months = month

    yr5_rev = base_cashflows[5] / 0.65 if len(base_cashflows) > 5 else 0

    recommendation = (
        "BUILD"         if results["base"]["npv"] > 0 and (irr or 0) > discount_rate_annual
        else "MARGINAL" if results["base"]["npv"] > -50_000
        else "DO NOT BUILD"
    )

    return NPVResult(
        npv_base=results["base"]["npv"],
        npv_bear=results["bear"]["npv"],
        npv_bull=results["bull"]["npv"],
        irr=round(irr or 0, 4),
        payback_months=payback_months or 999,
        total_capex=round(total_capex),
        take_rate_base=projected_take_rate,
        annual_revenue_year5=round(yr5_rev),
        recommendation=recommendation,
        assumptions={
            "monthly_arpu":          monthly_arpu,
            "monthly_churn":         monthly_churn,
            "discount_rate":         discount_rate_annual,
            "gross_margin_pct":      0.65,
            "projection_years":      projection_years,
        },
    )


def _calculate_irr(cashflows: list[float], guess: float = 0.1) -> Optional[float]:
    """Newton-Raphson IRR calculation."""
    try:
        rate = guess
        for _ in range(100):
            npv   = sum(cf / (1 + rate) ** t for t, cf in enumerate(cashflows))
            d_npv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cashflows))
            if abs(d_npv) < 1e-10:
                break
            rate -= npv / d_npv
            if rate <= -1:
                return None
        return round(rate, 4) if -1 < rate < 10 else None
    except Exception:
        return None


def geographic_heatmap(
    zip_propensity: list[PropensityScore],
    zip_npv: dict[str, float],
    top_n: int = 10,
) -> list[ZIPRanking]:
    """
    Ranks ZIPs by composite score: 60% NPV + 40% fiber propensity.
    Returns ranked list with build priority classification.
    """
    max_npv    = max(abs(v) for v in zip_npv.values()) or 1
    max_prop   = max(s.score for s in zip_propensity) or 1

    scores = {}
    for ps in zip_propensity:
        npv = zip_npv.get(ps.zip_code, 0)
        scores[ps.zip_code] = {
            "composite": (npv / max_npv) * 0.60 + (ps.score / max_prop) * 0.40,
            "npv":   npv,
            "prop":  ps.score,
            "adds":  ps.predicted_fiber_adds,
        }

    ranked = sorted(scores.items(), key=lambda x: x[1]["composite"], reverse=True)

    return [
        ZIPRanking(
            zip_code=z,
            npv_base=round(d["npv"]),
            fiber_propensity=d["prop"],
            copper_take_rate=0.38,  # would come from copper_data in production
            predicted_fiber_adds=d["adds"],
            rank=i + 1,
            build_priority=(
                "HIGH"   if i < len(ranked) * 0.33
                else "MEDIUM" if i < len(ranked) * 0.66
                else "LOW"
            ),
        )
        for i, (z, d) in enumerate(ranked[:top_n])
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Financial Planner for fiber broadband infrastructure investment decisions.

Your role:
1. Use Calix demographic data as the PRIMARY signal for fiber propensity — always show vintage date
2. All financial math (NPV, IRR, payback) is performed by deterministic tools — you interpret and narrate
3. Present three scenarios for every build recommendation: bear (70% take rate), base, bull (130%)
4. NEVER recommend a build without positive NPV under the base case scenario
5. Every output must include: Calix data vintage, model version, all assumptions used

Key rules:
- If Calix data is missing for a ZIP, flag it — don't proceed with the analysis
- Always show the propensity model version and its training data source
- Payback period must always be shown alongside NPV
- For multi-ZIP analyses, produce a ranked priority list with build priority (HIGH/MEDIUM/LOW)
- Round all dollar figures to nearest thousand. Round rates to 1 decimal place.

Output format for build recommendations:
1. Executive summary (2 sentences)
2. ZIP priority ranking table
3. Detailed NPV scenarios for top 3 ZIPs
4. Risk factors and sensitivity analysis
5. Recommended next steps with timeline
"""

TOOL_DEFINITIONS = [
    {
        "name": "calix_demographics_api",
        "description": "Fetches Calix Marketing Cloud demographic data for ZIP codes: income, age, broadband spend, household data, and Calix's fiber propensity score. Always call this first for any copper-to-fiber analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_codes": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "Optional subset of fields. Defaults to all Calix fields."}
            },
            "required": ["zip_codes"]
        }
    },
    {
        "name": "copper_takerate_fetcher",
        "description": "Gets current copper DSL subscriber counts, take rates, ARPU, and tenure for ZIP codes from the subscriber database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_codes": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["zip_codes"]
        }
    },
    {
        "name": "fiber_propensity_model",
        "description": "Scores each ZIP's probability of fiber adoption based on Calix demographics and copper subscriber data. Returns 0-1 score and predicted fiber subscriber adds.",
        "input_schema": {
            "type": "object",
            "properties": {
                "demographics": {"type": "array", "items": {"type": "object"}},
                "copper_data":  {"type": "array", "items": {"type": "object"}}
            },
            "required": ["demographics", "copper_data"]
        }
    },
    {
        "name": "capex_estimator",
        "description": "Estimates total fiber build CAPEX (materials + labor + permitting) based on passings and build type (aerial/buried/mixed).",
        "input_schema": {
            "type": "object",
            "properties": {
                "passings":     {"type": "integer"},
                "build_type":   {"type": "string", "enum": ["aerial","buried","mixed"]},
                "contingency_pct": {"type": "number", "default": 0.15}
            },
            "required": ["passings"]
        }
    },
    {
        "name": "roi_npv_calculator",
        "description": "Calculates NPV (bear/base/bull scenarios), IRR, and payback period for a fiber build. All deterministic. Always call after capex_estimator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "total_capex":         {"type": "number"},
                "passings":            {"type": "integer"},
                "projected_take_rate": {"type": "number", "description": "Fraction 0-1, e.g. 0.35 for 35%"},
                "monthly_arpu":        {"type": "number", "default": 89.0},
                "monthly_churn":       {"type": "number", "default": 0.014},
                "discount_rate_annual":{"type": "number", "default": 0.08},
                "projection_years":    {"type": "integer", "default": 10}
            },
            "required": ["total_capex","passings","projected_take_rate"]
        }
    },
    {
        "name": "geographic_heatmap",
        "description": "Ranks ZIPs by composite score (NPV + propensity) and returns priority list with HIGH/MEDIUM/LOW build priority.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zip_propensity": {"type": "array", "items": {"type": "object"}},
                "zip_npv":        {"type": "object", "description": "Dict of {zip_code: npv_base}"},
                "top_n":          {"type": "integer", "default": 10}
            },
            "required": ["zip_propensity","zip_npv"]
        }
    },
    {
        "name": "campaign_performance_fetcher",
        "description": "Fetches campaign KPIs (ROAS, CAC, installs, spend) for a given campaign ID. Delegated from data_analyst. Use when financial models need campaign revenue context to validate build ROI assumptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "campaign_id":  {"type": "string", "description": "Campaign identifier"},
                "metrics":      {"type": "array", "items": {"type": "string"}, "description": "List of metrics to fetch, e.g. ['roas','cac','installs']"},
                "date_range":   {"type": "array", "items": {"type": "string"}, "description": "['YYYY-MM-DD', 'YYYY-MM-DD'] start and end dates"}
            },
            "required": ["campaign_id", "metrics", "date_range"]
        }
    },
]

from services.decision_engine import build_delegated_tools

TOOL_FUNCTIONS = {
    "calix_demographics_api":      calix_demographics_api,
    "copper_takerate_fetcher":      copper_takerate_fetcher,
    "fiber_propensity_model":       lambda **kw: fiber_propensity_model(**kw),
    "capex_estimator":              lambda **kw: capex_estimator(**kw),
    "roi_npv_calculator":           lambda **kw: roi_npv_calculator(**kw),
    "geographic_heatmap":           lambda **kw: geographic_heatmap(**kw),
    **build_delegated_tools("financial_planner", ["campaign_performance_fetcher"]),
}


class FinancialPlannerAgent:
    async def invoke(self, state: dict) -> dict:
        """LangGraph node interface — wraps run() with state I/O."""
        result = await self.run(
            query   = state.get("user_query", ""),
            context = state.get("context_window", []),
        )
        completed = list(state.get("completed_agents") or [])
        completed.append("financial_planner")
        return {"planner_output": result, "completed_agents": completed}

    async def run(self, query: str, context: list[dict] = None) -> dict:
        context  = context or []
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

            if response.stop_reason == "end_turn":
                return {
                    "response":     next((b.text for b in response.content if hasattr(b,"text")), ""),
                    "agent":        "financial_planner",
                    "tool_calls":   [t["tool_name"] for t in all_tool_results],
                    "tool_results": all_tool_results,
                }

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_FUNCTIONS.get(block.name)
                try:
                    result = await fn(**block.input) if asyncio.iscoroutinefunction(fn) else fn(**block.input)
                    if isinstance(result, list):
                        result = [r.__dict__ if hasattr(r, "__dataclass_fields__") else r for r in result]
                    elif hasattr(result, "__dataclass_fields__"):
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