"""
core/router.py
==============
Semantic query router — maps an incoming user query to the correct specialist agent.

Approach: keyword scoring (fast path) with tie-breaking by department context.
No LLM call required; the planner can override if the keyword score is ambiguous.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RouteResult:
    agent:      str    # "data_analyst" | "financial_planner" | "strategist" | "hr_docqa"
    intent:     str    # fine-grained intent label
    confidence: float  # 0.0–1.0
    reasoning:  str    # brief explanation for logging/debugging


# ── Keyword vocabularies ──────────────────────────────────────────────────────
# Each set represents strong signals for the corresponding agent.
# Words are lowercased; multi-word phrases are matched as substrings.

_HR_KEYWORDS = frozenset([
    "policy", "policies", "handbook", "benefits", "leave", "pto", "vacation",
    "sick day", "sick leave", "onboarding", "onboard", "org chart",
    "performance review", "employee", "reimbursement", "payroll", "remote work",
    "work from home", "wfh", "compliance training", "code of conduct",
    "expense", "health insurance", "dental", "401k", "retirement", "fmla",
    "ada", "accommodation", "grievance", "disciplinary", "termination",
    "resignation", "notice period", "background check",
])

_DATA_ANALYST_KEYWORDS = frozenset([
    "take rate", "takerate", "passings", "subscribers", "subs",
    "campaign performance", "campaign metrics", "impressions", "clicks",
    "leads", "conversions", "cac", "cpl", "roas", "return on ad",
    "funnel", "acquisition funnel", "drop-off", "drop off",
    "cohort", "retention", "churn", "arpu", "ltv", "clv",
    "anomaly", "anomalies", "spike", "drop in", "traffic",
    "benchmark", "kpi", "metric", "analytics", "trend",
    "penetration rate", "net adds", "channel performance",
    "geographic", "region", "market performance",
])

_FINANCIAL_KEYWORDS = frozenset([
    "copper", "dsl", "copper to fiber", "upgrade", "convert copper",
    "fiber build", "build fiber", "infrastructure",
    "npv", "irr", "net present value", "internal rate of return",
    "capex", "capital expenditure", "roi", "return on investment",
    "calix", "demographics", "household income", "fiber propensity",
    "payback period", "payback", "break even", "break-even",
    "zip code", "zipcode", "zip-code", "market analysis",
    "build decision", "should we build", "investment",
    "spending pattern", "broadband spend", "households",
    "home ownership", "multi dwelling", "mdu",
])

_STRATEGIST_KEYWORDS = frozenset([
    "content", "social media", "post", "linkedin", "instagram", "facebook",
    "twitter", "x.com", "youtube", "tiktok",
    "hashtag", "hashtags", "campaign brief", "brief",
    "brand", "strategy", "strategize", "ideate", "ideas",
    "calendar", "content calendar", "editorial",
    "competitor", "competition", "competitive", "what are competitors",
    "copy", "copywriting", "creative", "messaging",
    "audience", "targeting", "tone of voice", "voice",
    "press release", "announcement", "launch",
    "awareness", "engagement", "organic",
])


def _score(query_lower: str, keywords: frozenset[str]) -> int:
    return sum(1 for kw in keywords if kw in query_lower)


def route_query(
    query:                str,
    conversation_history: list[dict] | None = None,
    department:           str = "marketing",
) -> RouteResult:
    """
    Routes the query to the best-matching agent.

    Priority rules:
    1. Department == "hr"  → always hr_docqa (access control)
    2. Keyword scoring     → highest score wins
    3. Tie / zero scores   → data_analyst (safe default for marketing)
    """
    q = query.lower()

    # Department-level hard override
    if department == "hr":
        return RouteResult(
            agent      = "hr_docqa",
            intent     = _hr_intent(q),
            confidence = 1.0,
            reasoning  = "Department=hr forces hr_docqa routing",
        )

    scores = {
        "hr_docqa":          _score(q, _HR_KEYWORDS),
        "data_analyst":      _score(q, _DATA_ANALYST_KEYWORDS),
        "financial_planner": _score(q, _FINANCIAL_KEYWORDS),
        "strategist":        _score(q, _STRATEGIST_KEYWORDS),
    }

    best_agent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_agent]
    total      = sum(scores.values()) or 1

    # If the best is hr and we're not in the hr department, still allow it —
    # employees in marketing may ask HR questions.
    if best_score == 0:
        best_agent = "data_analyst"
        confidence = 0.40
        reasoning  = "No keyword match; defaulting to data_analyst"
    else:
        confidence = round(best_score / total, 3)
        reasoning  = f"Keyword scores: {scores}"

    intent = _intent_for(best_agent, q)

    return RouteResult(
        agent      = best_agent,
        intent     = intent,
        confidence = confidence,
        reasoning  = reasoning,
    )


# ── Intent labellers ──────────────────────────────────────────────────────────

def _hr_intent(q: str) -> str:
    if any(w in q for w in ["leave", "pto", "vacation", "sick"]): return "leave_policy"
    if any(w in q for w in ["benefit", "health", "dental", "401k"]):  return "benefits"
    if any(w in q for w in ["onboard", "onboarding"]):               return "onboarding"
    if any(w in q for w in ["remote", "wfh", "work from home"]):     return "remote_work"
    if any(w in q for w in ["org chart", "who reports", "team"]):    return "org_chart"
    if any(w in q for w in ["performance review", "pip"]):           return "performance_review_process"
    return "policy_lookup"


def _intent_for(agent: str, q: str) -> str:
    if agent == "hr_docqa":
        return _hr_intent(q)

    if agent == "data_analyst":
        if "take rate" in q or "takerate" in q: return "take_rate"
        if "campaign" in q:                      return "campaign_performance"
        if "funnel" in q:                        return "funnel_metrics"
        if "cohort" in q:                        return "cohort_analysis"
        if "anomaly" in q or "anomal" in q:      return "anomaly_detection"
        return "analytics_query"

    if agent == "financial_planner":
        if "copper" in q or "convert" in q:          return "copper_to_fiber"
        if "npv" in q or "irr" in q or "roi" in q:   return "roi_npv"
        if "calix" in q or "demographic" in q:       return "calix_demographics"
        if "capex" in q or "build cost" in q:        return "capex_planning"
        if "propensity" in q:                         return "fiber_propensity"
        return "financial_analysis"

    if agent == "strategist":
        if "content" in q or "post" in q:    return "social_content"
        if "calendar" in q:                  return "content_calendar"
        if "competitor" in q:                return "competitor_content"
        if "brief" in q:                     return "campaign_brief"
        if "hashtag" in q:                   return "hashtag_research"
        return "brand_strategy"

    return "general"
