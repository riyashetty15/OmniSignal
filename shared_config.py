"""
FiberOrbit — Shared Configuration
Single source of truth for all modules, models, thresholds, and API endpoints.
"""

# ─── Agent Models ─────────────────────────────────────────────────────────────
# HR gets haiku (isolated, cost-optimized). Marketing gets opus (full reasoning).
AGENT_MODELS = {
    "hr_docqa":          "claude-3-haiku-20240307",   # isolated, cheaper
    "data_analyst":      "claude-opus-4-6",
    "financial_planner": "claude-opus-4-6",
    "strategist":        "claude-opus-4-6",
    "validator":         "claude-opus-4-6",
    "planner":           "claude-opus-4-6",
}

# ─── Modules & Vector Stores (module siloing) ──────────────────────────────────
MODULES = ["hr", "campaign", "fiber_network", "competitive", "seo", "financial"]

MODULE_VECTOR_STORES = {
    "hr":           "hr_docs",           # HR agent ONLY — isolated RBAC
    "campaign":     "campaign_docs",
    "fiber_network":"fiber_network_docs",
    "competitive":  "competitive_docs",
    "seo":          "seo_docs",
    "financial":    "financial_docs",    # Calix + copper/fiber financials
}

# ─── Routing Intent Labels ─────────────────────────────────────────────────────
MARKETING_INTENTS = [
    "take_rate", "campaign_performance", "funnel_metrics",
    "cohort_analysis", "anomaly_detection",     # → Data Analyst
    "copper_to_fiber", "roi_npv", "calix_demographics",
    "capex_planning", "fiber_propensity",        # → Financial Planner
    "social_content", "content_calendar",
    "brand_strategy", "competitor_content",      # → Strategist
]
HR_INTENTS = [
    "policy_lookup", "benefits", "onboarding",
    "leave_policy", "remote_work", "compliance_training",
    "org_chart", "performance_review_process",
]

# ─── Quality Thresholds ────────────────────────────────────────────────────────
FIDELITY_THRESHOLD    = 0.92   # minimum output fidelity vs validated baselines
EVIDENCE_SUFFICIENT   = 0.82   # retrieval score for sufficient coverage
EVIDENCE_PARTIAL      = 0.60   # retrieval score for partial (answer with caveat)

# ─── Industry Benchmarks (used for sanity checks — never LLM-estimated) ────────
INDUSTRY_BENCHMARKS = {
    "fiber_take_rate_national": 0.18,   # 18% national avg fiber take rate
    "fiber_take_rate_mature":   0.42,   # mature market (5yr+ deployment)
    "roas_typical":             3.5,
    "cac_residential":          350,    # $350 typical residential fiber CAC
    "cac_business":             1200,
    "clv_residential":          2400,   # $2400 36-month CLV
    "clv_business":             8500,
    "monthly_churn_residential":0.014,  # 1.4%/month
    "copper_to_fiber_switch_rate":0.65, # 65% of copper subs switch to fiber when available
    "install_backlog_sla_days": 14,
}

# ─── Calix API Configuration ───────────────────────────────────────────────────
CALIX_BASE_URL   = "https://api.calix.com/marketing-cloud/v2"
CALIX_FIELDS = [
    "median_household_income",
    "age_distribution",
    "avg_broadband_spend_monthly",
    "household_size",
    "current_broadband_provider",
    "fiber_propensity_score",        # Calix proprietary 0-100 score
    "multi_dwelling_unit_pct",
    "home_ownership_rate",
    "number_of_households",
]

# ─── Social Media Configuration ────────────────────────────────────────────────
SUPPORTED_PLATFORMS = ["linkedin", "meta", "x", "youtube"]
CONTENT_PILLARS = [
    "fiber_speed_reliability",
    "business_connectivity",
    "smart_home_enablement",
    "community_investment",
    "customer_success_stories",
    "technical_education",
]
BRAND_HASHTAGS = ["#FiberFirst", "#ConnectedCommunity", "#GigLife"]

# FTC/NAD claim triggers — any of these in copy must be flagged for legal review
COMPLIANCE_CLAIM_TRIGGERS = [
    "fastest", "best", "guaranteed", "unlimited", "no data cap",
    "rated #1", "award-winning", "studies show", "proven",
    r"up to \d+ (mbps|gbps|gig)",
]

# ─── Azure Configuration ───────────────────────────────────────────────────────
LLM_SERVER_PORT  = 8000
RAG_SERVER_PORT  = 8081
RAG_SERVER_URL   = f"http://marketorbit-rag:{RAG_SERVER_PORT}"

AZURE_SQL_SCHEMA = "marketing_dw"
AZURE_BLOB_HR_CONTAINER    = "hr-documents"
AZURE_BLOB_MKTG_CONTAINER  = "marketing-documents"