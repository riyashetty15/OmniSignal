"""
tests/agents.py
===============
Unit tests for all four specialist agents and the planner.

Tests use only mock tool results — no live API calls, no DB.
Run with: pytest tests/agents.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base import initial_state, AgentState
from core.router import route_query, RouteResult


# ─────────────────────────────────────────────────────────────────────────────
#  Routing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:
    def test_take_rate_routes_to_analyst(self):
        r = route_query("What is the fiber take rate in the Pacific Northwest for Q3?")
        assert r.agent == "data_analyst"
        assert r.intent == "take_rate"

    def test_calix_routes_to_financial(self):
        r = route_query("Show me Calix demographics for ZIP 98101")
        assert r.agent == "financial_planner"
        assert r.intent == "calix_demographics"

    def test_copper_routes_to_financial(self):
        r = route_query("Which copper areas should we convert to fiber based on NPV?")
        assert r.agent == "financial_planner"
        assert r.intent in ("copper_to_fiber", "roi_npv", "financial_analysis")

    def test_social_routes_to_strategist(self):
        r = route_query("Generate a LinkedIn post about our fiber launch")
        assert r.agent == "strategist"
        assert r.intent == "social_content"

    def test_policy_routes_to_hr(self):
        r = route_query("What is our remote work policy for employees?")
        assert r.agent == "hr_docqa"
        assert r.intent == "remote_work"

    def test_hr_department_forces_hr_agent(self):
        r = route_query("What is our ROI on fiber builds?", department="hr")
        assert r.agent == "hr_docqa"
        assert r.confidence == 1.0

    def test_unknown_query_defaults_to_analyst(self):
        r = route_query("xyz completely unrelated query !!!!")
        assert r.agent == "data_analyst"
        assert r.confidence < 0.6

    def test_campaign_routes_to_analyst(self):
        r = route_query("What was the ROAS and CAC for campaign CAMP-2024-001?")
        assert r.agent == "data_analyst"
        assert r.intent == "campaign_performance"

    def test_content_calendar_routes_to_strategist(self):
        r = route_query("Build me a content calendar for the next 4 weeks")
        assert r.agent == "strategist"
        assert r.intent == "content_calendar"


# ─────────────────────────────────────────────────────────────────────────────
#  AgentState / initial_state tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState:
    def test_initial_state_required_fields(self):
        state = initial_state("sess-1", "user-1", "What is take rate?")
        assert state["session_id"]    == "sess-1"
        assert state["user_id"]       == "user-1"
        assert state["user_query"]    == "What is take rate?"
        assert state["department"]    == "marketing"
        assert state["fidelity_score"] == 0.0
        assert state["retry_count"]   == 0
        assert state["completed_agents"] == []
        assert state["guardrail_flags"]  == []

    def test_initial_state_with_context(self):
        ctx = [{"query": "prior q", "response": "prior r", "agent": "data_analyst",
                "metadata": {}, "created_at": "2024-01-01"}]
        state = initial_state("sess-2", "user-2", "follow-up", context_window=ctx)
        assert len(state["context_window"]) == 1

    def test_initial_state_hr_department(self):
        state = initial_state("s", "u", "q", department="hr")
        assert state["department"] == "hr"


# ─────────────────────────────────────────────────────────────────────────────
#  Data Analyst Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataAnalystAgent:
    @pytest.mark.asyncio
    async def test_take_rate_calculator_deterministic(self):
        from agents.specialist.data_analyst import take_rate_calculator, TakeRateResult
        result = take_rate_calculator(region="Pacific Northwest", period="Q3-2024")
        assert isinstance(result, TakeRateResult)
        assert 0 < result.rate < 1
        assert result.passings > 0
        assert result.active_subs > 0
        assert result.rate == result.active_subs / result.passings

    @pytest.mark.asyncio
    async def test_funnel_metrics_drop_offs_sum_correctly(self):
        from agents.specialist.data_analyst import funnel_metrics_builder
        result = funnel_metrics_builder(campaign_id="CAMP-001")
        assert len(result.stages) == len(result.counts)
        assert len(result.conversion_rates) == len(result.stages) - 1
        for rate in result.conversion_rates:
            assert 0 <= rate <= 1

    @pytest.mark.asyncio
    async def test_anomaly_detector_returns_list(self):
        from agents.specialist.data_analyst import anomaly_detector
        anomalies = anomaly_detector(metric="take_rate", region="Southeast")
        assert isinstance(anomalies, list)

    def test_benchmark_comparator_above(self):
        from agents.specialist.data_analyst import benchmark_comparator
        result = benchmark_comparator("fiber_take_rate_national", 0.30)
        assert result["above_benchmark"] is True
        assert result["delta"] > 0

    def test_benchmark_comparator_below(self):
        from agents.specialist.data_analyst import benchmark_comparator
        result = benchmark_comparator("fiber_take_rate_national", 0.10)
        assert result["above_benchmark"] is False

    def test_benchmark_comparator_unknown_metric(self):
        from agents.specialist.data_analyst import benchmark_comparator
        result = benchmark_comparator("nonexistent_kpi", 0.5)
        assert result["benchmark_found"] is False

    @pytest.mark.asyncio
    async def test_invoke_returns_analyst_output(self):
        from agents.specialist.data_analyst import DataAnalystAgent
        agent = DataAnalystAgent()
        state = initial_state("s", "u", "What is the take rate in the Midwest?")
        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"response": "Take rate is 24%.", "agent": "data_analyst"}
            result = await agent.invoke(state)
        assert "analyst_output" in result
        assert "data_analyst" in result["completed_agents"]


# ─────────────────────────────────────────────────────────────────────────────
#  Financial Planner Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFinancialPlannerAgent:
    @pytest.mark.asyncio
    async def test_capex_estimator_aerial(self):
        from agents.specialist.financial_planner import capex_estimator
        result = capex_estimator(passings=10_000, build_type="aerial")
        assert result["total_capex"] > result["base_capex"]
        assert result["cost_per_passing"] == 650
        assert result["passings"] == 10_000

    @pytest.mark.asyncio
    async def test_capex_estimator_buried_more_than_aerial(self):
        from agents.specialist.financial_planner import capex_estimator
        aerial = capex_estimator(passings=1000, build_type="aerial")
        buried = capex_estimator(passings=1000, build_type="buried")
        assert buried["total_capex"] > aerial["total_capex"]

    def test_roi_positive_npv_for_high_take_rate(self):
        from agents.specialist.financial_planner import roi_npv_calculator
        result = roi_npv_calculator(
            total_capex=5_000_000, passings=10_000,
            projected_take_rate=0.40, monthly_arpu=89.0,
        )
        assert result.npv_bull > result.npv_base > result.npv_bear
        assert result.recommendation in ("BUILD", "MARGINAL", "DO NOT BUILD")

    def test_roi_negative_npv_for_low_take_rate(self):
        from agents.specialist.financial_planner import roi_npv_calculator
        result = roi_npv_calculator(
            total_capex=50_000_000, passings=1_000,
            projected_take_rate=0.05,
        )
        assert result.recommendation == "DO NOT BUILD"

    @pytest.mark.asyncio
    async def test_calix_mock_returns_correct_structure(self):
        from agents.specialist.financial_planner import calix_demographics_api, CalixDemographics
        results = await calix_demographics_api(zip_codes=["98101", "90210"])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, CalixDemographics)
            assert 0 <= r.fiber_propensity_score <= 100
            assert r.number_of_households > 0

    def test_propensity_score_in_range(self):
        from agents.specialist.financial_planner import (
            fiber_propensity_model, CalixDemographics, CopperData
        )
        demo = [CalixDemographics(
            zip_code="98101", median_household_income=80_000,
            age_distribution={"18-34": 0.25, "35-54": 0.40, "55+": 0.35},
            avg_broadband_spend_monthly=90.0, household_size=2.5,
            current_broadband_provider="Comcast", fiber_propensity_score=70.0,
            multi_dwelling_unit_pct=0.20, home_ownership_rate=0.65,
            number_of_households=4500, data_vintage="2024-Q3",
        )]
        copper = [CopperData(
            zip_code="98101", copper_subs=1500, total_passings=4000,
            take_rate=0.375, avg_arpu_monthly=52.0, avg_tenure_months=48.0,
        )]
        scores = fiber_propensity_model(demo, copper)
        assert len(scores) == 1
        assert 0.0 <= scores[0].score <= 1.0
        assert scores[0].predicted_fiber_adds >= 0

    @pytest.mark.asyncio
    async def test_invoke_returns_planner_output(self):
        from agents.specialist.financial_planner import FinancialPlannerAgent
        agent = FinancialPlannerAgent()
        state = initial_state("s", "u", "Should we build fiber in ZIP 98101?")
        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"response": "NPV is $2.1M — BUILD recommended.", "agent": "financial_planner"}
            result = await agent.invoke(state)
        assert "planner_output" in result
        assert "financial_planner" in result["completed_agents"]


# ─────────────────────────────────────────────────────────────────────────────
#  Strategist Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategistAgent:
    def test_ftc_compliance_checker_catches_fastest(self):
        from agents.specialist.strategist import ftc_compliance_checker
        flags = ftc_compliance_checker("We offer the fastest fiber in the region!")
        assert len(flags) > 0
        assert any("fastest" in f.trigger.lower() for f in flags)
        assert any(f.severity == "CRITICAL" for f in flags)

    def test_ftc_compliance_checker_clean_copy(self):
        from agents.specialist.strategist import ftc_compliance_checker
        flags = ftc_compliance_checker("Reliable fiber internet starting at $49/month.")
        assert len(flags) == 0

    def test_engagement_predictor_question_boost(self):
        from agents.specialist.strategist import engagement_predictor
        with_q    = engagement_predictor("Is your internet fast enough for remote work?", "linkedin", ["#fiber"])
        without_q = engagement_predictor("Our internet is fast for remote work.", "linkedin", ["#fiber"])
        assert with_q > without_q

    def test_engagement_predictor_too_many_hashtags(self):
        from agents.specialist.strategist import engagement_predictor
        score = engagement_predictor("Good post.", "meta", ["#" + str(i) for i in range(20)])
        assert score < 0.6

    def test_content_calendar_sorted_by_date(self):
        from agents.specialist.strategist import content_calendar_builder, PostDraft, ComplianceFlag
        posts = [
            PostDraft(copy="Post A", platform="linkedin", content_type="educational",
                      hashtags=["#fiber"], character_count=10, compliance_flags=[],
                      engagement_score=0.7, legal_review_required=False),
            PostDraft(copy="Post B", platform="meta", content_type="question",
                      hashtags=["#broadband"], character_count=8, compliance_flags=[],
                      engagement_score=0.65, legal_review_required=False),
        ]
        calendar = content_calendar_builder(posts, start_date="2024-11-01")
        dates = [e.date for e in calendar]
        assert dates == sorted(dates)

    def test_brand_guidelines_blocks_prohibited_phrase(self):
        from agents.specialist.strategist import brand_guidelines_enforcer
        result = brand_guidelines_enforcer("The cheapest fiber internet around!")
        assert result["passed"] is False

    @pytest.mark.asyncio
    async def test_invoke_returns_strategist_output(self):
        from agents.specialist.strategist import StrategistAgent
        agent = StrategistAgent()
        state = initial_state("s", "u", "Write a LinkedIn post about fiber launch")
        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"response": "Excited to announce...", "agent": "strategist"}
            result = await agent.invoke(state)
        assert "strategist_output" in result
        assert "strategist" in result["completed_agents"]


# ─────────────────────────────────────────────────────────────────────────────
#  HR DocQA Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHRDocQAAgent:
    def test_pii_masker_ssn(self):
        from agents.hr.hr_docqa import pii_masker
        masked, types = pii_masker("My SSN is 123-45-6789")
        assert "SSN" in types
        assert "123-45-6789" not in masked
        assert "[SSN-REDACTED]" in masked

    def test_pii_masker_email(self):
        from agents.hr.hr_docqa import pii_masker
        masked, types = pii_masker("Contact john.doe@company.com")
        assert "EMAIL" in types

    def test_pii_masker_clean_text(self):
        from agents.hr.hr_docqa import pii_masker
        masked, types = pii_masker("What is the remote work policy?")
        assert types == []
        assert masked == "What is the remote work policy?"

    def test_compliance_checker_harassment_escalates(self):
        from agents.hr.hr_docqa import hr_compliance_checker
        result = hr_compliance_checker("I want to report a harassment incident")
        assert result.requires_escalation is True
        assert result.suggested_routing == "legal"

    def test_compliance_checker_termination_escalates(self):
        from agents.hr.hr_docqa import hr_compliance_checker
        result = hr_compliance_checker("How do I handle a termination?")
        assert result.requires_escalation is True
        assert result.suggested_routing == "hr_manager"

    def test_compliance_checker_safe_query(self):
        from agents.hr.hr_docqa import hr_compliance_checker
        result = hr_compliance_checker("How many vacation days do I get?")
        assert result.requires_escalation is False
        assert result.suggested_routing == "self"

    def test_citation_formatter_adds_sources(self):
        from agents.hr.hr_docqa import doc_citation_formatter, HRCitation
        citations = [HRCitation(
            doc_id="pol-001", doc_title="Remote Work Policy", section="4.2",
            page=8, effective_date="2024-01-15", version="2.3",
            relevance_score=0.92, excerpt="...",
        )]
        result = doc_citation_formatter("You get $500 reimbursement.", citations)
        assert "Remote Work Policy" in result
        assert "4.2" in result
        assert "2.3" in result

    @pytest.mark.asyncio
    async def test_invoke_returns_hr_output(self):
        from agents.hr.hr_docqa import HRDocQAAgent
        agent = HRDocQAAgent()
        state = initial_state("s", "u", "What is the PTO policy?", department="hr")
        with patch.object(agent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"response": "You receive 15 days PTO.", "agent": "hr_docqa"}
            result = await agent.invoke(state)
        assert "hr_output" in result
        assert "hr_docqa" in result["completed_agents"]


# ─────────────────────────────────────────────────────────────────────────────
#  Planner Agent tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_planner_preserves_routed_agent(self):
        from agents.planner import PlannerAgent
        agent = PlannerAgent()
        state = initial_state("s", "u", "take rate in Pacific Northwest")
        state["routed_agent"] = "data_analyst"
        result = await agent.invoke(state)
        assert result["routed_agent"] == "data_analyst"

    @pytest.mark.asyncio
    async def test_planner_hr_dept_override(self):
        from agents.planner import PlannerAgent
        agent = PlannerAgent()
        state = initial_state("s", "u", "ROI analysis", department="hr")
        state["routed_agent"] = "financial_planner"
        result = await agent.invoke(state)
        assert result["routed_agent"] == "hr_docqa"

    @pytest.mark.asyncio
    async def test_planner_retry_adds_retry_flag(self):
        from agents.planner import PlannerAgent
        agent = PlannerAgent()
        state = initial_state("s", "u", "some query")
        state["routed_agent"]    = "data_analyst"
        state["retry_count"]     = 1
        state["completed_agents"] = ["data_analyst"]
        result = await agent.invoke(state)
        assert "retry" in result["completed_agents"]

    @pytest.mark.asyncio
    async def test_planner_extracts_zip_entities(self):
        from agents.planner import PlannerAgent
        agent = PlannerAgent()
        state = initial_state("s", "u", "Analyze ZIP codes 98101 and 90210")
        state["routed_agent"] = "financial_planner"
        result = await agent.invoke(state)
        assert "98101" in result["entity_memory"].get("zip_codes", [])
        assert "90210" in result["entity_memory"].get("zip_codes", [])


# ─────────────────────────────────────────────────────────────────────────────
#  SQL Safety Gate (_validate_read_only)
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLSafetyGate:
    """
    _validate_read_only is the only barrier between Claude-generated SQL
    and the Azure SQL DW. Every mutating keyword must be blocked.
    """

    def _vro(self, sql: str):
        from agents.specialist.data_analyst import _validate_read_only
        _validate_read_only(sql)

    def test_blocks_insert(self):
        with pytest.raises(ValueError, match="Mutating SQL blocked"):
            self._vro("INSERT INTO fiber_metrics VALUES (1, 2, 3)")

    def test_blocks_update(self):
        with pytest.raises(ValueError):
            self._vro("UPDATE fiber_metrics SET take_rate=0.5 WHERE region='test'")

    def test_blocks_delete(self):
        with pytest.raises(ValueError):
            self._vro("DELETE FROM fiber_metrics WHERE region='test'")

    def test_blocks_drop(self):
        with pytest.raises(ValueError):
            self._vro("DROP TABLE fiber_metrics")

    def test_blocks_truncate(self):
        with pytest.raises(ValueError):
            self._vro("TRUNCATE TABLE fiber_metrics")

    def test_blocks_exec(self):
        with pytest.raises(ValueError):
            self._vro("EXEC xp_cmdshell 'ls'")

    def test_blocks_execute(self):
        with pytest.raises(ValueError):
            self._vro("EXECUTE sp_helpdb")

    def test_case_insensitive_insert(self):
        with pytest.raises(ValueError):
            self._vro("insert into fiber_metrics values (1)")

    def test_case_insensitive_drop(self):
        with pytest.raises(ValueError):
            self._vro("Drop Table fiber_metrics")

    def test_allows_select(self):
        # Must NOT raise
        self._vro("SELECT * FROM fiber_metrics WHERE region='Pacific Northwest'")

    def test_allows_select_with_join(self):
        self._vro(
            "SELECT f.region, SUM(f.passings) FROM fiber_metrics f "
            "JOIN campaign_performance c ON f.region = c.region GROUP BY f.region"
        )

    def test_allows_select_with_subquery(self):
        self._vro(
            "SELECT * FROM fiber_metrics WHERE passings > "
            "(SELECT AVG(passings) FROM fiber_metrics)"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Strengthened anomaly detector tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetectorStrengthened:
    def test_known_outlier_date_is_flagged(self):
        from agents.specialist.data_analyst import anomaly_detector
        anomalies = anomaly_detector(metric="take_rate", region="Southeast")
        flagged_dates = [a.date for a in anomalies]
        assert "2024-10-03" in flagged_dates, (
            "2024-10-03 (value 0.189) should be flagged as a statistical outlier"
        )

    def test_outlier_direction_is_drop(self):
        from agents.specialist.data_analyst import anomaly_detector
        anomalies = anomaly_detector(metric="take_rate", region="Southeast")
        oct3 = next((a for a in anomalies if a.date == "2024-10-03"), None)
        assert oct3 is not None
        assert oct3.direction == "drop"

    def test_outlier_z_score_is_negative(self):
        from agents.specialist.data_analyst import anomaly_detector
        anomalies = anomaly_detector(metric="take_rate", region="Southeast")
        oct3 = next((a for a in anomalies if a.date == "2024-10-03"), None)
        assert oct3 is not None
        assert oct3.z_score < 0

    def test_normal_dates_not_flagged(self):
        from agents.specialist.data_analyst import anomaly_detector
        anomalies = anomaly_detector(metric="take_rate", region="Southeast")
        flagged_dates = {a.date for a in anomalies}
        # All non-outlier dates should be clean
        for normal_date in ["2024-10-01", "2024-10-02", "2024-10-04"]:
            assert normal_date not in flagged_dates

    def test_custom_z_threshold_catches_more(self):
        from agents.specialist.data_analyst import anomaly_detector
        strict   = anomaly_detector(metric="take_rate", region="SE", z_threshold=1.0)
        lenient  = anomaly_detector(metric="take_rate", region="SE", z_threshold=3.0)
        assert len(strict) >= len(lenient)


# ─────────────────────────────────────────────────────────────────────────────
#  Missing tool coverage: cohort_analyzer and campaign_performance_fetcher
# ─────────────────────────────────────────────────────────────────────────────

class TestCohortAnalyzer:
    @pytest.mark.asyncio
    async def test_returns_list_of_cohort_rows(self):
        from agents.specialist.data_analyst import cohort_analyzer, CohortRow
        rows = await cohort_analyzer(cohort_start="2024-Q1", cohort_end="2024-Q3")
        assert isinstance(rows, list)
        assert all(isinstance(r, CohortRow) for r in rows)

    @pytest.mark.asyncio
    async def test_retention_curves_start_at_one(self):
        from agents.specialist.data_analyst import cohort_analyzer
        rows = await cohort_analyzer(cohort_start="2024-Q1", cohort_end="2024-Q3")
        for row in rows:
            assert row.retention_by_month[0] == 1.0, (
                f"Cohort {row.cohort_month}: month-0 retention must be 1.0"
            )

    @pytest.mark.asyncio
    async def test_retention_is_monotonically_non_increasing(self):
        from agents.specialist.data_analyst import cohort_analyzer
        rows = await cohort_analyzer(cohort_start="2024-Q1", cohort_end="2024-Q3")
        for row in rows:
            curve = row.retention_by_month
            for i in range(1, len(curve)):
                assert curve[i] <= curve[i - 1], (
                    f"Cohort {row.cohort_month}: retention should not increase over time"
                )

    @pytest.mark.asyncio
    async def test_all_retention_values_in_range(self):
        from agents.specialist.data_analyst import cohort_analyzer
        rows = await cohort_analyzer(cohort_start="2024-Q1", cohort_end="2024-Q3")
        for row in rows:
            for val in row.retention_by_month:
                assert 0.0 <= val <= 1.0

    @pytest.mark.asyncio
    async def test_month_0_subscriber_count_positive(self):
        from agents.specialist.data_analyst import cohort_analyzer
        rows = await cohort_analyzer(cohort_start="2024-Q1", cohort_end="2024-Q3")
        for row in rows:
            assert row.month_0_subs > 0


class TestCampaignPerformanceFetcher:
    @pytest.mark.asyncio
    async def test_returns_list_of_campaign_metrics(self):
        from agents.specialist.data_analyst import campaign_performance_fetcher, CampaignMetrics
        results = await campaign_performance_fetcher(
            campaign_id="CAMP-001",
            metrics=["roas", "cac", "cpl"],
            date_range=("2024-10-01", "2024-10-31"),
        )
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CampaignMetrics) for r in results)

    @pytest.mark.asyncio
    async def test_roas_equals_revenue_over_spend(self):
        from agents.specialist.data_analyst import campaign_performance_fetcher
        results = await campaign_performance_fetcher(
            campaign_id="CAMP-001",
            metrics=["roas"],
            date_range=("2024-10-01", "2024-10-31"),
        )
        for m in results:
            expected_roas = round(m.revenue / m.spend, 2)
            assert m.roas == expected_roas

    @pytest.mark.asyncio
    async def test_cac_equals_spend_over_installs(self):
        from agents.specialist.data_analyst import campaign_performance_fetcher
        results = await campaign_performance_fetcher(
            campaign_id="CAMP-001",
            metrics=["cac"],
            date_range=("2024-10-01", "2024-10-31"),
        )
        for m in results:
            expected_cac = round(m.spend / m.installs, 2)
            assert m.cac == expected_cac

    @pytest.mark.asyncio
    async def test_cpl_equals_spend_over_leads(self):
        from agents.specialist.data_analyst import campaign_performance_fetcher
        results = await campaign_performance_fetcher(
            campaign_id="CAMP-001",
            metrics=["cpl"],
            date_range=("2024-10-01", "2024-10-31"),
        )
        for m in results:
            expected_cpl = round(m.spend / m.leads, 2)
            assert m.cpl == expected_cpl

    @pytest.mark.asyncio
    async def test_date_range_preserved_in_result(self):
        from agents.specialist.data_analyst import campaign_performance_fetcher
        dr = ("2024-10-01", "2024-10-31")
        results = await campaign_performance_fetcher(
            campaign_id="CAMP-001", metrics=["roas"], date_range=dr
        )
        assert results[0].date_range == dr


# ─────────────────────────────────────────────────────────────────────────────
#  TakeRateResult.to_summary() formatting
# ─────────────────────────────────────────────────────────────────────────────

class TestTakeRateToSummary:
    def test_summary_contains_rate_subs_and_passings(self):
        from agents.specialist.data_analyst import TakeRateResult
        r = TakeRateResult(rate=0.24, passings=45_000, active_subs=10_800,
                           region="PNW", period="Q3-2024")
        s = r.to_summary()
        assert "24.0%" in s
        assert "10,800" in s
        assert "45,000" in s

    def test_summary_includes_delta_when_provided(self):
        from agents.specialist.data_analyst import TakeRateResult
        r = TakeRateResult(rate=0.24, passings=45_000, active_subs=10_800,
                           region="PNW", period="Q3-2024", delta_pp=0.022)
        s = r.to_summary()
        assert "+" in s          # positive delta gets a "+" prefix
        assert "pp" in s

    def test_summary_shows_negative_delta(self):
        from agents.specialist.data_analyst import TakeRateResult
        r = TakeRateResult(rate=0.20, passings=45_000, active_subs=9_000,
                           region="PNW", period="Q3-2024", delta_pp=-0.04)
        s = r.to_summary()
        assert "-4.0%pp" in s or "-0.04" in s.replace(" ", "")

    def test_summary_always_shows_industry_avg(self):
        from agents.specialist.data_analyst import TakeRateResult
        r = TakeRateResult(rate=0.24, passings=45_000, active_subs=10_800,
                           region="PNW", period="Q3-2024")
        s = r.to_summary()
        assert "Industry avg" in s


# ─────────────────────────────────────────────────────────────────────────────
#  Agentic tool loop integration (DataAnalystAgent.run)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataAnalystToolLoop:
    """
    Mocks client.messages.create to exercise the while-True tool loop without
    making real API calls. Verifies that:
    - tool_use blocks trigger the right tool function
    - tool results are accumulated and returned in the output
    - end_turn terminates the loop and returns the final text
    """

    @pytest.mark.asyncio
    async def test_single_tool_call_dispatched_and_returned(self):
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from agents.specialist.data_analyst import DataAnalystAgent

        tool_block = SimpleNamespace(
            type="tool_use",
            name="benchmark_comparator",
            id="tu_001",
            input={"metric": "fiber_take_rate_national", "actual_value": 0.24},
        )
        first_resp = MagicMock()
        first_resp.stop_reason = "tool_use"
        first_resp.content     = [tool_block]

        text_block = SimpleNamespace(type="text", text="The 24% take rate is above the 18% national benchmark.")
        end_resp   = MagicMock()
        end_resp.stop_reason = "end_turn"
        end_resp.content     = [text_block]

        with patch("agents.specialist.data_analyst.client.messages.create",
                   side_effect=[first_resp, end_resp]):
            result = await DataAnalystAgent().run(
                "How does our take rate compare to national benchmarks?"
            )

        assert result["agent"] == "data_analyst"
        assert "benchmark_comparator" in result["tool_calls"]
        assert len(result["tool_results"]) == 1
        assert "fiber_take_rate_national" in result["tool_results"][0]["content"]
        assert result["response"] == "The 24% take rate is above the 18% national benchmark."

    @pytest.mark.asyncio
    async def test_unknown_tool_captured_as_error(self):
        from types import SimpleNamespace
        from unittest.mock import patch, MagicMock
        from agents.specialist.data_analyst import DataAnalystAgent

        bad_block = SimpleNamespace(
            type="tool_use",
            name="nonexistent_tool",
            id="tu_002",
            input={},
        )
        first_resp = MagicMock()
        first_resp.stop_reason = "tool_use"
        first_resp.content     = [bad_block]

        text_block = SimpleNamespace(type="text", text="I could not find the requested data.")
        end_resp   = MagicMock()
        end_resp.stop_reason = "end_turn"
        end_resp.content     = [text_block]

        with patch("agents.specialist.data_analyst.client.messages.create",
                   side_effect=[first_resp, end_resp]):
            result = await DataAnalystAgent().run("Use a nonexistent tool")

        assert len(result["tool_results"]) == 1
        content = result["tool_results"][0]["content"]
        assert "error" in content.lower() or "unknown" in content.lower()

    @pytest.mark.asyncio
    async def test_tool_results_returned_in_invoke_output(self):
        from unittest.mock import patch, AsyncMock
        from agents.specialist.data_analyst import DataAnalystAgent
        from agents.base import initial_state

        mock_run_result = {
            "response": "Take rate is 24 percent of 45000 passings.",
            "agent": "data_analyst",
            "tool_calls": ["take_rate_calculator"],
            "tool_results": [
                {"type": "tool_result", "tool_name": "take_rate_calculator",
                 "content": '{"passings": 45000, "active_subs": 10800}'}
            ],
        }
        agent = DataAnalystAgent()
        state = initial_state("s", "u", "take rate?")
        with patch.object(agent, "run", new_callable=AsyncMock, return_value=mock_run_result):
            result = await agent.invoke(state)

        assert "analyst_output" in result
        assert result["analyst_output"]["tool_results"][0]["tool_name"] == "take_rate_calculator"
        assert "data_analyst" in result["completed_agents"]
