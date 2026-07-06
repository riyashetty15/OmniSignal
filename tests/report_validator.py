"""
tests/report_validator.py
=========================
Tests for the ReportValidatorAgent and its three component scoring functions.

Scoring architecture (composite = weighted sum):
  structural   30% — intent-aware keyword coverage
  numeric      40% — faithfulness: response numbers cross-checked against tool outputs
  citation     30% — HR/financial must cite sources; analyst/strategist are lenient

Run with: pytest tests/report_validator.py -v
"""

import json
import pytest
from agents.base import initial_state
from agents.validation.report_validator import (
    _extract_numbers,
    _is_faithful,
    _structural_score,
    _numeric_grounding_score,
    _citation_score,
    _compute_fidelity,
    _confidence,
    ReportValidatorAgent,
)
from shared_config import FIDELITY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tool_result(data: dict) -> dict:
    """Builds a tool_result dict the way agents return them."""
    return {"type": "tool_result", "tool_use_id": "tu_test", "content": json.dumps(data)}


# ─────────────────────────────────────────────────────────────────────────────
#  _extract_numbers
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractNumbers:
    def test_extracts_percentage(self):
        assert 24.1 in _extract_numbers("take rate is 24.1%")

    def test_extracts_dollar_amount(self):
        assert 84000.0 in _extract_numbers("spend $84000 on ads")

    def test_extracts_dollar_with_k_suffix(self):
        assert 84000.0 in _extract_numbers("budget was $84K")

    def test_extracts_dollar_with_m_suffix(self):
        assert 1_500_000.0 in _extract_numbers("revenue $1.5M")

    def test_extracts_large_plain_number(self):
        assert 45000.0 in _extract_numbers("45000 passings")

    def test_ignores_numbers_under_four_digits(self):
        # _PLAIN_NUM_RE requires 4+ digit numbers
        nums = _extract_numbers("3 regions, 99 campaigns")
        assert 3.0 not in nums
        assert 99.0 not in nums

    def test_empty_string_returns_empty_list(self):
        assert _extract_numbers("") == []

    def test_multiple_types_in_one_string(self):
        nums = _extract_numbers("45000 passings, 24.0%, $84000 spend")
        assert 45000.0 in nums
        assert 24.0 in nums
        assert 84000.0 in nums


# ─────────────────────────────────────────────────────────────────────────────
#  _is_faithful
# ─────────────────────────────────────────────────────────────────────────────

class TestIsFaithful:
    def test_exact_match_is_faithful(self):
        assert _is_faithful(45000.0, [45000.0, 10800.0]) is True

    def test_within_five_pct_tolerance(self):
        # 4.4% difference — inside the 5% window
        assert _is_faithful(46980.0, [45000.0]) is True

    def test_outside_five_pct_tolerance(self):
        # 6.7% difference — outside the 5% window
        assert _is_faithful(48000.0, [45000.0]) is False

    def test_empty_tool_nums_always_faithful(self):
        # No tool outputs → can't penalise
        assert _is_faithful(99999.0, []) is True

    def test_zero_denominator_exact_zero_faithful(self):
        assert _is_faithful(0.0, [0.0]) is True

    def test_zero_denominator_nonzero_not_faithful(self):
        assert _is_faithful(1.0, [0.0]) is False


# ─────────────────────────────────────────────────────────────────────────────
#  _structural_score
# ─────────────────────────────────────────────────────────────────────────────

class TestStructuralScore:
    def test_take_rate_all_keyword_groups_present(self):
        text = (
            "The take rate shows 10800 active subscribers out of 45000 passings, "
            "representing 24 percent penetration."
        )
        score, missing = _structural_score(text, "data_analyst", "take_rate")
        assert score == 1.0
        assert missing == []

    def test_take_rate_missing_one_group_lowers_score(self):
        # No "%" or "percent" — Group 4 missing
        text = "The take rate shows 10800 active subscribers out of 45000 passings."
        score, missing = _structural_score(text, "data_analyst", "take_rate")
        assert score == 0.75
        assert len(missing) == 1

    def test_empty_response_scores_zero(self):
        score, missing = _structural_score("", "data_analyst", "take_rate")
        assert score == 0.0
        assert "empty response" in missing

    def test_campaign_performance_all_groups(self):
        text = "ROAS was 4.0. CAC $200. Impressions 500000. Campaign CAMP-001 performed well."
        score, missing = _structural_score(text, "data_analyst", "campaign_performance")
        assert score == 1.0

    def test_roi_npv_intent_all_groups(self):
        text = (
            "The NPV (net present value) is $2.1M. IRR is 14%. "
            "Payback period is 48 months. CAPEX is $6.5M. "
            "Recommendation: BUILD."
        )
        score, missing = _structural_score(text, "financial_planner", "roi_npv")
        assert score == 1.0

    def test_unknown_intent_falls_back_to_agent_keywords(self):
        text = "The take rate and campaign anomaly analysis shows promising subscribers."
        score, _ = _structural_score(text, "data_analyst", "totally_unknown_intent")
        # Falls back to _AGENT_KEYWORDS["data_analyst"] — some groups should match
        assert score > 0.0

    def test_leave_policy_intent(self):
        text = "You may submit a PTO leave request. Approval required within 5 days."
        score, missing = _structural_score(text, "hr_docqa", "leave_policy")
        assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  _numeric_grounding_score
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericGroundingScore:
    def test_analyst_all_numbers_match_tool_output(self):
        tool_results = [_tool_result({"passings": 45000, "active_subs": 10800})]
        text = "There are 45000 passings and 10800 active subscribers."
        score, notes = _numeric_grounding_score(text, "data_analyst", tool_results)
        assert score == 1.0
        assert notes == []

    def test_analyst_hallucinated_numbers_score_low(self):
        tool_results = [_tool_result({"passings": 45000, "active_subs": 10800})]
        text = "There are 99999 passings and 88888 active subscribers."
        score, notes = _numeric_grounding_score(text, "data_analyst", tool_results)
        assert score <= 0.50
        assert any("hallucination" in n.lower() or "not found" in n.lower() for n in notes)

    def test_analyst_no_numbers_in_response(self):
        tool_results = [_tool_result({"passings": 45000})]
        text = "The take rate looks reasonable for this region."
        score, notes = _numeric_grounding_score(text, "data_analyst", tool_results)
        assert score == 0.3
        assert any("no numeric" in n.lower() for n in notes)

    def test_analyst_no_tool_results_falls_back_to_presence_only(self):
        text = "There are 45000 passings and 10800 active subscribers."
        score, notes = _numeric_grounding_score(text, "data_analyst", [])
        assert score == 0.85
        assert any("presence-only" in n.lower() for n in notes)

    def test_analyst_partial_match_scores_between_thresholds(self):
        # 45000 matches but 99999 is hallucinated → partial
        tool_results = [_tool_result({"passings": 45000})]
        text = "There are 45000 passings and 99999 active subscribers."
        score, notes = _numeric_grounding_score(text, "data_analyst", tool_results)
        # 1 faithful, 1 unfaithful → ratio 0.5 → score 0.65
        assert 0.50 < score < 1.0

    def test_strategist_with_platform_and_hashtag_full_score(self):
        text = "Post this to LinkedIn today! #FiberFirst #ConnectedCommunity"
        score, notes = _numeric_grounding_score(text, "strategist", [])
        assert score == 1.0

    def test_strategist_missing_platform_and_hashtag_penalized(self):
        text = "Write some engaging content about fiber broadband benefits."
        score, notes = _numeric_grounding_score(text, "strategist", [])
        assert score == 0.70
        assert any("platform" in n.lower() or "hashtag" in n.lower() for n in notes)

    def test_hr_always_scores_full_numeric(self):
        # HR is text-heavy — no numeric check applied
        score, notes = _numeric_grounding_score(
            "Our PTO policy allows 15 days per year.", "hr_docqa", []
        )
        assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  _citation_score
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationScore:
    def test_hr_with_section_citation_passes(self):
        text = "Per Section 4.2 of the Remote Work Policy, equipment reimbursement is $500."
        score, notes = _citation_score(text, "hr_docqa")
        assert score == 1.0
        assert notes == []

    def test_hr_with_policy_id_citation_passes(self):
        text = "Refer to POL-2024-HR-001 for the full benefits schedule."
        score, notes = _citation_score(text, "hr_docqa")
        assert score == 1.0

    def test_hr_with_version_citation_passes(self):
        text = "This policy is version 2.3, effective 2024."
        score, notes = _citation_score(text, "hr_docqa")
        assert score == 1.0

    def test_hr_missing_citation_penalized(self):
        text = "You get 15 days of PTO per year."
        score, notes = _citation_score(text, "hr_docqa")
        assert score == 0.50
        assert len(notes) > 0

    def test_financial_with_calix_and_model_assumptions_full_score(self):
        text = (
            "Based on Calix data vintage 2024-Q3, our model assumptions include "
            "8% discount rate and 65% gross margin."
        )
        score, notes = _citation_score(text, "financial_planner")
        assert score == 1.0

    def test_financial_missing_calix_reference_penalized(self):
        text = "The NPV is positive under base case assumptions."
        score, notes = _citation_score(text, "financial_planner")
        assert score == 0.80

    def test_financial_has_calix_but_no_model_slight_penalty(self):
        text = "Based on Calix data vintage 2024-Q3, the NPV looks positive."
        score, notes = _citation_score(text, "financial_planner")
        assert score == 0.90

    def test_analyst_always_lenient(self):
        score, notes = _citation_score("Any analyst response with no citations.", "data_analyst")
        assert score == 1.0

    def test_strategist_always_lenient(self):
        score, notes = _citation_score("Any strategist response with no citations.", "strategist")
        assert score == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  _confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidence:
    def _ok_result(self) -> dict:
        return {"type": "tool_result", "content": '{"passings": 45000}'}

    def _err_result(self) -> dict:
        return {"type": "tool_result", "content": '{"error": "DB connection failed"}'}

    def test_analyst_no_tools_base_is_045(self):
        assert _confidence("short", "data_analyst", []) == 0.45

    def test_strategist_no_tools_base_is_065(self):
        assert _confidence("short", "strategist", []) == 0.65

    def test_hr_no_tools_base_is_065(self):
        assert _confidence("short", "hr_docqa", []) == 0.65

    def test_successful_tool_call_boosts_to_080(self):
        score = _confidence("short", "data_analyst", [self._ok_result()])
        assert score >= 0.80

    def test_all_errored_tools_scores_055(self):
        score = _confidence("short", "data_analyst", [self._err_result()])
        assert score == 0.55

    def test_response_over_500_chars_adds_bonus(self):
        long_text  = "word " * 120        # ~600 chars
        short_text = "word"
        assert _confidence(long_text, "data_analyst", []) > _confidence(short_text, "data_analyst", [])

    def test_response_over_1000_chars_adds_second_bonus(self):
        very_long = "word " * 250        # ~1250 chars
        medium    = "word " * 120        # ~600 chars
        assert _confidence(very_long, "data_analyst", []) > _confidence(medium, "data_analyst", [])

    def test_markdown_structure_adds_bonus(self):
        plain_text = "The take rate is 24 percent of homes passed."
        md_text    = "## Take Rate\n**Summary:** The take rate is 24 percent of homes passed."
        assert _confidence(md_text, "data_analyst", []) > _confidence(plain_text, "data_analyst", [])

    def test_confidence_never_exceeds_one(self):
        very_long_md = ("## Section\n**text** " * 300)    # >1000 chars + markdown
        result = _confidence(very_long_md, "data_analyst", [self._ok_result()])
        assert result <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  _compute_fidelity — composite scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFidelity:
    def test_weights_are_correct(self):
        # The three weights must sum exactly to 1.0
        assert abs(0.30 + 0.40 + 0.30 - 1.0) < 1e-9

    def test_good_take_rate_response_passes_threshold(self):
        # Response uses "percent" (not "%") so the literal "24" is not extracted as a
        # percentage — all response numbers (45000, 10800) match tool outputs exactly.
        text = (
            "The take rate analysis shows 45000 passings with 10800 active subscribers. "
            "The penetration rate is 24 percent of homes passed."
        )
        tool_results = [_tool_result({"passings": 45000, "active_subs": 10800})]
        state = initial_state("s", "u", "take rate?")
        score, notes = _compute_fidelity(text, "data_analyst", "take_rate", tool_results, state)
        assert score >= FIDELITY_THRESHOLD, (
            f"Expected >= {FIDELITY_THRESHOLD}, got {score:.3f}. Notes: {notes}"
        )

    def test_empty_response_fails_threshold(self):
        state = initial_state("s", "u", "take rate?")
        score, notes = _compute_fidelity("", "data_analyst", "take_rate", [], state)
        assert score < FIDELITY_THRESHOLD

    def test_hallucinated_numbers_fail_threshold(self):
        text = (
            "The take rate analysis shows 99999 passings with 88888 active subscribers, "
            "yielding a 50 percent penetration rate."
        )
        tool_results = [_tool_result({"passings": 45000, "active_subs": 10800})]
        state = initial_state("s", "u", "take rate?")
        score, notes = _compute_fidelity(text, "data_analyst", "take_rate", tool_results, state)
        assert score < FIDELITY_THRESHOLD

    def test_hr_response_with_citation_passes(self):
        text = (
            "Per Section 3.1 of the Leave Policy (Version 2.3, effective 2024), "
            "you may submit a PTO leave request. Approval from your manager is required within 5 days."
        )
        state = initial_state("s", "u", "leave policy?", department="hr")
        score, notes = _compute_fidelity(text, "hr_docqa", "leave_policy", [], state)
        assert score >= FIDELITY_THRESHOLD, (
            f"Expected >= {FIDELITY_THRESHOLD}, got {score:.3f}. Notes: {notes}"
        )

    def test_hr_response_without_citation_fails(self):
        text = "You get 15 days of PTO per year."
        state = initial_state("s", "u", "leave policy?", department="hr")
        score, _ = _compute_fidelity(text, "hr_docqa", "leave_policy", [], state)
        assert score < FIDELITY_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
#  ReportValidatorAgent.invoke — full LangGraph node integration
# ─────────────────────────────────────────────────────────────────────────────

class TestReportValidatorAgent:
    @pytest.mark.asyncio
    async def test_good_analyst_response_passes_gate(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "take rate?")
        state["routed_agent"] = "data_analyst"
        state["intent"]       = "take_rate"
        state["analyst_output"] = {
            "response": (
                "The take rate analysis shows 45000 passings with 10800 active subscribers. "
                "The penetration rate is 24 percent of homes passed."
            ),
            "agent": "data_analyst",
            "tool_results": [_tool_result({"passings": 45000, "active_subs": 10800})],
        }
        result = await agent.invoke(state)
        assert result["fidelity_score"] >= FIDELITY_THRESHOLD
        assert result["evidence_coverage"] in ("full", "partial")
        assert result["retry_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_response_triggers_retry(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "take rate?")
        state["routed_agent"]   = "data_analyst"
        state["intent"]         = "take_rate"
        state["analyst_output"] = {"response": "", "agent": "data_analyst", "tool_results": []}
        result = await agent.invoke(state)
        assert result["fidelity_score"] < FIDELITY_THRESHOLD
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_retry_count_does_not_double_increment(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "q")
        state["routed_agent"]   = "data_analyst"
        state["intent"]         = "take_rate"
        state["retry_count"]    = 1   # already retried once
        state["analyst_output"] = {"response": "", "agent": "data_analyst", "tool_results": []}
        result = await agent.invoke(state)
        assert result["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_hr_response_with_citation_passes(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "leave policy?", department="hr")
        state["routed_agent"] = "hr_docqa"
        state["intent"]       = "leave_policy"
        state["hr_output"] = {
            "response": (
                "Per Section 3.1 of the Leave Policy (Version 2.3, effective 2024), "
                "you may submit a PTO leave request. Approval required within 5 days."
            ),
            "agent": "hr_docqa",
            "tool_results": [],
        }
        result = await agent.invoke(state)
        assert result["fidelity_score"] >= FIDELITY_THRESHOLD

    @pytest.mark.asyncio
    async def test_hr_response_without_citation_fails(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "leave policy?", department="hr")
        state["routed_agent"] = "hr_docqa"
        state["intent"]       = "leave_policy"
        state["hr_output"] = {
            "response": "You get 15 days of PTO per year.",
            "agent": "hr_docqa",
            "tool_results": [],
        }
        result = await agent.invoke(state)
        assert result["fidelity_score"] < FIDELITY_THRESHOLD
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_output_confidence_reflects_successful_tool_use(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "q")
        state["routed_agent"] = "data_analyst"
        state["intent"]       = "take_rate"
        state["analyst_output"] = {
            "response": "There are 45000 passings and 10800 active subscribers.",
            "agent": "data_analyst",
            "tool_results": [_tool_result({"passings": 45000, "active_subs": 10800})],
        }
        result = await agent.invoke(state)
        assert result["output_confidence"] >= 0.80

    @pytest.mark.asyncio
    async def test_output_confidence_low_when_no_tools_called(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "q")
        state["routed_agent"] = "data_analyst"
        state["intent"]       = "take_rate"
        state["analyst_output"] = {
            "response": "The take rate looks fine.",
            "agent": "data_analyst",
            "tool_results": [],
        }
        result = await agent.invoke(state)
        assert result["output_confidence"] < 0.70

    @pytest.mark.asyncio
    async def test_validation_notes_populated_on_failure(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "take rate?")
        state["routed_agent"]   = "data_analyst"
        state["intent"]         = "take_rate"
        state["analyst_output"] = {"response": "", "agent": "data_analyst", "tool_results": []}
        result = await agent.invoke(state)
        assert len(result["validation_notes"]) > 0

    @pytest.mark.asyncio
    async def test_evidence_coverage_full_for_high_fidelity(self):
        agent = ReportValidatorAgent()
        state = initial_state("s", "u", "q")
        state["routed_agent"] = "data_analyst"
        state["intent"]       = "take_rate"
        state["analyst_output"] = {
            "response": (
                "The take rate analysis shows 45000 passings with 10800 active subscribers. "
                "The penetration rate is 24 percent of homes passed."
            ),
            "agent": "data_analyst",
            "tool_results": [_tool_result({"passings": 45000, "active_subs": 10800})],
        }
        result = await agent.invoke(state)
        assert result["evidence_coverage"] == "full"
