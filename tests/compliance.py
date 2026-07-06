"""
tests/compliance.py
====================
Tests for the full compliance stack: engine, TCPA, FTC/FCC, CAN-SPAM, privacy, state laws.
Run with: pytest tests/compliance.py -v
"""

from compliance.engine       import ComplianceEngine
from compliance.tcpa         import check_tcpa, requires_written_consent
from compliance.fcc_ftc_canspam import check_ftc_fcc, check_canspam
from compliance.privacy      import check_privacy, detect_sensitive_data, check_retention
from compliance.state_laws   import check_state_laws


# ─────────────────────────────────────────────────────────────────────────────
#  TCPA tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTCPA:
    def test_automated_text_without_consent_flagged(self):
        flags = check_tcpa("Send an automated text to all copper subscribers")
        assert len(flags) > 0
        assert any("Automated Outreach" in f.rule for f in flags)
        assert any(f.severity == "CRITICAL" for f in flags)

    def test_automated_text_with_optout_passes(self):
        flags = check_tcpa("Send automated text. Reply STOP to opt out.")
        critical = [f for f in flags if f.severity == "CRITICAL"]
        assert len(critical) == 0

    def test_cold_call_flagged(self):
        flags = check_tcpa("Cold call all residential numbers in the area")
        assert any("Do Not Call" in f.rule for f in flags)

    def test_purchased_list_flagged(self):
        flags = check_tcpa("Upload the purchased list to the dialer")
        assert any("purchased" in f.detail.lower() for f in flags)

    def test_sms_missing_frequency_disclosure(self):
        flags = check_tcpa("Welcome to FiberCo!", outreach_type="sms")
        assert any("Frequency" in f.rule for f in flags)

    def test_sms_with_full_disclosure_passes(self):
        flags = check_tcpa(
            "Welcome! Msg frequency varies. Msg & data rates may apply. Reply STOP to cancel.",
            outreach_type="sms",
        )
        assert not any(f.severity in ("CRITICAL", "BLOCK") for f in flags)

    def test_written_consent_required_for_sms(self):
        assert requires_written_consent("sms_marketing") is True

    def test_written_consent_not_required_for_service_call(self):
        assert requires_written_consent("service_call") is False

    def test_robocall_reference_flagged(self):
        flags = check_tcpa("Use the autodialer for the win-back campaign")
        assert len(flags) > 0

    def test_clean_copy_passes(self):
        flags = check_tcpa("Fiber internet for your home. Learn more at our website.")
        critical_or_block = [f for f in flags if f.severity in ("CRITICAL", "BLOCK")]
        assert len(critical_or_block) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  FTC / FCC tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFTCFCC:
    def test_fastest_flagged_critical(self):
        flags = check_ftc_fcc("We have the fastest fiber internet!")
        assert any(f.trigger.lower() == "fastest" for f in flags)
        assert any(f.severity == "CRITICAL" for f in flags)

    def test_rated_number_one_flagged(self):
        flags = check_ftc_fcc("Rated #1 by customers")
        assert len(flags) > 0
        assert any(f.severity == "CRITICAL" for f in flags)

    def test_guaranteed_flagged(self):
        flags = check_ftc_fcc("Guaranteed 1 Gbps speeds")
        assert any("guaranteed" in f.trigger.lower() for f in flags)

    def test_studies_show_flagged(self):
        flags = check_ftc_fcc("Studies show our network is 30% more reliable")
        assert any("studies" in f.trigger.lower() for f in flags)

    def test_unlimited_flagged_warn(self):
        flags = check_ftc_fcc("Unlimited data, no restrictions")
        assert any(f.severity == "WARN" for f in flags)

    def test_up_to_speed_flagged(self):
        flags = check_ftc_fcc("Get up to 1 Gbps speeds")
        assert any("up to" in f.trigger.lower() for f in flags)

    def test_clean_copy_no_flags(self):
        flags = check_ftc_fcc("Reliable fiber internet. Prices starting at $49/month.")
        assert len(flags) == 0

    def test_award_winning_is_warn_not_critical(self):
        flags = check_ftc_fcc("Our award-winning service")
        assert all(f.severity != "CRITICAL" for f in flags)


# ─────────────────────────────────────────────────────────────────────────────
#  CAN-SPAM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCANSPAM:
    def test_missing_optout_blocks(self):
        flags = check_canspam(
            subject_line = "Special offer",
            body         = "Check out our fiber plans.",
            has_opt_out  = False,
            has_address  = True,
        )
        assert any(f.severity == "BLOCK" for f in flags)
        assert any("Opt-Out" in f.rule for f in flags)

    def test_missing_address_blocks(self):
        flags = check_canspam(
            subject_line = "Special offer",
            body         = "Unsubscribe here.",
            has_opt_out  = True,
            has_address  = False,
        )
        assert any(f.severity == "BLOCK" for f in flags)
        assert any("Address" in f.rule for f in flags)

    def test_body_optout_language_satisfies_requirement(self):
        flags = check_canspam(
            subject_line = "Great offer",
            body         = "Click here to unsubscribe from our emails.",
            has_opt_out  = False,
            has_address  = True,
        )
        # body scan should find "unsubscribe" and satisfy the requirement
        assert not any(f.rule == "Opt-Out Mechanism Required" for f in flags)

    def test_deceptive_re_subject_flagged(self):
        flags = check_canspam(
            subject_line = "Re: Your account",
            body         = "Click unsubscribe. 123 Main St.",
            has_opt_out  = True,
            has_address  = True,
        )
        assert any("Deceptive" in f.rule for f in flags)

    def test_compliant_email_passes(self):
        flags = check_canspam(
            subject_line = "New fiber plans available",
            body         = "Check plans. Unsubscribe here. 123 Main St, Seattle WA.",
            has_opt_out  = True,
            has_address  = True,
        )
        blocking = [f for f in flags if f.severity == "BLOCK"]
        assert len(blocking) == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Privacy tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPrivacy:
    def test_detect_sensitive_health_data(self):
        cats = detect_sensitive_data("We have medical data for all subscribers")
        assert any("Health" in c for c in cats)

    def test_detect_sensitive_biometric(self):
        cats = detect_sensitive_data("Store biometric data for authentication")
        assert any("Biometric" in c for c in cats)

    def test_detect_no_sensitive_data(self):
        cats = detect_sensitive_data("Fiber internet starts at $49/month")
        assert cats == []

    def test_retention_over_limit_flagged(self):
        risk = check_retention("calix_demographic_cache", actual_days=120)
        assert risk is not None
        assert risk.severity == "HIGH"
        assert "90" in risk.detail

    def test_retention_within_limit_passes(self):
        risk = check_retention("calix_demographic_cache", actual_days=30)
        assert risk is None

    def test_sensitive_data_in_content_raises_risk(self):
        risks = check_privacy("We target customers based on their precise geolocation data")
        assert len(risks) > 0
        assert any(r.severity == "HIGH" for r in risks)


# ─────────────────────────────────────────────────────────────────────────────
#  State laws tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStateLaws:
    def test_florida_sms_flagged(self):
        flags = check_state_laws(
            content       = "Send automated text to all FL customers",
            target_states = ["FL"],
        )
        assert any(f.state == "FL" for f in flags)
        assert any(f.severity == "CRITICAL" for f in flags)

    def test_illinois_biometric_blocked(self):
        flags = check_state_laws(
            content       = "Collect biometric data for authentication in Illinois",
            target_states = ["IL"],
        )
        assert any(f.state == "IL" for f in flags)
        assert any(f.severity == "BLOCK" for f in flags)

    def test_california_ccpa_triggered(self):
        flags = check_state_laws(
            content       = "Marketing campaign for California",
            context       = {"processes_ca_residents": True},
            target_states = ["CA"],
        )
        assert any(f.state == "CA" for f in flags)

    def test_no_flags_for_irrelevant_state(self):
        flags = check_state_laws(
            content       = "Standard fiber promotion",
            target_states = ["MT"],   # Montana — no specific rules in registry
        )
        assert len(flags) == 0

    def test_state_detection_from_content(self):
        flags = check_state_laws(
            content = "Launch fiber in Florida and Illinois this quarter",
            context = {},
        )
        states = {f.state for f in flags}
        # Should detect FL and IL from content text
        assert "FL" in states or "IL" in states


# ─────────────────────────────────────────────────────────────────────────────
#  Compliance Engine integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComplianceEngine:
    def setup_method(self):
        self.engine = ComplianceEngine()

    def test_clean_copy_returns_go(self):
        report = self.engine.check(
            content   = "Reliable fiber internet. From $49/month. Terms apply.",
            operation = "social_post",
        )
        assert report.go_no_go == "GO"
        assert report.requires_legal is False

    def test_fastest_claim_returns_no_go(self):
        report = self.engine.check(
            content   = "The fastest internet in the region, guaranteed!",
            operation = "social_post",
        )
        assert report.go_no_go == "NO-GO"
        assert report.requires_legal is True

    def test_email_without_optout_returns_no_go(self):
        report = self.engine.check(
            content   = "Check out our new fiber plans.",
            operation = "email",
            context   = {"has_opt_out": False, "has_address": True, "subject_line": "New plans"},
        )
        assert report.go_no_go == "NO-GO"

    def test_report_to_dict_serializable(self):
        import json
        report = self.engine.check("Fiber internet for your home.", "social_post")
        d = report.to_dict()
        # Must be JSON-serializable
        serialized = json.dumps(d)
        assert "go_no_go" in serialized

    def test_ftc_warn_returns_go_with_review(self):
        report = self.engine.check(
            content   = "Unlimited fiber internet with no data caps.",
            operation = "social_post",
        )
        assert report.go_no_go in ("GO-WITH-REVIEW", "NO-GO")

    def test_has_blocks_property(self):
        report = self.engine.check(
            content   = "Email without opt-out.",
            operation = "email",
            context   = {"has_opt_out": False, "has_address": False, "subject_line": "Offer"},
        )
        assert report.has_blocks is True
