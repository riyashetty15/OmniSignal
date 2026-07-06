"""
compliance/engine.py
=====================
ComplianceEngine — single entry point that aggregates all compliance checks.

Agents call this before finalizing any content that will be:
  - Published (social media, email, website copy)
  - Sent as outreach (SMS, calls)
  - Used in data operations (demographic targeting, list uploads)

Returns a ComplianceReport with all flags, a severity summary, and a
go/no-go recommendation. "BLOCK" severity on any single flag = NO-GO.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from compliance.tcpa          import check_tcpa, TCPAFlag
from compliance.fcc_ftc_canspam import check_ftc_fcc, check_canspam, ComplianceFlag
from compliance.privacy       import check_privacy, PrivacyRisk
from compliance.state_laws    import check_state_laws, StateComplianceFlag


@dataclass
class ComplianceReport:
    content_snippet:  str                      # first 200 chars of checked content
    operation:        str                      # what kind of content this is
    go_no_go:         str                      # "GO" | "NO-GO" | "GO-WITH-REVIEW"
    requires_legal:   bool
    tcpa_flags:       list[TCPAFlag]           = field(default_factory=list)
    ftc_fcc_flags:    list[ComplianceFlag]     = field(default_factory=list)
    canspam_flags:    list[ComplianceFlag]     = field(default_factory=list)
    privacy_risks:    list[PrivacyRisk]        = field(default_factory=list)
    state_flags:      list[StateComplianceFlag] = field(default_factory=list)
    summary:          str                      = ""

    @property
    def all_flags(self) -> list:
        return (
            self.tcpa_flags +
            self.ftc_fcc_flags +
            self.canspam_flags +
            self.privacy_risks +
            self.state_flags
        )

    @property
    def has_blocks(self) -> bool:
        return any(
            getattr(f, "severity", "") in ("BLOCK",)
            for f in self.all_flags
        )

    @property
    def has_criticals(self) -> bool:
        return any(
            getattr(f, "severity", "") == "CRITICAL"
            for f in self.all_flags
        )

    def to_dict(self) -> dict:
        def _flag_to_dict(f: Any) -> dict:
            return {k: v for k, v in f.__dict__.items()}

        return {
            "go_no_go":        self.go_no_go,
            "requires_legal":  self.requires_legal,
            "summary":         self.summary,
            "tcpa_flags":      [_flag_to_dict(f) for f in self.tcpa_flags],
            "ftc_fcc_flags":   [_flag_to_dict(f) for f in self.ftc_fcc_flags],
            "canspam_flags":   [_flag_to_dict(f) for f in self.canspam_flags],
            "privacy_risks":   [_flag_to_dict(f) for f in self.privacy_risks],
            "state_flags":     [_flag_to_dict(f) for f in self.state_flags],
            "total_flags":     len(self.all_flags),
        }


class ComplianceEngine:
    """
    Aggregates all compliance checkers into one call.

    Usage:
        engine = ComplianceEngine()
        report = engine.check(
            content   = "Our fastest fiber internet...",
            operation = "social_post",
            context   = {"target_states": ["CA", "FL"]},
        )
        if report.go_no_go == "NO-GO":
            raise ValueError(report.summary)
    """

    def check(
        self,
        content:   str,
        operation: str = "general",
        context:   dict | None = None,
    ) -> ComplianceReport:
        """
        Runs all applicable compliance checks for the given content and operation.

        operation: "social_post" | "email" | "sms" | "call_script" |
                   "data_operation" | "campaign_brief" | "general"

        context keys (all optional):
          target_states       : list[str]  — e.g. ["CA", "FL"]
          is_email            : bool
          subject_line        : str        — email subject (for CAN-SPAM)
          has_opt_out         : bool
          has_address         : bool
          outreach_type       : str        — "sms" | "call" | "email"
          processes_ca_residents: bool
          processes_va_residents: bool
          processes_co_residents: bool
          data_type           : str
          retention_days      : int
        """
        ctx = context or {}

        # 1. TCPA
        tcpa = check_tcpa(
            content       = content,
            outreach_type = ctx.get("outreach_type", "general"),
        )

        # 2. FTC / FCC advertising
        ftc_fcc = check_ftc_fcc(content)

        # 3. CAN-SPAM (only for email operations)
        canspam: list[ComplianceFlag] = []
        if operation == "email" or ctx.get("is_email"):
            canspam = check_canspam(
                subject_line = ctx.get("subject_line", ""),
                body         = content,
                has_opt_out  = ctx.get("has_opt_out", False),
                has_address  = ctx.get("has_address", False),
            )

        # 4. Privacy
        privacy = check_privacy(content, operation, ctx)

        # 5. State-specific
        state = check_state_laws(
            content       = content,
            context       = ctx,
            target_states = ctx.get("target_states"),
        )

        # ── Aggregate decision ─────────────────────────────────────────────────
        all_flags = tcpa + ftc_fcc + canspam + privacy + state

        has_block    = any(getattr(f, "severity", "") == "BLOCK"    for f in all_flags)
        has_critical = any(getattr(f, "severity", "") == "CRITICAL" for f in all_flags)
        has_warn     = any(getattr(f, "severity", "") == "WARN"     for f in all_flags)

        if has_block:
            go_no_go       = "NO-GO"
            requires_legal = True
            summary        = f"BLOCKED: {sum(1 for f in all_flags if getattr(f,'severity','')=='BLOCK')} blocking issue(s) found. Legal review required before proceeding."
        elif has_critical:
            go_no_go       = "NO-GO"
            requires_legal = True
            summary        = f"CRITICAL compliance issues ({sum(1 for f in all_flags if getattr(f,'severity','')=='CRITICAL')} found). Must resolve before publishing."
        elif has_warn:
            go_no_go       = "GO-WITH-REVIEW"
            requires_legal = False
            summary        = f"Proceed with caution: {sum(1 for f in all_flags if getattr(f,'severity','')=='WARN')} warning(s). Review before publishing."
        else:
            go_no_go       = "GO"
            requires_legal = False
            summary        = "No compliance issues detected."

        return ComplianceReport(
            content_snippet = content[:200],
            operation       = operation,
            go_no_go        = go_no_go,
            requires_legal  = requires_legal,
            tcpa_flags      = tcpa,
            ftc_fcc_flags   = ftc_fcc,
            canspam_flags   = canspam,
            privacy_risks   = privacy,
            state_flags     = state,
            summary         = summary,
        )
