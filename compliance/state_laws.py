"""
compliance/state_laws.py
=========================
State-specific telecom marketing regulations.

States with notable stricter-than-federal rules:
  California — CCPA/CPRA, CPUC rate notice requirements
  Florida    — FTSA (stricter TCPA, $500/call statutory damages)
  Illinois   — BIPA (biometric data)
  Texas      — Ch. 305 (email spam)
  New York   — SHIELD Act (breach notification)
  Virginia   — VCDPA (privacy)
  Colorado   — CPA (privacy)
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class StateComplianceFlag:
    state:       str
    law:         str
    severity:    str
    detail:      str
    remediation: str


_STATE_RULES: dict[str, list[dict]] = {

    "CA": [
        {
            "law":      "CCPA/CPRA (Cal. Civ. Code §1798.100)",
            "trigger":  lambda c, ctx: ctx.get("processes_ca_residents", False),
            "severity": "CRITICAL",
            "detail":   "CA residents' PI is subject to CCPA: delete, access, opt-out of sale rights",
            "remediation": "Ensure privacy policy, deletion mechanism, and 'Do Not Sell' link are live.",
        },
        {
            "law":      "Cal. Public Utilities Code §2872",
            "trigger":  lambda c, ctx: any(kw in c.lower() for kw in ["rate increase", "price change", "price increase"]),
            "severity": "WARN",
            "detail":   "CPUC requires 30-day advance written notice to CA residential customers before rate increases",
            "remediation": "Send written notice 30 days before any residential rate change in California.",
        },
    ],

    "FL": [
        {
            "law":      "Florida Telephone Solicitation Act (FTSA) §501.059",
            "trigger":  lambda c, ctx: any(kw in c.lower() for kw in ["text", "sms", "automated call", "dialer"]),
            "severity": "CRITICAL",
            "detail":   (
                "FTSA is stricter than federal TCPA: requires prior express WRITTEN consent "
                "for ALL automated calls/texts, including informational. $500/call uncapped damages."
            ),
            "remediation": (
                "For FL subscribers: obtain written consent before any automated communication. "
                "Document: timestamp, IP address, and exact consent language shown."
            ),
        },
    ],

    "IL": [
        {
            "law":      "Illinois Biometric Information Privacy Act (BIPA) 740 ILCS 14",
            "trigger":  lambda c, ctx: any(kw in c.lower() for kw in ["biometric", "fingerprint", "facial recognition", "voice print"]),
            "severity": "BLOCK",
            "detail":   "BIPA governs collection of biometric identifiers. $1,000-$5,000 per violation.",
            "remediation": (
                "Before collecting biometric data from IL residents: "
                "(1) publish written retention policy; (2) get written consent; "
                "(3) never sell biometric data."
            ),
        },
    ],

    "TX": [
        {
            "law":      "Texas Business & Commerce Code Ch. 305",
            "trigger":  lambda c, ctx: ctx.get("is_email", False),
            "severity": "WARN",
            "detail":   "TX anti-spam: requires accurate sender identity and opt-out honored within 30 days",
            "remediation": "Ensure sender identity is accurate; process opt-outs within 30 days.",
        },
    ],

    "NY": [
        {
            "law":      "NY SHIELD Act (General Business Law §899-aa)",
            "trigger":  lambda c, ctx: any(kw in c.lower() for kw in ["breach", "security incident", "data leak"]),
            "severity": "CRITICAL",
            "detail":   "NY SHIELD Act: breach notification to NY residents and NY AG within 72 hours",
            "remediation": "Notify NY AG and affected NY residents within 72 hours of a confirmed breach.",
        },
    ],

    "VA": [
        {
            "law":      "Virginia Consumer Data Protection Act (VCDPA)",
            "trigger":  lambda c, ctx: ctx.get("processes_va_residents", False),
            "severity": "WARN",
            "detail":   "VCDPA: opt-out of targeted advertising required for 100,000+ VA consumers/year",
            "remediation": "Add opt-out of targeted advertising for Virginia residents in privacy policy.",
        },
    ],

    "CO": [
        {
            "law":      "Colorado Privacy Act (CPA) C.R.S. §6-1-1301",
            "trigger":  lambda c, ctx: ctx.get("processes_co_residents", False),
            "severity": "WARN",
            "detail":   "CPA: opt-out of profiling for marketing required for 100,000+ CO consumers/year",
            "remediation": "Provide opt-out of profiling/targeted advertising for Colorado residents.",
        },
    ],
}

_STATE_ABBREV_RE  = re.compile(r"\b(CA|FL|IL|TX|NY|VA|CO|WA|MA|NJ|OH|PA|MI|GA|NC|AZ)\b")
_STATE_NAME_MAP   = {
    "california": "CA", "florida": "FL", "illinois": "IL", "texas": "TX",
    "new york": "NY", "virginia": "VA", "colorado": "CO",
    "washington": "WA", "massachusetts": "MA", "new jersey": "NJ",
}


def _extract_states(content: str) -> set[str]:
    states = set(_STATE_ABBREV_RE.findall(content.upper()))
    for name, abbrev in _STATE_NAME_MAP.items():
        if name in content.lower():
            states.add(abbrev)
    return states


def check_state_laws(
    content:       str,
    context:       dict | None = None,
    target_states: list[str] | None = None,
) -> list[StateComplianceFlag]:
    context = context or {}
    states  = set(s.upper() for s in target_states) if target_states else _extract_states(content)
    flags: list[StateComplianceFlag] = []

    for state in states:
        for rule in _STATE_RULES.get(state, []):
            try:
                triggered = rule["trigger"](content, context)
            except Exception:
                triggered = False
            if triggered:
                flags.append(StateComplianceFlag(
                    state       = state,
                    law         = rule["law"],
                    severity    = rule["severity"],
                    detail      = rule["detail"],
                    remediation = rule["remediation"],
                ))

    return flags
