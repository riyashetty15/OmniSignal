"""
compliance/tcpa.py
==================
Telephone Consumer Protection Act (TCPA) compliance rules.

Relevant to a fiber telecom ISP because:
  - Automated or pre-recorded marketing calls/texts require prior express written consent
  - Using an ATDS (automatic telephone dialing system) without consent is $500-$1,500/call
  - Do Not Call (DNC) list applies to residential landlines and mobile numbers
  - TCPA is the most-litigated telecom marketing law; violations are class-action targets

This module checks marketing copy, campaign designs, and outreach strategies
for TCPA compliance issues before content is approved or campaigns are launched.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class TCPAFlag:
    rule:        str
    severity:    str          # "BLOCK" | "CRITICAL" | "WARN"
    detail:      str
    remediation: str


# ── Consent trigger patterns ───────────────────────────────────────────────────

_AUTOMATED_OUTREACH_PATTERNS = [
    (re.compile(r"\b(automated?\s+(text|sms|call|message|dialer))\b", re.I),
     "Automated outreach reference detected"),
    (re.compile(r"\b(robocall|robo.call|auto.dialer|autodialer)\b", re.I),
     "ATDS reference detected"),
    (re.compile(r"\b(blast\s+(text|sms|email|call))\b", re.I),
     "'Blast' messaging implies automated distribution"),
    (re.compile(r"\b(mass\s+(text|sms|outreach))\b", re.I),
     "Mass text/SMS campaigns require individual consent under TCPA"),
]

_CONSENT_LANGUAGE_REQUIRED = [
    "reply stop to opt out",
    "opt out",
    "unsubscribe",
    "to stop receiving",
    "msg & data rates may apply",
    "message frequency varies",
]

_DNC_RISK_PHRASES = [
    (re.compile(r"\b(cold\s+call|cold-call)\b", re.I),
     "Cold calling residential numbers without prior consent violates TCPA"),
    (re.compile(r"\b(unsolicited\s+(call|text|sms))\b", re.I),
     "Unsolicited calls/texts to non-consenting parties violate TCPA"),
    (re.compile(r"\bpurchased\s+(list|leads?|data)\b", re.I),
     "Purchased contact lists do not carry TCPA consent"),
]


def check_tcpa(content: str, outreach_type: str = "general") -> list[TCPAFlag]:
    """
    Checks marketing content or campaign descriptions for TCPA compliance issues.
    outreach_type: "sms" | "call" | "email" | "general"
    """
    flags: list[TCPAFlag] = []
    c = content.lower()

    for pattern, description in _AUTOMATED_OUTREACH_PATTERNS:
        if pattern.search(content):
            has_consent = any(phrase in c for phrase in _CONSENT_LANGUAGE_REQUIRED)
            if not has_consent:
                flags.append(TCPAFlag(
                    rule       = "TCPA §227(b) — Automated Outreach Consent",
                    severity   = "CRITICAL",
                    detail     = f"{description} but no opt-out/consent language found",
                    remediation = (
                        "Add explicit opt-out instructions (e.g., 'Reply STOP to opt out') "
                        "and confirm prior express written consent exists for this list."
                    ),
                ))

    for pattern, description in _DNC_RISK_PHRASES:
        if pattern.search(content):
            flags.append(TCPAFlag(
                rule       = "TCPA §227(c) — Do Not Call Registry",
                severity   = "CRITICAL",
                detail     = description,
                remediation = (
                    "Scrub contact list against National DNC Registry before outreach. "
                    "Residential numbers on DNC cannot be called for marketing without written consent."
                ),
            ))

    if outreach_type == "sms":
        has_freq = "message frequency" in c or "msg frequency" in c
        has_rate = "msg & data rates" in c or "message and data rates" in c
        if not has_freq:
            flags.append(TCPAFlag(
                rule       = "TCPA SMS Best Practices — Frequency Disclosure",
                severity   = "WARN",
                detail     = "SMS campaign missing 'Message frequency varies' disclosure",
                remediation = "Add: 'Msg frequency varies. Msg & data rates may apply. Reply STOP to cancel.'",
            ))
        if not has_rate:
            flags.append(TCPAFlag(
                rule       = "TCPA SMS Best Practices — Data Rate Disclosure",
                severity   = "WARN",
                detail     = "SMS campaign missing 'Msg & data rates may apply' disclosure",
                remediation = "Add standard carrier disclosure to all SMS templates.",
            ))

    return flags


def requires_written_consent(campaign_type: str) -> bool:
    """Returns True if this campaign type requires prior express WRITTEN consent."""
    written_consent_required = {
        "sms_marketing", "text_marketing", "automated_call_marketing",
        "robocall_marketing", "email_marketing",
    }
    return campaign_type.lower().replace(" ", "_") in written_consent_required
