"""
compliance/privacy.py
=====================
GDPR and CCPA/CPRA privacy compliance for marketing data operations.

Why this matters for a fiber ISP:
  - Subscriber data (name, address, IP, usage patterns) is personal data under GDPR
  - California residents' broadband data is protected under CCPA/CPRA
  - Calix demographic data used in targeting must have a lawful processing basis
  - Marketing segmentation must comply with data minimization principles
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum


class LawfulBasis(str, Enum):
    CONSENT             = "consent"
    CONTRACT            = "contract"
    LEGAL_OBLIGATION    = "legal_obligation"
    VITAL_INTERESTS     = "vital_interests"
    PUBLIC_TASK         = "public_task"
    LEGITIMATE_INTEREST = "legitimate_interest"
    NONE                = "none"


@dataclass
class PrivacyRisk:
    regulation:  str
    article:     str
    severity:    str
    detail:      str
    remediation: str


# ── Sensitive data categories ──────────────────────────────────────────────────

_SENSITIVE_PATTERNS = [
    (re.compile(r"\b(racial|ethnic)\s+(origin|group|background)\b", re.I),
     "Racial/ethnic origin — GDPR Art. 9 special category"),
    (re.compile(r"\b(health|medical|disability)\s+(data|information|status)\b", re.I),
     "Health data — GDPR Art. 9 special category"),
    (re.compile(r"\b(biometric)\s+(data|information|identifier)\b", re.I),
     "Biometric data — GDPR Art. 9 special category"),
    (re.compile(r"\b(political\s+opinion|religious\s+belief|trade\s+union)\b", re.I),
     "Political/religious/union data — GDPR Art. 9 special category"),
    (re.compile(r"\b(sexual\s+orientation|gender\s+identity)\b", re.I),
     "Sexual orientation — GDPR Art. 9 / CCPA sensitive PI"),
    (re.compile(r"\b(precise\s+geolocation|exact\s+location|gps\s+tracking)\b", re.I),
     "Precise geolocation — CCPA sensitive PI"),
    (re.compile(r"\b(social\s+security|ssn|tax\s+id|driver.s\s+license)\b", re.I),
     "Government ID — sensitive personal information"),
]


def detect_sensitive_data(content: str) -> list[str]:
    return [label for pattern, label in _SENSITIVE_PATTERNS if pattern.search(content)]


# ── Lawful basis mapping ───────────────────────────────────────────────────────

_MARKETING_LAWFUL_BASIS: dict[str, LawfulBasis] = {
    "email_marketing":        LawfulBasis.CONSENT,
    "sms_marketing":          LawfulBasis.CONSENT,
    "behavioral_targeting":   LawfulBasis.CONSENT,
    "lookalike_modeling":     LawfulBasis.LEGITIMATE_INTEREST,
    "service_communications": LawfulBasis.CONTRACT,
    "calix_demographic_use":  LawfulBasis.LEGITIMATE_INTEREST,
    "analytics_reporting":    LawfulBasis.LEGITIMATE_INTEREST,
    "retention_modeling":     LawfulBasis.LEGITIMATE_INTEREST,
}


def get_required_lawful_basis(operation: str) -> LawfulBasis:
    return _MARKETING_LAWFUL_BASIS.get(operation.lower(), LawfulBasis.LEGITIMATE_INTEREST)


# ── Data retention limits ─────────────────────────────────────────────────────

MAX_RETENTION_DAYS: dict[str, int] = {
    "marketing_consent":       730,
    "campaign_performance":    1095,
    "subscriber_data":         2555,
    "calix_demographic_cache": 90,
    "session_logs":            90,
    "analytics_events":        365,
}


def check_retention(data_type: str, actual_days: int) -> PrivacyRisk | None:
    max_days = MAX_RETENTION_DAYS.get(data_type)
    if max_days and actual_days > max_days:
        return PrivacyRisk(
            regulation  = "GDPR Art. 5(1)(e) / CCPA",
            article     = "Storage Limitation",
            severity    = "HIGH",
            detail      = f"'{data_type}' retained {actual_days} days; max is {max_days}",
            remediation = f"Implement auto-deletion/anonymization at {max_days} days for {data_type}.",
        )
    return None


# ── CCPA rights infrastructure ────────────────────────────────────────────────

def check_ccpa_rights(
    has_deletion_mechanism: bool,
    has_access_mechanism:   bool,
    has_opt_out_of_sale:    bool,
    has_privacy_policy:     bool,
    processes_ca_residents: bool,
) -> list[PrivacyRisk]:
    if not processes_ca_residents:
        return []

    risks = []

    if not has_privacy_policy:
        risks.append(PrivacyRisk(
            regulation  = "CCPA §1798.100",
            article     = "Privacy Policy Required",
            severity    = "BLOCK",
            detail      = "CCPA requires disclosure of data collection practices",
            remediation = "Publish a CCPA-compliant privacy policy.",
        ))

    if not has_deletion_mechanism:
        risks.append(PrivacyRisk(
            regulation  = "CCPA §1798.105",
            article     = "Right to Delete",
            severity    = "HIGH",
            detail      = "No deletion mechanism for California residents' personal information",
            remediation = "Implement a verified consumer deletion request process (respond within 45 days).",
        ))

    if not has_access_mechanism:
        risks.append(PrivacyRisk(
            regulation  = "CCPA §1798.100",
            article     = "Right to Know / Access",
            severity    = "HIGH",
            detail      = "No data access request mechanism for California residents",
            remediation = "Implement a verified access request process with 12-month PI lookback.",
        ))

    if not has_opt_out_of_sale:
        risks.append(PrivacyRisk(
            regulation  = "CCPA §1798.120",
            article     = "Right to Opt Out of Sale/Sharing",
            severity    = "HIGH",
            detail      = "'Do Not Sell or Share My Personal Information' link not found",
            remediation = "Add 'Do Not Sell or Share My Personal Information' link to homepage footer.",
        ))

    return risks


# ── Main checker ───────────────────────────────────────────────────────────────

def check_privacy(
    content:   str,
    operation: str = "general",
    context:   dict | None = None,
) -> list[PrivacyRisk]:
    context = context or {}
    risks: list[PrivacyRisk] = []

    for category in detect_sensitive_data(content):
        risks.append(PrivacyRisk(
            regulation  = "GDPR Art. 9 / CCPA",
            article     = "Special Category / Sensitive PI",
            severity    = "HIGH",
            detail      = f"Sensitive data category detected: {category}",
            remediation = (
                "Sensitive categories require explicit consent (GDPR) or "
                "sensitive PI opt-in rights disclosure (CCPA §1798.121). "
                "Consult Legal before using this data in marketing operations."
            ),
        ))

    if context.get("data_type") and context.get("retention_days"):
        risk = check_retention(context["data_type"], context["retention_days"])
        if risk:
            risks.append(risk)

    return risks
