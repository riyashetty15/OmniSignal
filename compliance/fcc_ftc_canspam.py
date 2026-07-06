"""
compliance/fcc_ftc_canspam.py
==============================
FCC, FTC, and CAN-SPAM compliance rules for telecom marketing.

FCC  — Broadband service claims, speed advertising, and geographic availability.
FTC  — Advertising substantiation (Section 5). Superlatives must be backed by evidence.
CAN-SPAM — Commercial email: opt-out, physical address, honest subject lines.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class ComplianceFlag:
    law:         str
    rule:        str
    severity:    str
    trigger:     str
    detail:      str
    remediation: str


# ── FTC / FCC Advertising Claim Rules ─────────────────────────────────────────

_FTC_RULES: list[tuple] = [
    (
        re.compile(r"\bfastest\b", re.I),
        "FTC §5 / NAD", "CRITICAL",
        "Superlative 'fastest' is an unsubstantiated comparative claim",
        "Must have current, third-party verified speed data (e.g. Ookla) for the specific market. Otherwise remove.",
    ),
    (
        re.compile(r"\b(rated\s+#\s*1|number\s+one|#1\s+rated)\b", re.I),
        "FTC §5 / NAD", "CRITICAL",
        "'Rated #1' requires a named, current, third-party source",
        "Cite: rated by [source], [year], in [category], in [market]. Without this, remove the claim.",
    ),
    (
        re.compile(r"\baward.winning\b", re.I),
        "FTC §5", "WARN",
        "Award claims must identify the specific award, awarding body, and year",
        "Add: 'Winner of [Award Name] from [Organization], [Year]'",
    ),
    (
        re.compile(r"\bguaranteed?\b", re.I),
        "FTC §5", "CRITICAL",
        "'Guaranteed' creates an enforceable consumer promise — must state exact terms",
        "Replace with specific guarantee terms or remove.",
    ),
    (
        re.compile(r"\bstudies?\s+show\b", re.I),
        "FTC §5", "CRITICAL",
        "Third-party study reference without citation is deceptive",
        "Cite the specific study: title, publisher, date, sample size, and methodology.",
    ),
    (
        re.compile(r"\bunlimited\b", re.I),
        "FTC §5 / FCC", "WARN",
        "'Unlimited' cannot be used if the plan throttles or has usage-based restrictions",
        "If any throttling/deprioritization exists, remove 'unlimited' or add a clear disclosure footnote.",
    ),
    (
        re.compile(r"\bno\s+data\s+cap\b", re.I),
        "FCC Broadband Facts", "WARN",
        "'No data cap' must be accurate for the advertised plan tier",
        "Verify this is accurate for the specific plan being advertised.",
    ),
    (
        re.compile(r"\bup\s+to\s+\d+\s*(gbps|mbps|gig)\b", re.I),
        "FCC Broadband Labeling Rule (2024)", "WARN",
        "FCC requires 'typical' speed disclosure alongside 'up to' claims",
        "Add: 'Typical download speed: X Mbps. Actual speeds may vary.' per FCC Broadband Facts Label.",
    ),
    (
        re.compile(r"\bfree\s+installation\b", re.I),
        "FTC §5", "WARN",
        "Promotional fee claims must disclose they are limited-time and state standard pricing",
        "Add: 'For new customers. Standard installation fee is $X. Offer expires [date].'",
    ),
]


def check_ftc_fcc(copy: str) -> list[ComplianceFlag]:
    """Scans copy for FTC/FCC advertising compliance issues."""
    flags = []
    for pattern, law, severity, detail, remediation in _FTC_RULES:
        match = pattern.search(copy)
        if match:
            flags.append(ComplianceFlag(
                law         = law,
                rule        = "Advertising Substantiation",
                severity    = severity,
                trigger     = match.group(0),
                detail      = detail,
                remediation = remediation,
            ))
    return flags


# ── CAN-SPAM Email Compliance ──────────────────────────────────────────────────

_DECEPTIVE_SUBJECT_PATTERNS = [
    (re.compile(r"^re:", re.I),  "Subject line starts with 'Re:' but is not a reply"),
    (re.compile(r"^fwd:", re.I), "Subject line starts with 'Fwd:' but is not a forward"),
]


_OPT_OUT_SIGNALS = [
    "unsubscribe", "opt out", "opt-out", "remove me", "stop receiving",
    "manage preferences", "email preferences",
]

_ADDRESS_SIGNALS = [
    re.compile(r"\b\d{1,5}\s+\w+\s+(st|ave|blvd|rd|dr|ln|way|ct|pl)\b", re.I),
    re.compile(r"\bpo\s+box\s+\d+\b", re.I),
]


def check_canspam(
    subject_line: str,
    body:         str,
    has_opt_out:  bool,
    has_address:  bool,
) -> list[ComplianceFlag]:
    """
    Checks a commercial email for CAN-SPAM Act compliance.
    Augments the caller-supplied has_opt_out / has_address flags by also
    scanning the body text directly, so partial implementations still get caught.
    """
    flags = []
    body_lower = body.lower()

    # Augment caller flags with body scan
    body_has_opt_out = has_opt_out or any(s in body_lower for s in _OPT_OUT_SIGNALS)
    body_has_address = has_address or any(p.search(body) for p in _ADDRESS_SIGNALS)

    for pattern, description in _DECEPTIVE_SUBJECT_PATTERNS:
        if pattern.search(subject_line):
            flags.append(ComplianceFlag(
                law         = "CAN-SPAM §7704(a)(2)",
                rule        = "No Deceptive Subject Lines",
                severity    = "CRITICAL",
                trigger     = subject_line[:80],
                detail      = description,
                remediation = "Rewrite subject line to accurately reflect email content.",
            ))

    if not body_has_opt_out:
        flags.append(ComplianceFlag(
            law         = "CAN-SPAM §7704(a)(3)",
            rule        = "Opt-Out Mechanism Required",
            severity    = "BLOCK",
            trigger     = "(missing opt-out)",
            detail      = "Commercial email must include a clear, conspicuous unsubscribe mechanism",
            remediation = "Add an unsubscribe link. Honor opt-outs within 10 business days.",
        ))

    if not body_has_address:
        flags.append(ComplianceFlag(
            law         = "CAN-SPAM §7704(a)(5)(A)(iii)",
            rule        = "Physical Address Required",
            severity    = "BLOCK",
            trigger     = "(missing address)",
            detail      = "Commercial email must include sender's valid physical postal address",
            remediation = "Add company's physical mailing address to email footer.",
        ))

    return flags
