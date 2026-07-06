"""
compliance/integration.py
==========================
Hooks the ComplianceEngine into the agent system.

Provides:
  compliance_gate()   — called by the Strategist agent before delivering any copy
  pre_publish_check() — called by the LLM server before any content leaves the system
  audit_log_flag()    — persists compliance events to SQLite for audit trails
"""

from __future__ import annotations
import json
from datetime import datetime, timezone

from compliance.engine import ComplianceEngine, ComplianceReport

_engine = ComplianceEngine()


def compliance_gate(
    content:   str,
    operation: str = "social_post",
    context:   dict | None = None,
) -> dict:
    """
    Thin wrapper used directly as an agent tool.
    Returns a serialisable dict that the LLM can read and act on.
    """
    report = _engine.check(content=content, operation=operation, context=context)
    return report.to_dict()


def pre_publish_check(
    content:       str,
    channel:       str,
    target_states: list[str] | None = None,
) -> ComplianceReport:
    """
    Final gate before content is sent to a social media API, email ESP, or SMS platform.
    Raises ValueError if GO-NO-GO is "NO-GO".
    """
    ctx = {
        "is_email":     channel == "email",
        "outreach_type": "sms" if channel in ("sms", "text") else channel,
        "target_states": target_states or [],
    }
    report = _engine.check(content=content, operation=channel, context=ctx)

    if report.go_no_go == "NO-GO":
        raise ValueError(
            f"Content blocked by compliance engine: {report.summary}\n"
            f"Flags: {json.dumps([f.__dict__ for f in report.all_flags], default=str)}"
        )
    return report


async def audit_log_flag(
    db_path: str,
    user_id: str,
    report:  ComplianceReport,
) -> None:
    """
    Persists a compliance event to the SQLite audit log.
    Table: compliance_audit (created lazily on first write).
    """
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS compliance_audit (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT    NOT NULL,
                operation       TEXT    NOT NULL,
                go_no_go        TEXT    NOT NULL,
                requires_legal  INTEGER NOT NULL,
                total_flags     INTEGER NOT NULL,
                summary         TEXT    NOT NULL,
                content_snippet TEXT    NOT NULL,
                flags_json      TEXT    NOT NULL,
                created_at      TEXT    NOT NULL
            )
        """)
        await db.execute(
            """INSERT INTO compliance_audit
               (user_id, operation, go_no_go, requires_legal, total_flags,
                summary, content_snippet, flags_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                user_id,
                report.operation,
                report.go_no_go,
                int(report.requires_legal),
                len(report.all_flags),
                report.summary,
                report.content_snippet,
                json.dumps(report.to_dict(), default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()
