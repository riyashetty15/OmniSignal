"""
HR Document QA Agent
ISOLATED model instance — accesses ONLY the hr_docs vector store.
No marketing, financial, or operational data is accessible.
Handles: policy lookups, benefits, onboarding, leave, compliance training.

Model: claude-3-haiku (cost-optimized, faster for simple policy lookups)
"""

from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from shared_config import AGENT_MODELS

client = Anthropic()
MODEL  = AGENT_MODELS["hr_docqa"]   # haiku — isolated, cheaper


# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HRCitation:
    doc_id: str
    doc_title: str
    section: str
    page: Optional[int]
    effective_date: str
    version: str
    relevance_score: float
    excerpt: str


@dataclass
class PolicyMeta:
    policy_id: str
    title: str
    version: str
    effective_date: str
    next_review_date: str
    approving_authority: str
    doc_type: str
    department: str


@dataclass
class ComplianceResult:
    requires_escalation: bool
    flags: list[str]
    escalation_reason: Optional[str]
    suggested_routing: str   # "hr_manager" | "legal" | "payroll" | "it"


# ─────────────────────────────────────────────────────────────────────────────
#  PII Detection and Masking
# ─────────────────────────────────────────────────────────────────────────────

PII_PATTERNS = [
    (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",                    "SSN"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  "EMAIL"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",                    "PHONE"),
    (r"\bEMP[-#]?\d{4,8}\b",                                    "EMPLOYEE_ID"),
    (r"\$\d[\d,]+\b",                                           "SALARY_AMOUNT"),
]

def pii_masker(text: str) -> tuple[str, list[str]]:
    """
    Detects and redacts employee PII before any processing.
    GDPR/CCPA Layer 1 compliance requirement.
    Returns: (masked_text, list_of_pii_types_found)
    """
    found_types = []
    masked = text
    for pattern, label in PII_PATTERNS:
        matches = re.findall(pattern, masked, re.IGNORECASE)
        if matches:
            masked = re.sub(pattern, f"[{label}-REDACTED]", masked, flags=re.IGNORECASE)
            found_types.append(label)
    return masked, found_types


# ─────────────────────────────────────────────────────────────────────────────
#  HR Compliance Checker
# ─────────────────────────────────────────────────────────────────────────────

SENSITIVE_TOPICS = {
    "termination":      ("hr_manager", "Employment termination queries require HR manager involvement"),
    "dismissal":        ("hr_manager", "Employment dismissal queries require HR manager involvement"),
    "salary":           ("hr_manager", "Compensation queries require HR manager — cannot discuss specific figures"),
    "compensation":     ("hr_manager", "Compensation queries require HR manager"),
    "disciplinary":     ("hr_manager", "Disciplinary action requires HR manager and possibly legal review"),
    "investigation":    ("legal",      "Investigation-related queries require Legal and HR involvement"),
    "harassment":       ("legal",      "Harassment queries require immediate Legal and HR escalation"),
    "discrimination":   ("legal",      "Discrimination queries require Legal and HR escalation"),
    "medical leave":    ("hr_manager", "Medical leave details may involve ADA/FMLA — HR manager review required"),
    "performance improvement": ("hr_manager", "PIP queries require HR manager involvement"),
    "lawsuit":          ("legal",      "Any litigation reference requires Legal department"),
    "union":            ("legal",      "Union-related queries require Legal review"),
}

def hr_compliance_checker(query: str) -> ComplianceResult:
    """
    Flags sensitive HR topics that require escalation.
    Prevents AI from answering questions that need a human HR professional.
    """
    query_lower = query.lower()
    flags = []
    routing = "self"

    for topic, (route, reason) in SENSITIVE_TOPICS.items():
        if topic in query_lower:
            flags.append(topic)
            if routing == "self" or (route == "legal" and routing == "hr_manager"):
                routing = route

    if flags:
        return ComplianceResult(
            requires_escalation=True,
            flags=flags,
            escalation_reason=SENSITIVE_TOPICS[flags[0]][1],
            suggested_routing=routing,
        )

    return ComplianceResult(
        requires_escalation=False,
        flags=[], escalation_reason=None, suggested_routing="self",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Vector Search (HR-Isolated)
# ─────────────────────────────────────────────────────────────────────────────

async def hr_vector_search(
    query: str,
    filters: dict = None,
    pgvector_client=None,
    top_k: int = 5,
) -> list[HRCitation]:
    """
    Semantic search over the hr_docs PGVector table ONLY.
    Module-siloed — physically different table from all marketing stores.
    RBAC enforced at DB level: this connection string has read access to hr_docs only.
    """
    filters = filters or {}

    # In production:
    # results = await pgvector_client.search(
    #     query=query,
    #     table="hr_docs",          # isolated table — no access to campaign_docs etc.
    #     filters=filters,          # doc_type, effective_date, department, version
    #     top_k=top_k,
    #     similarity_threshold=0.72,
    #     conn_string=HR_DB_CONN    # separate connection with hr_docs ONLY access
    # )

    # Mock scaffold
    return [
        HRCitation(
            doc_id="hr_policy_remote_work_v2_3",
            doc_title="Remote Work & Flexible Arrangements Policy",
            section="4.2 — Home Office Equipment Reimbursement",
            page=8,
            effective_date="2024-01-15",
            version="2.3",
            relevance_score=0.92,
            excerpt="Employees working remotely are eligible for a home office equipment reimbursement of up to $500 per calendar year. Eligible items include monitors, keyboards, webcams, and ergonomic accessories. Reimbursement requires manager approval and receipts submitted through the expense portal within 90 days of purchase.",
        )
    ]


async def hr_policy_lookup(
    policy_id: str = None,
    topic: str = None,
    db_client=None,
) -> list[PolicyMeta]:
    """
    Structured lookup for HR policies by ID or topic.
    Returns metadata: version, effective date, approving authority, next review date.
    """
    # In production: query HR policy registry
    # if policy_id:
    #     return await hr_db.get_policy_by_id(policy_id)
    # return await hr_db.search_policies(topic)

    return [
        PolicyMeta(
            policy_id="POL-2024-RW-001",
            title="Remote Work & Flexible Arrangements Policy",
            version="2.3",
            effective_date="2024-01-15",
            next_review_date="2025-01-15",
            approving_authority="Chief People Officer",
            doc_type="HR Policy",
            department="People & Culture",
        )
    ]


def doc_citation_formatter(
    answer: str,
    citations: list[HRCitation],
) -> str:
    """
    Formats the LLM's answer with exact policy citations.
    Ensures every factual claim is traceable to a specific document version.
    """
    if not citations:
        return answer + "\n\n⚠️ Note: No specific policy document was found for this query."

    citation_block = "\n\n---\n**Sources:**\n"
    for c in citations:
        citation_block += (
            f"• {c.doc_title}, {c.section}\n"
            f"  Version {c.version} | Effective: {c.effective_date} | Page {c.page or 'N/A'}\n"
        )
    return answer + citation_block


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the company HR assistant. You answer employee questions about company policies.

STRICT RULES:
1. Answer ONLY from the policy documents provided in the tool results
2. ALWAYS cite: policy name, section number, effective date, version number
3. If the answer is not in the retrieved documents, say explicitly: "I could not find this in the current HR policies. Please contact HR directly."
4. NEVER infer, generalize, or extrapolate beyond what the documents say
5. NEVER provide legal advice — always recommend HR or Legal for complex situations
6. If a compliance checker flags the query for escalation, provide the routing guidance and stop

Tone: Professional, clear, empathetic. These are real employee questions that affect people's lives.
Format: 
- Lead with the direct answer
- Follow with the policy citation
- End with "For questions not covered here, contact People & Culture at hr@company.com"
"""

TOOL_DEFINITIONS = [
    {
        "name": "pii_masker",
        "description": "Detects and masks employee PII in the query before processing. Always call first.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"]
        }
    },
    {
        "name": "hr_compliance_checker",
        "description": "Checks if query involves sensitive topics requiring HR manager or Legal escalation. Call before searching.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "hr_vector_search",
        "description": "Searches the HR document corpus for relevant policy sections. Returns chunks with policy name, section, and effective date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object", "description": "Optional: {doc_type, department, effective_after}"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "hr_policy_lookup",
        "description": "Structured lookup for a specific policy by topic or ID. Returns metadata including version, effective date, and approving authority.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "policy_id": {"type": "string"}
            }
        }
    },
]

TOOL_FUNCTIONS = {
    "pii_masker":           lambda **kw: pii_masker(**kw),
    "hr_compliance_checker":lambda **kw: hr_compliance_checker(**kw),
    "hr_vector_search":     hr_vector_search,
    "hr_policy_lookup":     hr_policy_lookup,
}


class HRDocQAAgent:
    """
    Isolated HR agent. Runs on claude-3-haiku.
    Has NO access to marketing, financial, or operational data.
    """

    async def invoke(self, state: dict) -> dict:
        """LangGraph node interface — wraps run() with state I/O."""
        result = await self.run(
            query   = state.get("user_query", ""),
            context = state.get("context_window", []),
        )
        completed = list(state.get("completed_agents") or [])
        completed.append("hr_docqa")
        return {"hr_output": result, "completed_agents": completed}

    async def run(self, query: str, context: list[dict] = None) -> dict:
        context  = context or []
        messages = context[-4:] + [{"role": "user", "content": query}]
        all_tool_results: list[dict] = []   # accumulated across all loop iterations

        while True:
            response = client.messages.create(
                model=MODEL, max_tokens=2048,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final = next((b.text for b in response.content if hasattr(b,"text")), "")
                return {
                    "response":     final,
                    "agent":        "hr_docqa",
                    "model":        MODEL,
                    "data_access":  ["hr_docs"],   # audit trail — only hr_docs accessed
                    "tool_calls":   [t["tool_name"] for t in all_tool_results],
                    "tool_results": all_tool_results,
                }

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                fn = TOOL_FUNCTIONS.get(block.name)
                try:
                    result = await fn(**block.input) if asyncio.iscoroutinefunction(fn) else fn(**block.input)
                    if isinstance(result, list):
                        result = [r.__dict__ if hasattr(r,"__dataclass_fields__") else r for r in result]
                    elif hasattr(result, "__dataclass_fields__"):
                        result = result.__dict__
                except Exception as e:
                    result = {"error": str(e)}

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "tool_name":   block.name,
                    "content":     json.dumps(result, default=str),
                })
            all_tool_results.extend(tool_results)
            messages.append({"role": "user", "content": tool_results})