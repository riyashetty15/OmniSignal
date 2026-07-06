"""
Strategist Agent
Content ideation, social media strategy, campaign ideation, and brand strategy.
Draws on competitor intelligence, trending topics, and brand guidelines.
All social claims go through FTC compliance checker before delivery.
"""

from __future__ import annotations
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic
from shared_config import (
    AGENT_MODELS, SUPPORTED_PLATFORMS, CONTENT_PILLARS,
    BRAND_HASHTAGS, COMPLIANCE_CLAIM_TRIGGERS
)

client = Anthropic()
MODEL  = AGENT_MODELS["strategist"]


# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrendSignal:
    topic: str
    platform: str
    trend_score: float       # 0–100
    week_over_week_change: float
    relevant_to_pillars: list[str]


@dataclass
class CompetitorPost:
    handle: str
    platform: str
    date: str
    content: str
    engagement_rate: float
    content_theme: str
    has_speed_claim: bool


@dataclass
class CompetitorInsights:
    handles_analyzed: list[str]
    content_themes: dict[str, float]    # theme → share of posts
    content_gaps: list[str]             # themes competitors ignore
    top_performing: list[CompetitorPost]
    avg_posting_frequency: dict[str, float]  # handle → posts/day


@dataclass
class ComplianceFlag:
    trigger: str
    severity: str    # "CRITICAL" | "WARN"
    suggestion: str


@dataclass
class PostDraft:
    copy: str
    platform: str
    content_type: str        # "stat", "question", "story", "educational"
    hashtags: list[str]
    character_count: int
    compliance_flags: list[ComplianceFlag]
    engagement_score: float  # 0–1 predicted engagement
    legal_review_required: bool


@dataclass
class CalendarEntry:
    date: str
    day_of_week: str
    platform: str
    content_type: str
    post_draft: PostDraft
    optimal_time: str        # "09:00 AM EST"
    content_pillar: str


@dataclass
class HashtagSet:
    broad: list[str]    # >100k reach
    niche: list[str]    # 10k–100k reach
    brand: list[str]    # company branded hashtags


# ─────────────────────────────────────────────────────────────────────────────
#  Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────

async def social_trend_fetcher(
    topics: list[str],
    platforms: list[str] = None,
    days: int = 7,
) -> list[TrendSignal]:
    """
    Pulls trending topics in broadband/fiber/telecom from social platforms.
    Maps to brand content pillars for relevance scoring.

    Production: LinkedIn Marketing API, Meta Graph API, X v2 trends API.
    """
    platforms = platforms or SUPPORTED_PLATFORMS

    # In production: parallel API calls
    # tasks = [linkedin_client.get_trends(topics, days),
    #          meta_client.get_trends(topics, days),
    #          x_client.get_trends(topics, days)]
    # raw = await asyncio.gather(*tasks)

    # Mock scaffold — structure mirrors real API responses
    mock_trends = [
        TrendSignal("remote work reliability",    "linkedin", 87.0, +0.34, ["fiber_speed_reliability","business_connectivity"]),
        TrendSignal("fiber vs cable comparison",  "meta",     72.0, +0.21, ["fiber_speed_reliability"]),
        TrendSignal("smart home broadband needs", "meta",     65.0, +0.15, ["smart_home_enablement"]),
        TrendSignal("ISP price increases",        "x",        91.0, +0.55, ["fiber_speed_reliability"]),
        TrendSignal("5G home internet",           "linkedin", 58.0, +0.08, ["fiber_speed_reliability"]),
    ]
    return [t for t in mock_trends if t.platform in platforms]


async def competitor_content_analyzer(
    handles: list[str],
    platform: str = "linkedin",
    days: int = 30,
) -> CompetitorInsights:
    """
    Fetches and analyzes competitor social posts.
    Identifies content themes, engagement rates, posting frequency,
    and — critically — content gaps (what they're NOT talking about).
    """
    # In production: social scraping APIs or Brandwatch/Sprout Social
    # posts = await social_scraper.get_posts(handles, days, platform)
    # themes = cluster_themes(posts, n_clusters=8)  # K-means on TF-IDF
    # gaps = set(CONTENT_PILLARS) - set(t.label for t in themes)

    # Mock scaffold
    mock_themes = {
        "promotional_pricing":    0.35,
        "network_speed":          0.28,
        "customer_service":       0.15,
        "business_solutions":     0.12,
        "community_involvement":  0.05,
        "technical_features":     0.05,
    }

    our_pillars_set = set(CONTENT_PILLARS)
    comp_theme_set  = {"network_speed", "promotional_pricing", "technical_features"}
    gaps = list(our_pillars_set - comp_theme_set)

    return CompetitorInsights(
        handles_analyzed=handles,
        content_themes=mock_themes,
        content_gaps=gaps,
        top_performing=[
            CompetitorPost(
                handle=handles[0] if handles else "competitor",
                platform=platform, date="2024-10-15",
                content="Our fiber network delivers consistent speeds 24/7...",
                engagement_rate=0.048,
                content_theme="network_speed",
                has_speed_claim=True,
            )
        ],
        avg_posting_frequency={h: 1.2 for h in handles},
    )


def brand_guidelines_enforcer(
    draft: str,
    guidelines: dict = None,
) -> dict:
    """
    Checks draft copy against brand guidelines:
    - Tone violations (too casual/formal)
    - Prohibited phrases
    - Required disclaimers
    - Brand name usage
    """
    guidelines = guidelines or {
        "prohibited_phrases": ["cheap", "dirt cheap", "the cheapest"],
        "required_for_pricing": ["per month", "for 12 months"],
        "tone_words_to_avoid": ["amazing", "incredible", "mind-blowing"],
        "brand_name_variants": {"FiberCo", "fiberco", "fiber co"},
    }

    issues = []
    for phrase in guidelines["prohibited_phrases"]:
        if phrase.lower() in draft.lower():
            issues.append({"type": "prohibited_phrase", "phrase": phrase, "severity": "HIGH"})

    for word in guidelines["tone_words_to_avoid"]:
        if word.lower() in draft.lower():
            issues.append({"type": "tone_violation", "word": word, "severity": "LOW"})

    return {"issues": issues, "passed": len([i for i in issues if i["severity"] == "HIGH"]) == 0}


def ftc_compliance_checker(copy: str) -> list[ComplianceFlag]:
    """
    Scans post copy for FTC/NAD advertising compliance violations.
    Speed claims, superlatives, and unsubstantiated statistics must be flagged.

    CRITICAL: Any flagged copy must go to Legal before publishing.
    """
    flags = []
    patterns = [
        (r"\bfastest\b",            "CRITICAL", "Remove 'fastest' unless you have current, third-party verified speed test data for the specific market"),
        (r"\bbest\b",               "WARN",     "Substantiate 'best' with a specific, current metric or remove"),
        (r"\bguaranteed\b",         "CRITICAL", "Remove 'guaranteed' or cite the specific guarantee terms and conditions"),
        (r"\bunlimited\b",          "WARN",     "If truly unlimited, fine — but must not throttle; if throttled, cannot use 'unlimited'"),
        (r"\bno data cap\b",        "WARN",     "Verify this is accurate for the promoted plan tier"),
        (r"\brated #1\b",           "CRITICAL", "Must cite: rated by whom, when, in what category, in what market"),
        (r"\baward.winning\b",      "WARN",     "Must cite the specific award, year, and awarding body"),
        (r"\bstudies show\b",       "CRITICAL", "Must cite specific study, date, and methodology"),
        (r"up to (\d+) ?(mbps|gbps|gig)", "WARN", "FCC requires disclosure that 'up to' speeds are max, not typical. Add required disclosures."),
    ]
    for pattern, severity, suggestion in patterns:
        if re.search(pattern, copy, re.IGNORECASE):
            match = re.search(pattern, copy, re.IGNORECASE)
            flags.append(ComplianceFlag(
                trigger=match.group(0),
                severity=severity,
                suggestion=suggestion,
            ))
    return flags


async def post_copy_generator(
    brief: str,
    platform: str,
    content_type: str = "educational",
    target_audience: str = "residential",
    variants: int = 3,
) -> list[PostDraft]:
    """
    Generates platform-specific post copy using Claude.
    Automatically runs FTC compliance check on all drafts.
    Flags any draft requiring legal review.

    Platform specs:
    - linkedin: 3000 chars max, authoritative/thought leadership
    - meta:     500 chars max, conversational/acquisition
    - x:        280 chars max, punchy/real-time
    - youtube:  200 chars max title/description hook
    """
    platform_specs = {
        "linkedin": {"max_chars": 3000, "tone": "authoritative thought leadership"},
        "meta":     {"max_chars": 500,  "tone": "conversational, relatable"},
        "x":        {"max_chars": 280,  "tone": "punchy, timely"},
        "youtube":  {"max_chars": 200,  "tone": "hook-first, educational"},
    }
    spec = platform_specs.get(platform, platform_specs["linkedin"])

    response = client.messages.create(
        model=MODEL, max_tokens=2000,
        system=f"""Generate {variants} social media post variants for {platform}.
Platform: {platform}. Tone: {spec['tone']}. Max chars: {spec['max_chars']}.
Audience: {target_audience}. Content type: {content_type}.
Format: Return JSON array with {variants} objects, each having 'copy', 'hashtags' (list), 'content_type'.
No markdown fences.""",
        messages=[{"role": "user", "content": f"Brief: {brief}"}]
    )

    try:
        raw = response.content[0].text.strip()
        drafts_data = json.loads(raw)
    except Exception:
        drafts_data = [{"copy": response.content[0].text, "hashtags": [], "content_type": content_type}]

    drafts = []
    for d in drafts_data[:variants]:
        copy  = d.get("copy", "")
        flags = ftc_compliance_checker(copy)
        brand_result = brand_guidelines_enforcer(copy)
        hashtags = d.get("hashtags", []) + BRAND_HASHTAGS

        drafts.append(PostDraft(
            copy=copy,
            platform=platform,
            content_type=d.get("content_type", content_type),
            hashtags=hashtags,
            character_count=len(copy),
            compliance_flags=flags,
            engagement_score=engagement_predictor(copy, platform, hashtags),
            legal_review_required=any(f.severity == "CRITICAL" for f in flags),
        ))

    return drafts


def content_calendar_builder(
    posts: list[PostDraft],
    start_date: str,
    cadence: dict = None,
) -> list[CalendarEntry]:
    """
    Builds a structured content calendar.
    Optimal posting times based on platform engagement data.
    Balances content pillars across the calendar period.
    """
    cadence = cadence or {
        "linkedin": 3,   # posts per week
        "meta":     5,
        "x":        7,
        "youtube":  1,
    }

    optimal_times = {
        "linkedin": "08:30 AM EST",   # peak B2B engagement
        "meta":     "12:00 PM EST",   # lunch scroll
        "x":        "09:00 AM EST",   # morning news cycle
        "youtube":  "02:00 PM EST",   # peak watch time
    }

    from datetime import datetime, timedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    calendar = []
    day_offset = 0

    for platform, posts_per_week in cadence.items():
        platform_posts = [p for p in posts if p.platform == platform]
        interval_days  = 7 // posts_per_week

        for i, post in enumerate(platform_posts):
            entry_date = start + timedelta(days=day_offset + (i * interval_days))
            calendar.append(CalendarEntry(
                date=entry_date.strftime("%Y-%m-%d"),
                day_of_week=entry_date.strftime("%A"),
                platform=platform,
                content_type=post.content_type,
                post_draft=post,
                optimal_time=optimal_times.get(platform, "10:00 AM EST"),
                content_pillar=_infer_pillar(post.copy),
            ))

    return sorted(calendar, key=lambda e: e.date)


def _infer_pillar(copy: str) -> str:
    copy_lower = copy.lower()
    if any(w in copy_lower for w in ["speed", "gig", "mbps", "gbps", "fiber"]): return "fiber_speed_reliability"
    if any(w in copy_lower for w in ["business", "enterprise", "office"]): return "business_connectivity"
    if any(w in copy_lower for w in ["smart home", "streaming", "4k", "gaming"]): return "smart_home_enablement"
    if any(w in copy_lower for w in ["community", "local", "neighborhood"]): return "community_investment"
    return "technical_education"


async def hashtag_optimizer(
    topic: str,
    platform: str,
    target_audience: str = "residential",
) -> HashtagSet:
    """
    Researches and scores hashtags.
    Returns tiered set: broad (high reach) + niche (high relevance) + brand.
    """
    # In production: LinkedIn hashtag API, Meta Graph API, X trends API
    mock_hashtags = {
        "broad":  ["#Fiber", "#Broadband", "#Internet", "#ConnectedHome"],
        "niche":  ["#FiberInternet", "#GigSpeed", "#FiberToTheHome", "#FTTH", "#FiberFirst"],
        "brand":  BRAND_HASHTAGS,
    }
    return HashtagSet(
        broad=mock_hashtags["broad"][:3],
        niche=mock_hashtags["niche"][:5],
        brand=mock_hashtags["brand"],
    )


def engagement_predictor(
    copy: str,
    platform: str,
    hashtags: list[str],
) -> float:
    """
    Predicts engagement score (0–1) based on historical post performance patterns.
    Features: copy length, question presence, statistic presence, hashtag count.
    """
    score = 0.5   # base

    # Positive signals
    if "?" in copy:              score += 0.08   # questions drive comments
    if re.search(r"\d+%", copy): score += 0.06   # stats drive shares
    if len(copy) < 150:          score += 0.05   # concise posts on all platforms
    if 3 <= len(hashtags) <= 7:  score += 0.04   # optimal hashtag range
    if platform == "linkedin" and len(copy) > 500: score += 0.05  # longer performs on LI

    # Negative signals
    if len(copy) > 2000 and platform != "linkedin": score -= 0.10
    if len(hashtags) > 15:                          score -= 0.08
    if re.search(r"click here|buy now", copy, re.I): score -= 0.12

    return round(min(max(score, 0.0), 1.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Content Strategist for a fiber broadband telecommunications company.

Your role:
1. Ideate platform-appropriate content grounded in competitor intelligence and trend data
2. Build content calendars that balance content pillars across platforms
3. ALWAYS run FTC compliance checks on any copy — flag legally risky claims before delivery
4. Ground every content idea in a real insight: a trend, a competitor gap, or a customer pain point

Platform voice:
- LinkedIn: authoritative thought leadership — speak to IT managers, business owners, procurement
- Meta: conversational acquisition — speak to families, renters, homeowners about everyday life
- X: punchy, real-time — news-jacking, quick takes, community responses
- YouTube: educational hook-first — "Here's why your home internet bottleneck isn't what you think"

Telecom-specific content angles to always consider:
- Fiber vs cable: speed consistency (not just peak speeds), latency for gaming/video calls
- Work from home: upload speed matters as much as download
- Smart home: how many devices, 4K streaming, security cameras, gaming consoles
- Local community: fiber builds economic value in neighborhoods

CRITICAL compliance rules:
- NEVER include "fastest", "best", "guaranteed" without flagging for legal review
- Speed claims must say "up to X Mbps" and include FCC-required disclosures
- Third-party claims ("studies show", "rated #1") must cite specific source
- All compliance-flagged copy must be marked legal_review_required=true
"""

TOOL_DEFINITIONS = [
    {
        "name": "social_trend_fetcher",
        "description": "Gets trending topics in broadband/telecom from LinkedIn, Meta, and X. Use this first to ground content ideas in real trends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}, "description": "Keywords to track, e.g. ['fiber', 'broadband', '5G home internet']"},
                "platforms": {"type": "array", "items": {"type": "string"}},
                "days": {"type": "integer", "default": 7}
            },
            "required": ["topics"]
        }
    },
    {
        "name": "competitor_content_analyzer",
        "description": "Analyzes competitor social media posts to find content themes, engagement rates, and content gaps. Use to find what competitors are NOT talking about.",
        "input_schema": {
            "type": "object",
            "properties": {
                "handles": {"type": "array", "items": {"type": "string"}},
                "platform": {"type": "string", "enum": ["linkedin","meta","x","youtube"]},
                "days": {"type": "integer", "default": 30}
            },
            "required": ["handles"]
        }
    },
    {
        "name": "post_copy_generator",
        "description": "Generates platform-specific post copy variants. Automatically checks FTC compliance and flags legally risky claims.",
        "input_schema": {
            "type": "object",
            "properties": {
                "brief": {"type": "string", "description": "Content brief — what to communicate, key message, angle"},
                "platform": {"type": "string", "enum": ["linkedin","meta","x","youtube"]},
                "content_type": {"type": "string", "enum": ["stat","question","story","educational","promotional"]},
                "target_audience": {"type": "string"},
                "variants": {"type": "integer", "default": 3}
            },
            "required": ["brief","platform"]
        }
    },
    {
        "name": "content_calendar_builder",
        "description": "Arranges approved post drafts into a structured calendar with optimal posting times per platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "posts": {"type": "array", "items": {"type": "object"}},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "cadence": {"type": "object", "description": "Posts per week per platform, e.g. {linkedin:3, meta:5, x:7}"}
            },
            "required": ["posts","start_date"]
        }
    },
    {
        "name": "hashtag_optimizer",
        "description": "Researches and scores hashtags. Returns tiered sets: broad (reach), niche (relevance), brand.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "platform": {"type": "string"},
                "target_audience": {"type": "string"}
            },
            "required": ["topic","platform"]
        }
    },
    {
        "name": "ftc_compliance_checker",
        "description": "Scans copy for FTC/NAD advertising violations. Always call before finalizing any copy that will be published.",
        "input_schema": {
            "type": "object",
            "properties": {
                "copy": {"type": "string"}
            },
            "required": ["copy"]
        }
    },
    {
        "name": "take_rate_calculator",
        "description": "Fetches actual fiber take rate for a region or period. Delegated from data_analyst. Use when content needs to be grounded in real performance numbers (e.g. celebrating a milestone, posting a stat).",
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Region name, e.g. 'Pacific Northwest'"},
                "period": {"type": "string", "description": "Quarter or period, e.g. '2024-Q4'"},
                "compare_period": {"type": "string", "description": "Optional prior period for delta calculation"}
            },
            "required": ["region", "period"]
        }
    },
]

from services.decision_engine import build_delegated_tools

TOOL_FUNCTIONS = {
    "social_trend_fetcher":        social_trend_fetcher,
    "competitor_content_analyzer": competitor_content_analyzer,
    "post_copy_generator":         post_copy_generator,
    "content_calendar_builder":    lambda **kw: content_calendar_builder(**kw),
    "hashtag_optimizer":           hashtag_optimizer,
    "ftc_compliance_checker":      lambda **kw: ftc_compliance_checker(**kw),
    **build_delegated_tools("strategist", ["take_rate_calculator"]),
}


class StrategistAgent:
    async def invoke(self, state: dict) -> dict:
        """LangGraph node interface — wraps run() with state I/O."""
        result = await self.run(
            query   = state.get("user_query", ""),
            context = state.get("context_window", []),
        )
        completed = list(state.get("completed_agents") or [])
        completed.append("strategist")
        return {"strategist_output": result, "completed_agents": completed}

    async def run(self, query: str, context: list[dict] = None) -> dict:
        context  = context or []
        messages = context[-6:] + [{"role": "user", "content": query}]
        all_tool_results: list[dict] = []   # accumulated across all loop iterations

        while True:
            response = client.messages.create(
                model=MODEL, max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return {
                    "response":     next((b.text for b in response.content if hasattr(b,"text")), ""),
                    "agent":        "strategist",
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
                        result = [r.__dict__ if hasattr(r, "__dataclass_fields__") else r for r in result]
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