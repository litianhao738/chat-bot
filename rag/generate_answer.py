#!/usr/bin/env python3
"""
Generate a lightweight structured answer from retrieved Project 8 chunks.

This is a scaffold for objective 4. For now it uses deterministic formatting
so the retrieval-and-answer pipeline is locally runnable even before any API or
LLM integration is added. It can optionally call a local Ollama-served model
to polish the structured answer draft.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.retrieve_chunks import load_pickle, load_topic_catalog, predict_topic, retrieve

DEFAULT_INDEX_PATH = "rag/output/retrieval_index.pkl"
DEFAULT_OUTPUT_PATH = "rag/output/answer.json"
DEFAULT_CLASSIFIER_PATH = "classification/output/inquiry_topic_classifier.pkl"
DEFAULT_TOPIC_CATALOG_PATH = "classification/output/topic_catalog.json"
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))
DEFAULT_OLLAMA_PATH = os.environ.get("OLLAMA_PATH", "")

GENERIC_KEYWORDS = {
    "ai", "agents", "agent", "company", "companies", "business", "platform", "software", "startup",
    "startups", "help", "helps", "team", "teams", "solution", "solutions", "tool", "tools", "data",
    "product", "products", "service", "services", "users", "customer", "customers",
}

THEME_MAP = {
    "patient/provider workflow": {"patient", "patients", "provider", "providers", "care", "clinical", "clinic"},
    "automation and agents": {"agent", "agents", "automation", "workflow", "autonomous", "assistant", "coding"},
    "data and infrastructure": {"api", "data", "ehr", "records", "integration", "connector", "infrastructure"},
    "compliance and risk": {"compliance", "risk", "insurance", "security", "regulation", "monitoring"},
    "marketplace and commerce": {"marketplace", "commerce", "retail", "brands", "sell", "stores"},
    "education delivery": {"education", "students", "learning", "school", "teachers", "student"},
}

LOCATION_TERMS = {
    "hong kong": {"hong kong", "hk", "hkg"},
}

COMPLIANCE_TERMS = {
    "compliance", "regulation", "regulatory", "legal", "license", "licensing", "privacy", "insurance",
    "security", "risk", "governance",
}

MARKET_TERMS = {
    "market", "customer", "customers", "buyer", "buyers", "demand", "segment", "adoption", "distribution",
}
OPERATIONS_TERMS = {
    "operations", "inventory", "supply", "chain", "warehouse", "fulfillment", "vendor", "vendors", "staffing",
    "pricing", "competition", "location",
}
FEEDBACK_TERMS = {
    "complaint", "complaints", "complain", "complaining", "feedback", "review", "reviews", "rating", "ratings",
    "comment", "comments", "sentiment", "reaction", "reactions", "social media", "online reviews",
    "customer reviews", "customer feedback", "public reaction",
}
FEEDBACK_THEME_MAP = {
    "delivery and fulfillment": {
        "delivery", "fulfillment", "shipping", "ship", "warehouse", "logistics", "lead time", "shipping volatility",
    },
    "inventory and product availability": {
        "inventory", "stockout", "stockouts", "availability", "replenishment", "forecasting", "supplier", "suppliers",
    },
    "pricing clarity and value": {
        "pricing", "price", "prices", "margin", "discount", "discounts", "profitability", "cost", "costs",
    },
    "customer service and experience": {
        "customer experience", "service", "service quality", "support", "personalization", "customer expectations", "checkout",
    },
    "returns and post-purchase handling": {
        "return", "returns", "refund", "refunds", "exchange", "exchanges",
    },
}
DEFAULT_FEEDBACK_THEMES = [
    "delivery and fulfillment",
    "inventory and product availability",
    "customer service and experience",
    "pricing clarity and value",
]
FEEDBACK_THEME_PRIORITY = {
    "delivery and fulfillment": 0,
    "inventory and product availability": 1,
    "customer service and experience": 2,
    "pricing clarity and value": 3,
    "returns and post-purchase handling": 4,
}

ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1B
    (?:
        \[[0-?]*[ -/]*[@-~]
        | \][^\x1B\x07]*(?:\x07|\x1B\\)
        | [PX^_][^\x1B]*(?:\x1B\\)
        | [@-_]
    )
    """,
    re.VERBOSE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_spaces(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_terminal_control_sequences(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return text


def split_sentences(text: str, max_sentences: int = 2) -> str:
    text = normalize_spaces(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [part.strip() for part in parts if part.strip()]
    return " ".join(parts[:max_sentences])


def summarize_matches(matches: List[Dict[str, object]], limit: int = 3) -> List[Dict[str, object]]:
    evidence = []
    for match in matches[:limit]:
        evidence.append(
            {
                "company_name": match.get("company_name") or "Unknown Company",
                "tagline": normalize_spaces(match.get("tagline", "")),
                "score": match.get("score"),
                "url": match.get("url"),
                "keywords": list(match.get("keywords", []))[:6],
                "retrieval_reason": match.get("retrieval_reason", []),
                "summary_text": split_sentences(match.get("description") or match.get("text", ""), max_sentences=2),
                "source_family": match.get("source_family", ""),
                "source_name": match.get("source_name", ""),
                "trust_tier": match.get("trust_tier", ""),
                "jurisdiction": match.get("jurisdiction", ""),
                "business_stage": match.get("business_stage", ""),
                "category": match.get("category", ""),
            }
        )
    return evidence


def is_yc_source(source_family: str) -> bool:
    source_family = normalize_spaces(source_family)
    return source_family not in {"hk_official", "business_guide"}


def is_business_guide_intent(query_intent: Dict[str, object]) -> bool:
    return bool(
        query_intent.get("asks_business_guide")
        or query_intent.get("asks_retail")
        or query_intent.get("asks_operations")
        or query_intent.get("asks_pricing")
        or query_intent.get("asks_competition")
        or query_intent.get("asks_business_plan")
    )


def is_feedback_intent(query_intent: Dict[str, object]) -> bool:
    return bool(query_intent.get("asks_feedback"))


def extract_common_signals(matches: List[Dict[str, object]]) -> Dict[str, object]:
    keyword_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()

    for match in matches:
        keywords = [normalize_spaces(keyword).lower() for keyword in match.get("keywords", [])]
        for keyword in keywords:
            if not keyword or keyword in GENERIC_KEYWORDS:
                continue
            keyword_counter[keyword] += 1

        combined_text = " ".join(keywords + [normalize_spaces(match.get("description", "")).lower()])
        combined_terms = set(re.findall(r"[a-z0-9]+", combined_text))
        for theme_label, theme_terms in THEME_MAP.items():
            hits = len(combined_terms & theme_terms)
            if hits:
                theme_counter[theme_label] += hits

    top_keywords = [keyword for keyword, _ in keyword_counter.most_common(6)]
    top_themes = [theme for theme, _ in theme_counter.most_common(3)]
    return {
        "top_keywords": top_keywords,
        "top_themes": top_themes,
    }


def extract_feedback_findings(matches: List[Dict[str, object]]) -> Dict[str, object]:
    theme_counter: Counter[str] = Counter()
    theme_support: Dict[str, List[str]] = {}

    for match in matches[:5]:
        combined_text = " ".join(
            [
                normalize_spaces(match.get("description", "")).lower(),
                normalize_spaces(match.get("retrieval_text", "")).lower(),
                " ".join(normalize_spaces(keyword).lower() for keyword in match.get("keywords", [])),
            ]
        )
        company_name = normalize_spaces(match.get("company_name", ""))
        for theme_label, theme_terms in FEEDBACK_THEME_MAP.items():
            hits = sum(1 for term in theme_terms if term in combined_text)
            if not hits:
                continue
            theme_counter[theme_label] += hits
            if company_name:
                theme_support.setdefault(theme_label, [])
                if company_name not in theme_support[theme_label]:
                    theme_support[theme_label].append(company_name)

    ranked_themes = sorted(
        theme_counter,
        key=lambda theme: (FEEDBACK_THEME_PRIORITY.get(theme, 99), -theme_counter[theme], theme),
    )[:5]
    if not ranked_themes:
        ranked_themes = DEFAULT_FEEDBACK_THEMES[:]

    return {
        "ranked_themes": ranked_themes,
        "theme_support": {
            theme: theme_support.get(theme, [])[:2]
            for theme in ranked_themes
        },
    }


def join_labels(labels: List[str], limit: int = 4) -> str:
    cleaned = [normalize_spaces(label) for label in labels if normalize_spaces(label)]
    if not cleaned:
        return ""
    selected = cleaned[:limit]
    if len(selected) == 1:
        return selected[0]
    if len(selected) == 2:
        return f"{selected[0]} and {selected[1]}"
    return ", ".join(selected[:-1]) + f", and {selected[-1]}"


def parse_query_intent(query: str) -> Dict[str, object]:
    lowered = normalize_spaces(query).lower()
    locations = [label for label, terms in LOCATION_TERMS.items() if any(term in lowered for term in terms)]
    asks_retail = any(term in lowered for term in {"retail", "store", "stores", "brick-and-mortar", "brick and mortar"})
    asks_operations = any(term in lowered for term in {"operations", "inventory", "supply chain", "sourcing", "warehouse", "fulfillment", "vendor", "vendors", "staffing"})
    asks_pricing = any(term in lowered for term in {"pricing", "price", "margin"})
    asks_competition = any(term in lowered for term in {"competition", "competitor", "competitive"})
    asks_business_plan = any(term in lowered for term in {"business plan", "market research", "funding", "budget", "startup cost"})
    asks_feedback = any(term in lowered for term in FEEDBACK_TERMS)
    asks_business_guide = any(
        phrase in lowered
        for phrase in {
            "how to start a business",
            "start a business",
            "open a retail store",
            "target customer",
            "target market",
            "customer segment",
            "go to market",
        }
    ) or asks_retail or asks_operations or asks_pricing or asks_competition or asks_business_plan
    return {
        "raw_query": query,
        "locations": locations,
        "asks_startup_guidance": any(term in lowered for term in {"start", "launch", "begin", "build", "found"}),
        "asks_validation": any(term in lowered for term in {"validate", "validation", "test", "mvp", "demand"}),
        "asks_examples": any(term in lowered for term in {"example", "examples", "similar", "benchmark", "compare"}),
        "asks_compliance": any(term in lowered for term in {"legal", "compliance", "regulation", "regulatory", "license", "licensing"}),
        "asks_registration": any(term in lowered for term in {"register", "registration", "incorporation", "incorporate", "company name", "annual return", "filing"}),
        "asks_official_guidance": any(term in lowered for term in {"register", "registration", "incorporation", "incorporate", "annual return", "filing", "compliance", "license", "licensing", "permit"}),
        "asks_market": any(term in lowered for term in {"market", "customer", "customers", "demand", "segment"}),
        "asks_retail": asks_retail,
        "asks_operations": asks_operations,
        "asks_pricing": asks_pricing,
        "asks_competition": asks_competition,
        "asks_feedback": asks_feedback,
        "asks_business_plan": asks_business_plan,
        "asks_business_guide": asks_business_guide,
        "asks_next_step": any(term in lowered for term in {"next", "first", "step", "should i"}),
    }


def build_market_direction(
    predicted_topic: str,
    signals: Dict[str, object],
    query_intent: Dict[str, object],
    feedback_findings: Dict[str, object],
) -> str:
    top_keywords = signals.get("top_keywords", [])
    top_themes = signals.get("top_themes", [])
    locations = query_intent.get("locations", [])
    if query_intent.get("asks_registration") or query_intent.get("asks_official_guidance"):
        return "The retrieved evidence is mainly about official Hong Kong incorporation, filing, and company-registration procedures."
    if is_feedback_intent(query_intent):
        ranked_themes = feedback_findings.get("ranked_themes", []) or DEFAULT_FEEDBACK_THEMES
        return (
            "The question is mainly about customer complaint or review patterns. "
            "The available evidence is indirect business-guide material, which suggests likely complaint clusters around "
            f"{join_labels(ranked_themes)} rather than a direct ranked list from a dedicated review corpus."
        )
    if is_business_guide_intent(query_intent):
        if query_intent.get("asks_retail") or query_intent.get("asks_operations"):
            return "The retrieved evidence is mainly business-guide material about retail launch, customer targeting, pricing, competition, and day-to-day operations."
        return "The retrieved evidence is mainly business-guide material about planning, market research, funding, and practical launch steps."
    if top_keywords:
        location_hint = f" for {', '.join(locations)}" if locations else ""
        return f"The retrieved YC companies suggest that the strongest recurring direction{location_hint} is around {', '.join(top_keywords[:4])}."
    if top_themes:
        return f"The strongest recurring directions in the retrieved evidence are {', '.join(top_themes[:3])}."
    return f"The retrieved evidence is centered on the topic '{predicted_topic}'."


def build_recommendation(
    predicted_topic: str,
    query_intent: Dict[str, object],
    signals: Dict[str, object],
    companies: List[str],
    feedback_findings: Dict[str, object],
) -> str:
    lowered = predicted_topic.lower()
    if query_intent.get("asks_registration") or query_intent.get("asks_official_guidance"):
        return (
            "Use the Companies Registry guidance as the primary source for incorporation steps, required forms, company-name checks, and filing sequence. "
            "Treat startup-pattern knowledge as secondary until the local registration path is clear."
        )
    if is_feedback_intent(query_intent):
        ranked_themes = feedback_findings.get("ranked_themes", []) or DEFAULT_FEEDBACK_THEMES
        return (
            "The best working answer from the current evidence is that customers are most likely to complain about "
            f"{join_labels(ranked_themes)}. Treat that as an evidence-guided inference from retail planning and operations material, not as a direct complaint ranking from review data."
        )
    if is_business_guide_intent(query_intent):
        if query_intent.get("asks_retail") or query_intent.get("asks_operations"):
            base = (
                "Use the business-guide evidence as the primary scaffold: define the target customer, product mix, price point, "
                "competitive positioning, store or channel choice, vendor setup, inventory flow, and basic operating systems before trying to scale."
            )
            if query_intent.get("locations"):
                return base + " If you will operate in Hong Kong, pair that operating plan with local registration, permit, and compliance checks."
            return base
        return (
            "Use the business-guide evidence to sequence the startup work: validate the market first, write a simple business plan, "
            "estimate startup costs and funding needs, then finalize registration and launch setup."
        )
    if "health" in lowered or "medical" in lowered:
        base = "A practical direction is to validate one narrow healthcare pain point first, such as care coordination, patient communication, or provider data access, before broadening the product scope."
        if query_intent.get("locations"):
            return base + " For Hong Kong-specific decisions, you will still need local regulatory and provider workflow evidence."
        return base
    if "agent" in lowered or "code" in lowered:
        return (
            "The retrieved startups point toward tools that reduce engineering friction. A strong next move is to "
            "focus on one high-value workflow such as spec alignment, code generation, testing, or deployment, instead of trying to automate the whole engineering lifecycle at once."
        )
    if "financial" in lowered or "payment" in lowered or "insurance" in lowered:
        return (
            "The evidence suggests that clarity around compliance, risk, and customer segment matters early. Start "
            "with one tightly defined use case and validate regulatory constraints before expanding."
        )
    if "education" in lowered or "students" in lowered:
        return (
            "The strongest pattern is to solve a specific learner or school workflow first. Validate whether your "
            "primary buyer is the student, the teacher, or the institution."
        )
    if companies:
        return (
            f"Use {companies[0]} and the other retrieved startups as benchmarks, then define the one customer pain "
            "point your product should solve more clearly than they do."
        )
    return "Use the retrieved evidence to narrow the target user and the first pain point to validate."


def build_next_step(predicted_topic: str, query_intent: Dict[str, object]) -> str:
    lowered = predicted_topic.lower()
    if query_intent.get("asks_registration") or query_intent.get("asks_official_guidance"):
        return "Start by confirming the company type, checking the proposed company name, preparing the incorporation form and articles of association, and reviewing the current filing and business-registration requirements on the Companies Registry site."
    if is_feedback_intent(query_intent):
        return (
            "If you need a direct ranking, analyze actual online retail review text and count how often complaint themes such as delivery, inventory, pricing, returns, and support appear."
        )
    if is_business_guide_intent(query_intent):
        if query_intent.get("asks_retail") or query_intent.get("asks_operations"):
            return "Write a one-page operating plan covering the target customer, product mix, pricing, competitor set, location or channel, vendor list, inventory approach, and the first marketing test."
        return "Write down the target customer, problem statement, competitor alternatives, first-year budget, and funding assumptions before deciding the launch sequence."
    if "health" in lowered or "medical" in lowered:
        if query_intent.get("locations"):
            return "Interview Hong Kong providers, clinics, or care operators and list one workflow where coordination, data access, or patient communication is still painful."
        return "Interview providers or clinics and identify one workflow where delays, coordination, or data access are currently painful."
    if "agent" in lowered or "code" in lowered:
        return "Talk to engineering teams and test whether one repeated workflow is painful enough to justify automation."
    if "financial" in lowered or "payment" in lowered or "insurance" in lowered:
        return "List the regulatory and operational constraints first, then validate one customer problem that is frequent and expensive."
    if "education" in lowered or "students" in lowered:
        return "Run a small user validation round with students, teachers, or schools to confirm who values the product most."
    return "Review the retrieved evidence and validate which repeated customer problem appears most urgent in the market."


def build_reference_examples(evidence: List[Dict[str, object]]) -> List[str]:
    references = []
    seen = set()
    for item in evidence[:3]:
        company = safe_company = normalize_spaces(item.get("company_name", ""))
        tagline = normalize_spaces(item.get("tagline", ""))
        if item.get("source_family") in {"hk_official", "business_guide"}:
            source_name = normalize_spaces(item.get("source_name", "Official Source"))
            label = f"{source_name}: {safe_company}"
        elif safe_company and tagline:
            label = f"{safe_company}: {tagline}"
        elif safe_company:
            label = safe_company
        else:
            continue
        if label in seen:
            continue
        seen.add(label)
        references.append(label)
    return references


def detect_evidence_gaps(query_intent: Dict[str, object], matches: List[Dict[str, object]], evidence: List[Dict[str, object]]) -> List[str]:
    combined = " ".join(
        normalize_spaces(match.get("retrieval_text") or match.get("description") or match.get("text", "")).lower()
        for match in matches[:5]
    )
    gaps: List[str] = []
    official_hk_hits = [
        item for item in evidence
        if item.get("source_family") == "hk_official" and item.get("jurisdiction") == "hong_kong"
    ]
    business_guide_hits = [item for item in evidence if item.get("source_family") == "business_guide"]
    if query_intent.get("locations") and len(official_hk_hits) < 2:
        gaps.append("The current evidence does not yet include enough Hong Kong-specific market or regulatory guidance.")
    if is_feedback_intent(query_intent):
        gaps.append("The current retrieval set does not contain a dedicated customer-review corpus, so complaint rankings should be treated as incomplete.")
    if is_business_guide_intent(query_intent) and not business_guide_hits:
        gaps.append("The current evidence is weak on step-by-step business-guide material for planning, pricing, competition, or operations.")
    if query_intent.get("asks_compliance") and not any(term in combined for term in COMPLIANCE_TERMS):
        gaps.append("The retrieved evidence is weak on compliance and legal details, so regulatory conclusions should not be treated as complete.")
    if (query_intent.get("asks_market") or query_intent.get("asks_pricing") or query_intent.get("asks_competition")) and not any(term in combined for term in MARKET_TERMS | OPERATIONS_TERMS):
        gaps.append("The current evidence is stronger on product patterns than on direct market-demand validation.")
    return gaps


def build_evidence_summary(
    predicted_topic: str,
    signals: Dict[str, object],
    evidence: List[Dict[str, object]],
    query_intent: Dict[str, object],
    feedback_findings: Dict[str, object],
) -> str:
    official_pages = [item.get("company_name") for item in evidence if item.get("source_family") == "hk_official" and item.get("company_name")]
    guide_pages = [item.get("company_name") for item in evidence if item.get("source_family") == "business_guide" and item.get("company_name")]
    yc_examples = [item.get("company_name") for item in evidence if is_yc_source(item.get("source_family", "")) and item.get("company_name")]
    top_themes = signals.get("top_themes", [])
    top_keywords = signals.get("top_keywords", [])
    if is_feedback_intent(query_intent) and guide_pages:
        ranked_themes = feedback_findings.get("ranked_themes", []) or DEFAULT_FEEDBACK_THEMES
        theme_support = feedback_findings.get("theme_support", {}) or {}
        strongest_theme = ranked_themes[0]
        support_sources = join_labels(theme_support.get(strongest_theme, []), limit=2)
        support_hint = f" {support_sources} provide the strongest support for that pattern." if support_sources else ""
        return (
            f"The retrieved guide pages such as {', '.join(guide_pages[:2])} suggest that the most plausible complaint areas are "
            f"{join_labels(ranked_themes)}.{support_hint} These sources are still indirect business guidance rather than a direct customer-review dataset."
        )
    if official_pages and guide_pages and yc_examples:
        return (
            f"The retrieved evidence combines official Hong Kong guidance from {', '.join(official_pages[:2])}, "
            f"business-guide advice from {', '.join(guide_pages[:2])}, and YC startup patterns from {', '.join(yc_examples[:2])}. "
            f"The repeated signals include {', '.join(top_keywords[:3]) or predicted_topic}."
        )
    if official_pages and guide_pages:
        return (
            f"The retrieved evidence combines official Hong Kong pages such as {', '.join(official_pages[:2])} "
            f"with business-guide material such as {', '.join(guide_pages[:2])}, which together cover local process requirements and practical startup steps."
        )
    if guide_pages and yc_examples:
        return (
            f"The retrieved evidence combines business-guide material from {', '.join(guide_pages[:2])} "
            f"with YC startup patterns from {', '.join(yc_examples[:2])}. The strongest recurring signals are {', '.join(top_keywords[:3]) or predicted_topic}."
        )
    if official_pages and yc_examples:
        return (
            f"The retrieved evidence combines official Hong Kong guidance from {', '.join(official_pages[:2])} "
            f"with YC-style startup patterns from {', '.join(yc_examples[:2])}. The repeated pattern is {top_themes[0] if top_themes else predicted_topic}, "
            f"supported by signals such as {', '.join(top_keywords[:3])}."
        )
    if official_pages:
        return f"The strongest evidence comes from official Hong Kong pages such as {', '.join(official_pages[:2])}, which directly support questions about local filing, compliance, or registration."
    if guide_pages:
        return f"The strongest evidence comes from business-guide pages such as {', '.join(guide_pages[:2])}, which directly support questions about planning, pricing, competition, operations, and launch steps."
    if yc_examples and top_themes:
        return f"Across {', '.join(yc_examples[:2])}, the most repeated pattern is {top_themes[0]}, supported by signals such as {', '.join(top_keywords[:3])}."
    if yc_examples:
        return f"The top retrieved YC examples are {', '.join(yc_examples[:2])}, which point to repeated patterns around {', '.join(top_keywords[:3])}."
    return f"The available evidence is still centered on {predicted_topic}."


def build_source_specific_sections(
    predicted_topic: str,
    signals: Dict[str, object],
    evidence: List[Dict[str, object]],
    query_intent: Dict[str, object],
    feedback_findings: Dict[str, object],
) -> Dict[str, object]:
    official_pages = [item.get("company_name") for item in evidence if item.get("source_family") == "hk_official" and item.get("company_name")]
    guide_pages = [item.get("company_name") for item in evidence if item.get("source_family") == "business_guide" and item.get("company_name")]
    yc_examples = [item.get("company_name") for item in evidence if is_yc_source(item.get("source_family", "")) and item.get("company_name")]
    top_keywords = signals.get("top_keywords", [])

    official_text = ""
    guide_text = ""
    yc_text = ""
    if official_pages:
        official_text = (
            f"Official Hong Kong guidance is grounded in {', '.join(official_pages[:2])}, which is most useful for company setup, filing, and compliance steps."
        )
    if guide_pages:
        if is_feedback_intent(query_intent):
            ranked_themes = feedback_findings.get("ranked_themes", []) or DEFAULT_FEEDBACK_THEMES
            guide_text = (
                f"Business-guide evidence is grounded in {', '.join(guide_pages[:2])}. "
                f"It points to likely complaint themes such as {join_labels(ranked_themes)}, but it is not a direct review dataset."
            )
        else:
            guide_text = (
                f"Business-guide advice is grounded in {', '.join(guide_pages[:2])}, which is most useful for target-customer definition, pricing, competition, operations, and launch planning."
            )
    if yc_examples:
        yc_text = (
            f"YC startup patterns are illustrated by {', '.join(yc_examples[:2])}, which are better treated as benchmarks or inspiration than as official instructions."
        )
        if top_keywords:
            yc_text += f" The main recurring signals are {', '.join(top_keywords[:3])}."
    if not official_text and not guide_text and not yc_text:
        yc_text = f"The available evidence is still centered on {predicted_topic}."

    return {
        "hong_kong_official_guidance": official_text,
        "business_guide_advice": guide_text,
        "yc_startup_patterns": yc_text,
    }


def build_short_answer(
    query_intent: Dict[str, object],
    predicted_topic: str,
    signals: Dict[str, object],
    gaps: List[str],
    feedback_findings: Dict[str, object],
) -> str:
    top_keywords = signals.get("top_keywords", [])
    locations = query_intent.get("locations", [])
    location_hint = f" in {', '.join(locations)}" if locations else ""
    if query_intent.get("asks_registration") or query_intent.get("asks_official_guidance"):
        answer = f"Your question is mainly asking for official company-registration guidance{location_hint}, and the strongest evidence is about incorporation, filing, and registration requirements."
    elif is_feedback_intent(query_intent):
        ranked_themes = feedback_findings.get("ranked_themes", []) or DEFAULT_FEEDBACK_THEMES
        answer = (
            f"Your question is mainly asking about customer complaint or review patterns{location_hint}. "
            f"Based on the current retail guidance, the most likely complaint areas are {join_labels(ranked_themes)}. "
            f"This is an indirect inference rather than a direct ranking from a dedicated review dataset."
        )
    elif is_business_guide_intent(query_intent):
        if query_intent.get("asks_retail") or query_intent.get("asks_operations"):
            answer = f"Your question is mainly asking for practical retail or operations guidance{location_hint}, and the strongest evidence is guide-style advice on customer targeting, pricing, competition, and execution."
        else:
            answer = f"Your question is mainly asking for practical startup guidance{location_hint}, and the strongest evidence is guide-style advice on planning, funding, registration, and launch sequencing."
    elif top_keywords:
        answer = f"Your question is mainly about {predicted_topic}{location_hint}, and the strongest YC pattern is around {', '.join(top_keywords[:3])}."
    else:
        answer = f"Your question is mainly about {predicted_topic}{location_hint}."
    if gaps:
        first_gap = gaps[0]
        first_gap = first_gap[:1].lower() + first_gap[1:] if first_gap else first_gap
        answer += f" However, {first_gap}"
    return answer


def compose_answer_text(
    query_intent: Dict[str, object],
    short_answer: str,
    evidence_summary: str,
    recommendation: str,
    next_step: str,
    gaps: List[str],
) -> str:
    if is_feedback_intent(query_intent):
        parts = [short_answer, evidence_summary]
        if gaps:
            parts.append("Evidence gap: " + " ".join(gaps[:1]))
        parts.append(next_step)
        return " ".join(part for part in parts if part).strip()
    parts = [short_answer, evidence_summary, recommendation]
    if gaps:
        parts.append("Evidence gap: " + " ".join(gaps[:2]))
    parts.append(next_step)
    return " ".join(part for part in parts if part).strip()


def build_polish_prompt(payload: Dict[str, object]) -> str:
    evidence_lines = []
    for item in payload.get("evidence", [])[:3]:
        company_name = item.get("company_name") or "Unknown Company"
        tagline = normalize_spaces(item.get("tagline", ""))
        summary = normalize_spaces(item.get("summary_text", ""))
        keywords = ", ".join(item.get("keywords", [])[:5])
        source_family = normalize_spaces(item.get("source_family", ""))
        label = "Source" if source_family in {"hk_official", "business_guide"} else "Company"
        parts = [f"{label}: {company_name}"]
        if tagline:
            parts.append(f"Tagline: {tagline}")
        if summary:
            parts.append(f"Evidence: {summary}")
        if keywords:
            parts.append(f"Keywords: {keywords}")
        evidence_lines.append(" | ".join(parts))

    prompt = f"""You are polishing a structured entrepreneurship-support answer.

Rules:
- Keep the answer factual and grounded only in the provided evidence.
- Do not invent companies, markets, or claims.
- Keep the tone practical, concise, and advisor-like.
- Write 2 short paragraphs.
- Mention at most 2 company names from the evidence.

User query:
{payload.get("query", "")}

Predicted topic:
{payload.get("predicted_topic", "")}

Query intent:
{json.dumps(payload.get("query_intent", {}), ensure_ascii=False)}

Structured draft:
- Short answer: {payload.get("short_answer", "")}
- Evidence summary: {payload.get("evidence_summary", "")}
- Market direction: {payload.get("market_direction", "")}
- Recommendation: {payload.get("recommendation", "")}
- Evidence gaps: {" | ".join(payload.get("evidence_gaps", []))}
- Suggested next step: {payload.get("suggested_next_step", "")}

Key signals:
- Top keywords: {", ".join(payload.get("key_signals", {}).get("top_keywords", [])[:5])}
- Top themes: {", ".join(payload.get("key_signals", {}).get("top_themes", [])[:3])}

Evidence:
{os.linesep.join(evidence_lines) if evidence_lines else "No evidence provided."}

Return only the polished final answer text.
"""
    return prompt


def resolve_ollama_path() -> str:
    configured = normalize_spaces(DEFAULT_OLLAMA_PATH)
    if configured and os.path.exists(configured):
        return configured

    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"),
        os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def polish_answer_with_ollama(payload: Dict[str, object], model: str, timeout_seconds: int) -> Dict[str, object]:
    ollama_path = resolve_ollama_path()
    if not ollama_path:
        return {
            "used": False,
            "status": "ollama_not_found",
            "message": "Ollama CLI is not available on PATH and no default install path was found.",
        }

    prompt = build_polish_prompt(payload)
    try:
        completed = subprocess.run(
            [ollama_path, "run", "--nowordwrap", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "used": False,
            "status": "timeout",
            "message": f"Ollama call timed out after {timeout_seconds} seconds.",
        }
    except OSError as exc:
        return {
            "used": False,
            "status": "execution_error",
            "message": f"Failed to execute Ollama: {exc}",
        }

    stdout = normalize_spaces(strip_terminal_control_sequences(completed.stdout))
    stderr = normalize_spaces(strip_terminal_control_sequences(completed.stderr))
    if completed.returncode != 0:
        return {
            "used": False,
            "status": "ollama_error",
            "message": stderr or f"Ollama exited with code {completed.returncode}.",
        }
    if not stdout:
        return {
            "used": False,
            "status": "empty_response",
            "message": "Ollama returned an empty response.",
        }

    return {
        "used": True,
        "status": "ok",
        "model": model,
        "ollama_path": ollama_path,
        "message": "Polished answer generated with local Ollama.",
        "text": stdout,
    }


def build_answer(query: str, predicted_topic: str, matches: List[Dict[str, object]]) -> Dict[str, object]:
    query_intent = parse_query_intent(query)
    evidence = summarize_matches(matches, limit=3)
    companies = list(dict.fromkeys([item["company_name"] for item in evidence]))
    signals = extract_common_signals(matches[:5])
    feedback_findings = extract_feedback_findings(matches[:5]) if is_feedback_intent(query_intent) else {}
    evidence_gaps = detect_evidence_gaps(query_intent, matches, evidence)
    market_direction = build_market_direction(predicted_topic, signals, query_intent, feedback_findings)
    evidence_summary = build_evidence_summary(predicted_topic, signals, evidence, query_intent, feedback_findings)
    source_sections = build_source_specific_sections(predicted_topic, signals, evidence, query_intent, feedback_findings)
    recommendation = build_recommendation(predicted_topic, query_intent, signals, companies, feedback_findings)
    next_step = build_next_step(predicted_topic, query_intent)
    short_answer = build_short_answer(query_intent, predicted_topic, signals, evidence_gaps, feedback_findings)
    answer_text = compose_answer_text(query_intent, short_answer, evidence_summary, recommendation, next_step, evidence_gaps)
    reference_examples = build_reference_examples(evidence)

    return {
        "query": query,
        "predicted_topic": predicted_topic,
        "query_intent": query_intent,
        "short_answer": short_answer,
        "evidence_summary": evidence_summary,
        "market_direction": market_direction,
        "key_signals": signals,
        "feedback_findings": feedback_findings,
        "evidence_gaps": evidence_gaps,
        "recommendation": recommendation,
        "suggested_next_step": next_step,
        "reference_examples": reference_examples,
        "answer_sections": {
            "hong_kong_official_guidance": source_sections["hong_kong_official_guidance"],
            "business_guide_advice": source_sections["business_guide_advice"],
            "yc_startup_patterns": source_sections["yc_startup_patterns"],
            "what_yc_evidence_suggests": evidence_summary,
            "what_is_missing": evidence_gaps,
            "recommended_direction": recommendation,
            "next_step": next_step,
        },
        "answer_text": answer_text,
        "final_answer": answer_text,
        "answer_mode": "deterministic_structured",
        "relevant_companies": companies[:5],
        "evidence": evidence,
        "llm_polish": {
            "used": False,
            "status": "not_requested",
            "message": "Local-model polishing was not requested.",
        },
        "notes": [
            "This version uses deterministic evidence summarization instead of direct free-form generation.",
            "Optional next step: add API or local-model polishing on top of this structured draft.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a structured Project 8 answer from retrieved chunks.")
    parser.add_argument("--query", required=True, help="User enquiry")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to retrieval index pickle")
    parser.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH, help="Path to topic classifier pickle")
    parser.add_argument("--topic-catalog-path", default=DEFAULT_TOPIC_CATALOG_PATH, help="Path to topic catalog json")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save answer json")
    parser.add_argument("--top-k", type=int, default=5, help="How many retrieved chunks to use")
    parser.add_argument("--topic-candidates", type=int, default=3, help="How many predicted topics to search")
    parser.add_argument("--polish-with-local-llm", action="store_true", help="Polish the structured answer with a local Ollama model")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name for optional local polishing")
    parser.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT, help="Timeout in seconds for the optional Ollama polishing call")
    return parser.parse_args()


def safe_print_json(payload: Dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True, indent=2))


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.index_path):
        raise SystemExit(
            f"Retrieval index not found: {args.index_path}\n"
            "Please run: python rag/build_retrieval_index.py"
        )

    index = load_pickle(args.index_path)
    topic_catalog = load_topic_catalog(args.topic_catalog_path)
    best_topic, topic_ranking = predict_topic(
        query=args.query,
        classifier_path=args.classifier_path,
        topic_catalog_path=args.topic_catalog_path,
    )
    matches = retrieve(
        query=args.query,
        index=index,
        topic_ranking=topic_ranking,
        topic_catalog=topic_catalog,
        top_k=args.top_k,
        topic_candidates=max(1, args.topic_candidates),
    )
    payload = build_answer(query=args.query, predicted_topic=best_topic, matches=matches)
    if args.polish_with_local_llm:
        polish_result = polish_answer_with_ollama(
            payload=payload,
            model=args.ollama_model,
            timeout_seconds=max(1, args.ollama_timeout),
        )
        payload["llm_polish"] = polish_result
        if polish_result.get("used") and polish_result.get("text"):
            payload["final_answer"] = polish_result["text"]
            payload["answer_mode"] = "deterministic_structured_plus_local_polish"

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    safe_print_json(payload)


if __name__ == "__main__":
    main()
