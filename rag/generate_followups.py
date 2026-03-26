#!/usr/bin/env python3
"""
Generate likely follow-up questions for Project 8 objective 5.

This version is still deterministic, but it is now grounded in the
structured answer payload:
- reads the structured answer output
- uses topic, key signals, recommendation, and retrieved companies
- produces follow-up questions that are more specific to the current answer
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

DEFAULT_ANSWER_PATH = "rag/output/answer.json"
DEFAULT_OUTPUT_PATH = "rag/output/followups.json"


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_keyword(keyword: str) -> str:
    keyword = safe_text(keyword).replace("_", " ").strip()
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword


def topic_templates(topic_label: str, keyword_hint: str, company_hint: str) -> List[str]:
    lowered = topic_label.lower()
    if "health" in lowered or "medical" in lowered:
        return [
            f"What regulatory or compliance issues should I check next for a healthcare idea focused on {keyword_hint or 'patient care'}?",
            f"Which YC healthcare companies are most similar to {company_hint or 'this business model'}?",
            "How can I validate demand with clinics, providers, or patients in a narrow workflow first?",
        ]
    if "financial" in lowered or "payment" in lowered or "insurance" in lowered:
        return [
            f"What legal and compliance risks should I review next if the startup is centered on {keyword_hint or 'payments or fintech'}?",
            f"Which fintech or insurance startups are closest to {company_hint or 'this idea'}?",
            "What customer segment would be easiest to validate first in Hong Kong?",
        ]
    if "education" in lowered or "students" in lowered:
        return [
            f"Who should be the first target users if the product is focused on {keyword_hint or 'learning outcomes'}?",
            f"Which learning startups could be used as benchmarks against {company_hint or 'this idea'}?",
            "How can I validate whether students, teachers, or schools have the strongest demand?",
        ]
    if "agent" in lowered or "code" in lowered or "support" in lowered:
        return [
            f"Which engineering workflow should I narrow down first if the product focuses on {keyword_hint or 'coding agents'}?",
            f"How is {company_hint or 'the top retrieved company'} different from the other retrieved startups?",
            "What evidence would prove that teams are willing to adopt this workflow before automating more tasks?",
        ]
    return [
        f"Which YC companies in the retrieved results are the best benchmarks for {company_hint or 'this idea'}?",
        f"What customer problem appears most consistently around {keyword_hint or 'the main retrieved theme'}?",
        "What should be the next market validation step for this startup idea?",
    ]


def official_templates(page_hint: str) -> List[str]:
    return [
        f"What filing documents and sequence should I double-check next from {page_hint or 'the Companies Registry guidance'}?",
        "Which registration or compliance steps can be completed online, and which ones may still require extra documents or approvals?",
        "What local permits, licences, or post-incorporation obligations should I review after the basic registration step?",
    ]


def business_guide_templates(keyword_hint: str, guide_hint: str, has_hk_location: bool) -> List[str]:
    questions = [
        f"How should I validate the target customer, pricing, and competition before acting on {guide_hint or 'this guide-based plan'}?",
        f"What operating assumptions should I test first around {keyword_hint or 'inventory, location, and demand'}?",
        "What simple one-page plan should I write before spending money on launch or expansion?",
    ]
    if has_hk_location:
        questions.append("Which parts of this plan are general business advice, and which parts need Hong Kong-specific regulatory confirmation?")
    return questions


def feedback_templates(keyword_hint: str, guide_hint: str) -> List[str]:
    return [
        f"Which complaint categories should I track first for {keyword_hint or 'online retail reviews'}, such as pricing, delivery, returns, or support?",
        f"What review dataset or customer-feedback source should I collect before relying too heavily on {guide_hint or 'this indirect evidence'}?",
        "How should I turn repeated complaints into operational fixes and a simple prioritization list?",
    ]


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def build_followup_candidates(answer_payload: Dict[str, object]) -> List[str]:
    topic_label = safe_text(answer_payload.get("predicted_topic", "unknown"))
    key_signals = answer_payload.get("key_signals", {}) or {}
    top_keywords = [normalize_keyword(keyword) for keyword in key_signals.get("top_keywords", []) if normalize_keyword(keyword)]
    top_themes = [normalize_keyword(theme) for theme in key_signals.get("top_themes", []) if normalize_keyword(theme)]
    evidence = answer_payload.get("evidence", []) or []
    companies = [safe_text(company) for company in answer_payload.get("relevant_companies", []) if safe_text(company)]
    recommendation = safe_text(answer_payload.get("recommendation"))
    next_step = safe_text(answer_payload.get("suggested_next_step"))
    query_intent = answer_payload.get("query_intent", {}) or {}

    keyword_hint = ", ".join(top_keywords[:2])
    yc_companies = [safe_text(item.get("company_name")) for item in evidence if safe_text(item.get("source_family")) not in {"hk_official", "business_guide"} and safe_text(item.get("company_name"))]
    official_pages = [safe_text(item.get("company_name")) for item in evidence if safe_text(item.get("source_family")) == "hk_official" and safe_text(item.get("company_name"))]
    guide_pages = [safe_text(item.get("company_name")) for item in evidence if safe_text(item.get("source_family")) == "business_guide" and safe_text(item.get("company_name"))]
    company_hint = yc_companies[0] if yc_companies else (companies[0] if companies else "")

    if query_intent.get("asks_feedback"):
        followups = feedback_templates(keyword_hint, guide_pages[0] if guide_pages else "")
    elif query_intent.get("asks_registration") or query_intent.get("asks_official_guidance") or official_pages:
        followups = official_templates(official_pages[0] if official_pages else "")
    elif query_intent.get("asks_business_guide") or guide_pages:
        followups = business_guide_templates(keyword_hint, guide_pages[0] if guide_pages else "", bool(query_intent.get("locations")))
    else:
        followups = topic_templates(topic_label, keyword_hint, company_hint)

    if top_themes:
        followups.append(
            f"Among {', '.join(top_themes[:2])}, which theme should I prioritize first for an MVP?"
        )
    if recommendation:
        followups.append(
            f"What concrete experiment could I run next to test this recommendation: {recommendation}"
        )
    if next_step:
        followups.append(
            f"What interview or validation questions should I ask if my next step is: {next_step}"
        )
    if len(yc_companies) >= 2:
        followups.append(
            f"How should I compare {yc_companies[0]} and {yc_companies[1]} when defining my own positioning?"
        )

    return dedupe_keep_order(followups)


def build_followups(answer_payload: Dict[str, object]) -> Dict[str, object]:
    topic_label = safe_text(answer_payload.get("predicted_topic", "unknown"))
    followups = build_followup_candidates(answer_payload)[:5]
    return {
        "query": answer_payload.get("query"),
        "predicted_topic": topic_label,
        "follow_up_questions": followups,
        "follow_up_focus": {
            "top_keywords": (answer_payload.get("key_signals", {}) or {}).get("top_keywords", [])[:4],
            "top_themes": (answer_payload.get("key_signals", {}) or {}).get("top_themes", [])[:3],
            "reference_companies": answer_payload.get("relevant_companies", [])[:3],
        },
        "notes": [
            "This version uses deterministic but answer-aware follow-up generation.",
            "Optional next step: replace or polish these questions with API or local-model generation after the answer layer is stable.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate follow-up questions for Project 8.")
    parser.add_argument("--answer-path", default=DEFAULT_ANSWER_PATH, help="Path to answer json")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save follow-up json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.answer_path, "r", encoding="utf-8") as fh:
        answer_payload = json.load(fh)

    payload = build_followups(answer_payload)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
