#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PLACEHOLDER_NAMES = {
    "",
    "N/A",
    "Unknown Company",
    "The Problem",
    "The Solution",
    "Problem",
    "TL;DR",
    "tl;dr",
    "TLDR",
    "TLDR;",
}

WEIRD_NAME_MARKERS = [
    "TL;DR",
    "tl;dr",
    "TLDR",
    "The Problem",
    "The Solution",
    "What is ",
    "****",
    "<<<",
    ">>>",
    "Problem:",
]

SLUG_PATTERN = re.compile(r"/companies/([^/?#]+)")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def slug_to_name(slug: str) -> str:
    pieces = [p for p in re.split(r"[-_]+", slug) if p]
    return " ".join(piece.upper() if piece.isupper() else piece.capitalize() for piece in pieces)


def slug_name(url: str) -> str:
    match = SLUG_PATTERN.search(url or "")
    return slug_to_name(match.group(1)) if match else ""


def word_count(text: str) -> int:
    return len(str(text or "").split())


def normalize_name_for_quality(name: str) -> str:
    text = str(name or "").strip()
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def summarize_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    urls = [str(row.get("url", "")).strip() for row in rows]
    return {
        "total_records": len(rows),
        "unique_urls": len(set(urls)),
        "duplicate_urls": len(urls) - len(set(urls)),
        "batch_na": sum(1 for row in rows if str(row.get("batch")).strip() in {"", "N/A", "None"}),
        "location_na": sum(1 for row in rows if str(row.get("location")).strip() in {"", "N/A", "None"}),
        "team_size_na": sum(1 for row in rows if str(row.get("team_size")).strip() in {"", "N/A", "None"}),
        "founders_empty": sum(1 for row in rows if not row.get("founders")),
        "description_empty": sum(1 for row in rows if not str(row.get("description", "")).strip()),
        "website_empty": sum(1 for row in rows if not str(row.get("website", "")).strip()),
    }


def summarize_kb(rows: list[dict[str, Any]]) -> dict[str, Any]:
    company_ids = [str(row.get("company_id", "")).strip() for row in rows]
    canonical_names = [str(row.get("canonical_name", "")).strip() for row in rows]
    keyword_counts = [len(row.get("top_keywords", [])) for row in rows]
    clean_word_counts = [word_count(row.get("clean_text", "")) for row in rows]
    raw_weird_name_count = sum(1 for row in rows if is_weird_name(str(row.get("raw_company_name", ""))))
    canonical_weird_name_count = sum(1 for row in rows if is_weird_name(str(row.get("canonical_name", ""))))

    return {
        "total_records": len(rows),
        "unique_company_ids": len(set(company_ids)),
        "duplicate_company_ids": len(company_ids) - len(set(company_ids)),
        "unique_canonical_names": len(set(canonical_names)),
        "duplicate_canonical_names": len(canonical_names) - len(set(canonical_names)),
        "empty_clean_text": sum(1 for row in rows if not str(row.get("clean_text", "")).strip()),
        "empty_keywords": sum(1 for row in rows if not row.get("top_keywords")),
        "empty_topic_labels": sum(1 for row in rows if not str(row.get("topic_label", "")).strip()),
        "raw_weird_name_records": raw_weird_name_count,
        "canonical_weird_name_records": canonical_weird_name_count,
        "avg_keywords_per_record": round(sum(keyword_counts) / len(keyword_counts), 2),
        "min_keywords_per_record": min(keyword_counts),
        "max_keywords_per_record": max(keyword_counts),
        "avg_clean_text_words": round(sum(clean_word_counts) / len(clean_word_counts), 2),
        "min_clean_text_words": min(clean_word_counts),
        "max_clean_text_words": max(clean_word_counts),
    }


def is_weird_name(name: str) -> bool:
    text = normalize_name_for_quality(name)
    if text in PLACEHOLDER_NAMES:
        return True
    if any(marker in text for marker in WEIRD_NAME_MARKERS):
        return True
    return False


def sample_zero_keyword_records(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    sample = []
    for row in rows:
        if row.get("top_keywords"):
            continue
        sample.append(
            {
                "company_id": row.get("company_id"),
                "canonical_name": row.get("canonical_name"),
                "raw_company_name": row.get("raw_company_name"),
                "raw_one_liner": row.get("raw_one_liner"),
                "description": row.get("description"),
                "clean_text": row.get("clean_text"),
                "topic_label": row.get("topic_label"),
            }
        )
        if len(sample) >= limit:
            break
    return sample


def sample_weird_names(rows: list[dict[str, Any]], field: str, limit: int = 10) -> list[dict[str, Any]]:
    sample = []
    for row in rows:
        if not is_weird_name(str(row.get(field, ""))):
            continue
        sample.append(
            {
                "company_id": row.get("company_id"),
                "canonical_name": row.get("canonical_name"),
                "raw_company_name": row.get("raw_company_name"),
                "raw_one_liner": row.get("raw_one_liner"),
                "canonical_tagline": row.get("canonical_tagline"),
            }
        )
        if len(sample) >= limit:
            break
    return sample


def sample_duplicate_names(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    counts = Counter(str(row.get("canonical_name", "")).strip() for row in rows)
    sample = []
    for canonical_name, count in counts.most_common():
        if count <= 1:
            continue
        sample.append(
            {
                "canonical_name": canonical_name,
                "count": count,
                "company_ids": [
                    row.get("company_id")
                    for row in rows
                    if str(row.get("canonical_name", "")).strip() == canonical_name
                ][:5],
            }
        )
        if len(sample) >= limit:
            break
    return sample


def topic_distribution(rows: list[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    counts = Counter(str(row.get("topic_label", "")).strip() for row in rows)
    most_common = [{"topic_label": name, "count": count} for name, count in counts.most_common(limit)]
    least_common = [
        {"topic_label": name, "count": count}
        for name, count in sorted(counts.items(), key=lambda item: (item[1], item[0]))[:limit]
    ]
    return {
        "most_common": most_common,
        "least_common": least_common,
    }


def repair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    repaired = []
    for row in rows:
        canonical_name = str(row.get("canonical_name", "")).strip()
        raw_company_name = str(row.get("raw_company_name", "")).strip()
        derived_slug_name = slug_name(str(row.get("url", "")))
        if canonical_name and canonical_name == derived_slug_name and canonical_name != raw_company_name:
            repaired.append(
                {
                    "company_id": row.get("company_id"),
                    "raw_company_name": raw_company_name,
                    "raw_one_liner": row.get("raw_one_liner"),
                    "canonical_name": canonical_name,
                }
            )

    return {
        "repaired_from_slug_count": len(repaired),
        "sample": repaired[:10],
    }


def build_report(source_path: Path, kb_path: Path) -> dict[str, Any]:
    source_rows = load_jsonl(source_path)
    kb_rows = load_jsonl(kb_path)

    return {
        "source_path": str(source_path),
        "kb_path": str(kb_path),
        "source_summary": summarize_source(source_rows),
        "kb_summary": summarize_kb(kb_rows),
        "topic_distribution": topic_distribution(kb_rows),
        "repair_stats": repair_stats(kb_rows),
        "samples": {
            "canonical_weird_names": sample_weird_names(kb_rows, "canonical_name"),
            "raw_weird_names": sample_weird_names(kb_rows, "raw_company_name"),
            "zero_keyword_records": sample_zero_keyword_records(kb_rows),
            "duplicate_canonical_names": sample_duplicate_names(kb_rows),
        },
        "findings": [
            {
                "severity": "high",
                "title": "Source records are missing several structured fields",
                "details": "batch, location, team_size, and founders are effectively empty across the entire source dataset.",
            },
            {
                "severity": "medium",
                "title": "Raw source names still contain placeholder or noisy text",
                "details": "The original scraped fields still include values such as TL;DR, The Problem, and marketing-style titles.",
            },
            {
                "severity": "medium",
                "title": "Some records have no extracted keywords",
                "details": "These rows usually have empty descriptions or very short source text, which weakens retrieval quality.",
            },
            {
                "severity": "medium",
                "title": "Canonical name collisions remain",
                "details": "Multiple different companies share the same cleaned canonical_name, which can confuse reporting and chatbot retrieval.",
            },
            {
                "severity": "low",
                "title": "Topic distribution is uneven",
                "details": "A few topic clusters are much larger than others, suggesting the 20-cluster taxonomy is coarse for some business areas.",
            },
        ],
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Check Project 8 YC keyword knowledge base quality.")
    parser.add_argument("--source", default="yc_detailed_data.jsonl", help="Path to source JSONL data")
    parser.add_argument(
        "--kb",
        default="project8_kb/cleaned_company_kb.jsonl",
        help="Path to cleaned knowledge base JSONL",
    )
    parser.add_argument(
        "--out",
        default="project8_kb/data_quality_report.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    report = build_report(Path(args.source), Path(args.kb))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
