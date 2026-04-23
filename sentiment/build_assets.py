#!/usr/bin/env python3
"""
Build lightweight sentiment assets from the Kaggle social-media dataset.

Outputs (written to data/sentiment/ by default):
- data/sentiment/sentiment_cleaned.jsonl
- data/sentiment/sentiment_summary.json
- data/sentiment/sentiment_keyword_profiles.json

Usage:
    python sentiment/build_assets.py --dataset-path path/to/sentimentdataset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Resolve project root so we can import config regardless of cwd
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from config import SENTIMENT_DIR, SENTIMENT_CLEANED_PATH, SENTIMENT_KEYWORDS_PATH, SENTIMENT_SUMMARY_PATH

DEFAULT_DATASET_CANDIDATES = [
    str(_ROOT  / "sentiment" / "input" / "sentimentdataset.csv"),
]
DEFAULT_OUTPUT_DIR    = str(SENTIMENT_DIR)
DEFAULT_CLEANED_PATH  = SENTIMENT_CLEANED_PATH
DEFAULT_SUMMARY_PATH  = SENTIMENT_SUMMARY_PATH
DEFAULT_KEYWORDS_PATH = SENTIMENT_KEYWORDS_PATH

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "for", "from", "has", "have", "i", "in", "is", "it",
    "its", "just", "my", "of", "on", "or", "our", "so", "that", "the", "this", "to", "today", "was", "we", "with",
    "you", "your",
}


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def clean_social_text(text: str) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> str:
    lowered = safe_text(label).lower()
    if lowered.startswith("pos"):
        return "Positive"
    if lowered.startswith("neg"):
        return "Negative"
    return "Neutral"


def parse_hashtags(raw: str) -> List[str]:
    hashtags: List[str] = []
    for token in re.split(r"[\s,]+", safe_text(raw)):
        token = token.strip()
        if not token:
            continue
        if token.startswith("#"):
            token = token[1:]
        token = re.sub(r"[^a-zA-Z0-9_]+", "", token).lower()
        if token:
            hashtags.append(token)
    return hashtags


def tokenize(text: str) -> List[str]:
    text = clean_social_text(text)
    return [token for token in text.split() if token and token not in STOPWORDS and len(token) > 2]


def predict_vader_label(text: str, analyzer: SentimentIntensityAnalyzer) -> Dict[str, object]:
    scores = analyzer.polarity_scores(text)
    compound = float(scores["compound"])
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {
        "compound": round(compound, 4),
        "scores": {key: round(float(value), 4) for key, value in scores.items()},
        "label": label,
    }


def to_number(value: str) -> float:
    try:
        return float(safe_text(value) or 0)
    except ValueError:
        return 0.0


def resolve_dataset_path(cli_path: str) -> str:
    candidates: List[str] = []
    if safe_text(cli_path):
        candidates.append(cli_path)
    env_candidate = os.environ.get("SENTIMENT_DATASET_PATH", "")
    if safe_text(env_candidate):
        candidates.append(env_candidate)
    candidates.extend(DEFAULT_DATASET_CANDIDATES)

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit(
        "Sentiment dataset not found. Provide --dataset-path or set SENTIMENT_DATASET_PATH."
    )


def read_rows(path: str) -> List[Dict[str, object]]:
    analyzer = SentimentIntensityAnalyzer()
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, raw_row in enumerate(reader):
            text = safe_text(raw_row.get("Text"))
            if not text:
                continue
            sentiment_label = normalize_label(safe_text(raw_row.get("Sentiment")))
            cleaned_text = clean_social_text(text)
            hashtags = parse_hashtags(safe_text(raw_row.get("Hashtags")))
            vader_result = predict_vader_label(cleaned_text, analyzer)
            rows.append(
                {
                    "row_id": idx,
                    "text": text,
                    "clean_text": cleaned_text,
                    "dataset_sentiment": sentiment_label,
                    "vader_sentiment": vader_result["label"],
                    "vader_compound": vader_result["compound"],
                    "vader_scores": vader_result["scores"],
                    "timestamp": safe_text(raw_row.get("Timestamp")),
                    "user": safe_text(raw_row.get("User")),
                    "platform": safe_text(raw_row.get("Platform")),
                    "hashtags": hashtags,
                    "country": safe_text(raw_row.get("Country")),
                    "retweets": to_number(safe_text(raw_row.get("Retweets"))),
                    "likes": to_number(safe_text(raw_row.get("Likes"))),
                    "year": safe_text(raw_row.get("Year")),
                    "month": safe_text(raw_row.get("Month")),
                    "day": safe_text(raw_row.get("Day")),
                    "hour": safe_text(raw_row.get("Hour")),
                }
            )
    return rows


def top_counter_items(counter: Counter[str], limit: int = 10) -> List[Dict[str, object]]:
    return [{"value": item, "count": count} for item, count in counter.most_common(limit)]


def build_keyword_profiles(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    tokens_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    hashtags_by_label: Dict[str, Counter[str]] = defaultdict(Counter)
    platform_by_label: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        label = safe_text(row.get("dataset_sentiment")) or "Neutral"
        for token in tokenize(safe_text(row.get("text"))):
            tokens_by_label[label][token] += 1
        for hashtag in row.get("hashtags", []):
            hashtags_by_label[label][safe_text(hashtag)] += 1
        platform = safe_text(row.get("platform"))
        if platform:
            platform_by_label[label][platform] += 1

    return {
        "top_keywords_by_sentiment": {
            label: top_counter_items(counter, limit=20)
            for label, counter in tokens_by_label.items()
        },
        "top_hashtags_by_sentiment": {
            label: top_counter_items(counter, limit=15)
            for label, counter in hashtags_by_label.items()
        },
        "top_platforms_by_sentiment": {
            label: top_counter_items(counter, limit=10)
            for label, counter in platform_by_label.items()
        },
    }


def build_summary(rows: List[Dict[str, object]], dataset_path: str, keyword_profiles: Dict[str, object]) -> Dict[str, object]:
    label_counts = Counter(safe_text(row.get("dataset_sentiment")) for row in rows)
    platform_counts = Counter(safe_text(row.get("platform")) for row in rows if safe_text(row.get("platform")))
    country_counts = Counter(safe_text(row.get("country")) for row in rows if safe_text(row.get("country")))
    vader_agreement = sum(
        1 for row in rows
        if safe_text(row.get("dataset_sentiment")) == safe_text(row.get("vader_sentiment"))
    )

    return {
        "dataset_path": dataset_path,
        "row_count": len(rows),
        "dataset_sentiment_distribution": dict(label_counts),
        "top_platforms": top_counter_items(platform_counts, limit=10),
        "top_countries": top_counter_items(country_counts, limit=10),
        "vader_agreement_rate": round(vader_agreement / max(1, len(rows)), 4),
        "sample_rows": rows[:5],
        "keyword_profile_preview": {
            key: value
            for key, value in keyword_profiles.items()
        },
    }


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kaggle-backed sentiment assets for Project 8.")
    parser.add_argument("--dataset-path", default="", help="Path to sentimentdataset.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cleaned-path", default=DEFAULT_CLEANED_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--keywords-path", default=DEFAULT_KEYWORDS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_path)
    rows = read_rows(dataset_path)
    keyword_profiles = build_keyword_profiles(rows)
    summary = build_summary(rows, dataset_path=dataset_path, keyword_profiles=keyword_profiles)

    write_jsonl(args.cleaned_path, rows)
    write_json(args.summary_path, summary)
    write_json(args.keywords_path, keyword_profiles)

    print(json.dumps(
        {
            "dataset_path": dataset_path,
            "row_count": len(rows),
            "summary_path": args.summary_path,
            "keywords_path": args.keywords_path,
            "cleaned_path": args.cleaned_path,
        },
        ensure_ascii=True,
        indent=2,
    ))


if __name__ == "__main__":
    main()
