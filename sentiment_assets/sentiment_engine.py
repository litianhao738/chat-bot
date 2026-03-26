#!/usr/bin/env python3
"""
Lightweight sentiment engine for Project 8 demo use.

Uses:
- VADER sentiment scoring
- keyword profiles built from the uploaded Kaggle dataset
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = str(ROOT_DIR / "sentiment_assets" / "output" / "sentiment_summary.json")
DEFAULT_KEYWORDS_PATH = str(ROOT_DIR / "sentiment_assets" / "output" / "sentiment_keyword_profiles.json")

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "for", "from", "have", "i", "in", "is", "it", "my",
    "of", "on", "or", "our", "so", "that", "the", "this", "to", "was", "we", "with", "you", "your",
}

SENTIMENT_QUERY_TERMS = {
    "complaint", "complaints", "feedback", "review", "reviews", "rating", "ratings", "sentiment", "opinions",
    "reaction", "reactions", "comment", "comments", "customer feedback", "user feedback", "public reaction",
    "social media", "twitter", "instagram", "facebook", "reddit", "buzz", "word of mouth", "complain",
    "complaining",
    "praised", "praise", "criticized", "criticism", "negative feedback", "positive feedback", "frustration",
    "customer satisfaction", "what are people saying", "what do users think",
}

SENTIMENT_QUERY_PHRASES = {
    "customer feedback",
    "user feedback",
    "social media",
    "public reaction",
    "customer complaints",
    "negative feedback",
    "positive feedback",
    "what are people saying",
    "what do users think",
    "online reviews",
    "user reviews",
    "customer reviews",
    "market sentiment",
    "social sentiment",
    "brand sentiment",
}

STRONG_SENTIMENT_TERMS = {
    "complaint", "complaints", "complain", "complaining", "feedback", "review", "reviews", "rating", "ratings",
    "sentiment", "reaction", "reactions", "comment", "comments", "social media", "twitter", "instagram",
    "facebook", "reddit", "public reaction", "customer feedback", "user feedback", "online reviews",
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


def tokenize(text: str) -> List[str]:
    return [
        token for token in clean_social_text(text).split()
        if token and token not in STOPWORDS and len(token) > 2
    ]


def should_analyze_sentiment(text: str) -> Dict[str, object]:
    normalized = clean_social_text(text)
    token_set = set(tokenize(text))
    matched_phrases = sorted(
        phrase for phrase in SENTIMENT_QUERY_PHRASES
        if phrase in normalized
    )

    matched_terms = []
    for term in SENTIMENT_QUERY_TERMS:
        normalized_term = clean_social_text(term)
        if " " in normalized_term:
            if normalized_term in normalized:
                matched_terms.append(term)
        else:
            if normalized_term in token_set:
                matched_terms.append(term)

    strong_matches = sorted(set(matched_terms) & STRONG_SENTIMENT_TERMS)
    should_run = bool(matched_phrases) or bool(strong_matches) or len(matched_terms) >= 2
    reason_parts: List[str] = []
    if matched_phrases:
        reason_parts.append("matched sentiment phrases: " + ", ".join(matched_phrases[:4]))
    if strong_matches:
        reason_parts.append("matched strong sentiment terms: " + ", ".join(strong_matches[:6]))
    if matched_terms:
        reason_parts.append("matched sentiment terms: " + ", ".join(sorted(matched_terms)[:6]))
    if not reason_parts:
        reason_parts.append("query is not explicitly about feedback, complaints, reviews, or social-media reactions")

    return {
        "should_run": should_run,
        "matched_phrases": matched_phrases,
        "matched_strong_terms": strong_matches,
        "matched_terms": sorted(set(matched_terms)),
        "reason": "; ".join(reason_parts),
    }


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_keyword_lookup(keyword_profiles: Dict[str, object]) -> Dict[str, set[str]]:
    lookup: Dict[str, set[str]] = {}
    top_keywords = keyword_profiles.get("top_keywords_by_sentiment", {}) or {}
    for label, items in top_keywords.items():
        lookup[label] = {safe_text(item.get("value")).lower() for item in items if safe_text(item.get("value"))}
    return lookup


def label_from_score(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_text(text: str, summary_path: str = DEFAULT_SUMMARY_PATH, keywords_path: str = DEFAULT_KEYWORDS_PATH) -> Dict[str, object]:
    if not os.path.exists(summary_path) or not os.path.exists(keywords_path):
        raise FileNotFoundError(
            "Sentiment assets not found. Please run: python sentiment_assets/build_sentiment_assets.py"
        )

    summary = load_json(summary_path)
    keyword_profiles = load_json(keywords_path)
    keyword_lookup = build_keyword_lookup(keyword_profiles)
    analyzer = SentimentIntensityAnalyzer()

    cleaned_text = clean_social_text(text)
    tokens = tokenize(text)
    vader_scores = analyzer.polarity_scores(cleaned_text)

    matched_keywords = {
        label: sorted(set(tokens) & keywords)
        for label, keywords in keyword_lookup.items()
    }
    keyword_bias = len(matched_keywords.get("Positive", [])) - len(matched_keywords.get("Negative", []))
    combined_score = float(vader_scores["compound"]) + min(0.2, max(-0.2, keyword_bias * 0.04))
    predicted_sentiment = label_from_score(combined_score)

    reasons: List[str] = []
    if abs(float(vader_scores["compound"])) >= 0.35:
        reasons.append(f"VADER compound score is {round(float(vader_scores['compound']), 4)}")
    if matched_keywords.get("Positive"):
        reasons.append("matched positive keywords: " + ", ".join(matched_keywords["Positive"][:5]))
    if matched_keywords.get("Negative"):
        reasons.append("matched negative keywords: " + ", ".join(matched_keywords["Negative"][:5]))
    if not reasons:
        reasons.append("prediction is mostly driven by general VADER polarity signals")

    return {
        "input_text": text,
        "clean_text": cleaned_text,
        "tokens": tokens,
        "predicted_sentiment": predicted_sentiment,
        "combined_score": round(combined_score, 4),
        "vader_scores": {key: round(float(value), 4) for key, value in vader_scores.items()},
        "matched_keywords": matched_keywords,
        "reasoning": reasons,
        "dataset_context": {
            "row_count": summary.get("row_count", 0),
            "dataset_sentiment_distribution": summary.get("dataset_sentiment_distribution", {}),
            "top_platforms": (summary.get("top_platforms", []) or [])[:5],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Project 8 lightweight sentiment engine.")
    parser.add_argument("--text", required=True, help="Text to analyze")
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--keywords-path", default=DEFAULT_KEYWORDS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trigger = should_analyze_sentiment(args.text)
    payload = analyze_text(
        text=args.text,
        summary_path=args.summary_path,
        keywords_path=args.keywords_path,
    )
    payload["trigger"] = trigger
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
