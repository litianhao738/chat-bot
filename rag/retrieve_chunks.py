#!/usr/bin/env python3
"""
Retrieve the most relevant knowledge base chunks for a user query.

Pipeline:
- topic prediction from the existing classifier output
- multi-topic candidate recall
- query expansion
- lightweight reranking with quality-aware scoring
- JSON output for later answer generation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.build_retrieval_index import tokenize, safe_text

DEFAULT_INDEX_PATH = "rag/output/retrieval_index.pkl"
DEFAULT_TOPIC_CATALOG_PATH = "classification/output/topic_catalog.json"
DEFAULT_CLASSIFIER_PATH = "classification/output/inquiry_topic_classifier.pkl"
DEFAULT_OUTPUT_PATH = "rag/output/retrieval_results.json"

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "be", "business", "can", "company", "do", "for", "how", "i",
    "in", "is", "launch", "my", "of", "should", "start", "startup", "the", "to", "what",
    "with", "would", "you",
}

QUERY_EXPANSION = {
    "register": ["registration", "incorporation", "filing", "registry"],
    "registration": ["register", "incorporation", "filing", "registry"],
    "incorporation": ["register", "registration", "company name", "annual return"],
    "incorporate": ["register", "registration", "filing", "company name"],
    "company": ["registration", "incorporation", "annual return", "registry"],
    "compliance": ["annual return", "filing", "registry", "regulatory"],
    "legal": ["compliance", "registry", "filing", "registration"],
    "licence": ["license", "permit", "registration"],
    "license": ["licence", "permit", "registration"],
    "health": ["healthcare", "medical", "clinical", "patient", "provider"],
    "healthcare": ["medical", "clinical", "patient", "provider", "care"],
    "medical": ["healthcare", "clinical", "patient", "care"],
    "clinic": ["clinical", "provider", "patient", "care"],
    "fintech": ["financial", "payments", "banking", "compliance", "credit"],
    "financial": ["fintech", "payments", "banking", "capital", "credit"],
    "payment": ["payments", "fintech", "financial", "credit", "transaction"],
    "payments": ["payment", "fintech", "financial", "credit", "transaction"],
    "insurance": ["risk", "compliance", "coverage", "healthcare"],
    "education": ["students", "learning", "school", "teachers", "college"],
    "student": ["students", "education", "learning", "school"],
    "students": ["student", "education", "learning", "school"],
    "learning": ["education", "students", "teachers", "school"],
    "teacher": ["teachers", "education", "students", "school"],
    "teachers": ["teacher", "education", "students", "school"],
    "marketing": ["brand", "ads", "campaign", "social", "growth"],
    "brand": ["brands", "marketing", "commerce", "consumer"],
    "commerce": ["marketplace", "retail", "brands", "consumer", "sell"],
    "marketplace": ["commerce", "retail", "sell", "consumer", "brands"],
    "retail": ["commerce", "marketplace", "consumer", "stores", "inventory", "pricing", "location"],
    "store": ["retail", "stores", "foot traffic", "location", "inventory"],
    "stores": ["store", "retail", "location", "inventory", "operations"],
    "operations": ["inventory", "fulfillment", "logistics", "staffing", "vendors"],
    "inventory": ["operations", "stock", "supply", "warehouse", "fulfillment"],
    "pricing": ["price", "competition", "margin", "customers"],
    "competition": ["competitors", "market", "positioning", "pricing"],
    "customer": ["customers", "segment", "market", "pricing"],
    "customers": ["customer", "segment", "market", "pricing"],
    "market": ["customer", "customers", "competition", "pricing", "segment"],
    "funding": ["capital", "loan", "investors", "budget"],
    "plan": ["planning", "strategy", "market", "funding"],
    "supply": ["supply chain", "inventory", "logistics", "fulfillment"],
    "chain": ["supply chain", "inventory", "logistics", "warehouse"],
    "agent": ["agents", "automation", "assistant", "support"],
    "agents": ["agent", "automation", "assistant", "support"],
    "coding": ["code", "developer", "engineering", "api"],
    "code": ["coding", "developer", "engineering", "api"],
    "ai": ["model", "automation", "agents", "software"],
}

HONG_KONG_TERMS = {"hong", "kong", "hong kong", "hk"}
OFFICIAL_QUERY_TERMS = {
    "register", "registration", "incorporation", "incorporate", "company name", "annual return", "filing",
    "compliance", "registry", "license", "licence", "permit", "business registration", "brn",
}
BUSINESS_GUIDE_QUERY_TERMS = {
    "retail", "store", "stores", "operations", "inventory", "supply", "chain", "pricing", "competition",
    "competitor", "customer", "customers", "market", "segment", "funding", "finance", "budget", "location",
    "vendors", "vendor", "staffing", "launch", "plan", "planning", "profit", "profitability",
}
BUSINESS_GUIDE_QUERY_PHRASES = {
    "how to start a business",
    "start a business",
    "open a retail store",
    "retail store",
    "target customer",
    "target market",
    "market research",
    "customer segment",
    "supply chain",
    "business plan",
    "go to market",
    "pricing strategy",
    "competitive advantage",
}
YC_QUERY_TERMS = {
    "example", "examples", "similar", "benchmark", "benchmarks", "compare", "comparison", "pattern", "patterns",
    "case", "cases", "yc", "startup example", "startup examples",
}


def load_pickle(path: str) -> object:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_topic_catalog(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["topics"]


def predict_topic(query: str, classifier_path: str, topic_catalog_path: str) -> Tuple[str, List[Dict[str, object]]]:
    if os.path.exists(classifier_path):
        try:
            from classification.classify_inquiry_topics import TopicInquiryClassifier  # noqa: F401

            classifier = load_pickle(classifier_path)
            scores = classifier.predict_proba([query])[0]
            ranking = sorted(zip(classifier.classes_, scores), key=lambda item: item[1], reverse=True)
            return ranking[0][0], [{"topic_label": label, "score": round(score, 4)} for label, score in ranking[:3]]
        except Exception:
            pass

    topics = load_topic_catalog(topic_catalog_path)
    query_tokens = set(tokenize(query))
    scored = []
    for topic in topics:
        keywords = {safe_text(x).lower() for x in topic.get("top_keywords", [])}
        overlap = len(query_tokens & keywords)
        scored.append((safe_text(topic.get("topic_label")), overlap))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_topic = scored[0][0] if scored else "unknown"
    return best_topic, [{"topic_label": label, "score": float(score)} for label, score in scored[:3]]


def normalize_topic_ranking(topic_ranking: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not topic_ranking:
        return []
    max_score = max(float(item.get("score", 0.0)) for item in topic_ranking) or 1.0
    normalized = []
    for item in topic_ranking:
        normalized.append(
            {
                "topic_label": safe_text(item.get("topic_label")),
                "score": round(float(item.get("score", 0.0)), 4),
                "normalized_score": round(float(item.get("score", 0.0)) / max_score, 4),
            }
        )
    return normalized


def filtered_query_tokens(query: str) -> List[str]:
    return [token for token in tokenize(query) if token not in QUERY_STOPWORDS]


def build_expanded_query_tokens(
    query: str,
    topic_ranking: List[Dict[str, object]],
    topic_catalog: List[Dict[str, object]],
    topic_candidates: int,
) -> Dict[str, List[str]]:
    original = filtered_query_tokens(query)
    expanded: List[str] = []

    for token in original:
        expanded.extend(QUERY_EXPANSION.get(token, []))

    catalog_lookup = {
        safe_text(topic.get("topic_label")): topic
        for topic in topic_catalog
    }
    for topic_info in topic_ranking[:topic_candidates]:
        topic_meta = catalog_lookup.get(safe_text(topic_info.get("topic_label")))
        if not topic_meta:
            continue
        for keyword in topic_meta.get("top_keywords", [])[:4]:
            expanded.extend(tokenize(safe_text(keyword)))

    expanded = [token for token in expanded if token not in QUERY_STOPWORDS]

    weighted_tokens: List[str] = []
    weighted_tokens.extend(original * 3)
    weighted_tokens.extend(expanded)

    return {
        "original_tokens": original,
        "expanded_tokens": expanded,
        "weighted_tokens": weighted_tokens,
    }


def detect_query_facets(query: str, expanded_query: Dict[str, List[str]]) -> Dict[str, object]:
    normalized_query = safe_text(query).lower()
    original_terms = set(expanded_query["original_tokens"])
    raw_terms = set(tokenize(query))
    asks_hong_kong = "hong kong" in normalized_query or " hk " in f" {normalized_query} " or any(term in original_terms for term in {"hong", "kong", "hk"})
    asks_official = any(term in normalized_query for term in OFFICIAL_QUERY_TERMS) or any(term in original_terms for term in {"register", "registration", "incorporation", "incorporate", "compliance", "license", "licence", "permit"})
    asks_business_guide = (
        any(phrase in normalized_query for phrase in BUSINESS_GUIDE_QUERY_PHRASES)
        or len(raw_terms & BUSINESS_GUIDE_QUERY_TERMS) >= 2
        or any(term in original_terms for term in {"retail", "store", "operations", "inventory", "pricing", "competition", "market", "customer", "funding"})
    )
    asks_operations = any(term in raw_terms for term in {"operations", "inventory", "supply", "chain", "warehouse", "fulfillment", "vendor", "vendors", "staffing"})
    asks_examples = any(term in normalized_query for term in YC_QUERY_TERMS) or any(term in raw_terms for term in {"example", "examples", "similar", "benchmark", "compare", "yc"})

    preferred_source_families: List[str] = []
    if asks_official:
        preferred_source_families.append("hk_official")
    if asks_business_guide:
        preferred_source_families.append("business_guide")
    if asks_examples:
        preferred_source_families.append("yc")

    return {
        "asks_hong_kong": asks_hong_kong,
        "asks_official": asks_official,
        "asks_business_guide": asks_business_guide,
        "asks_operations": asks_operations,
        "asks_examples": asks_examples,
        "preferred_source_families": preferred_source_families,
        "preferred_jurisdiction": "hong_kong" if asks_hong_kong else "",
    }


def make_query_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    weighted = {
        term: (1.0 + math.log(freq)) * idf.get(term, 0.0)
        for term, freq in tf.items()
        if term in idf
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in weighted.items()}


def dot_product(left: Dict[str, float], right: Dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def make_doc_vector(tf: Dict[str, int], idf: Dict[str, float]) -> Dict[str, float]:
    weighted = {
        term: (1.0 + math.log(freq)) * idf.get(term, 0.0)
        for term, freq in tf.items()
        if term in idf
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in weighted.items()}


def overlap_ratio(query_terms: set[str], doc_terms: List[str]) -> Tuple[float, List[str]]:
    if not query_terms or not doc_terms:
        return 0.0, []
    matched = sorted(query_terms & set(doc_terms))
    if not matched:
        return 0.0, []
    return len(matched) / max(1, len(query_terms)), matched[:6]


def diversify_results(
    scored: List[Dict[str, object]],
    top_k: int,
    query_facets: Dict[str, object],
) -> List[Dict[str, object]]:
    if len(scored) <= 1:
        return scored[:top_k]

    selected: List[Dict[str, object]] = []
    selected_ids = set()
    company_counts = Counter()
    url_counts = Counter()
    source_family_counts = Counter()
    category_counts = Counter()

    while len(selected) < min(top_k, len(scored)):
        best_idx = None
        best_adjusted = -1.0

        for idx, candidate in enumerate(scored):
            chunk_id = safe_text(candidate.get("chunk_id"))
            if chunk_id in selected_ids:
                continue

            adjusted = float(candidate.get("score", 0.0))
            company_id = safe_text(candidate.get("company_id"))
            url = safe_text(candidate.get("url"))
            source_family = safe_text(candidate.get("source_family")) or "yc"
            category = safe_text(candidate.get("category"))

            if company_counts[company_id]:
                adjusted *= 0.58 if source_family == "hk_official" else 0.72
            if url_counts[url]:
                adjusted *= 0.55
            if category and category_counts[category]:
                adjusted *= 0.88

            if query_facets.get("asks_hong_kong") and not query_facets.get("asks_official"):
                if source_family_counts["hk_official"] >= 1 and source_family == "hk_official":
                    adjusted *= 0.82
                if source_family_counts["yc"] == 0 and source_family == "yc":
                    adjusted *= 1.05

            if query_facets.get("asks_official"):
                if source_family == "hk_official" and company_counts[company_id] == 0:
                    adjusted *= 1.06

            if query_facets.get("asks_business_guide"):
                if source_family == "business_guide" and company_counts[company_id] == 0:
                    adjusted *= 1.05
                if source_family_counts["business_guide"] >= 2 and source_family == "business_guide":
                    adjusted *= 0.86
                if source_family == "yc" and not query_facets.get("asks_examples"):
                    adjusted *= 0.94

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_idx = idx

        if best_idx is None:
            break

        chosen = dict(scored[best_idx])
        chosen["diversified_score"] = round(best_adjusted, 4)
        selected.append(chosen)
        selected_ids.add(safe_text(chosen.get("chunk_id")))
        company_counts[safe_text(chosen.get("company_id"))] += 1
        url_counts[safe_text(chosen.get("url"))] += 1
        source_family_counts[safe_text(chosen.get("source_family")) or "yc"] += 1
        category = safe_text(chosen.get("category"))
        if category:
            category_counts[category] += 1

    return selected


def retrieve(
    query: str,
    index: Dict[str, object],
    topic_ranking: List[Dict[str, object]],
    topic_catalog: List[Dict[str, object]],
    top_k: int = 5,
    topic_candidates: int = 3,
) -> List[Dict[str, object]]:
    idf = index["idf"]
    normalized_topics = normalize_topic_ranking(topic_ranking)
    topic_prior = {
        item["topic_label"]: float(item["normalized_score"])
        for item in normalized_topics[:topic_candidates]
    }
    expanded_query = build_expanded_query_tokens(
        query=query,
        topic_ranking=normalized_topics,
        topic_catalog=topic_catalog,
        topic_candidates=topic_candidates,
    )
    query_facets = detect_query_facets(query, expanded_query)
    query_vector = make_query_vector(expanded_query["weighted_tokens"], idf)

    doc_id_set = set()
    for topic_info in normalized_topics[:topic_candidates]:
        doc_id_set.update(index["topic_to_doc_ids"].get(topic_info["topic_label"], []))
    for preferred_source in query_facets["preferred_source_families"]:
        doc_id_set.update(index.get("source_family_to_doc_ids", {}).get(preferred_source, []))
    preferred_jurisdiction = safe_text(query_facets.get("preferred_jurisdiction"))
    if preferred_jurisdiction:
        doc_id_set.update(index.get("jurisdiction_to_doc_ids", {}).get(preferred_jurisdiction, []))
    doc_ids = sorted(doc_id_set) if doc_id_set else list(range(len(index["documents"])))

    original_token_set = set(expanded_query["original_tokens"])
    expanded_token_set = set(expanded_query["expanded_tokens"]) | original_token_set

    scored = []
    for doc_id in doc_ids:
        doc = index["documents"][doc_id]
        if doc.get("is_low_info") and doc.get("quality_score", 0) < 0.35:
            continue

        doc_vector = make_doc_vector(doc["tf"], idf)
        coarse_score = dot_product(query_vector, doc_vector)

        keyword_overlap, matched_keywords = overlap_ratio(expanded_token_set, doc.get("keyword_tokens", []))
        topic_overlap, matched_topic_terms = overlap_ratio(expanded_token_set, doc.get("topic_tokens", []))
        company_overlap, matched_company_terms = overlap_ratio(original_token_set, doc.get("company_tokens", []))
        description_overlap, matched_description_terms = overlap_ratio(
            original_token_set,
            doc.get("description_tokens", []),
        )

        if coarse_score <= 0 and keyword_overlap == 0 and topic_overlap == 0 and company_overlap == 0:
            continue

        topic_match_score = topic_prior.get(doc["topic_label"], 0.0)
        quality_score = float(doc.get("quality_score", 0.5))
        quality_multiplier = 0.75 if doc.get("is_low_info") else 1.0
        source_family = safe_text(doc.get("source_family")) or "yc"
        jurisdiction = safe_text(doc.get("jurisdiction"))
        trust_tier = safe_text(doc.get("trust_tier"))

        source_bonus = 0.0
        if source_family == "hk_official":
            source_bonus += 0.02
            if query_facets["asks_official"]:
                source_bonus += 0.22
            if query_facets["asks_business_guide"] and not query_facets["asks_official"]:
                source_bonus -= 0.03
        if source_family == "business_guide":
            source_bonus += 0.02
            if query_facets["asks_business_guide"]:
                source_bonus += 0.18
            if query_facets["asks_operations"]:
                source_bonus += 0.07
        if source_family == "yc":
            if query_facets["asks_examples"]:
                source_bonus += 0.12
            if query_facets["asks_business_guide"] and not query_facets["asks_examples"]:
                source_bonus -= 0.04
        if jurisdiction == "hong_kong" and query_facets["asks_hong_kong"]:
            source_bonus += 0.03
        if trust_tier == "official":
            source_bonus += 0.03

        rerank_score = (
            coarse_score * 0.58
            + keyword_overlap * 0.16
            + topic_overlap * 0.10
            + description_overlap * 0.07
            + company_overlap * 0.04
            + topic_match_score * 0.05
            + source_bonus
        ) * quality_score * quality_multiplier

        if rerank_score > 0:
            reasons = []
            if topic_match_score > 0:
                reasons.append("matched predicted topic")
            if source_family == "hk_official":
                reasons.append("official Hong Kong source")
            if source_family == "business_guide":
                reasons.append("business guide source")
            if source_family == "yc" and query_facets["asks_examples"]:
                reasons.append("startup example source")
            if matched_keywords:
                reasons.append(f"matched keywords: {', '.join(matched_keywords[:4])}")
            if matched_topic_terms:
                reasons.append(f"matched topic terms: {', '.join(matched_topic_terms[:4])}")
            if matched_description_terms:
                reasons.append(f"matched content terms: {', '.join(matched_description_terms[:4])}")
            if quality_score >= 0.9:
                reasons.append("high-information description")

            scored.append(
                {
                    "score": round(rerank_score, 4),
                    "coarse_score": round(coarse_score, 4),
                    "topic_match_score": round(topic_match_score, 4),
                    "keyword_overlap": round(keyword_overlap, 4),
                    "quality_score": round(quality_score, 4),
                    "chunk_id": doc["chunk_id"],
                    "company_id": doc["company_id"],
                    "company_name": doc["company_name"],
                    "topic_label": doc["topic_label"],
                    "url": doc["url"],
                    "website": doc["website"],
                    "source": doc.get("source", ""),
                    "source_family": source_family,
                    "source_name": doc.get("source_name", ""),
                    "trust_tier": trust_tier,
                    "jurisdiction": jurisdiction,
                    "business_stage": doc.get("business_stage", ""),
                    "category": doc.get("category", ""),
                    "keywords": doc["keywords"],
                    "tagline": doc.get("tagline", ""),
                    "description": doc.get("description", ""),
                    "retrieval_text": doc.get("retrieval_text", ""),
                    "retrieval_reason": reasons or ["matched retrieval text"],
                    "text": doc["text"],
                }
            )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return diversify_results(scored, top_k=top_k, query_facets=query_facets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve Project 8 knowledge base chunks for a query.")
    parser.add_argument("--query", required=True, help="User enquiry")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH, help="Path to retrieval index pickle")
    parser.add_argument("--classifier-path", default=DEFAULT_CLASSIFIER_PATH, help="Path to topic classifier pickle")
    parser.add_argument("--topic-catalog-path", default=DEFAULT_TOPIC_CATALOG_PATH, help="Path to topic catalog json")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="Path to save retrieval results")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to return")
    parser.add_argument("--topic-candidates", type=int, default=3, help="How many predicted topics to search")
    return parser.parse_args()


def safe_print_json(payload: Dict[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2)
    print(text)


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
        args.query,
        index=index,
        topic_ranking=topic_ranking,
        topic_catalog=topic_catalog,
        top_k=args.top_k,
        topic_candidates=max(1, args.topic_candidates),
    )

    payload = {
        "query": args.query,
        "predicted_topic": best_topic,
        "topic_ranking": normalize_topic_ranking(topic_ranking),
        "query_facets": detect_query_facets(args.query, build_expanded_query_tokens(args.query, topic_ranking, topic_catalog, max(1, args.topic_candidates))),
        "topic_candidates": max(1, args.topic_candidates),
        "top_k": args.top_k,
        "matches": matches,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    safe_print_json(payload)


if __name__ == "__main__":
    main()
