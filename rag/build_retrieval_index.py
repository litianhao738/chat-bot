#!/usr/bin/env python3
"""
Build a higher-quality local retrieval index for Project 8 objective 3.

Compared with the first scaffold version, this index:
- builds a dedicated retrieval text instead of using raw chunk text only
- pulls richer fields from the cleaned company KB when available
- stores field-level tokens for later reranking
- computes a lightweight quality score so low-information chunks can be downweighted
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

DEFAULT_KB_PATH = "project8_kb/knowledge_base_chunks.jsonl"
DEFAULT_COMPANY_KB_PATH = "project8_kb/cleaned_company_kb.jsonl"
DEFAULT_EXTRA_KB_PATHS = [
    "external_sources/processed/hk_official_kb_chunks.jsonl",
    "external_sources/processed/business_guide_kb_chunks.jsonl",
]
DEFAULT_EXTRA_COMPANY_KB_PATHS = [
    "external_sources/processed/hk_official_kb.jsonl",
    "external_sources/processed/business_guide_kb.jsonl",
]
DEFAULT_OUTPUT_DIR = "rag/output"
DEFAULT_INDEX_PATH = "rag/output/retrieval_index.pkl"
DEFAULT_SUMMARY_PATH = "rag/output/retrieval_index_summary.json"


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def tokenize(text: str) -> List[str]:
    text = safe_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_keywords(keywords: object) -> List[str]:
    if not isinstance(keywords, list):
        return []
    return [safe_text(keyword) for keyword in keywords if safe_text(keyword)]


def strip_structured_markers(text: str) -> str:
    text = safe_text(text)
    text = re.sub(r"\b(?:Company|Tagline|Description|Keywords|Topic):", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_company_lookup(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        safe_text(row.get("company_id")): row
        for row in rows
        if safe_text(row.get("company_id"))
    }


def build_retrieval_fields(
    chunk_row: Dict[str, object],
    company_row: Dict[str, object] | None,
) -> Dict[str, object]:
    company_row = company_row or {}

    company_name = safe_text(company_row.get("canonical_name")) or safe_text(chunk_row.get("company_name"))
    topic_label = safe_text(chunk_row.get("topic_label")) or safe_text(company_row.get("topic_label")) or "unknown"
    keywords = normalize_keywords(chunk_row.get("keywords")) or normalize_keywords(company_row.get("top_keywords"))
    tagline = safe_text(company_row.get("canonical_tagline"))
    description = safe_text(company_row.get("description"))
    raw_text = strip_structured_markers(safe_text(chunk_row.get("text")))
    aliases = company_row.get("aliases", []) if isinstance(company_row.get("aliases"), list) else []
    aliases = [safe_text(alias) for alias in aliases if safe_text(alias)][:3]

    if not description and raw_text:
        description = raw_text

    company_tokens = tokenize(company_name)
    alias_tokens = tokenize(" ".join(aliases))
    topic_tokens = tokenize(topic_label)
    tagline_tokens = tokenize(tagline)
    description_tokens = tokenize(description)[:180]
    keyword_tokens = tokenize(" ".join(keywords))
    raw_text_tokens = tokenize(raw_text)[:140]

    weighted_tokens: List[str] = []
    weighted_tokens.extend(company_tokens * 4)
    weighted_tokens.extend(alias_tokens * 2)
    weighted_tokens.extend(topic_tokens * 3)
    weighted_tokens.extend(tagline_tokens * 2)
    weighted_tokens.extend(keyword_tokens * 3)
    weighted_tokens.extend(description_tokens)
    weighted_tokens.extend(raw_text_tokens[:80])

    retrieval_text = " ".join(
        part
        for part in [
            company_name,
            " ".join(aliases),
            topic_label,
            " ".join(keywords),
            tagline,
            description,
        ]
        if part
    ).strip()

    description_len = len(description_tokens)
    keyword_count = len(keywords)
    retrieval_len = len(weighted_tokens)

    quality_score = 0.25
    quality_score += min(description_len, 120) / 180.0
    quality_score += min(keyword_count, 8) / 20.0
    quality_score += 0.12 if tagline_tokens else 0.0
    quality_score += 0.08 if len(company_tokens) >= 1 else 0.0
    if description_len < 8:
        quality_score *= 0.65
    if retrieval_len < 20:
        quality_score *= 0.75
    quality_score = max(0.15, min(1.25, quality_score))

    is_low_info = description_len < 8 and keyword_count < 3

    return {
        "company_name": company_name,
        "topic_label": topic_label,
        "keywords": keywords,
        "tagline": tagline,
        "description": description,
        "aliases": aliases,
        "retrieval_text": retrieval_text,
        "retrieval_tokens": weighted_tokens,
        "company_tokens": company_tokens,
        "topic_tokens": topic_tokens,
        "keyword_tokens": keyword_tokens,
        "description_tokens": description_tokens,
        "quality_score": round(quality_score, 4),
        "is_low_info": is_low_info,
    }


def build_index(rows: List[Dict[str, object]], company_lookup: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    doc_freq = Counter()
    topic_to_doc_ids: Dict[str, List[int]] = defaultdict(list)
    source_family_to_doc_ids: Dict[str, List[int]] = defaultdict(list)
    jurisdiction_to_doc_ids: Dict[str, List[int]] = defaultdict(list)
    source_family_counts: Counter[str] = Counter()
    documents = []
    low_info_count = 0

    for idx, row in enumerate(rows):
        company_id = safe_text(row.get("company_id"))
        retrieval_fields = build_retrieval_fields(row, company_lookup.get(company_id))
        text = safe_text(row.get("text"))
        tokens = retrieval_fields["retrieval_tokens"]
        tf = Counter(tokens)
        for term in set(tokens):
            doc_freq[term] += 1

        topic_label = retrieval_fields["topic_label"]
        source_family = safe_text(row.get("source_family")) or "yc"
        jurisdiction = safe_text(row.get("jurisdiction")) or ("hong_kong" if source_family == "hk_official" else "global")
        topic_to_doc_ids[topic_label].append(idx)
        source_family_to_doc_ids[source_family].append(idx)
        jurisdiction_to_doc_ids[jurisdiction].append(idx)
        source_family_counts[source_family] += 1
        if retrieval_fields["is_low_info"]:
            low_info_count += 1

        documents.append(
            {
                "doc_id": idx,
                "chunk_id": safe_text(row.get("chunk_id")),
                "company_id": company_id,
                "company_name": retrieval_fields["company_name"],
                "topic_id": row.get("topic_id"),
                "topic_label": topic_label,
                "url": safe_text(row.get("url")),
                "website": safe_text(row.get("website")),
                "source": safe_text(row.get("source")),
                "source_family": source_family,
                "source_name": safe_text(row.get("source_name")),
                "trust_tier": safe_text(row.get("trust_tier")),
                "jurisdiction": jurisdiction,
                "business_stage": safe_text(row.get("business_stage")),
                "category": safe_text(row.get("category")),
                "keywords": retrieval_fields["keywords"],
                "tagline": retrieval_fields["tagline"],
                "description": retrieval_fields["description"],
                "aliases": retrieval_fields["aliases"],
                "text": text,
                "retrieval_text": retrieval_fields["retrieval_text"],
                "tf": dict(tf),
                "length": sum(tf.values()),
                "company_tokens": retrieval_fields["company_tokens"],
                "topic_tokens": retrieval_fields["topic_tokens"],
                "keyword_tokens": retrieval_fields["keyword_tokens"],
                "description_tokens": retrieval_fields["description_tokens"],
                "quality_score": retrieval_fields["quality_score"],
                "is_low_info": retrieval_fields["is_low_info"],
            }
        )

    n_docs = len(documents)
    idf = {
        term: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for term, freq in doc_freq.items()
    }

    return {
        "n_docs": n_docs,
        "idf": idf,
        "topic_to_doc_ids": dict(topic_to_doc_ids),
        "source_family_to_doc_ids": dict(source_family_to_doc_ids),
        "jurisdiction_to_doc_ids": dict(jurisdiction_to_doc_ids),
        "source_family_counts": dict(source_family_counts),
        "documents": documents,
        "low_info_count": low_info_count,
    }


def ensure_output_dir(path: str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_pickle(path: str, payload: object) -> None:
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project 8 local retrieval index.")
    parser.add_argument("--kb-path", default=DEFAULT_KB_PATH, help="Path to knowledge_base_chunks.jsonl")
    parser.add_argument(
        "--company-kb-path",
        default=DEFAULT_COMPANY_KB_PATH,
        help="Path to cleaned_company_kb.jsonl for richer retrieval fields",
    )
    parser.add_argument(
        "--extra-kb-paths",
        nargs="*",
        default=DEFAULT_EXTRA_KB_PATHS,
        help="Optional extra chunk JSONL files to merge into the retrieval index",
    )
    parser.add_argument(
        "--extra-company-kb-paths",
        nargs="*",
        default=DEFAULT_EXTRA_COMPANY_KB_PATHS,
        help="Optional extra page-level KB JSONL files for richer retrieval metadata",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.kb_path):
        raise SystemExit(f"Knowledge base chunks file not found: {args.kb_path}")

    ensure_output_dir(args.output_dir)
    rows = read_jsonl(args.kb_path)
    used_extra_kb_paths = []
    for extra_path in args.extra_kb_paths:
        if os.path.exists(extra_path):
            rows.extend(read_jsonl(extra_path))
            used_extra_kb_paths.append(extra_path)
    company_lookup = {}
    if os.path.exists(args.company_kb_path):
        company_lookup = build_company_lookup(read_jsonl(args.company_kb_path))
    used_extra_company_kb_paths = []
    for extra_company_path in args.extra_company_kb_paths:
        if os.path.exists(extra_company_path):
            company_lookup.update(build_company_lookup(read_jsonl(extra_company_path)))
            used_extra_company_kb_paths.append(extra_company_path)

    index = build_index(rows, company_lookup)

    index_path = os.path.join(args.output_dir, Path(DEFAULT_INDEX_PATH).name)
    summary_path = os.path.join(args.output_dir, Path(DEFAULT_SUMMARY_PATH).name)

    save_pickle(index_path, index)
    save_json(
        summary_path,
        {
            "kb_path": args.kb_path,
            "company_kb_path": args.company_kb_path if os.path.exists(args.company_kb_path) else None,
            "extra_kb_paths": used_extra_kb_paths,
            "extra_company_kb_paths": used_extra_company_kb_paths,
            "n_docs": index["n_docs"],
            "n_topics": len(index["topic_to_doc_ids"]),
            "low_info_count": index["low_info_count"],
            "source_family_counts": index["source_family_counts"],
            "output_files": {
                "index_path": index_path,
                "summary_path": summary_path,
            },
        },
    )

    print(json.dumps({"index_path": index_path, "summary_path": summary_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
