#!/usr/bin/env python3
"""
Build a Hong Kong official sub-knowledge-base from Companies Registry pages.

Outputs:
- external_sources/processed/hk_official_kb.jsonl
- external_sources/processed/hk_official_kb_chunks.jsonl
- external_sources/processed/hk_official_kb_summary.json
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_MANIFEST_PATH = "external_sources/source_manifest.json"
DEFAULT_OUTPUT_DIR = "external_sources/processed"
DEFAULT_KB_PATH = "external_sources/processed/hk_official_kb.jsonl"
DEFAULT_CHUNKS_PATH = "external_sources/processed/hk_official_kb_chunks.jsonl"
DEFAULT_SUMMARY_PATH = "external_sources/processed/hk_official_kb_summary.json"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
}

STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "by", "can",
    "company", "companies", "companies registry", "company registry", "cr", "do", "for", "from", "have", "how",
    "if", "in", "into", "is", "it", "its", "local", "may", "more", "new", "of", "on", "or", "other", "our",
    "should", "that", "the", "their", "them", "there", "these", "this", "to", "under", "use", "using", "what",
    "when", "which", "who", "will", "with", "you", "your", "hong", "kong", "registry", "page", "pages", "faq",
    "electronic", "services", "service", "portal", "information", "limited"
}


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slugify(text: str) -> str:
    text = safe_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def read_manifest(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_html(url: str) -> str:
    response = requests.get(url, timeout=40, headers=REQUEST_HEADERS)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding or "utf-8"
    return response.text


def extract_main_html(page_html: str) -> str:
    match = re.search(r"<main\b[^>]*>(.*)</main>", page_html, re.I | re.S)
    return match.group(1) if match else page_html


def extract_title(page_html: str) -> str:
    h1_match = re.search(r"<h1\b[^>]*>(.*?)</h1>", page_html, re.I | re.S)
    if h1_match:
        title = strip_tags(h1_match.group(1))
        if title:
            return title
    title_match = re.search(r"<title>(.*?)</title>", page_html, re.I | re.S)
    if title_match:
        title = strip_tags(title_match.group(1))
        title = re.sub(r"^Companies Registry\s*-\s*", "", title, flags=re.I)
        if title:
            return title
    return "Companies Registry Page"


def strip_tags(fragment: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", fragment)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<noscript.*?>.*?</noscript>", " ", text)
    text = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|li|ul|ol|h1|h2|h3|h4|h5|h6|table|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def normalize_line(line: str) -> str:
    line = safe_text(line)
    line = re.sub(r"\s+", " ", line)
    line = line.replace("�", " ")
    line = line.replace("鈥淯", "\"").replace("鈥?", "\"").replace("鈥檚", "'s").replace("鈥", "\"")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -:")


def extract_text(page_html: str) -> str:
    main_html = extract_main_html(page_html)
    text = strip_tags(main_html)
    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line and len(line) > 2]
    filtered: List[str] = []
    seen_recent = set()
    for line in lines:
        lowered = line.lower()
        if lowered in {"skip to content", "back to top", "breadcrumb", "search"}:
            continue
        if len(line) < 4:
            continue
        if any(
            noise in lowered
            for noise in [
                "the detail of this page",
                "play / pause the auto play",
                "previous next",
                "go to slide",
                "collapse all expand all",
                "read more",
                "enter",
                "quick links",
                "all news",
            ]
        ):
            continue
        if lowered in seen_recent:
            continue
        seen_recent.add(lowered)
        filtered.append(line)
    return "\n".join(filtered)


def split_sentences(text: str, limit: int = 3) -> str:
    parts = re.split(r"(?<=[.!?])\s+", safe_text(text))
    parts = [part.strip() for part in parts if part.strip()]
    return " ".join(parts[:limit])


def tokenize(text: str) -> List[str]:
    text = safe_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if token]


def top_keywords_from_text(text: str, limit: int = 12) -> List[str]:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 2]
    unigram_counts = Counter(tokens)

    bigrams = []
    for left, right in zip(tokens, tokens[1:]):
        if left in STOPWORDS or right in STOPWORDS:
            continue
        bigrams.append(f"{left} {right}")
    bigram_counts = Counter(bigrams)

    merged: List[Tuple[str, int]] = []
    merged.extend(unigram_counts.most_common(limit * 3))
    merged.extend([(term, score) for term, score in bigram_counts.most_common(limit * 2) if score >= 2])
    merged.sort(key=lambda item: (-item[1], item[0]))

    keywords: List[str] = []
    seen = set()
    for term, _score in merged:
        if term in seen:
            continue
        seen.add(term)
        keywords.append(term)
        if len(keywords) >= limit:
            break
    return keywords


def chunk_text(text: str, max_chars: int = 1200, min_chars: int = 300) -> List[str]:
    paragraphs = [normalize_line(part) for part in text.splitlines() if normalize_line(part)]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append(current.strip())
            current = paragraph
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars:
            merged[-1] = f"{merged[-1]}\n{chunk}".strip()
        else:
            merged.append(chunk)
    return merged or [normalize_line(text)]


def build_record(page: Dict[str, object], source_meta: Dict[str, object]) -> Dict[str, object]:
    url = safe_text(page["url"])
    page_html = fetch_html(url)
    title = extract_title(page_html)
    extracted_text = extract_text(page_html)
    description = split_sentences(extracted_text, limit=4)
    top_keywords = top_keywords_from_text(f"{title}\n{extracted_text}")
    website = source_meta.get("website", "")
    domain = urlparse(url).netloc

    merged_text = " | ".join(
        part for part in [title, safe_text(page.get("category")), description, url] if part
    )

    return {
        "company_id": f"hk-{slugify(page['page_id'])}",
        "canonical_name": title,
        "aliases": [safe_text(alias) for alias in page.get("aliases", []) if safe_text(alias)],
        "website": website,
        "url": url,
        "raw_company_name": title,
        "raw_one_liner": safe_text(page.get("category")),
        "description": description,
        "canonical_tagline": safe_text(page.get("business_stage")).replace("_", " "),
        "merged_text": merged_text,
        "clean_text": " ".join(tokenize(extracted_text)),
        "top_keywords": top_keywords,
        "topic_id": page.get("topic_id"),
        "topic_label": safe_text(page.get("topic_label")),
        "source_family": "hk_official",
        "source_name": safe_text(source_meta.get("source_name")),
        "trust_tier": safe_text(source_meta.get("trust_tier")),
        "jurisdiction": safe_text(source_meta.get("jurisdiction")),
        "business_stage": safe_text(page.get("business_stage")),
        "category": safe_text(page.get("category")),
        "content_type": "official_webpage",
        "domain": domain,
        "full_text": extracted_text,
    }


def build_chunk_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    chunks = chunk_text(record["full_text"])
    rows: List[Dict[str, object]] = []
    base_keywords = record.get("top_keywords", [])[:12]
    for idx, chunk in enumerate(chunks):
        chunk_keywords = top_keywords_from_text(chunk, limit=10) or list(base_keywords)
        text = (
            f"Source: {record['source_name']}. Page: {record['canonical_name']}. "
            f"Category: {record['category']}. Business stage: {record['business_stage']}. "
            f"Description: {chunk}. Keywords: {', '.join(chunk_keywords)}. "
            f"Topic: {record['topic_label']}."
        )
        rows.append(
            {
                "chunk_id": f"{record['company_id']}__{idx}",
                "source": "hk_official_kb",
                "company_id": record["company_id"],
                "company_name": record["canonical_name"],
                "website": record["website"],
                "url": record["url"],
                "topic_id": record["topic_id"],
                "topic_label": record["topic_label"],
                "keywords": chunk_keywords,
                "text": text,
                "source_family": record["source_family"],
                "source_name": record["source_name"],
                "trust_tier": record["trust_tier"],
                "jurisdiction": record["jurisdiction"],
                "business_stage": record["business_stage"],
                "category": record["category"],
            }
        )
    return rows


def write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Hong Kong official sub-knowledge-base from Companies Registry pages.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kb-path", default=DEFAULT_KB_PATH)
    parser.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest_path)
    hk_manifest = manifest["hk_official"]
    pages = hk_manifest["pages"]

    records: List[Dict[str, object]] = []
    chunk_rows: List[Dict[str, object]] = []
    for page in pages:
        record = build_record(page, hk_manifest)
        records.append(record)
        chunk_rows.extend(build_chunk_rows(record))

    write_jsonl(args.kb_path, records)
    write_jsonl(args.chunks_path, chunk_rows)

    summary = {
        "source_family": "hk_official",
        "source_name": hk_manifest.get("source_name"),
        "website": hk_manifest.get("website"),
        "jurisdiction": hk_manifest.get("jurisdiction"),
        "trust_tier": hk_manifest.get("trust_tier"),
        "page_count": len(records),
        "chunk_count": len(chunk_rows),
        "pages": [
            {
                "company_id": record["company_id"],
                "title": record["canonical_name"],
                "url": record["url"],
                "topic_label": record["topic_label"],
                "business_stage": record["business_stage"],
            }
            for record in records
        ],
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
