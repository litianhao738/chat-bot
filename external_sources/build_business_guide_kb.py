#!/usr/bin/env python3
"""
Build a business-guide sub-knowledge-base from curated entrepreneurship pages.

Outputs:
- external_sources/processed/business_guide_kb.jsonl
- external_sources/processed/business_guide_kb_chunks.jsonl
- external_sources/processed/business_guide_kb_summary.json

Notes:
- Live fetching is attempted first.
- If a page blocks requests or returns unusable content, the builder falls back to
  the curated summary stored in external_sources/source_manifest.json.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import requests

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover - optional dependency at runtime
    curl_requests = None

DEFAULT_MANIFEST_PATH = "external_sources/source_manifest.json"
DEFAULT_OUTPUT_DIR = "external_sources/processed"
DEFAULT_KB_PATH = "external_sources/processed/business_guide_kb.jsonl"
DEFAULT_CHUNKS_PATH = "external_sources/processed/business_guide_kb_chunks.jsonl"
DEFAULT_SUMMARY_PATH = "external_sources/processed/business_guide_kb_summary.json"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0 Safari/537.36"
}

STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "been", "business",
    "but", "by", "can", "customers", "do", "for", "from", "get", "guide", "how", "if", "in", "into", "is",
    "it", "its", "more", "most", "need", "of", "on", "or", "out", "page", "retail", "should", "small",
    "startup", "store", "that", "the", "their", "them", "there", "these", "this", "to", "up", "use", "what",
    "when", "which", "with", "you", "your",
}

NOISE_SUBSTRINGS = {
    "advertising disclosure",
    "editor verified",
    "editorial guidelines",
    "business news daily earns commissions",
    "get free guidance",
    "join free",
    "search input",
    "small business resources",
    "best small business loans",
    "best payroll services",
    "best pos systems",
    "best crm software",
    "table of contents icon",
    "did you know",
    "key takeaway",
    "recommended",
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


def fetch_html(url: str) -> Tuple[str, str]:
    errors: List[str] = []

    try:
        response = requests.get(url, timeout=40, headers=REQUEST_HEADERS)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or response.encoding or "utf-8"
        return response.text, "requests"
    except Exception as exc:
        errors.append(f"requests={type(exc).__name__}: {exc}")

    if curl_requests is not None:
        try:
            response = curl_requests.get(url, impersonate="chrome136", timeout=60)
            response.raise_for_status()
            return response.text, "curl_cffi"
        except Exception as exc:
            errors.append(f"curl_cffi={type(exc).__name__}: {exc}")

    raise RuntimeError(" | ".join(errors) if errors else "No fetch backend available.")


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
    line = line.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", "\"").replace("\u201d", "\"")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" -:")


def extract_title(page_html: str, fallback_title: str) -> str:
    h1_match = re.search(r"<h1\b[^>]*>(.*?)</h1>", page_html, re.I | re.S)
    if h1_match:
        title = normalize_line(strip_tags(h1_match.group(1)))
        if title:
            return title
    title_match = re.search(r"<title>(.*?)</title>", page_html, re.I | re.S)
    if title_match:
        title = normalize_line(strip_tags(title_match.group(1)))
        title = re.sub(r"\s*\|\s*.+$", "", title)
        if title:
            return title
    return fallback_title


def clean_lines(page_html: str) -> List[str]:
    text = strip_tags(page_html)
    lines = [normalize_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line and len(line) > 2]
    cleaned: List[str] = []
    previous_line = ""
    for line in lines:
        lowered = line.lower()
        if any(noise in lowered for noise in NOISE_SUBSTRINGS):
            continue
        if lowered == previous_line:
            continue
        previous_line = lowered
        cleaned.append(line)
    return cleaned


def trim_to_article(lines: List[str], page: Dict[str, object], fallback_title: str) -> List[str]:
    start_markers = [safe_text(item).lower() for item in page.get("start_markers", []) if safe_text(item)]
    end_markers = [safe_text(item).lower() for item in page.get("end_markers", []) if safe_text(item)]
    if fallback_title:
        start_markers.append(fallback_title.lower())

    start_idx = 0
    matched_indices: List[int] = []
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(marker and marker in lowered for marker in start_markers):
            matched_indices.append(idx)
    if matched_indices:
        start_idx = matched_indices[-1]
    trimmed = lines[start_idx:]

    if end_markers:
        for idx, line in enumerate(trimmed):
            lowered = line.lower()
            if any(marker and marker in lowered for marker in end_markers):
                trimmed = trimmed[:idx]
                break

    filtered: List[str] = []
    for line in trimmed:
        lowered = line.lower()
        if len(line) < 5:
            continue
        if lowered.startswith("best ") and " for 20" in lowered:
            continue
        if lowered.startswith("read related"):
            continue
        filtered.append(line)

    return filtered[:120]


def extract_text(page_html: str, page: Dict[str, object]) -> str:
    lines = clean_lines(page_html)
    lines = trim_to_article(lines, page=page, fallback_title=safe_text(page.get("fallback_title")))
    return "\n".join(lines).strip()


def split_sentences(text: str, limit: int = 4) -> str:
    parts = re.split(r"(?<=[.!?])\s+", safe_text(text))
    parts = [part.strip() for part in parts if part.strip()]
    return " ".join(parts[:limit])


def tokenize(text: str) -> List[str]:
    text = safe_text(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return [token for token in text.split() if token]


def top_keywords_from_text(text: str, seed_keywords: List[str], limit: int = 12) -> List[str]:
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
    merged.extend((safe_text(keyword).lower(), limit + 3) for keyword in seed_keywords if safe_text(keyword))
    merged.sort(key=lambda item: (-item[1], item[0]))

    keywords: List[str] = []
    seen = set()
    for term, _score in merged:
        term = safe_text(term)
        if not term or term in seen:
            continue
        seen.add(term)
        keywords.append(term)
        if len(keywords) >= limit:
            break
    return keywords


def chunk_text(text: str, max_chars: int = 1000, min_chars: int = 260) -> List[str]:
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


def choose_content(page: Dict[str, object], page_html: str, fetch_error: str, fetch_backend: str) -> Tuple[str, str]:
    extracted = extract_text(page_html, page) if page_html else ""
    fallback = safe_text(page.get("fallback_summary"))
    if fetch_error:
        return fallback, "manifest_fallback"
    if len(tokenize(extracted)) < 80:
        return fallback or extracted, "manifest_fallback"
    return extracted, f"live_fetch_{fetch_backend}"


def build_record(page: Dict[str, object], manifest_meta: Dict[str, object]) -> Dict[str, object]:
    url = safe_text(page["url"])
    page_html = ""
    fetch_error = ""
    fetch_backend = ""
    try:
        page_html, fetch_backend = fetch_html(url)
    except Exception as exc:
        fetch_error = f"{type(exc).__name__}: {exc}"

    content_text, access_mode = choose_content(page, page_html, fetch_error, fetch_backend)
    title = extract_title(page_html, safe_text(page.get("fallback_title"))) if page_html else safe_text(page.get("fallback_title"))
    title = title or safe_text(page.get("fallback_title")) or safe_text(page.get("page_id"))
    description = split_sentences(content_text, limit=4)
    seed_keywords = [safe_text(item) for item in page.get("seed_keywords", []) if safe_text(item)]
    top_keywords = top_keywords_from_text(f"{title}\n{content_text}", seed_keywords=seed_keywords)
    website = safe_text(page.get("website"))
    domain = urlparse(url).netloc
    merged_text = " | ".join(
        part for part in [title, safe_text(page.get("category")), description, url] if part
    )

    return {
        "company_id": f"bg-{slugify(page['page_id'])}",
        "canonical_name": title,
        "aliases": [safe_text(alias) for alias in page.get("aliases", []) if safe_text(alias)],
        "website": website,
        "url": url,
        "raw_company_name": title,
        "raw_one_liner": safe_text(page.get("category")),
        "description": description,
        "canonical_tagline": safe_text(page.get("business_stage")).replace("_", " "),
        "merged_text": merged_text,
        "clean_text": " ".join(tokenize(content_text)),
        "top_keywords": top_keywords,
        "topic_id": page.get("topic_id"),
        "topic_label": safe_text(page.get("topic_label")),
        "source_family": safe_text(manifest_meta.get("source_family")) or "business_guide",
        "source_name": safe_text(page.get("source_name")),
        "trust_tier": safe_text(page.get("trust_tier")) or "editorial",
        "jurisdiction": safe_text(manifest_meta.get("default_jurisdiction")) or "global",
        "business_stage": safe_text(page.get("business_stage")),
        "category": safe_text(page.get("category")),
        "content_type": "business_guide_webpage" if access_mode.startswith("live_fetch") else "business_guide_curated_summary",
        "domain": domain,
        "full_text": content_text,
        "access_mode": access_mode,
        "fetch_backend": fetch_backend,
        "fetch_error": fetch_error,
    }


def build_chunk_rows(record: Dict[str, object]) -> List[Dict[str, object]]:
    chunks = chunk_text(record["full_text"])
    rows: List[Dict[str, object]] = []
    base_keywords = record.get("top_keywords", [])[:12]
    for idx, chunk in enumerate(chunks):
        chunk_keywords = top_keywords_from_text(chunk, seed_keywords=base_keywords, limit=10) or list(base_keywords)
        text = (
            f"Source: {record['source_name']}. Guide: {record['canonical_name']}. "
            f"Category: {record['category']}. Business stage: {record['business_stage']}. "
            f"Description: {chunk}. Keywords: {', '.join(chunk_keywords)}. "
            f"Topic: {record['topic_label']}."
        )
        rows.append(
            {
                "chunk_id": f"{record['company_id']}__{idx}",
                "source": "business_guide_kb",
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
    parser = argparse.ArgumentParser(description="Build the business-guide sub-knowledge-base from curated pages.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kb-path", default=DEFAULT_KB_PATH)
    parser.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest_path)
    business_manifest = manifest["business_guide"]
    pages = business_manifest["pages"]

    records: List[Dict[str, object]] = []
    chunk_rows: List[Dict[str, object]] = []
    for page in pages:
        record = build_record(page, business_manifest)
        records.append(record)
        chunk_rows.extend(build_chunk_rows(record))

    write_jsonl(args.kb_path, records)
    write_jsonl(args.chunks_path, chunk_rows)

    summary = {
        "source_family": "business_guide",
        "page_count": len(records),
        "chunk_count": len(chunk_rows),
        "live_fetch_count": sum(1 for record in records if safe_text(record.get("access_mode")).startswith("live_fetch")),
        "fallback_count": sum(1 for record in records if not safe_text(record.get("access_mode")).startswith("live_fetch")),
        "pages": [
            {
                "company_id": record["company_id"],
                "title": record["canonical_name"],
                "url": record["url"],
                "source_name": record["source_name"],
                "topic_label": record["topic_label"],
                "business_stage": record["business_stage"],
                "access_mode": record["access_mode"],
                "fetch_backend": record["fetch_backend"],
                "fetch_error": record["fetch_error"],
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
