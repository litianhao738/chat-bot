
#!/usr/bin/env python3
"""
Build Project 8 objective-1 keyword knowledge base from yc_detailed_data.jsonl.

What it does
------------
1. Load and clean YC company records
2. Repair noisy company/name/tagline fields with simple heuristics
3. Extract document-level keywords using TF-IDF n-grams
4. Build a global keyword catalog
5. Cluster companies into topical groups and derive taxonomy labels
6. Export knowledge-base-ready JSONL chunks for later RAG / chatbot use

Outputs
-------
- cleaned_company_kb.jsonl
- company_keywords.csv
- global_keywords.csv
- topic_taxonomy.json
- knowledge_base_chunks.jsonl
- summary.json
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

LOCAL_VENDOR = os.path.join(os.path.dirname(__file__), ".vendor")


def clear_scientific_modules() -> None:
    stale_modules = [
        name
        for name in sys.modules
        if name == "numpy" or name.startswith("numpy.") or name == "sklearn" or name.startswith("sklearn.")
    ]
    for name in stale_modules:
        sys.modules.pop(name, None)


def import_scientific_stack():
    try:
        import numpy as np  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # type: ignore
        from sklearn.preprocessing import normalize  # type: ignore

        return np, KMeans, TruncatedSVD, ENGLISH_STOP_WORDS, TfidfVectorizer, normalize
    except Exception as primary_exc:
        if os.path.isdir(LOCAL_VENDOR) and LOCAL_VENDOR not in sys.path:
            clear_scientific_modules()
            sys.path.insert(0, LOCAL_VENDOR)
            try:
                import numpy as np  # type: ignore
                from sklearn.cluster import KMeans  # type: ignore
                from sklearn.decomposition import TruncatedSVD  # type: ignore
                from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer  # type: ignore
                from sklearn.preprocessing import normalize  # type: ignore

                return np, KMeans, TruncatedSVD, ENGLISH_STOP_WORDS, TfidfVectorizer, normalize
            except Exception as vendor_exc:
                raise SystemExit(
                    "Failed to import numpy / scikit-learn for keyword building.\n"
                    "The project first tried the active Python environment, then the local .vendor fallback.\n"
                    "Please run:\n"
                    "  python -m pip install -r requirements.txt\n"
                    f"Primary import error: {primary_exc}\n"
                    f"Vendor fallback error: {vendor_exc}"
                ) from vendor_exc
        raise SystemExit(
            "Missing dependency for keyword builder.\n"
            "Please run:\n"
            "  python -m pip install -r requirements.txt\n"
            f"Import error: {primary_exc}"
        ) from primary_exc


np, KMeans, TruncatedSVD, ENGLISH_STOP_WORDS, TfidfVectorizer, normalize = import_scientific_stack()


GENERIC_STOPWORDS = {
    "startup", "startups", "company", "companies", "team", "teams", "business", "businesses",
    "platform", "platforms", "product", "products", "customers", "customer", "helps", "help",
    "build", "building", "built", "developer", "developers", "tool", "tools", "software",
    "service", "services", "solution", "solutions", "use", "using", "users", "make", "makes",
    "making", "used", "new", "next", "best", "better", "way", "free", "ai", "data", "work",
    "works", "working", "founded", "founder", "founders", "today", "future", "modern",
    "enable", "enables", "enabled", "empower", "empowers", "powered", "power", "automation",
    "automate", "automates", "automated", "real", "time", "used", "across", "every", "one",
    "two", "three", "first", "our", "we", "us", "you", "your", "they", "them", "like", "world",
    "based", "create", "creates", "created", "easy", "end", "people", "online", "mobile",
    "technology", "technologies", "app", "apps", "faster", "fast", "management"
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | GENERIC_STOPWORDS

PERSON_TOKENS = {
    "raj", "jason", "steven", "nicole", "john", "michael", "david", "daniel", "andrew", "alex",
    "james", "kevin", "ryan", "emma", "sarah", "li", "chen", "wang", "zhang", "fan", "ma",
    "kim", "lee", "patel", "singh", "kadiyala", "bartel"
}

BAD_NAME_EXACT = {
    "problem", "the problem", "solution", "the solution",
    "tldr", "tl;dr", "tldr;", "tl;dr;", "tl/dr",
    "team", "unknown company",
}

MARKETING_PREFIXES = {
    "introducing", "introduce", "announcing", "announce", "launching", "launch",
    "meet", "presenting", "present", "discover",
}

DEFAULT_INPUT_CANDIDATES = [
    "data/raw/yc_detailed_data.repaired.v2.jsonl",
    "data/raw/yc_detailed_data.repaired.jsonl",
    "data/raw/yc_detailed_data.jsonl",
]


@dataclass
class CompanyRecord:
    company_id: str
    canonical_name: str
    aliases: List[str]
    website: str
    url: str
    raw_company_name: str
    raw_one_liner: str
    description: str
    canonical_tagline: str
    merged_text: str
    clean_text: str
    top_keywords: List[str]
    topic_id: int = -1
    topic_label: str = ""


def safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def slug_from_url(url: str) -> str:
    m = re.search(r"/companies/([^/?#]+)", url or "")
    if m:
        return m.group(1).strip().lower()
    return ""


def domain_from_website(url: str) -> str:
    url = url or ""
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0]
    url = re.sub(r"^www\.", "", url)
    return url.lower().strip()


def slug_to_name(slug: str) -> str:
    if not slug:
        return ""
    pieces = [p for p in re.split(r"[-_]+", slug) if p]
    return " ".join(piece.upper() if piece.isupper() else piece.capitalize() for piece in pieces)


def looks_like_person_name(s: str) -> bool:
    s = safe_text(s)
    if not s:
        return False
    if ":" in s or "." in s or len(s.split()) < 2 or len(s.split()) > 4:
        return False
    tokens = [re.sub(r"[^A-Za-z]", "", t).lower() for t in s.split()]
    if not all(tokens):
        return False
    title_like = all(t[0].isalpha() for t in tokens)
    knownish = sum(t in PERSON_TOKENS for t in tokens) >= 1
    # Very simple fallback heuristic
    return title_like and (knownish or all(len(t) > 1 for t in tokens))


def looks_like_tagline(s: str) -> bool:
    s = safe_text(s)
    if not s:
        return False
    word_count = len(s.split())
    return (":" in s) or (word_count >= 5) or any(ch in s for ch in ".!?/")


def strip_leading_noise(s: str) -> str:
    s = html.unescape(safe_text(s))
    s = s.replace("—", " - ").replace("–", " - ")
    s = re.sub(r"^[^A-Za-z0-9]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_bad_name_candidate(s: str) -> bool:
    s = strip_leading_noise(s)
    if not s:
        return True

    lowered = s.lower().strip(" -:")
    if lowered in BAD_NAME_EXACT:
        return True
    if lowered.startswith("what is "):
        return True
    if lowered.startswith("tl;dr") or lowered.startswith("tldr"):
        return True
    if lowered.startswith("the problem") or lowered.startswith("the solution"):
        return True
    if lowered.split()[0] in MARKETING_PREFIXES:
        return True
    return False


def normalize_company_candidate(s: str) -> str:
    s = strip_leading_noise(s)
    if not s:
        return ""

    if re.match(r"(?i)^what is [^?]+\??$", s):
        return ""

    for sep in [":", " - "]:
        if sep in s:
            left = s.split(sep, 1)[0].strip()
            if left and not is_bad_name_candidate(left) and not looks_like_person_name(left) and len(left.split()) <= 6:
                return left

    if is_bad_name_candidate(s) or looks_like_person_name(s):
        return ""
    if len(s.split()) <= 6:
        return s
    return ""

def clean_visible_text(text: str) -> str:
    text = html.unescape(safe_text(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^A-Za-z0-9&/\-+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_nlp(text: str) -> str:
    text = clean_visible_text(text).lower()
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9+\-& ]+", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def choose_name_and_tagline(row: Dict[str, object]) -> Tuple[str, str, List[str]]:
    raw_name = safe_text(row.get("company_name"))
    raw_one = safe_text(row.get("one_liner"))
    desc = safe_text(row.get("description"))
    slug_name = slug_to_name(slug_from_url(safe_text(row.get("url"))))
    domain_name = slug_to_name(domain_from_website(safe_text(row.get("website"))).split(".")[0])

    aliases = []
    for val in [raw_name, raw_one, slug_name, domain_name]:
        val = safe_text(val)
        if val and val not in aliases:
            aliases.append(val)

    canonical_name = ""
    tagline = ""

    normalized_raw_name = normalize_company_candidate(raw_name)
    normalized_raw_one = normalize_company_candidate(raw_one)
    normalized_slug_name = normalize_company_candidate(slug_name) or slug_name
    normalized_domain_name = normalize_company_candidate(domain_name) or domain_name

    if ":" in strip_leading_noise(raw_name):
        left, right = strip_leading_noise(raw_name).split(":", 1)
        left, right = left.strip(), right.strip()
        if left and not is_bad_name_candidate(left) and not looks_like_person_name(left) and len(left.split()) <= 6:
            canonical_name = left
            tagline = right or desc
    if not canonical_name and normalized_raw_name:
        canonical_name = normalized_raw_name
    if not canonical_name and normalized_raw_one:
        canonical_name = normalized_raw_one
    if not canonical_name:
        canonical_name = normalized_slug_name or normalized_domain_name or slug_name or domain_name or "Unknown Company"

    if not tagline:
        candidates = [raw_name, raw_one, desc]
        for c in candidates:
            if c and c != canonical_name and looks_like_tagline(c) and not is_bad_name_candidate(c) and not looks_like_person_name(c):
                tagline = c
                break
    if not tagline:
        fallback_tagline_candidates = [desc, raw_one, raw_name]
        for candidate in fallback_tagline_candidates:
            if candidate and candidate != canonical_name and not is_bad_name_candidate(candidate):
                tagline = candidate
                break

    if ":" in tagline:
        parts = tagline.split(":", 1)
        if parts[0].strip().lower() == canonical_name.lower():
            tagline = parts[1].strip()

    canonical_name = re.sub(r"\s+", " ", strip_leading_noise(canonical_name)).strip(" -:")
    tagline = re.sub(r"\s+", " ", safe_text(tagline)).strip(" -:")
    return canonical_name, tagline, aliases


def load_records(path: str) -> List[CompanyRecord]:
    records: List[CompanyRecord] = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            url = safe_text(row.get("url"))
            company_id = slug_from_url(url) or f"company_{idx:05d}"
            if company_id in seen_ids:
                continue
            seen_ids.add(company_id)

            canonical_name, tagline, aliases = choose_name_and_tagline(row)
            description = safe_text(row.get("description"))
            website = safe_text(row.get("website"))
            raw_company_name = safe_text(row.get("company_name"))
            raw_one_liner = safe_text(row.get("one_liner"))

            merged_text = " | ".join(
                [x for x in [canonical_name, tagline, description, website, url] if x]
            )
            clean_text = normalize_for_nlp(" ".join([canonical_name, tagline, description]))

            records.append(
                CompanyRecord(
                    company_id=company_id,
                    canonical_name=canonical_name,
                    aliases=aliases,
                    website=website,
                    url=url,
                    raw_company_name=raw_company_name,
                    raw_one_liner=raw_one_liner,
                    description=description,
                    canonical_tagline=tagline,
                    merged_text=merged_text,
                    clean_text=clean_text,
                    top_keywords=[],
                )
            )
    return records


def tfidf_keywords(docs: List[str], top_k: int = 12, min_df: int = 2, max_features: int = 80000):
    vectorizer = TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=0.6,
        strip_accents="unicode",
        max_features=max_features,
    )
    X = vectorizer.fit_transform(docs)
    features = np.asarray(vectorizer.get_feature_names_out())

    per_doc_keywords: List[List[str]] = []
    per_doc_scores: List[List[Tuple[str, float]]] = []

    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            per_doc_keywords.append([])
            per_doc_scores.append([])
            continue
        order = np.argsort(row.data)[::-1][:top_k]
        kws = [(features[row.indices[j]], float(row.data[j])) for j in order]
        per_doc_scores.append(kws)
        per_doc_keywords.append([kw for kw, _ in kws])

    global_scores = np.asarray(X.sum(axis=0)).ravel()
    global_order = np.argsort(global_scores)[::-1]
    global_keywords = [(features[i], float(global_scores[i])) for i in global_order if global_scores[i] > 0]

    return vectorizer, X, per_doc_keywords, per_doc_scores, global_keywords


def cluster_topics(X, docs: List[str], n_clusters: int = 20, top_k: int = 10):
    n_docs = X.shape[0]
    n_clusters = max(5, min(n_clusters, n_docs // 50 if n_docs >= 250 else max(5, n_docs // 10)))
    n_clusters = min(n_clusters, n_docs)

    # Dense semantic-ish representation via SVD on TF-IDF
    svd_dim = int(max(10, min(100, X.shape[1] - 1 if X.shape[1] > 1 else 1)))
    X_svd = TruncatedSVD(n_components=svd_dim, random_state=42).fit_transform(X)
    X_svd = normalize(X_svd)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_svd)

    topic_docs = {i: [] for i in range(n_clusters)}
    topic_indices = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        topic_docs[label].append(docs[idx])
        topic_indices[label].append(idx)

    topic_taxonomy = []
    topic_label_lookup = {}

    for topic_id in range(n_clusters):
        agg_text = " ".join(topic_docs[topic_id])
        if not agg_text.strip():
            label = f"topic_{topic_id}"
            keywords = []
        else:
            vec = TfidfVectorizer(
                stop_words=list(STOPWORDS),
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8,
                strip_accents="unicode",
                max_features=25000,
            )
            Xt = vec.fit_transform(topic_docs[topic_id])
            feats = np.asarray(vec.get_feature_names_out())
            sums = np.asarray(Xt.sum(axis=0)).ravel()
            order = np.argsort(sums)[::-1]
            keywords = [feats[i] for i in order[:top_k] if sums[i] > 0]
            label = " / ".join(keywords[:3]) if keywords else f"topic_{topic_id}"

        topic_label_lookup[topic_id] = label
        topic_taxonomy.append({
            "topic_id": topic_id,
            "topic_label": label,
            "top_keywords": keywords,
            "num_companies": len(topic_indices[topic_id]),
        })

    return labels, topic_taxonomy, topic_label_lookup


def make_kb_chunks(records: List[CompanyRecord]) -> List[Dict[str, object]]:
    chunks = []
    for rec in records:
        chunk_text = (
            f"Company: {rec.canonical_name}. "
            f"Tagline: {rec.canonical_tagline}. "
            f"Description: {rec.description}. "
            f"Keywords: {', '.join(rec.top_keywords)}. "
            f"Topic: {rec.topic_label}."
        )
        chunks.append({
            "chunk_id": f"{rec.company_id}__0",
            "source": "yc_detailed_data",
            "company_id": rec.company_id,
            "company_name": rec.canonical_name,
            "website": rec.website,
            "url": rec.url,
            "topic_id": rec.topic_id,
            "topic_label": rec.topic_label,
            "keywords": rec.top_keywords,
            "text": chunk_text,
        })
    return chunks


def write_jsonl(path: str, items: Iterable[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pipeline(input_path: str, outdir: str, top_k_keywords: int, n_clusters: int) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)

    records = load_records(input_path)
    docs = [r.clean_text for r in records]

    vectorizer, X, per_doc_keywords, per_doc_scores, global_keywords = tfidf_keywords(
        docs,
        top_k=top_k_keywords,
        min_df=max(2, len(records) // 1000),
        max_features=90000,
    )

    labels, topic_taxonomy, topic_label_lookup = cluster_topics(X, docs, n_clusters=n_clusters, top_k=12)

    for rec, kws, label in zip(records, per_doc_keywords, labels):
        rec.top_keywords = kws
        rec.topic_id = int(label)
        rec.topic_label = topic_label_lookup[int(label)]

    kb_chunks = make_kb_chunks(records)

    cleaned_path = os.path.join(outdir, "cleaned_company_kb.jsonl")
    write_jsonl(cleaned_path, [asdict(r) for r in records])

    company_kw_csv = os.path.join(outdir, "company_keywords.csv")
    write_csv(
        company_kw_csv,
        [
            {
                "company_id": r.company_id,
                "canonical_name": r.canonical_name,
                "topic_id": r.topic_id,
                "topic_label": r.topic_label,
                "keywords": " | ".join(r.top_keywords),
                "website": r.website,
                "url": r.url,
            }
            for r in records
        ],
        ["company_id", "canonical_name", "topic_id", "topic_label", "keywords", "website", "url"],
    )

    global_kw_csv = os.path.join(outdir, "global_keywords.csv")
    write_csv(
        global_kw_csv,
        [{"rank": i + 1, "keyword": kw, "score": score} for i, (kw, score) in enumerate(global_keywords[:500])],
        ["rank", "keyword", "score"],
    )

    taxonomy_path = os.path.join(outdir, "topic_taxonomy.json")
    with open(taxonomy_path, "w", encoding="utf-8") as f:
        json.dump(topic_taxonomy, f, ensure_ascii=False, indent=2)

    kb_chunks_path = os.path.join(outdir, "knowledge_base_chunks.jsonl")
    write_jsonl(kb_chunks_path, kb_chunks)

    summary = {
        "input_path": input_path,
        "n_companies": len(records),
        "n_topics": len(topic_taxonomy),
        "top_20_global_keywords": [kw for kw, _ in global_keywords[:20]],
        "output_files": {
            "cleaned_company_kb": cleaned_path,
            "company_keywords_csv": company_kw_csv,
            "global_keywords_csv": global_kw_csv,
            "topic_taxonomy_json": taxonomy_path,
            "knowledge_base_chunks_jsonl": kb_chunks_path,
        },
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def resolve_default_input() -> str:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return "yc_detailed_data.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Build Project 8 keyword knowledge base from YC JSONL.")
    parser.add_argument("--input", default=resolve_default_input(), help="Path to YC company JSONL input")
    parser.add_argument("--outdir", default="data/kb", help="Output folder")
    parser.add_argument("--top_k_keywords", type=int, default=12, help="Top keywords per company")
    parser.add_argument("--n_clusters", type=int, default=20, help="Target number of topic clusters")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    summary = build_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        top_k_keywords=args.top_k_keywords,
        n_clusters=args.n_clusters,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
