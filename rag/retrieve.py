"""
Dense retrieval with source-aware re-ranking.

Pipeline:
  1. Detect query intent (official / business-guide / examples / sentiment)
  2. Embed query with sentence-transformers
  3. Recall Top-K candidates from ChromaDB
  4. Re-rank with source boost + diversity penalty
  5. Infer a human-readable topic label from intent + top results
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    BUSINESS_GUIDE_PHRASES,
    BUSINESS_GUIDE_TERMS,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OFFICIAL_TERMS,
    SOURCE_BOOST,
    TOP_K_DENSE,
    TOP_K_FINAL,
)


# ── cached singletons ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_collection(COLLECTION_NAME)


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokens(text: str) -> frozenset:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return frozenset(cleaned.split())


# ── intent term sets (module-level, built once) ───────────────────────────────

_EXAMPLE_TOKENS: frozenset = frozenset({
    "example", "examples", "similar", "benchmark", "compare", "yc", "pattern",
})

# Single-word sentiment tokens — can be matched against the token set
_SENTIMENT_TOKENS: frozenset = frozenset({
    "complaint", "complaints", "feedback", "review", "reviews",
    "sentiment", "reaction", "comment", "comments", "complain",
    "twitter", "instagram",
})

# Multi-word sentiment phrases — must be matched as substrings in the full query
# (multi-word strings can never appear as elements of a single-word token set,
#  so they were silently dead when placed inside the tokens & {...} check before)
_SENTIMENT_PHRASES: frozenset = frozenset({
    "social media",
    "what are people saying",
    "customer feedback",
    "user feedback",
    "online reviews",
    "what do users think",
    "public reaction",
})


# ── intent detection ──────────────────────────────────────────────────────────

def detect_intent(query: str) -> Dict[str, object]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"query must be a non-empty string, got {query!r}")

    q = query.lower()
    tokens = _tokens(q)

    asks_hk       = "hong kong" in q or " hk " in f" {q} "
    asks_official = bool(tokens & frozenset(OFFICIAL_TERMS))
    asks_bg       = (
        bool(tokens & frozenset(BUSINESS_GUIDE_TERMS))
        or any(p in q for p in BUSINESS_GUIDE_PHRASES)
    )
    asks_examples  = bool(tokens & _EXAMPLE_TOKENS)
    asks_sentiment = (
        bool(tokens & _SENTIMENT_TOKENS)
        or any(p in q for p in _SENTIMENT_PHRASES)
    )

    preferred: List[str] = []
    if asks_official:
        preferred.append("hk_official")
    if asks_bg:
        preferred.append("business_guide")
    if asks_examples:
        preferred.append("yc")

    return {
        "asks_hk":             asks_hk,
        "asks_official":       asks_official,
        "asks_business_guide": asks_bg,
        "asks_examples":       asks_examples,
        "asks_sentiment":      asks_sentiment,
        "preferred_sources":   preferred,
    }


# ── topic label inference (no classifier needed) ──────────────────────────────

def infer_topic_label(intent: Dict[str, object], matches: List[Dict]) -> str:
    """
    Derive a human-readable topic label purely from intent flags and
    the top retrieved source family. No classifier or external file needed.
    """
    if intent.get("asks_official"):
        return "HK Company Registration & Compliance"
    if intent.get("asks_business_guide"):
        return "Business Operations & Planning"
    if intent.get("asks_sentiment"):
        return "Social Media Sentiment"

    # Fall back to the most common topic_label in top results
    if matches:
        labels = [
            m["metadata"].get("topic_label", "")
            for m in matches
            if m["metadata"].get("topic_label")
        ]
        if labels:
            return Counter(labels).most_common(1)[0][0]

    return "Startup / YC Patterns"


# ── re-ranking ────────────────────────────────────────────────────────────────

def _rerank(
    candidates: List[Dict[str, object]],
    intent: Dict[str, object],
    top_k: int,
) -> List[Dict[str, object]]:
    scored: List[Tuple[float, Dict]] = []

    for c in candidates:
        base = float(c.get("dense_score", 0.0))
        sf   = c["metadata"].get("source_family", "yc")

        boost = 0.0
        if intent["asks_official"] and sf == "hk_official":
            boost += SOURCE_BOOST.get(("hk_official", "asks_official"), 0)
        if intent["asks_business_guide"] and sf == "business_guide":
            boost += SOURCE_BOOST.get(("business_guide", "asks_business_guide"), 0)
        if intent["asks_examples"] and sf == "yc":
            boost += SOURCE_BOOST.get(("yc", "asks_examples"), 0)

        scored.append((base + boost, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Dict] = []
    company_seen: Counter = Counter()
    url_seen:     Counter = Counter()

    for raw_score, c in scored:
        if len(selected) >= top_k:
            break
        meta    = c["metadata"]
        company = meta.get("company_id", "")
        url     = meta.get("url", "")

        adjusted = raw_score
        if company and company_seen[company]:
            adjusted *= 0.70
        if url and url_seen[url]:
            adjusted *= 0.50

        c["final_score"] = round(adjusted, 4)
        selected.append(c)
        company_seen[company] += 1
        url_seen[url]         += 1

    return selected


# ── public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K_FINAL,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Returns:
        matches  – re-ranked list of dicts with metadata + final_score
        intent   – detected query intent flags
    """
    intent = detect_intent(query)

    model      = _embedding_model()
    collection = _chroma_collection()

    query_embedding = model.encode([query])[0].tolist()
    n_candidates    = max(TOP_K_DENSE, top_k * 4)

    results   = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_candidates,
        include=["metadatas", "distances", "documents"],
    )

    candidates: List[Dict] = []
    for id_, meta, dist, doc in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["documents"][0],
    ):
        dense_score = max(0.0, 1.0 - dist / 2.0)
        candidates.append({
            "id":          id_,
            "metadata":    meta,
            "document":    doc,
            "dense_score": round(dense_score, 4),
        })

    matches = _rerank(candidates, intent, top_k)
    return matches, intent


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "How do I register a company in Hong Kong?"
    print(f"Query: {query}\n")
    matches, intent = retrieve(query)
    topic = infer_topic_label(intent, matches)
    print(f"Inferred topic : {topic}")
    print(f"Intent         : {intent}\n")
    for i, m in enumerate(matches, 1):
        meta = m["metadata"]
        print(f"[{i}] score={m['final_score']:.4f}  source={meta['source_family']}")
        print(f"     {meta['company_name']}  |  {meta['topic_label']}")
        print()
