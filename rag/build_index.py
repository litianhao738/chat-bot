#!/usr/bin/env python3
"""
Build (or rebuild) the ChromaDB dense-retrieval index from existing KB chunks.

Usage:
    python rag/build_index.py            # build with progress output
    python rag/build_index.py --reset    # drop & rebuild from scratch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    KB_CHUNK_PATHS,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def safe(value: object) -> str:
    return str(value).strip() if value is not None else ""


def load_chunks() -> List[Dict[str, object]]:
    """Read all KB chunk JSONL files and merge into one list."""
    rows: List[Dict[str, object]] = []
    for path in KB_CHUNK_PATHS:
        p = Path(path)
        if not p.exists():
            print(f"  [skip] not found: {path}")
            continue
        count_before = len(rows)
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        added = len(rows) - count_before
        family = rows[-1].get("source_family", "yc") if rows else "?"
        print(f"  loaded {added:>5} chunks  ({family})  from {p.name}")
    return rows


def chunk_to_embed_text(row: Dict[str, object]) -> str:
    """
    Build the string that gets embedded.
    We use the full `text` field (already rich) plus keywords for weight.
    """
    text = safe(row.get("text"))
    keywords = ", ".join(safe(k) for k in row.get("keywords", []) if safe(k))
    topic = safe(row.get("topic_label"))
    parts = [text]
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if topic:
        parts.append(f"Topic: {topic}")
    return " ".join(parts)


def build_metadata(row: Dict[str, object]) -> Dict[str, str]:
    """
    ChromaDB metadata must be flat str/int/float/bool.
    We store the fields needed for source routing and display.
    """
    return {
        "chunk_id":     safe(row.get("chunk_id")),
        "company_id":   safe(row.get("company_id")),
        "company_name": safe(row.get("company_name")),
        "source_family": safe(row.get("source_family")) or "yc",
        "source_name":  safe(row.get("source_name")),
        "topic_label":  safe(row.get("topic_label")),
        "url":          safe(row.get("url")),
        "trust_tier":   safe(row.get("trust_tier")),
        "jurisdiction": safe(row.get("jurisdiction")),
        "category":     safe(row.get("category")),
        "business_stage": safe(row.get("business_stage")),
        "keywords":     ", ".join(safe(k) for k in row.get("keywords", []) if safe(k)),
        "tagline":      safe(row.get("tagline", "")),
        "text":         safe(row.get("text", ""))[:2000],   # cap for storage
    }


# ── main build ────────────────────────────────────────────────────────────────

def build(reset: bool = False) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*60}")
    print("Loading KB chunks …")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}\n")

    print(f"Loading embedding model: {EMBEDDING_MODEL} …")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Opening ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("  existing collection deleted")
        except Exception:
            pass

    # Get-or-create with cosine distance
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if safe(c.get("chunk_id")) not in existing_ids]
    print(f"  already indexed: {len(existing_ids)}, to add: {len(new_chunks)}\n")

    if not new_chunks:
        print("Index is up to date. Nothing to do.")
        return

    # Embed and upsert in batches
    total = len(new_chunks)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = new_chunks[start : start + EMBEDDING_BATCH_SIZE]
        texts = [chunk_to_embed_text(c) for c in batch]
        ids   = [safe(c["chunk_id"]) for c in batch]
        metas = [build_metadata(c) for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        done = min(start + EMBEDDING_BATCH_SIZE, total)
        print(f"  indexed {done:>5} / {total}")

    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} documents.")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Drop the existing collection and rebuild from scratch")
    args = parser.parse_args()
    build(reset=args.reset)


if __name__ == "__main__":
    main()
