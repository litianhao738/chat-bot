"""
Central configuration for yc-pro.
All paths, model names, and tunable constants live here.

KB data flow (build once, run forever):
  scrapers/  →  data/raw/  →  builders/  →  data/kb/  →  ChromaDB (data/chroma_db/)

At runtime this project has zero dependency on any external directory.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"        # scraped JSONL before processing
KB_DIR          = DATA_DIR / "kb"         # processed chunk JSONL files
CHROMA_DB_PATH  = str(DATA_DIR / "chroma_db")
SENTIMENT_DIR   = DATA_DIR / "sentiment"  # sentiment assets output

# ── KB chunk files (output of builders, input to build_index.py) ──────────────
KB_CHUNK_PATHS = [
    KB_DIR / "knowledge_base_chunks.jsonl",      # YC companies
    KB_DIR / "hk_official_kb_chunks.jsonl",      # HK Companies Registry
    KB_DIR / "business_guide_kb_chunks.jsonl",   # Business guides
]

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "yc_kb"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_DENSE = 20    # candidates from ChromaDB before re-ranking
TOP_K_FINAL = 5     # chunks passed to LLM

# ── Source routing ────────────────────────────────────────────────────────────
OFFICIAL_TERMS = {
    "register", "registration", "incorporation", "incorporate",
    "annual return", "filing", "compliance", "registry",
    "license", "licence", "permit", "brn",
}
BUSINESS_GUIDE_TERMS = {
    "retail", "store", "stores", "operations", "inventory",
    "supply", "chain", "pricing", "competition", "competitor",
    "customer", "customers", "market", "segment", "funding",
    "vendor", "staffing", "location", "profit",
}
BUSINESS_GUIDE_PHRASES = {
    "how to start a business", "start a business",
    "open a retail store", "target customer", "target market",
    "market research", "business plan", "go to market",
    "pricing strategy", "startup plan", "first-time founder",
    "stockouts", "seasonal demand", "demand spikes",
}

# Source-family boost: (source_family, intent_flag) → additive score boost
SOURCE_BOOST = {
    ("hk_official",    "asks_official"):        0.30,
    ("business_guide", "asks_business_guide"):  0.25,
    ("yc",             "asks_examples"):         0.15,
}

# ── Sentiment ─────────────────────────────────────────────────────────────────
SENTIMENT_SUMMARY_PATH  = str(SENTIMENT_DIR / "sentiment_summary.json")
SENTIMENT_KEYWORDS_PATH = str(SENTIMENT_DIR / "sentiment_keyword_profiles.json")
SENTIMENT_CLEANED_PATH  = str(SENTIMENT_DIR / "sentiment_cleaned.jsonl")
