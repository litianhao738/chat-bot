"""
RAG answer generation via Ollama.

Two modes:
  stream_answer()  – generator that yields text chunks (for Streamlit st.write_stream)
  full_answer()    – blocking call that returns the complete answer string
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    BUSINESS_GUIDE_PHRASES,
    BUSINESS_GUIDE_TERMS,
    OFFICIAL_TERMS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
)


# ── prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert entrepreneurship advisor specialising in Hong Kong startups \
and the Y Combinator startup ecosystem.

Rules you must follow:
- Answer ONLY using information from the provided context. Do not invent facts.
- If the context does not contain enough information, say so clearly.
- Be concise and practical. Write 2–4 short paragraphs.
- When citing specific companies or sources, name them.
- Do not repeat the question back to the user.
"""

def _format_chunk(i: int, meta: Dict[str, str], doc: str) -> str:
    sf = meta.get("source_family", "yc")
    if sf == "hk_official":
        header = f"[Source {i}] Official HK guidance – {meta.get('source_name', '')} | {meta.get('category', '')}"
    elif sf == "business_guide":
        header = f"[Source {i}] Business guide – {meta.get('source_name', '')} | {meta.get('category', '')}"
    else:
        name = meta.get("company_name", "Unknown")
        topic = meta.get("topic_label", "")
        header = f"[Source {i}] YC company: {name} | Topic: {topic}"

    # Use stored text for richer context (already capped at 2000 chars in indexer)
    body = meta.get("text") or doc
    return f"{header}\n{body[:800]}"


def build_prompt(
    query: str,
    matches: List[Dict[str, object]],
    topic_label: str,
    sentiment_context: str = "",
) -> str:
    context_blocks = []
    for i, m in enumerate(matches, 1):
        context_blocks.append(_format_chunk(i, m["metadata"], m.get("document", "")))

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    sentiment_section = ""
    if sentiment_context:
        sentiment_section = f"\n\n--- SENTIMENT ANALYSIS ---\n{sentiment_context}\n--- END SENTIMENT ---"

    prompt = f"""{_SYSTEM_PROMPT}

--- CONTEXT START ---
{context}{sentiment_section}
--- CONTEXT END ---

Topic: {topic_label}

Question: {query}

Answer:"""
    return prompt


# ── query rewriting ──────────────────────────────────────────────────────────

_QUERY_REWRITE_PROMPT = """\
You are a query clarification assistant for a Hong Kong startup / Y Combinator \
knowledge base.

Your task:
1. Read the user's question carefully.
2. Decide whether it can be rewritten to be more specific using the domain \
vocabulary below.
3. If yes, rewrite it to better match the knowledge base's terminology — \
preserve the original intent, improve the precision.
4. If the question is already precise, return it unchanged.
5. Return ONLY the final question. No explanation, no preamble, no quotes.

--- DOMAIN VOCABULARY ---
{vocab}
--- END VOCABULARY ---

User question: {query}

Rewritten question:"""


@lru_cache(maxsize=1)
def _build_vocab_context() -> str:
    """
    Assemble all predefined domain vocabulary from across the project into a
    structured string for the query-rewriting prompt.
    Built once and cached for the lifetime of the process.
    """
    lines: List[str] = []

    # 1. Registration & compliance (config.py)
    lines.append("Registration & Compliance terms: " + ", ".join(sorted(OFFICIAL_TERMS)))

    # 2. Business operations (config.py)
    lines.append("Business Operations terms: " + ", ".join(sorted(BUSINESS_GUIDE_TERMS)))
    lines.append("Business Operations phrases: " + "; ".join(sorted(BUSINESS_GUIDE_PHRASES)))

    # 3. YC / Examples (rag/retrieve.py)
    try:
        from rag.retrieve import _EXAMPLE_TOKENS
        lines.append("YC / Examples terms: " + ", ".join(sorted(_EXAMPLE_TOKENS)))
    except ImportError:
        pass

    # 4. Sentiment — prefer the richer set from sentiment/engine.py
    try:
        from sentiment.engine import SENTIMENT_QUERY_PHRASES, SENTIMENT_QUERY_TERMS
        single = sorted(t for t in SENTIMENT_QUERY_TERMS if " " not in t)
        multi  = sorted(SENTIMENT_QUERY_PHRASES)
        lines.append("Sentiment / Feedback terms: " + ", ".join(single))
        lines.append("Sentiment / Feedback phrases: " + "; ".join(multi))
    except ImportError:
        from rag.retrieve import _SENTIMENT_PHRASES, _SENTIMENT_TOKENS
        lines.append("Sentiment / Feedback terms: " + ", ".join(sorted(_SENTIMENT_TOKENS)))
        lines.append("Sentiment / Feedback phrases: " + "; ".join(sorted(_SENTIMENT_PHRASES)))

    # 5. Topic taxonomy (data/kb/topic_taxonomy.json)
    taxonomy_path = ROOT / "data" / "kb" / "topic_taxonomy.json"
    if taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        topic_lines = []
        for entry in taxonomy:
            label    = entry.get("topic_label", "")
            keywords = entry.get("top_keywords", [])[:5]
            if label and keywords:
                topic_lines.append(f"  {label}: {', '.join(keywords)}")
        if topic_lines:
            lines.append("Topic categories and key terms:\n" + "\n".join(topic_lines))

    # 6. Source aliases (builders/source_manifest.json)
    manifest_path = ROOT / "builders" / "source_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aliases: List[str] = []
        for source_data in manifest.values():
            for page in source_data.get("pages", []):
                aliases.extend(page.get("aliases", []))
        if aliases:
            lines.append("Source / page aliases: " + "; ".join(sorted(set(aliases))))

    return "\n".join(lines)


def rewrite_query(
    query: str,
    model: str = OLLAMA_MODEL,
) -> Tuple[str, bool]:
    """
    Pre-process the user query with an LLM before retrieval and generation.

    Sends the query together with all predefined domain vocabulary to Ollama
    and asks it to rewrite the question using more precise terminology if
    possible.  If the query is already clear, the original is returned.

    Returns:
        (rewritten_query, was_rewritten)
        was_rewritten is False when the LLM returned the same string or when
        the Ollama call fails — the original query is always returned as
        fallback so this step never blocks the pipeline.
    """
    vocab  = _build_vocab_context()
    prompt = _QUERY_REWRITE_PROMPT.format(vocab=vocab, query=query)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        rewritten = resp.json().get("response", "").strip()
    except Exception:
        return query, False  # graceful fallback — never block the pipeline

    if not rewritten or len(rewritten) < 5:
        return query, False

    was_rewritten = rewritten.lower().strip() != query.lower().strip()
    return rewritten, was_rewritten


# ── Ollama calls ──────────────────────────────────────────────────────────────

def stream_answer(
    query: str,
    matches: List[Dict[str, object]],
    topic_label: str,
    model: str = OLLAMA_MODEL,
    sentiment_context: str = "",
) -> Generator[str, None, None]:
    """
    Yield answer text chunks as they arrive from Ollama.
    Compatible with Streamlit's st.write_stream().
    """
    prompt = build_prompt(query, matches, topic_label, sentiment_context)
    url = f"{OLLAMA_BASE_URL}/api/generate"

    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield (
            "⚠️ Cannot connect to Ollama. "
            "Make sure Ollama is running (`ollama serve`) and "
            f"the model `{model}` is pulled (`ollama pull {model}`)."
        )
        return
    except requests.exceptions.RequestException as exc:
        yield f"⚠️ Ollama request failed: {exc}"
        return

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            data = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        token = data.get("response", "")
        if token:
            yield token
        if data.get("done"):
            break


def full_answer(
    query: str,
    matches: List[Dict[str, object]],
    topic_label: str,
    model: str = OLLAMA_MODEL,
    sentiment_context: str = "",
) -> str:
    """Blocking version – collects all streamed tokens and returns full text."""
    return "".join(stream_answer(query, matches, topic_label, model, sentiment_context))


# ── follow-up generation ──────────────────────────────────────────────────────

_FOLLOWUP_PROMPT_TEMPLATE = """\
You are an entrepreneurship advisor. Based on the question and answer below, \
suggest exactly 3 concise follow-up questions the user might want to ask next. \
Return ONLY the 3 questions, one per line, no numbering.

Original question: {query}

Answer summary: {answer_snippet}
"""

def generate_followups(
    query: str,
    answer_text: str,
    model: str = OLLAMA_MODEL,
) -> List[str]:
    """Return up to 3 follow-up question strings."""
    snippet = answer_text[:400].replace("\n", " ")
    prompt = _FOLLOWUP_PROMPT_TEMPLATE.format(query=query, answer_snippet=snippet)

    url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except Exception:
        return []

    lines = [l.strip(" -•123456789.") for l in raw.strip().splitlines() if l.strip()]
    return [l for l in lines if len(l) > 10][:3]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke test: prints the answer for a hard-coded query
    from rag.retrieve import retrieve

    query = " ".join(sys.argv[1:]) or "How do I register a company in Hong Kong?"
    print(f"Query: {query}\n{'─'*60}")
    topic, matches, intent = retrieve(query)
    print(f"Topic: {topic}  |  Sources: {[m['metadata']['source_family'] for m in matches]}\n")
    print("Answer:\n")
    for chunk in stream_answer(query, matches, topic):
        print(chunk, end="", flush=True)
    print("\n")
