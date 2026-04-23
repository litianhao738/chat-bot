#!/usr/bin/env python3
"""
yc-pro  –  Upgraded Entrepreneurship Chatbot
Streamlit UI with dense retrieval + Ollama streaming generation.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    OLLAMA_MODEL,
    SENTIMENT_KEYWORDS_PATH,
    SENTIMENT_SUMMARY_PATH,
    TOP_K_FINAL,
)
from rag.generate import generate_followups, rewrite_query, stream_answer
from rag.retrieve import infer_topic_label, retrieve


# ── sentiment helpers ─────────────────────────────────────────────────────────

def _sentiment_assets_ready() -> bool:
    return os.path.exists(SENTIMENT_SUMMARY_PATH) and os.path.exists(SENTIMENT_KEYWORDS_PATH)


def _run_sentiment(query: str) -> dict | None:
    if not _sentiment_assets_ready():
        return None
    try:
        from sentiment.engine import analyze_text, should_analyze_sentiment
        trigger = should_analyze_sentiment(query)
        if not trigger.get("should_run"):
            return None
        payload = analyze_text(query, SENTIMENT_SUMMARY_PATH, SENTIMENT_KEYWORDS_PATH)
        payload["trigger"] = trigger
        return payload
    except Exception:
        return None


def _sentiment_to_context(payload: dict) -> str:
    lines = [
        f"Predicted sentiment: {payload.get('predicted_sentiment', 'Unknown')}",
        f"VADER compound score: {payload.get('vader_scores', {}).get('compound', 'N/A')}",
    ]
    pos_kws = payload.get("matched_keywords", {}).get("Positive", [])
    neg_kws = payload.get("matched_keywords", {}).get("Negative", [])
    if pos_kws:
        lines.append(f"Matched positive signals: {', '.join(pos_kws[:5])}")
    if neg_kws:
        lines.append(f"Matched negative signals: {', '.join(neg_kws[:5])}")
    ctx = payload.get("dataset_context", {})
    if ctx.get("row_count"):
        lines.append(
            f"Dataset: {ctx['row_count']} social-media posts "
            f"({', '.join(p['value'] for p in (ctx.get('top_platforms') or [])[:3])})"
        )
    return "\n".join(lines)


# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Entrepreneurship Chatbot (Pro)",
    page_icon="🚀",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ollama_model   = st.text_input("Ollama model", value=OLLAMA_MODEL)
    top_k          = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=TOP_K_FINAL)
    show_sources   = st.checkbox("Show retrieved sources", value=True)
    show_followups = st.checkbox("Generate follow-up questions", value=True)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.caption("**Architecture**")
    st.caption("• Retrieval: sentence-transformers + ChromaDB")
    st.caption("• Generation: Ollama (streaming)")
    st.caption("• Source routing: hk_official / business_guide / yc")
    if _sentiment_assets_ready():
        st.caption("• Sentiment: VADER + Kaggle dataset ✓")
    else:
        st.caption("• Sentiment: not built  →  `python sentiment/build_assets.py`")


# ── title ─────────────────────────────────────────────────────────────────────
st.title("🚀 Entrepreneurship Chatbot")
st.caption(
    "Dense retrieval (all-MiniLM-L6-v2 + ChromaDB) · "
    f"Streaming generation ({ollama_model}) · "
    "Source-aware routing"
)


# ── source panel ──────────────────────────────────────────────────────────────
def render_sources(matches: list) -> None:
    if not matches:
        return
    with st.expander("📚 Retrieved sources", expanded=False):
        for i, m in enumerate(matches, 1):
            meta  = m["metadata"]
            sf    = meta.get("source_family", "yc")
            icon  = {"hk_official": "🏛️", "business_guide": "📖"}.get(sf, "🔬")
            name  = meta.get("company_name") or meta.get("source_name") or "Unknown"
            score = m.get("final_score", 0.0)
            url   = meta.get("url", "")
            label = f"{icon} [{i}] **{name}** · `{sf}` · score `{score:.3f}`"
            if url:
                label += f"  \n{url}"
            st.markdown(label)
            topic = meta.get("topic_label", "")
            kws   = meta.get("keywords", "")
            if topic:
                st.caption(f"topic: {topic}" + (f"  |  kws: {kws[:80]}" if kws else ""))
            st.divider()


# ── sentiment panel ───────────────────────────────────────────────────────────
def render_sentiment(payload: dict) -> None:
    with st.expander("📊 Sentiment Lens", expanded=True):
        col1, col2 = st.columns(2)
        predicted = payload.get("predicted_sentiment", "Unknown")
        color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(predicted, "❔")
        col1.metric("Predicted Sentiment", f"{color} {predicted}")
        col1.metric("Combined Score", round(payload.get("combined_score", 0), 4))
        pos_kws = payload.get("matched_keywords", {}).get("Positive", [])
        neg_kws = payload.get("matched_keywords", {}).get("Negative", [])
        with col2:
            if pos_kws:
                st.write("Positive signals:", ", ".join(pos_kws[:5]))
            if neg_kws:
                st.write("Negative signals:", ", ".join(neg_kws[:5]))
        if payload.get("reasoning"):
            st.caption("Reasoning: " + " · ".join(payload["reasoning"][:3]))
        st.caption(
            "Note: dataset is heavily neutral-skewed — treat as supplementary signal."
        )


# ── render past turns ─────────────────────────────────────────────────────────
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if show_sources:
            render_sources(turn.get("matches", []))
        if turn.get("sentiment"):
            render_sentiment(turn["sentiment"])
        if show_followups and turn.get("followups"):
            st.markdown("**Suggested follow-ups:**")
            for q in turn["followups"]:
                st.markdown(f"- {q}")


# ── chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a startup / entrepreneurship question …")

if query and query.strip():
    original_query = query.strip()

    with st.chat_message("user"):
        st.write(original_query)

    # query rewriting — clarify with domain vocabulary before retrieval
    with st.spinner("Clarifying question …"):
        query, was_rewritten = rewrite_query(original_query, model=ollama_model)
    if was_rewritten:
        st.caption(f"✏️ Interpreted as: *{query}*")

    # retrieval
    with st.spinner("Retrieving …"):
        try:
            matches, intent = retrieve(query, top_k=top_k)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

    topic_label = infer_topic_label(intent, matches)

    tags = []
    if intent.get("asks_official"):       tags.append("`🏛️ HK official`")
    if intent.get("asks_business_guide"): tags.append("`📖 Business guide`")
    if intent.get("asks_examples"):       tags.append("`🔬 YC examples`")
    if intent.get("asks_sentiment"):      tags.append("`📊 Sentiment`")
    if tags:
        st.caption(f"Routing: {' · '.join(tags)}  |  topic: `{topic_label}`")
    print(f"\n--- Sentiment Analysis Debug ---\n")
    print(f"Sentiment Payload: {intent}")
    print(f"\n--- Sentiment Analysis Debug ---\n")
    # sentiment (conditional)
    sentiment_payload = None
    sentiment_ctx     = ""
    if intent.get("asks_sentiment"):
        with st.spinner("Running sentiment analysis …"):
            sentiment_payload = _run_sentiment(query)
            print(f"\n--- Sentiment Analysis Debug ---\n")
            print(f"Sentiment Payload: {sentiment_payload}")
            print(f"\n--- Sentiment Analysis Debug ---\n")
        if sentiment_payload:
            sentiment_ctx = _sentiment_to_context(sentiment_payload)

    # streaming generation
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        for token in stream_answer(
            query, matches, topic_label,
            model=ollama_model,
            sentiment_context=sentiment_ctx,
        ):
            full_answer += token
            placeholder.markdown(full_answer + "▌")

        placeholder.markdown(full_answer)

        if show_sources:
            render_sources(matches)
        if sentiment_payload:
            render_sentiment(sentiment_payload)

    # follow-ups
    followups: list = []
    if show_followups and full_answer and not full_answer.startswith("⚠️"):
        with st.spinner("Generating follow-up questions …"):
            followups = generate_followups(query, full_answer, model=ollama_model)
        if followups:
            with st.chat_message("assistant"):
                st.markdown("**Suggested follow-ups:**")
                for q in followups:
                    st.markdown(f"- {q}")

    st.session_state.history.append({
        "query":     original_query,
        "answer":    full_answer,
        "topic":     topic_label,
        "intent":    intent,
        "matches":   matches,
        "sentiment": sentiment_payload,
        "followups": followups,
    })
