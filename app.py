#!/usr/bin/env python3
"""
yc-pro - Upgraded Entrepreneurship Chatbot
Streamlit UI with dense retrieval, Ollama streaming generation, and analytics.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from textwrap import fill
from typing import Any

import streamlit as st

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import (
    KB_CHUNK_PATHS,
    KB_DIR,
    OLLAMA_MODEL,
    SENTIMENT_CLEANED_PATH,
    SENTIMENT_KEYWORDS_PATH,
    SENTIMENT_SUMMARY_PATH,
    TOP_K_FINAL,
)
from rag.generate import generate_followups, rewrite_query, stream_answer
from rag.retrieve import infer_topic_label, retrieve

PROJECT_SUMMARY_PATH = KB_DIR / "summary.json"
CLEANED_COMPANY_KB_PATH = KB_DIR / "cleaned_company_kb.jsonl"
EVALUATION_SUMMARY_PATH = ROOT / "data" / "evaluation" / "summary.json"
EVALUATION_REPORT_PATH = ROOT / "data" / "evaluation" / "classification_report.csv"
EVALUATION_SAMPLES_PATH = ROOT / "data" / "evaluation" / "topk_retrieval_samples.csv"
ANALYTICS_DIR = ROOT / "data" / "analytics"
FOLLOWUP_LOG_PATH = ANALYTICS_DIR / "followup_events.jsonl"


def _sentiment_assets_ready() -> bool:
    return (
        os.path.exists(SENTIMENT_SUMMARY_PATH)
        and os.path.exists(SENTIMENT_KEYWORDS_PATH)
        and os.path.exists(SENTIMENT_CLEANED_PATH)
    )


def _analytics_dependencies_ready() -> bool:
    return pd is not None


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
        platforms = ", ".join(p["value"] for p in (ctx.get("top_platforms") or [])[:3])
        lines.append(f"Dataset: {ctx['row_count']} social-media posts ({platforms})")
    return "\n".join(lines)


def _load_optional_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_followup_event(
    event_type: str,
    turn_id: str,
    query: str,
    followup_text: str,
    followup_rank: int,
    topic: str = "",
) -> None:
    _append_jsonl(
        FOLLOWUP_LOG_PATH,
        {
            "timestamp": _now_iso(),
            "session_id": st.session_state.get("session_id", ""),
            "event_type": event_type,
            "turn_id": turn_id,
            "query": query,
            "followup_text": followup_text,
            "followup_rank": followup_rank,
            "topic": topic,
        },
    )


@lru_cache(maxsize=None)
def _load_jsonl_rows(path_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path_text)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_uncached_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _wrap_axis_label(value: object, width: int = 18) -> str:
    text = str(value).strip()
    return fill(text or "Unknown", width=width)


def _empty_frame():
    return pd.DataFrame() if pd is not None else None


def build_source_family_frame():
    if pd is None:
        return None
    counts: Counter[str] = Counter()
    for path in KB_CHUNK_PATHS:
        for row in _load_jsonl_rows(str(path)):
            counts[str(row.get("source_family") or "yc")] += 1
    frame = pd.DataFrame(
        [{"source_family": source_family, "count": count} for source_family, count in counts.items()]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("source_family")


def build_topic_distribution_frame(limit: int = 10):
    if pd is None:
        return None
    counts = Counter(
        str(row.get("topic_label", "unknown")).strip() or "unknown"
        for row in _load_jsonl_rows(str(CLEANED_COMPANY_KB_PATH))
    )
    frame = pd.DataFrame([{"topic_label": label, "count": count} for label, count in counts.items()])
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).head(limit).set_index("topic_label")


def build_sentiment_distribution_frame():
    if pd is None:
        return None
    summary = _load_optional_json(SENTIMENT_SUMMARY_PATH)
    frame = pd.DataFrame(
        [
            {"sentiment": label, "count": count}
            for label, count in (summary.get("dataset_sentiment_distribution", {}) or {}).items()
        ]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("sentiment")


def build_sentiment_platform_frame():
    if pd is None:
        return None
    rows = _load_jsonl_rows(str(SENTIMENT_CLEANED_PATH))
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame(rows)
    if "platform" not in frame or "dataset_sentiment" not in frame:
        return _empty_frame()
    return frame.groupby(["platform", "dataset_sentiment"]).size().unstack(fill_value=0).sort_index()


def build_engagement_frame():
    if pd is None:
        return None
    rows = _load_jsonl_rows(str(SENTIMENT_CLEANED_PATH))
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame(rows)
    if "platform" not in frame:
        return _empty_frame()
    frame["likes"] = pd.to_numeric(frame.get("likes"), errors="coerce").fillna(0.0)
    frame["retweets"] = pd.to_numeric(frame.get("retweets"), errors="coerce").fillna(0.0)
    return (
        frame.groupby("platform")[["likes", "retweets"]]
        .mean()
        .round(2)
        .sort_values("likes", ascending=False)
        .reset_index()
    )


def build_keyword_profile_frame(label: str, limit: int = 10):
    if pd is None:
        return None
    profiles = _load_optional_json(SENTIMENT_KEYWORDS_PATH)
    items = ((profiles.get("top_keywords_by_sentiment", {}) or {}).get(label, []) or [])[:limit]
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    return frame.rename(columns={"value": "keyword"}).set_index("keyword")


def build_session_retrieval_mix_frame(chat_history: list[dict[str, Any]]):
    if pd is None:
        return None
    counts: Counter[str] = Counter()
    for turn in chat_history:
        for match in turn.get("matches", []):
            meta = match.get("metadata", {}) if isinstance(match, dict) else {}
            counts[str(meta.get("source_family") or "unknown")] += 1
    frame = pd.DataFrame(
        [{"source_family": source_family, "count": count} for source_family, count in counts.items()]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("source_family")


def build_followup_summary() -> dict[str, Any]:
    rows = _load_uncached_jsonl_rows(FOLLOWUP_LOG_PATH)
    shown = [row for row in rows if row.get("event_type") == "shown"]
    clicked = [row for row in rows if row.get("event_type") == "clicked"]
    shown_keys = {
        (row.get("turn_id"), row.get("followup_rank"), row.get("followup_text"))
        for row in shown
    }
    clicked_keys = {
        (row.get("turn_id"), row.get("followup_rank"), row.get("followup_text"))
        for row in clicked
    }
    shown_count = len(shown_keys)
    clicked_count = len(clicked_keys)
    return {
        "shown": shown_count,
        "clicked": clicked_count,
        "ctr": clicked_count / shown_count if shown_count else 0.0,
        "events": len(rows),
    }


def build_followup_rank_frame():
    if pd is None:
        return None
    rows = _load_uncached_jsonl_rows(FOLLOWUP_LOG_PATH)
    if not rows:
        return _empty_frame()
    frame = pd.DataFrame(rows)
    if "followup_rank" not in frame or "event_type" not in frame:
        return _empty_frame()
    frame["followup_rank"] = pd.to_numeric(frame["followup_rank"], errors="coerce").fillna(0).astype(int)
    grouped = (
        frame.groupby(["followup_rank", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("followup_rank")
    )
    if "shown" not in grouped:
        grouped["shown"] = 0
    if "clicked" not in grouped:
        grouped["clicked"] = 0
    grouped["ctr"] = grouped.apply(
        lambda row: float(row["clicked"]) / float(row["shown"]) if row["shown"] else 0.0,
        axis=1,
    )
    grouped["rank_label"] = grouped["followup_rank"].apply(lambda value: f"Suggestion {int(value)}")
    return grouped


def build_followup_journey_frame(limit: int = 10):
    if pd is None:
        return None
    clicked = [
        row for row in _load_uncached_jsonl_rows(FOLLOWUP_LOG_PATH)
        if row.get("event_type") == "clicked"
    ]
    if not clicked:
        return _empty_frame()
    counts = Counter(
        (
            str(row.get("query", "")).strip()[:80],
            str(row.get("followup_text", "")).strip()[:80],
        )
        for row in clicked
    )
    frame = pd.DataFrame(
        [
            {"from_query": source, "to_followup": target, "clicks": count}
            for (source, target), count in counts.most_common(limit)
        ]
    )
    return frame


def render_horizontal_bar_chart(frame, label_column: str, value_column: str, color: str, height: int = 340) -> None:
    chart_frame = frame.reset_index().copy()
    chart_frame["wrapped_label"] = chart_frame[label_column].apply(_wrap_axis_label)
    chart_spec = {
        "data": {"values": chart_frame.to_dict(orient="records")},
        "mark": {"type": "bar", "cornerRadiusEnd": 4},
        "encoding": {
            "y": {
                "field": "wrapped_label",
                "type": "nominal",
                "sort": "-x",
                "axis": {"labelAngle": 0, "title": None, "labelLimit": 220},
            },
            "x": {"field": value_column, "type": "quantitative", "axis": {"title": None}},
            "tooltip": [
                {"field": label_column, "type": "nominal", "title": "Label"},
                {"field": value_column, "type": "quantitative", "title": "Value"},
            ],
            "color": {"value": color},
        },
        "height": height,
    }
    st.vega_lite_chart(chart_spec, use_container_width=True)


def render_donut_chart(
    frame,
    label_column: str,
    value_column: str,
    title: str,
    colors: list[str] | None = None,
    height: int = 250,
) -> None:
    chart_frame = frame.reset_index().copy()
    total = float(chart_frame[value_column].sum())
    chart_frame["percentage"] = chart_frame[value_column].apply(
        lambda value: f"{(float(value) / total):.1%}" if total else "0.0%"
    )
    color_encoding: dict[str, Any] = {
        "field": label_column,
        "type": "nominal",
        "legend": {"title": title, "orient": "bottom"},
    }
    if colors:
        color_encoding["scale"] = {"range": colors}

    chart_spec = {
        "data": {"values": chart_frame.to_dict(orient="records")},
        "mark": {"type": "arc", "innerRadius": 58, "outerRadius": 96, "stroke": "#ffffff", "strokeWidth": 2},
        "encoding": {
            "theta": {"field": value_column, "type": "quantitative", "stack": True},
            "color": color_encoding,
            "tooltip": [
                {"field": label_column, "type": "nominal", "title": title},
                {"field": value_column, "type": "quantitative", "title": "Count"},
                {"field": "percentage", "type": "nominal", "title": "Share"},
            ],
        },
        "height": height,
        "view": {"stroke": None},
    }
    st.vega_lite_chart(chart_spec, use_container_width=True)


def render_grouped_sentiment_chart(frame, height: int = 320) -> None:
    chart_frame = frame.reset_index().rename(columns={"index": "platform"})
    value_columns = [column for column in chart_frame.columns if column != "platform"]
    melted = chart_frame.melt(
        id_vars=["platform"],
        value_vars=value_columns,
        var_name="sentiment",
        value_name="count",
    )
    chart_spec = {
        "data": {"values": melted.to_dict(orient="records")},
        "mark": {"type": "bar", "cornerRadiusEnd": 3},
        "encoding": {
            "x": {"field": "platform", "type": "nominal", "axis": {"title": None, "labelAngle": 0}},
            "y": {"field": "count", "type": "quantitative", "axis": {"title": None}},
            "xOffset": {"field": "sentiment"},
            "color": {
                "field": "sentiment",
                "type": "nominal",
                "scale": {
                    "domain": ["Positive", "Neutral", "Negative"],
                    "range": ["#2f855a", "#718096", "#c53030"],
                },
                "legend": {"title": "Sentiment"},
            },
            "tooltip": [
                {"field": "platform", "type": "nominal"},
                {"field": "sentiment", "type": "nominal"},
                {"field": "count", "type": "quantitative"},
            ],
        },
        "height": height,
    }
    st.vega_lite_chart(chart_spec, use_container_width=True)


def render_engagement_scatter(frame, height: int = 320) -> None:
    chart_spec = {
        "data": {"values": frame.to_dict(orient="records")},
        "layer": [
            {
                "mark": {"type": "circle", "size": 220, "opacity": 0.8},
                "encoding": {
                    "x": {"field": "likes", "type": "quantitative", "axis": {"title": "Average Likes"}},
                    "y": {"field": "retweets", "type": "quantitative", "axis": {"title": "Average Retweets"}},
                    "color": {"field": "platform", "type": "nominal", "legend": {"title": "Platform"}},
                    "tooltip": [
                        {"field": "platform", "type": "nominal"},
                        {"field": "likes", "type": "quantitative"},
                        {"field": "retweets", "type": "quantitative"},
                    ],
                },
            },
            {
                "mark": {"type": "text", "dy": -12, "fontSize": 11},
                "encoding": {
                    "x": {"field": "likes", "type": "quantitative"},
                    "y": {"field": "retweets", "type": "quantitative"},
                    "text": {"field": "platform", "type": "nominal"},
                    "color": {"value": "#1a202c"},
                },
            },
        ],
        "height": height,
    }
    st.vega_lite_chart(chart_spec, use_container_width=True)


def render_overview_metrics() -> None:
    project_summary = _load_optional_json(PROJECT_SUMMARY_PATH)
    sentiment_summary = _load_optional_json(SENTIMENT_SUMMARY_PATH) if _sentiment_assets_ready() else {}
    source_frame = build_source_family_frame()
    indexed_docs = int(source_frame["count"].sum()) if source_frame is not None and not source_frame.empty else 0

    metric_columns = st.columns(5)
    metric_columns[0].metric("Companies", int(project_summary.get("n_companies", 0)))
    metric_columns[1].metric("Indexed Chunks", indexed_docs)
    metric_columns[2].metric("Topics", int(project_summary.get("n_topics", 0)))
    metric_columns[3].metric("Sentiment Rows", int(sentiment_summary.get("row_count", 0)))
    metric_columns[4].metric("Chat Turns", len(st.session_state.history))


def render_evaluation_results() -> None:
    summary = _load_optional_json(EVALUATION_SUMMARY_PATH)
    if not summary:
        st.markdown("**Evaluation Results**")
        st.info("Run `python evaluation/run_evaluation.py` to generate offline evaluation metrics.")
        return

    st.markdown("**Evaluation Results**")
    st.caption(
        "Offline checks from `evaluation/test_queries.csv`: topic routing, retrieval ranking, "
        "answer support, and response latency."
    )

    cols = st.columns(5)
    cols[0].metric("Topic Accuracy", f"{summary.get('topic_accuracy', 0):.1%}")
    cols[1].metric("Weighted F1", f"{summary.get('topic_weighted_f1', 0):.1%}")
    cols[2].metric("Recall@5", f"{summary.get('relevant_recall_at_5', 0):.1%}")
    cols[3].metric("MRR", f"{summary.get('relevant_mrr', 0):.3f}")
    cols[4].metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f} ms")

    metric_rows = [
        {"metric": "Source Recall@1", "value": summary.get("source_recall_at_1", 0)},
        {"metric": "Source Recall@3", "value": summary.get("source_recall_at_3", 0)},
        {"metric": "Source Recall@5", "value": summary.get("source_recall_at_5", 0)},
        {"metric": "Relevant Recall@1", "value": summary.get("relevant_recall_at_1", 0)},
        {"metric": "Relevant Recall@3", "value": summary.get("relevant_recall_at_3", 0)},
        {"metric": "Relevant Recall@5", "value": summary.get("relevant_recall_at_5", 0)},
        {"metric": "ROUGE-1 / Coverage", "value": summary.get("rouge_1", 0)},
        {"metric": "ROUGE-L / Coverage", "value": summary.get("rouge_l", 0)},
    ]
    if pd is not None:
        eval_frame = pd.DataFrame(metric_rows)
        chart_spec = {
            "data": {"values": eval_frame.to_dict(orient="records")},
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "y": {"field": "metric", "type": "nominal", "sort": "-x", "axis": {"title": None}},
                "x": {"field": "value", "type": "quantitative", "axis": {"format": "%", "title": None}},
                "color": {"value": "#2b6cb0"},
                "tooltip": [
                    {"field": "metric", "type": "nominal"},
                    {"field": "value", "type": "quantitative", "format": ".2%"},
                ],
            },
            "height": 280,
        }
        st.vega_lite_chart(chart_spec, use_container_width=True)

        if EVALUATION_REPORT_PATH.exists():
            report_frame = pd.read_csv(EVALUATION_REPORT_PATH)
            st.dataframe(report_frame, use_container_width=True)

        if EVALUATION_SAMPLES_PATH.exists():
            with st.expander("Top-K retrieval samples", expanded=False):
                samples = pd.read_csv(EVALUATION_SAMPLES_PATH).head(8)
                st.dataframe(samples, use_container_width=True)


def render_followup_analytics() -> None:
    st.markdown("**Follow-Up Prediction Analytics**")
    summary = build_followup_summary()
    cols = st.columns(4)
    cols[0].metric("Suggestions Shown", int(summary["shown"]))
    cols[1].metric("Suggestions Clicked", int(summary["clicked"]))
    cols[2].metric("Follow-Up CTR", f"{summary['ctr']:.1%}")
    cols[3].metric("Logged Events", int(summary["events"]))

    rank_frame = build_followup_rank_frame()
    journey_frame = build_followup_journey_frame(limit=10)

    left, right = st.columns(2)
    with left:
        if rank_frame is not None and not rank_frame.empty:
            chart_spec = {
                "data": {"values": rank_frame.to_dict(orient="records")},
                "mark": {"type": "bar", "cornerRadiusEnd": 4},
                "encoding": {
                    "x": {"field": "rank_label", "type": "nominal", "axis": {"title": None, "labelAngle": 0}},
                    "y": {"field": "ctr", "type": "quantitative", "axis": {"title": "CTR", "format": "%"}},
                    "color": {"value": "#805ad5"},
                    "tooltip": [
                        {"field": "rank_label", "type": "nominal", "title": "Suggestion"},
                        {"field": "shown", "type": "quantitative", "title": "Shown"},
                        {"field": "clicked", "type": "quantitative", "title": "Clicked"},
                        {"field": "ctr", "type": "quantitative", "title": "CTR", "format": ".1%"},
                    ],
                },
                "height": 240,
            }
            st.vega_lite_chart(chart_spec, use_container_width=True)
        else:
            st.info("Generate and click suggested follow-ups to populate CTR by rank.")

    with right:
        if journey_frame is not None and not journey_frame.empty:
            st.dataframe(journey_frame, use_container_width=True)
        else:
            st.info("Clicked follow-ups will appear here as user journey transitions.")


def render_key_insights() -> None:
    project_summary = _load_optional_json(PROJECT_SUMMARY_PATH)
    sentiment_summary = _load_optional_json(SENTIMENT_SUMMARY_PATH) if _sentiment_assets_ready() else {}
    source_frame = build_source_family_frame()

    insights: list[str] = []
    if source_frame is not None and not source_frame.empty:
        top_source = source_frame.reset_index().iloc[0]
        insights.append(
            f"**Data Coverage**\n\nThe retrieval knowledge base is led by `{top_source['source_family']}` "
            f"with {int(top_source['count'])} chunks, while HK official and business-guide sources add focused procedural context."
        )

    if sentiment_summary:
        distribution = sentiment_summary.get("dataset_sentiment_distribution", {}) or {}
        total_rows = int(sentiment_summary.get("row_count", 0))
        neutral_count = int(distribution.get("Neutral", 0))
        if total_rows:
            insights.append(
                f"**Sentiment Limitation**\n\nThe sentiment dataset is neutral-heavy: "
                f"{neutral_count}/{total_rows} rows ({neutral_count / total_rows:.1%}) are neutral, "
                "so sentiment should be treated as supplementary evidence."
            )

    insights.append(
        f"**Knowledge Base Scale**\n\nThe core YC knowledge base covers "
        f"{int(project_summary.get('n_companies', 0))} companies across "
        f"{int(project_summary.get('n_topics', 0))} topic clusters."
    )

    session_frame = build_session_retrieval_mix_frame(st.session_state.history)
    if session_frame is not None and not session_frame.empty:
        top_session_source = session_frame.reset_index().iloc[0]
        insights.append(
            f"**Live Session Insight**\n\nIn this chat session, `{top_session_source['source_family']}` "
            "has been the most used source family, which helps explain the answer style."
        )

    st.markdown("\n\n".join(insights))


def render_analytics_dashboard() -> None:
    st.subheader("Analytics Dashboard")
    st.caption(
        "This dashboard summarizes knowledge-base coverage, source balance, startup topics, "
        "sentiment assets, and the current chat session retrieval mix."
    )

    if not _analytics_dependencies_ready():
        st.warning("Analytics dashboard requires pandas. Please run: python -m pip install pandas")
        return

    render_overview_metrics()
    render_evaluation_results()
    render_followup_analytics()
    st.divider()

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("**Knowledge Source Distribution**")
        source_frame = build_source_family_frame()
        if source_frame is not None and not source_frame.empty:
            render_donut_chart(
                source_frame,
                "source_family",
                "count",
                title="Source",
                colors=["#2b6cb0", "#319795", "#dd6b20", "#805ad5"],
                height=250,
            )
            st.caption("The unified retrieval store combines YC examples, HK official guidance, and business guides.")
        else:
            st.info("Source summary is not available yet.")

    with top_right:
        st.markdown("**Top Knowledge-Base Topics**")
        topic_frame = build_topic_distribution_frame(limit=10)
        if topic_frame is not None and not topic_frame.empty:
            render_horizontal_bar_chart(topic_frame, "topic_label", "count", color="#dd6b20", height=360)
            st.caption("These are the most common startup themes represented in the current YC knowledge base.")
        else:
            st.info("Topic distribution is not available yet.")

    middle_left, middle_right = st.columns(2)
    with middle_left:
        st.markdown("**Sentiment Label Distribution**")
        if _sentiment_assets_ready():
            sentiment_distribution = build_sentiment_distribution_frame()
            if sentiment_distribution is not None and not sentiment_distribution.empty:
                render_donut_chart(
                    sentiment_distribution,
                    "sentiment",
                    "count",
                    title="Sentiment",
                    colors=["#718096", "#2f855a", "#c53030", "#805ad5"],
                    height=250,
                )
            else:
                st.info("Sentiment summary is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    with middle_right:
        st.markdown("**Platform by Sentiment**")
        if _sentiment_assets_ready():
            platform_frame = build_sentiment_platform_frame()
            if platform_frame is not None and not platform_frame.empty:
                render_grouped_sentiment_chart(platform_frame, height=320)
                st.caption("This chart shows sentiment-labelled posts across Twitter, Instagram, and Facebook.")
            else:
                st.info("Platform-level sentiment data is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("**Average Engagement by Platform**")
        if _sentiment_assets_ready():
            engagement_frame = build_engagement_frame()
            if engagement_frame is not None and not engagement_frame.empty:
                render_engagement_scatter(engagement_frame, height=320)
                st.dataframe(engagement_frame, use_container_width=True)
            else:
                st.info("Engagement data is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    with lower_right:
        st.markdown("**Current Session Retrieval Mix**")
        session_frame = build_session_retrieval_mix_frame(st.session_state.history)
        if session_frame is not None and not session_frame.empty:
            render_horizontal_bar_chart(session_frame, "source_family", "count", color="#319795", height=220)
            st.caption("This chart updates as the chatbot retrieves sources during the current session.")
        else:
            st.info("Ask a few questions in the chatbot tab to generate this live chart.")

    keyword_left, keyword_right = st.columns(2)
    with keyword_left:
        st.markdown("**Top Positive Keywords**")
        if _sentiment_assets_ready():
            positive_keywords = build_keyword_profile_frame("Positive", limit=10)
            if positive_keywords is not None and not positive_keywords.empty:
                render_horizontal_bar_chart(positive_keywords, "keyword", "count", color="#38a169", height=340)
            else:
                st.info("Positive keyword profile is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    with keyword_right:
        st.markdown("**Top Neutral Keywords**")
        if _sentiment_assets_ready():
            neutral_keywords = build_keyword_profile_frame("Neutral", limit=10)
            if neutral_keywords is not None and not neutral_keywords.empty:
                render_horizontal_bar_chart(neutral_keywords, "keyword", "count", color="#718096", height=340)
            else:
                st.info("Neutral keyword profile is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    st.divider()
    st.markdown("**Key Insights**")
    render_key_insights()

    st.markdown("**Diagnostic Notes**")
    st.write("- The retrieval data is YC-heavy, so specialized sources are important for Hong Kong procedures and practical business guidance.")
    st.write("- The sentiment dataset is highly imbalanced toward neutral posts, so it should be presented as a descriptive analytics signal.")
    st.write("- The current session retrieval mix helps explain which source families shaped the chatbot's answers.")


st.set_page_config(
    page_title="Entrepreneurship Chatbot (Pro)",
    page_icon="YC",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "queued_query" not in st.session_state:
    st.session_state.queued_query = ""


with st.sidebar:
    st.header("Settings")
    ollama_model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    top_k = st.slider("Top-K retrieved chunks", min_value=2, max_value=8, value=TOP_K_FINAL)
    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_followups = st.checkbox("Generate follow-up questions", value=True)

    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.queued_query = ""
        st.rerun()

    st.divider()
    st.caption("**Architecture**")
    st.caption("- Retrieval: sentence-transformers + ChromaDB")
    st.caption("- Generation: Ollama (streaming)")
    st.caption("- Analytics: Streamlit + Vega-Lite")
    st.caption("- Source routing: hk_official / business_guide / yc")
    if _sentiment_assets_ready():
        st.caption("- Sentiment: VADER + Kaggle dataset")
    else:
        st.caption("- Sentiment: not built. Run `python sentiment/build_assets.py`")


st.title("Entrepreneurship Chatbot")
st.caption(
    "Dense retrieval (all-MiniLM-L6-v2 + ChromaDB) | "
    f"Streaming generation ({ollama_model}) | "
    "Analytics dashboard"
)


def render_sources(matches: list) -> None:
    if not matches:
        return
    with st.expander("Retrieved sources", expanded=False):
        for i, m in enumerate(matches, 1):
            meta = m["metadata"]
            sf = meta.get("source_family", "yc")
            source_label = {
                "hk_official": "HK official",
                "business_guide": "Business guide",
            }.get(sf, "YC")
            name = meta.get("company_name") or meta.get("source_name") or "Unknown"
            score = m.get("final_score", 0.0)
            url = meta.get("url", "")
            label = f"**{source_label} [{i}] {name}** | `{sf}` | score `{score:.3f}`"
            if url:
                label += f"  \n{url}"
            st.markdown(label)
            topic = meta.get("topic_label", "")
            kws = meta.get("keywords", "")
            if topic:
                st.caption(f"topic: {topic}" + (f" | kws: {kws[:80]}" if kws else ""))
            st.divider()


def render_sentiment(payload: dict) -> None:
    with st.expander("Sentiment Lens", expanded=True):
        col1, col2 = st.columns(2)
        predicted = payload.get("predicted_sentiment", "Unknown")
        label = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}.get(
            predicted, "Unknown"
        )
        col1.metric("Predicted Sentiment", label)
        col1.metric("Combined Score", round(payload.get("combined_score", 0), 4))
        pos_kws = payload.get("matched_keywords", {}).get("Positive", [])
        neg_kws = payload.get("matched_keywords", {}).get("Negative", [])
        with col2:
            if pos_kws:
                st.write("Positive signals:", ", ".join(pos_kws[:5]))
            if neg_kws:
                st.write("Negative signals:", ", ".join(neg_kws[:5]))
        if payload.get("reasoning"):
            st.caption("Reasoning: " + " | ".join(payload["reasoning"][:3]))
        st.caption("Note: dataset is heavily neutral-skewed; treat as a supplementary signal.")


def render_followup_buttons(turn: dict[str, Any], key_prefix: str) -> None:
    followups = turn.get("followups") or []
    if not followups:
        return
    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(min(3, len(followups)))
    for index, question in enumerate(followups, 1):
        column = cols[(index - 1) % len(cols)]
        with column:
            if st.button(question, key=f"{key_prefix}_followup_{index}", use_container_width=True):
                log_followup_event(
                    "clicked",
                    str(turn.get("turn_id", "")),
                    str(turn.get("query", "")),
                    str(question),
                    index,
                    str(turn.get("topic", "")),
                )
                st.session_state.queued_query = str(question)
                st.rerun()


chatbot_tab, analytics_tab = st.tabs(["Chatbot", "Analytics Dashboard"])

with chatbot_tab:
    messages = st.container()
    with messages:
        for turn_index, turn in enumerate(st.session_state.history):
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if show_sources:
                    render_sources(turn.get("matches", []))
                if turn.get("sentiment"):
                    render_sentiment(turn["sentiment"])
                if show_followups and turn.get("followups"):
                    turn_id = str(turn.get("turn_id") or f"history_{turn_index}")
                    render_followup_buttons(turn, f"turn_{turn_id}")

    typed_query = st.chat_input("Ask a startup or entrepreneurship question")
    queued_query = str(st.session_state.get("queued_query", "")).strip()
    query = queued_query or typed_query
    if queued_query:
        st.session_state.queued_query = ""

    if query and query.strip():
        original_query = query.strip()
        turn_id = uuid.uuid4().hex

        with messages:
            with st.chat_message("user"):
                st.write(original_query)

            with st.spinner("Clarifying question..."):
                query, was_rewritten = rewrite_query(original_query, model=ollama_model)
            if was_rewritten:
                st.caption(f"Interpreted as: *{query}*")

            with st.spinner("Retrieving..."):
                try:
                    matches, intent = retrieve(query, top_k=top_k)
                except Exception as exc:
                    st.error(f"Retrieval failed: {exc}")
                    st.stop()

            topic_label = infer_topic_label(intent, matches)

            tags = []
            if intent.get("asks_official"):
                tags.append("`HK official`")
            if intent.get("asks_business_guide"):
                tags.append("`Business guide`")
            if intent.get("asks_examples"):
                tags.append("`YC examples`")
            if intent.get("asks_sentiment"):
                tags.append("`Sentiment`")
            if tags:
                st.caption(f"Routing: {' | '.join(tags)} | topic: `{topic_label}`")

            sentiment_payload = None
            sentiment_ctx = ""
            if intent.get("asks_sentiment"):
                with st.spinner("Running sentiment analysis..."):
                    sentiment_payload = _run_sentiment(query)
                if sentiment_payload:
                    sentiment_ctx = _sentiment_to_context(sentiment_payload)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_answer = ""

                for token in stream_answer(
                    query,
                    matches,
                    topic_label,
                    model=ollama_model,
                    sentiment_context=sentiment_ctx,
                ):
                    full_answer += token
                    placeholder.markdown(full_answer + "...")

                placeholder.markdown(full_answer)

                if show_sources:
                    render_sources(matches)
                if sentiment_payload:
                    render_sentiment(sentiment_payload)

            followups: list = []
            if show_followups and full_answer and not full_answer.startswith("Warning:"):
                with st.spinner("Generating follow-up questions..."):
                    followups = generate_followups(query, full_answer, model=ollama_model)
                for rank, followup in enumerate(followups, 1):
                    log_followup_event(
                        "shown",
                        turn_id,
                        original_query,
                        followup,
                        rank,
                        topic_label,
                    )
                if followups:
                    current_turn = {
                        "turn_id": turn_id,
                        "query": original_query,
                        "topic": topic_label,
                        "followups": followups,
                    }
                    with st.chat_message("assistant"):
                        render_followup_buttons(current_turn, f"turn_{turn_id}")

        st.session_state.history.append({
            "turn_id": turn_id,
            "query": original_query,
            "answer": full_answer,
            "topic": topic_label,
            "intent": intent,
            "matches": matches,
            "sentiment": sentiment_payload,
            "followups": followups,
        })

with analytics_tab:
    render_analytics_dashboard()
