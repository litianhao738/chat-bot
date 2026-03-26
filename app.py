#!/usr/bin/env python3
"""
Streamlit demo for Project 8 objectives 3-6 plus supplementary sentiment assets.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from textwrap import fill
from typing import Dict, List

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from rag.generate_answer import build_answer, polish_answer_with_ollama
from rag.generate_followups import build_followups
from rag.retrieve_chunks import load_pickle, load_topic_catalog, predict_topic, retrieve
from sentiment_assets.sentiment_engine import analyze_text, should_analyze_sentiment

ROOT_DIR = Path(__file__).resolve().parent
INDEX_PATH = "rag/output/retrieval_index.pkl"
CLASSIFIER_PATH = "classification/output/inquiry_topic_classifier.pkl"
TOPIC_CATALOG_PATH = "classification/output/topic_catalog.json"
CLASSIFIER_REPORT_PATH = "classification/output/inquiry_topic_classifier_report.json"
RETRIEVAL_SUMMARY_PATH = "rag/output/retrieval_index_summary.json"
PROJECT_SUMMARY_PATH = "project8_kb/summary.json"
CLEANED_COMPANY_KB_PATH = "project8_kb/cleaned_company_kb.jsonl"
SENTIMENT_SUMMARY_PATH = str(ROOT_DIR / "sentiment_assets" / "output" / "sentiment_summary.json")
SENTIMENT_KEYWORDS_PATH = str(ROOT_DIR / "sentiment_assets" / "output" / "sentiment_keyword_profiles.json")
SENTIMENT_CLEANED_PATH = str(ROOT_DIR / "sentiment_assets" / "output" / "sentiment_cleaned.jsonl")


def load_optional_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def load_jsonl_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sentiment_assets_ready() -> bool:
    return (
        os.path.exists(SENTIMENT_SUMMARY_PATH)
        and os.path.exists(SENTIMENT_KEYWORDS_PATH)
        and os.path.exists(SENTIMENT_CLEANED_PATH)
    )


def analytics_dependencies_ready() -> bool:
    return pd is not None


def wrap_axis_label(value: object, width: int = 18) -> str:
    text = str(value).strip()
    if not text:
        return "Unknown"
    return fill(text, width=width)


def build_source_family_frame() -> pd.DataFrame:
    summary = load_optional_json(RETRIEVAL_SUMMARY_PATH)
    frame = pd.DataFrame(
        [
            {"source_family": source_family, "count": count}
            for source_family, count in (summary.get("source_family_counts", {}) or {}).items()
        ]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("source_family")


def build_topic_distribution_frame(limit: int = 10) -> pd.DataFrame:
    counts = Counter(
        str(row.get("topic_label", "unknown")).strip() or "unknown"
        for row in load_jsonl_rows(CLEANED_COMPANY_KB_PATH)
    )
    frame = pd.DataFrame(
        [{"topic_label": label, "count": count} for label, count in counts.items()]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).head(limit).set_index("topic_label")


def build_classifier_performance_frame() -> pd.DataFrame:
    report = load_optional_json(CLASSIFIER_REPORT_PATH)
    rows = []
    for label, metrics in (report.get("per_label_report", {}) or {}).items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "topic_label": label,
                "f1_score": round(float(metrics.get("f1-score", 0.0)), 4),
                "precision": round(float(metrics.get("precision", 0.0)), 4),
                "recall": round(float(metrics.get("recall", 0.0)), 4),
                "support": int(metrics.get("support", 0)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("f1_score", ascending=False)


def build_sentiment_distribution_frame() -> pd.DataFrame:
    summary = load_optional_json(SENTIMENT_SUMMARY_PATH)
    frame = pd.DataFrame(
        [
            {"sentiment": label, "count": count}
            for label, count in (summary.get("dataset_sentiment_distribution", {}) or {}).items()
        ]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("sentiment")


def build_sentiment_platform_frame() -> pd.DataFrame:
    rows = load_jsonl_rows(SENTIMENT_CLEANED_PATH)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["platform", "dataset_sentiment"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    return grouped


def build_engagement_frame() -> pd.DataFrame:
    rows = load_jsonl_rows(SENTIMENT_CLEANED_PATH)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["likes"] = pd.to_numeric(frame.get("likes"), errors="coerce").fillna(0.0)
    frame["retweets"] = pd.to_numeric(frame.get("retweets"), errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby("platform")[["likes", "retweets"]]
        .mean()
        .round(2)
        .sort_values("likes", ascending=False)
        .reset_index()
    )
    return grouped


def build_keyword_profile_frame(label: str, limit: int = 10) -> pd.DataFrame:
    profiles = load_optional_json(SENTIMENT_KEYWORDS_PATH)
    items = ((profiles.get("top_keywords_by_sentiment", {}) or {}).get(label, []) or [])[:limit]
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    return frame.rename(columns={"value": "keyword"}).set_index("keyword")


def build_session_retrieval_mix_frame(chat_history: List[Dict[str, object]]) -> pd.DataFrame:
    counts = Counter()
    for turn in chat_history:
        for match in turn.get("matches", []):
            counts[str(match.get("source_family", "unknown"))] += 1

    frame = pd.DataFrame(
        [{"source_family": source_family, "count": count} for source_family, count in counts.items()]
    )
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False).set_index("source_family")


def render_horizontal_bar_chart(
    st,
    frame: pd.DataFrame,
    label_column: str,
    value_column: str,
    color: str,
    height: int = 340,
) -> None:
    chart_frame = frame.reset_index().copy()
    chart_frame["wrapped_label"] = chart_frame[label_column].apply(wrap_axis_label)
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
            "x": {
                "field": value_column,
                "type": "quantitative",
                "axis": {"title": None, "labelAngle": 0},
            },
            "tooltip": [
                {"field": label_column, "type": "nominal", "title": "Label"},
                {"field": value_column, "type": "quantitative", "title": "Value"},
            ],
            "color": {"value": color},
        },
        "height": height,
    }
    st.vega_lite_chart(chart_spec, use_container_width=True)


def render_grouped_sentiment_chart(st, frame: pd.DataFrame, height: int = 340) -> None:
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
            "x": {
                "field": "platform",
                "type": "nominal",
                "axis": {"title": None, "labelAngle": 0},
            },
            "y": {
                "field": "count",
                "type": "quantitative",
                "axis": {"title": None, "labelAngle": 0},
            },
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


def render_scatter_with_labels(st, frame: pd.DataFrame, height: int = 340) -> None:
    chart_spec = {
        "data": {"values": frame.to_dict(orient="records")},
        "layer": [
            {
                "mark": {"type": "circle", "size": 220, "opacity": 0.8},
                "encoding": {
                    "x": {"field": "likes", "type": "quantitative", "axis": {"title": "Average Likes", "labelAngle": 0}},
                    "y": {"field": "retweets", "type": "quantitative", "axis": {"title": "Average Retweets", "labelAngle": 0}},
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


def build_key_insights(chat_history: List[Dict[str, object]]) -> List[Dict[str, str]]:
    insights: List[Dict[str, str]] = []
    retrieval_summary = load_optional_json(RETRIEVAL_SUMMARY_PATH)
    classifier_report = load_optional_json(CLASSIFIER_REPORT_PATH)
    project_summary = load_optional_json(PROJECT_SUMMARY_PATH)

    source_counts = retrieval_summary.get("source_family_counts", {}) or {}
    if source_counts:
        dominant_source = max(source_counts.items(), key=lambda item: item[1])
        insights.append({
            "title": "Data Coverage",
            "detail": (
                f"The retrieval index is anchored by {dominant_source[0]} with "
                f"{dominant_source[1]} indexed chunks, so specialized sources are important for balance."
            ),
        })

    metrics = classifier_report.get("metrics", {}) or {}
    accuracy = float(metrics.get("accuracy", 0.0))
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    insights.append({
        "title": "Model Performance",
        "detail": (
            f"The inquiry router is usable for demo purposes with accuracy {accuracy:.4f} "
            f"and macro F1 {macro_f1:.4f}, but it still behaves like a lightweight router "
            f"rather than a high-confidence classifier."
        ),
    })

    per_label_report = classifier_report.get("per_label_report", {}) or {}
    label_rows = []
    for label, label_metrics in per_label_report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        label_rows.append(
            {
                "topic_label": str(label),
                "f1_score": float(label_metrics.get("f1-score", 0.0)),
            }
        )

    if label_rows:
        weakest = min(label_rows, key=lambda item: item["f1_score"])
        strongest = max(label_rows, key=lambda item: item["f1_score"])
        insights.append({
            "title": "Topic Diagnosis",
            "detail": (
                f"The strongest topic label is {strongest['topic_label']} at F1 {strongest['f1_score']:.4f}, "
                f"while {weakest['topic_label']} is currently the weakest and would benefit most from more training examples."
            ),
        })

    if sentiment_assets_ready():
        sentiment_summary = load_optional_json(SENTIMENT_SUMMARY_PATH)
        distribution = sentiment_summary.get("dataset_sentiment_distribution", {}) or {}
        neutral_count = int(distribution.get("Neutral", 0))
        total_rows = int(sentiment_summary.get("row_count", 0))
        if total_rows:
            neutral_share = neutral_count / total_rows
            insights.append({
                "title": "Data Limitation",
                "detail": (
                    f"The sentiment dataset is highly neutral-heavy: {neutral_count}/{total_rows} rows "
                    f"({neutral_share:.1%}) are neutral, so the sentiment lens should be presented as supplementary evidence."
                ),
            })

    if chat_history:
        session_mix = build_session_retrieval_mix_frame(chat_history)
        if not session_mix.empty:
            top_session_source = session_mix.reset_index().iloc[0]
            insights.append({
                "title": "Live Demo Insight",
                "detail": (
                    f"In the current session, {top_session_source['source_family']} has been used most often, "
                    f"which helps explain the answer style the user is seeing in the demo."
                ),
            })

    n_companies = int(project_summary.get("n_companies", 0))
    n_topics = int(project_summary.get("n_topics", 0))
    insights.append({
        "title": "Knowledge Base Scale",
        "detail": (
            f"The core YC knowledge base currently covers {n_companies} companies across "
            f"{n_topics} topic clusters, which gives the chatbot broad startup-pattern coverage."
        ),
    })
    return insights


def render_key_insights(st, chat_history: List[Dict[str, object]]) -> None:
    insights = build_key_insights(chat_history)
    if not insights:
        st.info("Key insights will appear here after the project summaries are loaded.")
        return

    lines = []
    for item in insights:
        title = item.get("title", "Insight").strip()
        detail = item.get("detail", "").strip()
        lines.append(f"**{title}**")
        lines.append(detail)
        lines.append("")

    st.markdown("\n".join(lines).strip())


def run_chat_turn(
    query: str,
    index: Dict[str, object],
    topic_catalog: List[Dict[str, object]],
    use_local_polish: bool,
    ollama_model: str,
    top_k: int,
    topic_candidates: int,
) -> Dict[str, object]:
    predicted_topic, topic_ranking = predict_topic(
        query=query,
        classifier_path=CLASSIFIER_PATH,
        topic_catalog_path=TOPIC_CATALOG_PATH,
    )
    matches = retrieve(
        query=query,
        index=index,
        topic_ranking=topic_ranking,
        topic_catalog=topic_catalog,
        top_k=top_k,
        topic_candidates=topic_candidates,
    )
    answer_payload = build_answer(query=query, predicted_topic=predicted_topic, matches=matches)
    if use_local_polish:
        answer_payload["llm_polish"] = polish_answer_with_ollama(
            payload=answer_payload,
            model=ollama_model.strip() or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
            timeout_seconds=120,
        )
        if answer_payload["llm_polish"].get("used") and answer_payload["llm_polish"].get("text"):
            answer_payload["final_answer"] = answer_payload["llm_polish"]["text"]
            answer_payload["answer_mode"] = "deterministic_structured_plus_local_polish"

    followup_payload = build_followups(answer_payload)
    sentiment_payload = None
    sentiment_trigger = should_analyze_sentiment(query)
    if sentiment_assets_ready() and sentiment_trigger.get("should_run"):
        try:
            sentiment_payload = analyze_text(
                text=query,
                summary_path=SENTIMENT_SUMMARY_PATH,
                keywords_path=SENTIMENT_KEYWORDS_PATH,
            )
            sentiment_payload["trigger"] = sentiment_trigger
        except Exception as exc:
            sentiment_payload = {
                "error": str(exc),
                "predicted_sentiment": "Unavailable",
                "reasoning": ["Sentiment analysis could not be completed for this query."],
                "trigger": sentiment_trigger,
            }

    return {
        "query": query,
        "predicted_topic": predicted_topic,
        "topic_ranking": topic_ranking,
        "matches": matches,
        "answer_payload": answer_payload,
        "followup_payload": followup_payload,
        "sentiment_payload": sentiment_payload,
        "sentiment_trigger": sentiment_trigger,
    }


def render_sentiment_snapshot(st, summary: Dict[str, object]) -> None:
    st.subheader("Sentiment Snapshot")
    left_col, right_col = st.columns(2)
    with left_col:
        st.metric("Rows", summary.get("row_count", 0))
        st.metric("VADER Agreement", summary.get("vader_agreement_rate", 0.0))
    with right_col:
        st.write("Dataset Sentiment Distribution")
        st.json(summary.get("dataset_sentiment_distribution", {}))


def render_turn(st, turn: Dict[str, object], turn_number: int) -> None:
    query = turn["query"]
    answer_payload = turn["answer_payload"]
    followup_payload = turn["followup_payload"]
    sentiment_payload = turn.get("sentiment_payload")
    sentiment_trigger = turn.get("sentiment_trigger", {})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(answer_payload.get("final_answer", answer_payload.get("answer_text", "")))

        with st.expander(f"Turn {turn_number} Details", expanded=False):
            st.subheader("Predicted Topic")
            st.write(turn.get("predicted_topic", "unknown"))
            st.json(turn.get("topic_ranking", []))

            st.subheader("Answer Sections")
            st.json(answer_payload.get("answer_sections", {}))

            if answer_payload.get("evidence_gaps"):
                st.subheader("Evidence Gaps")
                for gap in answer_payload["evidence_gaps"]:
                    st.write(f"- {gap}")

            if answer_payload.get("reference_examples"):
                st.subheader("Reference Examples")
                for item in answer_payload["reference_examples"]:
                    st.write(f"- {item}")

            st.subheader("Recommendation")
            st.write(answer_payload.get("recommendation", ""))

            st.subheader("Suggested Next Step")
            st.write(answer_payload.get("suggested_next_step", ""))

            st.subheader("Evidence")
            for item in answer_payload.get("evidence", []):
                score = item.get("score", "")
                with st.expander(f"{item.get('company_name', 'Unknown')} ({score})"):
                    if item.get("source_name"):
                        st.write(f"Source: {item['source_name']}")
                    if item.get("tagline"):
                        st.write(item["tagline"])
                    if item.get("keywords"):
                        st.write(", ".join(item["keywords"]))
                    st.write(item.get("summary_text", ""))
                    if item.get("retrieval_reason"):
                        st.write("Reasons: " + "; ".join(item["retrieval_reason"]))
                    if item.get("url"):
                        st.write(item["url"])

            st.subheader("Suggested Follow-up Questions")
            for question in followup_payload.get("follow_up_questions", []):
                st.write(f"- {question}")

            if sentiment_payload:
                st.subheader("Sentiment Lens")
                if sentiment_payload.get("error"):
                    st.write(sentiment_payload["error"])
                else:
                    if sentiment_payload.get("trigger", {}).get("reason"):
                        st.write("Trigger: " + sentiment_payload["trigger"]["reason"])
                    st.write(f"Predicted Sentiment: {sentiment_payload.get('predicted_sentiment', 'Unknown')}")
                    st.write(f"Combined Score: {sentiment_payload.get('combined_score', 'N/A')}")
                    if sentiment_payload.get("reasoning"):
                        st.write("Reasoning")
                        for item in sentiment_payload["reasoning"]:
                            st.write(f"- {item}")
                    if sentiment_payload.get("matched_keywords"):
                        st.write("Matched Keywords")
                        st.json(sentiment_payload["matched_keywords"])

            if answer_payload.get("llm_polish"):
                st.subheader("Local Polish Status")
                st.json(answer_payload["llm_polish"])


def render_overview_metrics(st) -> None:
    retrieval_summary = load_optional_json(RETRIEVAL_SUMMARY_PATH)
    classifier_report = load_optional_json(CLASSIFIER_REPORT_PATH)
    project_summary = load_optional_json(PROJECT_SUMMARY_PATH)
    sentiment_summary = load_optional_json(SENTIMENT_SUMMARY_PATH) if sentiment_assets_ready() else {}

    metric_columns = st.columns(5)
    metric_columns[0].metric("Companies", int(project_summary.get("n_companies", 0)))
    metric_columns[1].metric("Indexed Docs", int(retrieval_summary.get("n_docs", 0)))
    metric_columns[2].metric("Topics", int(retrieval_summary.get("n_topics", 0)))
    metric_columns[3].metric(
        "Classifier Accuracy",
        round(float(((classifier_report.get("metrics", {}) or {}).get("accuracy", 0.0))), 4),
    )
    metric_columns[4].metric("Sentiment Rows", int(sentiment_summary.get("row_count", 0)))


def render_analytics_dashboard(st, chat_history: List[Dict[str, object]]) -> None:
    st.subheader("Analytics Dashboard")
    st.caption(
        "This dashboard summarizes the current data sources, topic coverage, classifier performance, "
        "and supplementary social-media sentiment analytics behind the chatbot."
    )

    if not analytics_dependencies_ready():
        st.warning("Analytics dashboard requires pandas. Please run: python -m pip install pandas")
        return

    render_overview_metrics(st)
    st.divider()

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("**Knowledge Source Distribution**")
        source_frame = build_source_family_frame()
        if not source_frame.empty:
            render_horizontal_bar_chart(st, source_frame, "source_family", "count", color="#2b6cb0", height=250)
            st.caption("The unified retrieval index is still YC-heavy, with Hong Kong official pages and business guides as focused supplements.")
        else:
            st.info("Retrieval source summary is not available yet.")

    with top_right:
        st.markdown("**Top Knowledge-Base Topics**")
        topic_frame = build_topic_distribution_frame(limit=10)
        if not topic_frame.empty:
            render_horizontal_bar_chart(st, topic_frame, "topic_label", "count", color="#dd6b20", height=360)
            st.caption("These are the most common startup themes represented in the current knowledge base.")
        else:
            st.info("Topic distribution is not available yet.")

    middle_left, middle_right = st.columns(2)
    with middle_left:
        st.markdown("**Classifier F1 Score by Topic**")
        classifier_frame = build_classifier_performance_frame()
        if not classifier_frame.empty:
            render_horizontal_bar_chart(st, classifier_frame, "topic_label", "f1_score", color="#2f855a", height=430)
            weakest = classifier_frame.sort_values("f1_score", ascending=True).head(5)
            st.caption("Lowest-F1 topics highlight where routing is still weaker and more training examples may help.")
            st.dataframe(weakest, use_container_width=True)
        else:
            st.info("Classifier evaluation report is not available yet.")

    with middle_right:
        st.markdown("**Sentiment Label Distribution**")
        if sentiment_assets_ready():
            sentiment_distribution = build_sentiment_distribution_frame()
            if not sentiment_distribution.empty:
                render_horizontal_bar_chart(st, sentiment_distribution, "sentiment", "count", color="#805ad5", height=250)
                st.caption("The uploaded Kaggle dataset is heavily neutral, which is useful to mention as a project limitation.")
            else:
                st.info("Sentiment summary is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("**Platform by Sentiment**")
        if sentiment_assets_ready():
            platform_frame = build_sentiment_platform_frame()
            if not platform_frame.empty:
                render_grouped_sentiment_chart(st, platform_frame, height=320)
                st.caption("This chart shows how sentiment-labelled posts are distributed across Twitter, Instagram, and Facebook.")
            else:
                st.info("Platform-level sentiment data is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    with lower_right:
        st.markdown("**Average Engagement by Platform**")
        if sentiment_assets_ready():
            engagement_frame = build_engagement_frame()
            if not engagement_frame.empty:
                render_scatter_with_labels(st, engagement_frame, height=320)
                st.dataframe(engagement_frame, use_container_width=True)
                st.caption("Average likes and retweets provide a simple descriptive-analytics view of social-media engagement.")
            else:
                st.info("Engagement data is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    keyword_left, keyword_right = st.columns(2)
    with keyword_left:
        st.markdown("**Top Positive Keywords**")
        if sentiment_assets_ready():
            positive_keywords = build_keyword_profile_frame("Positive", limit=10)
            if not positive_keywords.empty:
                render_horizontal_bar_chart(st, positive_keywords, "keyword", "count", color="#38a169", height=340)
            else:
                st.info("Positive keyword profile is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    with keyword_right:
        st.markdown("**Top Neutral Keywords**")
        if sentiment_assets_ready():
            neutral_keywords = build_keyword_profile_frame("Neutral", limit=10)
            if not neutral_keywords.empty:
                render_horizontal_bar_chart(st, neutral_keywords, "keyword", "count", color="#718096", height=340)
            else:
                st.info("Neutral keyword profile is not available yet.")
        else:
            st.info("Build sentiment assets to unlock this chart.")

    st.divider()
    st.markdown("**Current Session Retrieval Mix**")
    session_frame = build_session_retrieval_mix_frame(chat_history)
    if not session_frame.empty:
        render_horizontal_bar_chart(st, session_frame, "source_family", "count", color="#319795", height=220)
        st.caption("This chart updates during the demo and shows which source families were actually used across the current chat session.")
    else:
        st.info("Ask a few questions in the chatbot tab to generate a live retrieval-source mix for this session.")

    st.markdown("**Key Insights**")
    render_key_insights(st, chat_history)

    st.markdown("**Diagnostic Notes**")
    st.write("- The retrieval index is dominated by YC data, so specialized source families are important for Hong Kong procedures and general business guidance.")
    st.write("- The inquiry classifier is usable for routing, but some topic labels still have weak F1 scores and low confidence.")
    st.write("- The sentiment dataset is highly imbalanced toward neutral posts, so the sentiment lens should be presented as supplementary evidence rather than a final decision tool.")


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed.\n"
            "Please run:\n"
            "  python -m pip install streamlit"
        ) from exc

    st.set_page_config(page_title="Entrepreneurship Chatbot", layout="wide")
    st.title("Entrepreneurship Chatbot")
    st.caption("Multi-source retrieval, structured answering, follow-up prediction, and supplementary sentiment analysis.")

    if not os.path.exists(INDEX_PATH):
        st.error("Retrieval index not found. Please run: python rag/build_retrieval_index.py")
        st.stop()

    index = load_pickle(INDEX_PATH)
    topic_catalog = load_topic_catalog(TOPIC_CATALOG_PATH)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Settings")
        use_local_polish = st.checkbox("Use local 3B polishing via Ollama", value=False)
        ollama_model = st.text_input("Ollama model", value=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
        top_k = st.slider("Top K Retrieval", min_value=3, max_value=8, value=5)
        topic_candidates = st.slider("Topic Candidates", min_value=1, max_value=5, value=3)
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        if sentiment_assets_ready():
            render_sentiment_snapshot(st, load_optional_json(SENTIMENT_SUMMARY_PATH))
        else:
            st.info("Sentiment assets not found yet. Run: python sentiment_assets/build_sentiment_assets.py")

    chatbot_tab, analytics_tab = st.tabs(["Chatbot", "Analytics Dashboard"])

    with chatbot_tab:
        for idx, turn in enumerate(st.session_state.chat_history, start=1):
            render_turn(st, turn, idx)

        st.divider()
        st.caption("Type a question and press Enter to send.")
        with st.form("chat_query_form", clear_on_submit=True):
            query = st.text_input("Ask a startup / entrepreneurship question", key="chat_query_input")
            action_left, action_right = st.columns([1, 1])
            submit_query = action_left.form_submit_button("Send")
            clear_chat = action_right.form_submit_button("Clear Chat History")

        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()

        if submit_query and query and query.strip():
            turn = run_chat_turn(
                query=query.strip(),
                index=index,
                topic_catalog=topic_catalog,
                use_local_polish=use_local_polish,
                ollama_model=ollama_model,
                top_k=top_k,
                topic_candidates=topic_candidates,
            )
            st.session_state.chat_history.append(turn)
            st.rerun()

    with analytics_tab:
        render_analytics_dashboard(st, st.session_state.chat_history)


if __name__ == "__main__":
    main()
