from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
ANALYTICS_DIR = ROOT / "data" / "analytics"
OUT_DIR = ROOT / "outputs" / "obj5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLLOWUP_EVENTS_PATH = ANALYTICS_DIR / "followup_events.jsonl"
SUMMARY_PATH = ANALYTICS_DIR / "followup_summary.json"
JOURNEYS_PATH = ANALYTICS_DIR / "followup_journeys.csv"


# =========================
# Helpers
# =========================
def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def unique_event_key(row: dict) -> tuple:
    return (
        row.get("session_id"),
        row.get("turn_id"),
        int(row.get("followup_rank") or 0),
        str(row.get("followup_text") or "").strip(),
    )


def wrap_label(value: object, width: int = 20) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def infer_journey_stage(text: object, topic: object = "") -> str:
    joined = f"{topic or ''} {text or ''}".lower()

    if any(token in joined for token in ["sentiment", "facebook", "reddit", "twitter", "social", "reaction"]):
        return "Market Sentiment"
    if any(token in joined for token in ["fund", "invest", "yc", "seed", "valuation", "venture", "raise"]):
        return "Seeking Funding"
    if any(token in joined for token in ["register", "incorpor", "brn", "ubi", "company name", "compliance", "deregister"]):
        return "Registering a Business"
    if any(token in joined for token in ["license", "licensing", "operation", "inventory", "pricing", "stockout", "supplier"]):
        return "Operations Planning"
    if any(token in joined for token in ["validate", "customer", "market research", "demand", "mvp", "prototype", "product-market"]):
        return "Idea Validation"
    if any(token in joined for token in ["launch", "startup", "business plan", "first-time founder"]):
        return "Startup Planning"
    return "General Exploration"


def build_frames(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(events)
    if frame.empty:
        return frame, pd.DataFrame(), pd.DataFrame()

    frame["followup_rank"] = pd.to_numeric(frame.get("followup_rank"), errors="coerce").fillna(0).astype(int)
    frame["event_key"] = frame.apply(unique_event_key, axis=1)

    shown = frame[frame["event_type"] == "shown"].drop_duplicates("event_key").copy()
    clicked = frame[frame["event_type"] == "clicked"].drop_duplicates("event_key").copy()

    shown_keys = set(shown["event_key"])
    clicked["matched_shown"] = clicked["event_key"].isin(shown_keys)

    return frame, shown, clicked


def compute_metrics(shown: pd.DataFrame, clicked: pd.DataFrame) -> dict:
    total_shown = int(len(shown))
    total_clicked = int(clicked["event_key"].nunique()) if not clicked.empty else 0
    total_turns = int(shown["turn_id"].nunique()) if not shown.empty and "turn_id" in shown else 0
    clicked_turns = int(clicked["turn_id"].nunique()) if not clicked.empty and "turn_id" in clicked else 0

    rank_metrics: dict[str, dict[str, float | int]] = {}
    ranks = sorted(set(shown["followup_rank"].tolist()) | set(clicked["followup_rank"].tolist()))
    for rank in ranks:
        if rank <= 0:
            continue
        rank_shown = shown[shown["followup_rank"] == rank]
        rank_clicked = clicked[clicked["followup_rank"] == rank]
        shown_count = int(rank_shown["event_key"].nunique())
        clicked_count = int(rank_clicked["event_key"].nunique())
        rank_metrics[str(rank)] = {
            "shown": shown_count,
            "clicked": clicked_count,
            "ctr": clicked_count / shown_count if shown_count else 0.0,
        }

    if total_clicked and ranks:
        alpha = 1.0
        clicked_rank_counts = clicked["followup_rank"].value_counts().to_dict()
        denominator = total_clicked + alpha * len(ranks)
        probabilities = [
            (clicked_rank_counts.get(int(rank), 0) + alpha) / denominator
            for rank in clicked["followup_rank"].tolist()
            if int(rank) > 0
        ]
        rank_choice_perplexity = float(math.exp(-np.mean(np.log(probabilities)))) if probabilities else None
    else:
        rank_choice_perplexity = None

    return {
        "events": int(total_shown + total_clicked),
        "suggestions_shown": total_shown,
        "suggestions_clicked": total_clicked,
        "overall_ctr": total_clicked / total_shown if total_shown else 0.0,
        "turns_with_suggestions": total_turns,
        "turns_with_click": clicked_turns,
        "turn_level_prediction_accuracy": clicked_turns / total_turns if total_turns else 0.0,
        "ctr_by_rank": rank_metrics,
        "rank_choice_perplexity": rank_choice_perplexity,
        "perplexity_note": (
            "This is an empirical rank-choice perplexity proxy based on clicked suggestion ranks. "
            "True token-level model perplexity requires model log probabilities, which are not available "
            "from the current Ollama follow-up generation call."
        ),
    }


def build_journey_edges(shown: pd.DataFrame, clicked: pd.DataFrame) -> pd.DataFrame:
    edges: Counter[tuple[str, str]] = Counter()

    if not shown.empty:
        first_turns = shown.drop_duplicates("turn_id")
        for _, row in first_turns.iterrows():
            stage = infer_journey_stage(row.get("query"), row.get("topic"))
            edges[("Start", stage)] += 1

    if not clicked.empty:
        for _, row in clicked.iterrows():
            source = infer_journey_stage(row.get("query"), row.get("topic"))
            target = infer_journey_stage(row.get("followup_text"), row.get("topic"))
            edges[(source, target)] += 1

    rows = [
        {"source_stage": source, "target_stage": target, "count": count}
        for (source, target), count in edges.most_common()
    ]
    return pd.DataFrame(rows)


def save_metrics_table(metrics: dict, out_path: Path):
    rows = [
        ["Suggestions shown", metrics["suggestions_shown"]],
        ["Suggestions clicked", metrics["suggestions_clicked"]],
        ["Overall CTR", f"{metrics['overall_ctr']:.1%}"],
        ["Turns with suggestions", metrics["turns_with_suggestions"]],
        ["Turns with clicked follow-up", metrics["turns_with_click"]],
        ["Turn-level prediction accuracy", f"{metrics['turn_level_prediction_accuracy']:.1%}"],
        ["Rank-choice perplexity", f"{metrics['rank_choice_perplexity']:.3f}" if metrics["rank_choice_perplexity"] else "N/A"],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    ax.set_title("Objective 5: Follow-Up Prediction Summary", fontsize=14, pad=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_bar(metrics: dict, out_path: Path):
    labels = ["Any suggestion\nclicked", "Suggestion 1\nCTR", "Suggestion 2\nCTR", "Suggestion 3\nCTR"]
    rank_metrics = metrics["ctr_by_rank"]
    values = [
        metrics["turn_level_prediction_accuracy"],
        rank_metrics.get("1", {}).get("ctr", 0.0),
        rank_metrics.get("2", {}).get("ctr", 0.0),
        rank_metrics.get("3", {}).get("ctr", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    bars = ax.bar(labels, values, color=["#2563eb", "#0f766e", "#0f766e", "#0f766e"])
    ax.set_ylim(0, max(0.2, max(values) + 0.15))
    ax.set_ylabel("Rate")
    ax.set_title(
        f"Objective 5: Follow-Up Acceptance and CTR "
        f"(n={metrics['suggestions_shown']} shown, {metrics['suggestions_clicked']} clicked)",
        fontsize=14,
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", alpha=0.25)

    counts = [
        f"{metrics['turns_with_click']}/{metrics['turns_with_suggestions']} turns",
        f"{rank_metrics.get('1', {}).get('clicked', 0)}/{rank_metrics.get('1', {}).get('shown', 0)}",
        f"{rank_metrics.get('2', {}).get('clicked', 0)}/{rank_metrics.get('2', {}).get('shown', 0)}",
        f"{rank_metrics.get('3', {}).get('clicked', 0)}/{rank_metrics.get('3', {}).get('shown', 0)}",
    ]
    for bar, value, count_label in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.1%}\n{count_label}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ctr_by_topic(shown: pd.DataFrame, clicked: pd.DataFrame, out_path: Path):
    if shown.empty or "topic" not in shown:
        return

    shown_counts = shown.groupby("topic")["event_key"].nunique()
    clicked_counts = clicked.groupby("topic")["event_key"].nunique() if not clicked.empty else pd.Series(dtype=float)
    rows = []
    for topic, shown_count in shown_counts.items():
        clicked_count = int(clicked_counts.get(topic, 0))
        rows.append({
            "topic": str(topic),
            "shown": int(shown_count),
            "clicked": clicked_count,
            "ctr": clicked_count / shown_count if shown_count else 0.0,
        })
    topic_df = pd.DataFrame(rows).sort_values("ctr", ascending=False)
    topic_df.to_csv(OUT_DIR / "ctr_by_topic.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.55 * len(topic_df) + 1.3)))
    y_pos = np.arange(len(topic_df))
    ax.barh(y_pos, topic_df["ctr"], color="#7c3aed")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([wrap_label(label, width=26) for label in topic_df["topic"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.2, float(topic_df["ctr"].max()) + 0.15))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_xlabel("CTR")
    ax.set_title("Objective 5: Follow-Up CTR by Topic", fontsize=14)
    ax.grid(axis="x", alpha=0.25)

    for idx, row in topic_df.reset_index(drop=True).iterrows():
        value = float(row["ctr"])
        ax.text(value + 0.01, idx, f"{value:.1%} ({int(row['clicked'])}/{int(row['shown'])})", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rank_distribution(clicked: pd.DataFrame, out_path: Path):
    if clicked.empty:
        return

    ranks = [1, 2, 3]
    rank_counts = clicked["followup_rank"].value_counts().to_dict()
    labels = [f"Suggestion {rank}" for rank in ranks]
    values = [int(rank_counts.get(rank, 0)) for rank in ranks]
    total = max(1, sum(values))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values, color=["#2563eb", "#0f766e", "#64748b"])
    ax.set_ylim(0, max(1.2, max(values) + 0.8))
    ax.set_ylabel("Clicked Count")
    ax.set_title(f"Objective 5: Clicked Suggestion Rank Distribution (n={sum(values)} clicks)", fontsize=14)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.06,
            f"{value}\n{value / total:.1%}",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sankey_style_flow(edges_df: pd.DataFrame, out_path: Path):
    if edges_df.empty:
        return

    start_edges = edges_df[edges_df["source_stage"] == "Start"]
    clicked_edges = edges_df[edges_df["source_stage"] != "Start"]

    middle_nodes = sorted(
        set(start_edges["target_stage"]).union(set(clicked_edges["source_stage"]))
    )
    right_nodes = sorted(set(clicked_edges["target_stage"]))
    if not right_nodes:
        right_nodes = middle_nodes

    layer_x = {"left": 0.08, "middle": 0.45, "right": 0.82}
    positions: dict[tuple[str, str], tuple[float, float]] = {("left", "Start"): (layer_x["left"], 0.5)}

    def assign(layer: str, nodes: list[str], x: float):
        if not nodes:
            return
        ys = np.linspace(0.78, 0.22, len(nodes))
        for node, y in zip(nodes, ys):
            positions[(layer, node)] = (x, float(y))

    assign("middle", middle_nodes, layer_x["middle"])
    assign("right", right_nodes, layer_x["right"])

    max_count = max(1, int(edges_df["count"].max()))
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off")
    ax.set_title("Objective 5: User Journey Flow from Clicked Follow-Ups", fontsize=14, pad=14)

    for _, row in edges_df.iterrows():
        source = row["source_stage"]
        target = row["target_stage"]
        count = int(row["count"])

        if source == "Start":
            source_key = ("left", "Start")
            target_key = ("middle", target)
        else:
            source_key = ("middle", source)
            target_key = ("right", target)
        if source_key not in positions or target_key not in positions:
            continue

        x1, y1 = positions[source_key]
        x2, y2 = positions[target_key]
        connection = "arc3,rad=0.16" if y1 != y2 else "arc3,rad=0.02"
        ax.annotate(
            "",
            xy=(x2 - 0.06, y2),
            xytext=(x1 + 0.06, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 1.4 + 4.0 * count / max_count,
                "alpha": 0.55,
                "color": "#2563eb",
                "connectionstyle": connection,
            },
        )
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.035,
            str(count),
            fontsize=9,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1"},
        )

    for (_, node), (x, y) in positions.items():
        ax.text(
            x,
            y,
            wrap_label(node, width=16),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#334155"},
        )

    ax.text(layer_x["left"], 0.96, "Entry", ha="center", fontsize=10, color="#475569")
    ax.text(layer_x["middle"], 0.96, "Current Need", ha="center", fontsize=10, color="#475569")
    ax.text(layer_x["right"], 0.96, "Predicted Next Need", ha="center", fontsize=10, color="#475569")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    events = read_jsonl(FOLLOWUP_EVENTS_PATH)
    frame, shown, clicked = build_frames(events)
    if frame.empty:
        raise FileNotFoundError(
            f"No follow-up events found at {FOLLOWUP_EVENTS_PATH}.\n"
            "Run the Streamlit chatbot, ask questions with follow-up generation enabled, "
            "and click suggested follow-up questions first."
        )

    metrics = compute_metrics(shown, clicked)
    edges_df = build_journey_edges(shown, clicked)

    shown.to_csv(OUT_DIR / "followup_shown_events.csv", index=False)
    clicked.to_csv(OUT_DIR / "followup_clicked_events.csv", index=False)
    edges_df.to_csv(OUT_DIR / "journey_stage_edges.csv", index=False)

    save_metrics_table(metrics, OUT_DIR / "followup_metrics_table.png")
    save_accuracy_bar(metrics, OUT_DIR / "prediction_accuracy_bar.png")
    save_ctr_by_topic(shown, clicked, OUT_DIR / "ctr_by_topic.png")
    save_rank_distribution(clicked, OUT_DIR / "clicked_rank_distribution.png")
    save_sankey_style_flow(edges_df, OUT_DIR / "user_journey_sankey.png")

    manifest = {
        **metrics,
        "input_log": str(FOLLOWUP_EVENTS_PATH.relative_to(ROOT)),
        "legacy_summary": str(SUMMARY_PATH.relative_to(ROOT)) if SUMMARY_PATH.exists() else None,
        "legacy_journeys": str(JOURNEYS_PATH.relative_to(ROOT)) if JOURNEYS_PATH.exists() else None,
        "outputs": {
            "user_journey_sankey": "outputs/obj5/user_journey_sankey.png",
            "prediction_accuracy_bar": "outputs/obj5/prediction_accuracy_bar.png",
            "followup_metrics_table": "outputs/obj5/followup_metrics_table.png",
            "ctr_by_topic": "outputs/obj5/ctr_by_topic.png",
            "clicked_rank_distribution": "outputs/obj5/clicked_rank_distribution.png",
            "journey_stage_edges": "outputs/obj5/journey_stage_edges.csv",
        },
    }
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print("[DONE] Objective 5 figures generated.")
    print(f"[DONE] Output directory: {OUT_DIR}")
    print(f"[DONE] Suggestions shown: {metrics['suggestions_shown']}")
    print(f"[DONE] Suggestions clicked: {metrics['suggestions_clicked']}")
    print(f"[DONE] Overall CTR: {metrics['overall_ctr']:.4f}")
    print(f"[DONE] Turn-level prediction accuracy: {metrics['turn_level_prediction_accuracy']:.4f}")
    if metrics["rank_choice_perplexity"]:
        print(f"[DONE] Rank-choice perplexity proxy: {metrics['rank_choice_perplexity']:.4f}")


if __name__ == "__main__":
    main()
