from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "evaluation"
OUT_DIR = ROOT / "outputs" / "obj4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_RESULTS_PATH = DATA_DIR / "evaluation_results.csv"
TOPK_SAMPLES_PATH = DATA_DIR / "topk_retrieval_samples.csv"
SUMMARY_PATH = DATA_DIR / "summary.json"


# =========================
# Helpers
# =========================
def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Please run answer evaluation first:\n"
            f".venv\\Scripts\\python.exe evaluation\\run_evaluation.py --with-generation"
        )


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wrap_text(value: object, width: int = 46) -> str:
    text = str(value or "").strip()
    return "\n".join(textwrap.wrap(text, width=width))


def save_table_image(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    font_size: int = 8,
    row_height: float = 0.9,
    col_width: float = 2.7,
):
    n_rows, n_cols = df.shape
    fig_w = max(10, n_cols * col_width)
    fig_h = max(3.2, n_rows * row_height + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.65)
    for cell in table.get_celld().values():
        cell.set_height(cell.get_height() * 1.25)
    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def truncate_words(value: object, max_words: int = 18) -> str:
    words = str(value or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def parse_topk_json(value: object) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def save_rouge_bar(summary: dict, out_path: Path):
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    values = [
        float(summary.get("rouge_1", 0.0)),
        float(summary.get("rouge_2", 0.0)),
        float(summary.get("rouge_l", 0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, max(0.2, max(values) + 0.05))
    ax.set_ylabel("Score")
    ax.set_title("Objective 4: Answer Similarity Scores", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_source_counts(samples_df: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for _, row in samples_df.iterrows():
        for rank in [1, 2, 3]:
            payload = parse_topk_json(row.get(f"top_{rank}"))
            source = payload.get("source")
            if source:
                counts[source] = counts.get(source, 0) + 1
    return pd.DataFrame(
        [{"source_family": key, "count": value} for key, value in sorted(counts.items())]
    )


def save_source_pie(source_df: pd.DataFrame, out_path: Path):
    if source_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.pie(
        source_df["count"],
        labels=source_df["source_family"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Source Attribution Across Retrieved Answer Contexts", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_answer_length_reduction(samples_df: pd.DataFrame, out_path: Path):
    df = samples_df.copy()
    df["gold_len"] = df["gold_answer"].fillna("").astype(str).str.split().map(len)
    df["answer_len"] = df["answer_preview"].fillna("").astype(str).str.split().map(len)
    df["compression_ratio"] = np.where(df["answer_len"] > 0, df["gold_len"] / df["answer_len"], 0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(min(20, len(df)))
    subset = df.head(20)
    width = 0.35
    ax.bar(x - width / 2, subset["answer_len"], width, label="Answer / Retrieved Context Preview")
    ax.bar(x + width / 2, subset["gold_len"], width, label="Gold Answer")
    ax.set_xlabel("Evaluation Query Index")
    ax.set_ylabel("Word Count")
    ax.set_title("Answer Preview vs Gold Answer Length", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return df[["query", "answer_len", "gold_len", "compression_ratio"]]


def save_rouge_by_topic(eval_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rouge_cols = ["rouge_1", "rouge_2", "rouge_l"]
    df = eval_df.copy()
    for col in rouge_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    topic_df = (
        df.groupby("expected_topic_group")[rouge_cols]
        .mean()
        .reset_index()
        .sort_values("expected_topic_group")
    )

    x = np.arange(len(topic_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(x - width, topic_df["rouge_1"], width, label="ROUGE-1")
    ax.bar(x, topic_df["rouge_2"], width, label="ROUGE-2")
    ax.bar(x + width, topic_df["rouge_l"], width, label="ROUGE-L")
    ax.set_ylabel("Average Score")
    ax.set_xlabel("Topic Group")
    ax.set_title("Answer Similarity Scores by Topic Group", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_text(t, width=16) for t in topic_df["expected_topic_group"]])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return topic_df


def build_comparison_table(samples_df: pd.DataFrame, max_rows: int = 2) -> pd.DataFrame:
    rows = []
    for _, row in samples_df.head(max_rows).iterrows():
        top1 = parse_topk_json(row.get("top_1"))
        rows.append({
            "Query": wrap_text(truncate_words(row.get("query"), 8), width=20),
            "Top Retrieved Snippet": wrap_text(truncate_words(top1.get("snippet", ""), 12), width=24),
            "Generated Answer Preview": wrap_text(truncate_words(row.get("answer_preview"), 12), width=24),
            "Gold Answer": wrap_text(truncate_words(row.get("gold_answer"), 10), width=22),
        })
    return pd.DataFrame(rows)


def save_comparison_cards(samples_df: pd.DataFrame, out_path: Path, max_rows: int = 2):
    rows = []
    for _, row in samples_df.head(max_rows).iterrows():
        top1 = parse_topk_json(row.get("top_1"))
        rows.append({
            "query": truncate_words(row.get("query"), 12),
            "source": f"{top1.get('source', '')} | {top1.get('id', '')} | score {float(top1.get('score', 0.0)):.3f}",
            "generated": truncate_words(row.get("answer_preview"), 22),
            "gold": truncate_words(row.get("gold_answer"), 18),
        })

    if not rows:
        return

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=1,
        figsize=(16.5, 4.8 * len(rows)),
        squeeze=False,
    )
    fig.suptitle("Objective 4: Retrieved Source vs Generated Answer (Sample)", fontsize=17, y=0.99)

    for idx, item in enumerate(rows):
        ax = axes[idx, 0]
        ax.axis("off")

        ax.add_patch(
            plt.Rectangle(
                (0.015, 0.05),
                0.96,
                0.90,
                fill=False,
                linewidth=1.2,
                transform=ax.transAxes,
            )
        )

        ax.text(
            0.04,
            0.88,
            f"Case {idx + 1} Query",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.04,
            0.80,
            wrap_text(item["query"], width=95),
            fontsize=10,
            transform=ax.transAxes,
            va="top",
        )

        ax.text(
            0.04,
            0.62,
            "Top Retrieved Source",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.04,
            0.54,
            wrap_text(item["source"], width=88),
            fontsize=10,
            transform=ax.transAxes,
            va="top",
        )

        ax.text(
            0.04,
            0.34,
            "Generated Answer Preview",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.04,
            0.26,
            wrap_text(item["generated"], width=68),
            fontsize=10,
            transform=ax.transAxes,
            va="top",
        )

        ax.text(
            0.53,
            0.34,
            "Gold Answer",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.53,
            0.26,
            wrap_text(item["gold"], width=56),
            fontsize=10,
            transform=ax.transAxes,
            va="top",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.8)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_full_comparison_csv(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in samples_df.iterrows():
        top1 = parse_topk_json(row.get("top_1"))
        rows.append({
            "query": row.get("query", ""),
            "top_retrieved_source": top1.get("source", ""),
            "top_retrieved_id": top1.get("id", ""),
            "top_retrieved_score": top1.get("score", ""),
            "top_retrieved_snippet": top1.get("snippet", ""),
            "generated_answer_preview": row.get("answer_preview", ""),
            "gold_answer": row.get("gold_answer", ""),
        })
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    require_file(EVALUATION_RESULTS_PATH)
    require_file(TOPK_SAMPLES_PATH)
    require_file(SUMMARY_PATH)

    eval_df = pd.read_csv(EVALUATION_RESULTS_PATH)
    samples_df = pd.read_csv(TOPK_SAMPLES_PATH)
    summary = load_summary(SUMMARY_PATH)

    mode = summary.get("answer_metric_mode", "unknown")
    if mode != "generated_answer":
        print("[WARN] Current evaluation summary is not generated-answer mode.")
        print("[WARN] Run this first for final Objective 4 figures:")
        print("[WARN] .venv\\Scripts\\python.exe evaluation\\run_evaluation.py --with-generation")
        print(f"[WARN] Current mode: {mode}")

    save_rouge_bar(summary, OUT_DIR / "rouge_scores_bar.png")

    source_df = build_source_counts(samples_df)
    source_df.to_csv(OUT_DIR / "source_attribution_counts.csv", index=False)
    save_source_pie(source_df, OUT_DIR / "source_attribution_pie.png")

    length_df = save_answer_length_reduction(samples_df, OUT_DIR / "answer_length_comparison.png")
    length_df.to_csv(OUT_DIR / "answer_length_stats.csv", index=False)

    topic_df = save_rouge_by_topic(eval_df, OUT_DIR / "rouge_by_topic_group.png")
    topic_df.to_csv(OUT_DIR / "rouge_by_topic_group.csv", index=False)

    metrics_df = pd.DataFrame([
        ["Evaluation Mode", mode],
        ["Test Queries", int(summary.get("test_query_count", len(eval_df)))],
        ["Generated Answers", summary.get("answer_mode_counts", {}).get("generated_answer", 0)],
        ["ROUGE-1", f"{float(summary.get('rouge_1', 0.0)):.4f}"],
        ["ROUGE-2", f"{float(summary.get('rouge_2', 0.0)):.4f}"],
        ["ROUGE-L", f"{float(summary.get('rouge_l', 0.0)):.4f}"],
        ["Relevant Recall@5", f"{float(summary.get('relevant_recall_at_5', 0.0)):.4f}"],
        ["Relevant MRR", f"{float(summary.get('relevant_mrr', 0.0)):.4f}"],
        ["Relevant NDCG@5", f"{float(summary.get('relevant_ndcg_at_5', 0.0)):.4f}"],
    ], columns=["Metric", "Value"])
    save_table_image(metrics_df, "Objective 4: Answer Evaluation Summary", OUT_DIR / "answer_evaluation_summary_table.png")

    full_comparison = build_full_comparison_csv(samples_df)
    full_comparison.to_csv(OUT_DIR / "retrieved_vs_answer_full.csv", index=False)

    save_comparison_cards(
        samples_df,
        OUT_DIR / "retrieved_vs_answer_table.png",
        max_rows=2,
    )

    manifest = {
        "input_files": {
            "evaluation_results": str(EVALUATION_RESULTS_PATH),
            "topk_retrieval_samples": str(TOPK_SAMPLES_PATH),
            "summary": str(SUMMARY_PATH),
        },
        "output_dir": str(OUT_DIR),
        "generated_files": sorted([p.name for p in OUT_DIR.glob("*")]),
        "statistics": {
            "answer_metric_mode": mode,
            "test_queries": int(summary.get("test_query_count", len(eval_df))),
            "generated_answers": int(summary.get("answer_mode_counts", {}).get("generated_answer", 0)),
            "rouge_1": float(summary.get("rouge_1", 0.0)),
            "rouge_2": float(summary.get("rouge_2", 0.0)),
            "rouge_l": float(summary.get("rouge_l", 0.0)),
            "source_families": source_df.to_dict(orient="records"),
        },
    }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] Objective 4 figures generated.")
    print(f"[DONE] Output directory: {OUT_DIR}")
    print(f"[DONE] Answer metric mode: {mode}")
    print(f"[DONE] ROUGE-1: {float(summary.get('rouge_1', 0.0)):.4f}")
    print(f"[DONE] ROUGE-2: {float(summary.get('rouge_2', 0.0)):.4f}")
    print(f"[DONE] ROUGE-L: {float(summary.get('rouge_l', 0.0)):.4f}")


if __name__ == "__main__":
    main()
