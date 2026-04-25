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
OUT_DIR = ROOT / "outputs" / "obj3"
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
            f"Please run retrieval evaluation first:\n"
            f".venv\\Scripts\\python.exe evaluation\\run_evaluation.py"
        )


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def has_value(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip().str.lower()
    return text.ne("") & text.ne("nan") & text.ne("none")


def wrap_text(value: object, width: int = 42) -> str:
    text = str(value or "").strip()
    return "\n".join(textwrap.wrap(text, width=width))


def save_table_image(df: pd.DataFrame, title: str, out_path: Path, font_size: int = 9):
    n_rows, n_cols = df.shape
    fig_w = max(10, n_cols * 2.4)
    fig_h = max(3.0, n_rows * 0.75 + 1.8)

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
    table.scale(1, 1.5)

    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_topk_json(value: object) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def compute_recall_at_k(eval_df: pd.DataFrame, rank_col: str, ks=(1, 3, 5), denominator_mask=None) -> dict[int, float]:
    ranks = as_numeric(eval_df[rank_col])
    if denominator_mask is None:
        denominator_mask = pd.Series([True] * len(eval_df), index=eval_df.index)
    denominator_mask = denominator_mask.fillna(False).astype(bool)
    denom = int(denominator_mask.sum())
    if denom == 0:
        return {k: 0.0 for k in ks}
    return {
        k: float(((ranks[denominator_mask] > 0) & (ranks[denominator_mask] <= k)).sum() / denom)
        for k in ks
    }


def save_recall_curve(source_recall: dict[int, float], relevant_recall: dict[int, float], out_path: Path):
    ks = sorted(set(source_recall) | set(relevant_recall))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(ks, [source_recall.get(k, 0.0) for k in ks], marker="o", linewidth=2.5, label="Source Recall@K")
    ax.plot(ks, [relevant_recall.get(k, 0.0) for k in ks], marker="o", linewidth=2.5, label="Relevant Chunk Recall@K")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall")
    ax.set_title("Objective 3: Recall@K Curve", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for k in ks:
        for offset, value in [(0.02, source_recall.get(k, 0.0)), (-0.06, relevant_recall.get(k, 0.0))]:
            ax.text(k, min(1.02, value + offset), f"{value:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_relevance_histogram(samples_df: pd.DataFrame, out_path: Path):
    score_rows = []
    for _, row in samples_df.iterrows():
        for rank in [1, 2, 3]:
            payload = parse_topk_json(row.get(f"top_{rank}"))
            if "score" in payload:
                score_rows.append({"rank": f"Top {rank}", "score": float(payload["score"])})

    score_df = pd.DataFrame(score_rows)
    if score_df.empty:
        return score_df

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(score_df["score"], bins=18, alpha=0.85)
    ax.set_xlabel("Final Retrieval Score")
    ax.set_ylabel("Retrieved Chunks")
    ax.set_title("Search Relevance Score Histogram", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    mean_score = score_df["score"].mean()
    ax.axvline(mean_score, linestyle="--", linewidth=2)
    ax.text(mean_score, ax.get_ylim()[1] * 0.9, f"Mean={mean_score:.3f}", rotation=90, va="top", ha="right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return score_df


def save_rank_distribution(eval_df: pd.DataFrame, out_path: Path):
    ranks = as_numeric(eval_df["relevant_rank"])
    labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4-5", "Not in Top 5"]
    counts = [
        int((ranks == 1).sum()),
        int((ranks == 2).sum()),
        int((ranks == 3).sum()),
        int(((ranks >= 4) & (ranks <= 5)).sum()),
        int((ranks.isna() | (ranks <= 0)).sum()),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar(labels, counts)
    ax.set_ylabel("Queries")
    ax.set_title("Relevant Chunk Rank Distribution", fontsize=14)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value), ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_source_recall_bar(eval_df: pd.DataFrame, out_path: Path):
    df = eval_df.copy()
    df["source_rank_num"] = as_numeric(df["source_rank"])
    rows = []
    for source, group in df.groupby("expected_source_family"):
        if not source:
            continue
        ranks = group["source_rank_num"]
        rows.append({
            "source": source,
            "recall_at_1": float((ranks == 1).sum() / len(group)),
            "recall_at_3": float(((ranks > 0) & (ranks <= 3)).sum() / len(group)),
            "queries": len(group),
        })

    source_df = pd.DataFrame(rows).sort_values("source")
    if source_df.empty:
        return source_df

    x = np.arange(len(source_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - width / 2, source_df["recall_at_1"], width, label="Recall@1")
    ax.bar(x + width / 2, source_df["recall_at_3"], width, label="Recall@3")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Expected Source Family")
    ax.set_title("Source-Family Retrieval Recall", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_text(s, width=14) for s in source_df["source"]])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for i, row in source_df.iterrows():
        pos = source_df.index.get_loc(i)
        ax.text(pos - width / 2, row["recall_at_1"] + 0.015, f"{row['recall_at_1']:.2f}", ha="center", fontsize=8)
        ax.text(pos + width / 2, row["recall_at_3"] + 0.015, f"{row['recall_at_3']:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return source_df


def build_topk_sample_table(samples_df: pd.DataFrame, max_rows: int = 8) -> pd.DataFrame:
    rows = []
    for _, row in samples_df.head(max_rows).iterrows():
        top_payloads = [parse_topk_json(row.get(f"top_{rank}")) for rank in [1, 2, 3]]
        rows.append({
            "Query": wrap_text(row.get("query"), width=38),
            "Expected Source": row.get("expected_source_family", ""),
            "Top 1": wrap_text(top_payloads[0].get("id", ""), width=24),
            "Top 1 Score": f"{float(top_payloads[0].get('score', 0.0)):.3f}" if top_payloads[0] else "",
            "Top 2": wrap_text(top_payloads[1].get("id", ""), width=24),
            "Top 3": wrap_text(top_payloads[2].get("id", ""), width=24),
        })
    return pd.DataFrame(rows)


# =========================
# Main
# =========================
def main():
    require_file(EVALUATION_RESULTS_PATH)
    require_file(TOPK_SAMPLES_PATH)

    eval_df = pd.read_csv(EVALUATION_RESULTS_PATH)
    samples_df = pd.read_csv(TOPK_SAMPLES_PATH)
    summary = load_summary(SUMMARY_PATH)

    source_mask = has_value(eval_df["expected_source_family"])
    relevant_mask = has_value(eval_df["relevant_id"])
    source_recall = compute_recall_at_k(eval_df, "source_rank", ks=(1, 3, 5), denominator_mask=source_mask)
    relevant_recall = compute_recall_at_k(eval_df, "relevant_rank", ks=(1, 3, 5), denominator_mask=relevant_mask)

    save_recall_curve(
        source_recall,
        relevant_recall,
        OUT_DIR / "recall_at_k_curve.png",
    )

    score_df = save_relevance_histogram(
        samples_df,
        OUT_DIR / "search_relevance_histogram.png",
    )

    save_rank_distribution(
        eval_df,
        OUT_DIR / "relevant_rank_distribution.png",
    )

    source_df = save_source_recall_bar(
        eval_df,
        OUT_DIR / "source_recall_by_family.png",
    )

    metrics_df = pd.DataFrame([
        ["Test Queries", len(eval_df)],
        ["Source Recall@1", f"{source_recall[1]:.4f}"],
        ["Source Recall@3", f"{source_recall[3]:.4f}"],
        ["Source Recall@5", f"{source_recall[5]:.4f}"],
        ["Relevant Recall@1", f"{relevant_recall[1]:.4f}"],
        ["Relevant Recall@3", f"{relevant_recall[3]:.4f}"],
        ["Relevant Recall@5", f"{relevant_recall[5]:.4f}"],
        ["Source MRR", f"{float(summary.get('source_mrr', 0.0)):.4f}"],
        ["Relevant MRR", f"{float(summary.get('relevant_mrr', 0.0)):.4f}"],
        ["Relevant NDCG@5", f"{float(summary.get('relevant_ndcg_at_5', 0.0)):.4f}"],
        ["Mean Retrieval Latency (ms)", f"{float(summary.get('avg_latency_ms', 0.0)):.2f}"],
    ], columns=["Metric", "Value"])

    save_table_image(
        metrics_df,
        "Objective 3: Retrieval Metrics",
        OUT_DIR / "retrieval_metrics_table.png",
    )

    topk_table = build_topk_sample_table(samples_df, max_rows=8)
    save_table_image(
        topk_table,
        "Objective 3: Top-K Retrieval Samples",
        OUT_DIR / "topk_retrieval_samples_table.png",
        font_size=7,
    )

    if not source_df.empty:
        source_df.to_csv(OUT_DIR / "source_recall_by_family.csv", index=False)
    if not score_df.empty:
        score_df.to_csv(OUT_DIR / "retrieval_score_distribution.csv", index=False)

    manifest = {
        "input_files": {
            "evaluation_results": str(EVALUATION_RESULTS_PATH),
            "topk_retrieval_samples": str(TOPK_SAMPLES_PATH),
            "summary": str(SUMMARY_PATH),
        },
        "output_dir": str(OUT_DIR),
        "generated_files": sorted([p.name for p in OUT_DIR.glob("*")]),
        "statistics": {
            "test_queries": int(len(eval_df)),
            "source_recall_at_1": source_recall[1],
            "source_recall_at_3": source_recall[3],
            "source_recall_at_5": source_recall[5],
            "relevant_recall_at_1": relevant_recall[1],
            "relevant_recall_at_3": relevant_recall[3],
            "relevant_recall_at_5": relevant_recall[5],
            "source_mrr": float(summary.get("source_mrr", 0.0)),
            "relevant_mrr": float(summary.get("relevant_mrr", 0.0)),
            "relevant_ndcg_at_5": float(summary.get("relevant_ndcg_at_5", 0.0)),
            "avg_latency_ms": float(summary.get("avg_latency_ms", 0.0)),
        },
    }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] Objective 3 figures generated.")
    print(f"[DONE] Output directory: {OUT_DIR}")
    print(f"[DONE] Source Recall@5: {source_recall[5]:.4f}")
    print(f"[DONE] Relevant Recall@5: {relevant_recall[5]:.4f}")
    print(f"[DONE] Relevant MRR: {float(summary.get('relevant_mrr', 0.0)):.4f}")
    print(f"[DONE] Relevant NDCG@5: {float(summary.get('relevant_ndcg_at_5', 0.0)):.4f}")


if __name__ == "__main__":
    main()
