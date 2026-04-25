from pathlib import Path
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import cohen_kappa_score


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "evaluation"
OUT_DIR = ROOT / "outputs" / "obj2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_REPORT_PATH = DATA_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = DATA_DIR / "confusion_matrix.csv"
EVALUATION_RESULTS_PATH = DATA_DIR / "evaluation_results.csv"
SUMMARY_PATH = DATA_DIR / "summary.json"


# =========================
# Helpers
# =========================
def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Please run evaluation first:\n"
            f".venv\\Scripts\\python.exe evaluation\\run_evaluation.py"
        )


def pretty_label(label: str) -> str:
    if not isinstance(label, str):
        return str(label)
    return label.replace("_", " ").title()


def wrap_label(label: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(pretty_label(label), width=width))


def save_table_image(df: pd.DataFrame, title: str, out_path: Path):
    n_rows, n_cols = df.shape
    fig_w = max(9, n_cols * 2.2)
    fig_h = max(2.8, n_rows * 0.55 + 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    ax.set_title(title, fontsize=14, pad=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm_df: pd.DataFrame, out_path: Path, normalize: bool = False):
    labels = cm_df.index.tolist()
    matrix = cm_df.values.astype(float)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )

    fig_size = max(7.5, len(labels) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([wrap_label(x) for x in labels], rotation=45, ha="right")
    ax.set_yticklabels([wrap_label(x) for x in labels])

    ax.set_xlabel("Predicted Topic")
    ax.set_ylabel("Expected Topic")
    ax.set_title(
        "Topic Classification Confusion Matrix"
        + (" Normalized" if normalize else " Counts"),
        fontsize=14,
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if normalize:
                text = f"{matrix[i, j]:.2f}"
            else:
                text = f"{int(matrix[i, j])}"
            ax.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_topic_metrics_bar(report_df: pd.DataFrame, out_path: Path):
    df = report_df.copy()

    metric_cols = ["precision", "recall", "f1"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, df["precision"], width, label="Precision")
    ax.bar(x, df["recall"], width, label="Recall")
    ax.bar(x + width, df["f1"], width, label="F1-score")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_xlabel("Topic")
    ax.set_title("Topic Classification Metrics by Topic", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(t, width=16) for t in df["topic_group"]])
    ax.legend()

    for idx in range(len(df)):
        for offset, col in [(-width, "precision"), (0, "recall"), (width, "f1")]:
            value = df.iloc[idx][col]
            if pd.notna(value):
                ax.text(
                    idx + offset,
                    value + 0.015,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_topic_distribution(eval_df: pd.DataFrame, out_path: Path):
    expected_counts = eval_df["expected_topic_group"].value_counts().sort_index()
    predicted_counts = eval_df["predicted_topic_group"].value_counts().sort_index()

    labels = sorted(set(expected_counts.index).union(set(predicted_counts.index)))
    expected = [expected_counts.get(x, 0) for x in labels]
    predicted = [predicted_counts.get(x, 0) for x in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, expected, width, label="Expected")
    ax.bar(x + width / 2, predicted, width, label="Predicted")

    ax.set_ylabel("Number of Queries")
    ax.set_xlabel("Topic")
    ax.set_title("Expected vs Predicted Topic Distribution", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(t, width=16) for t in labels])
    ax.legend()

    for i, value in enumerate(expected):
        ax.text(i - width / 2, value + 0.05, str(value), ha="center", fontsize=8)
    for i, value in enumerate(predicted):
        ax.text(i + width / 2, value + 0.05, str(value), ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tsne_topic_map(eval_df: pd.DataFrame, out_path: Path):
    df = eval_df.copy()
    df = df.dropna(subset=["query", "predicted_topic_group"])

    queries = df["query"].astype(str).tolist()
    topics = df["predicted_topic_group"].astype(str).tolist()

    if len(queries) < 3:
        print("[WARN] Not enough queries for t-SNE plot.")
        return

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=1000,
    )
    X = vectorizer.fit_transform(queries)

    # Reduce dimensionality first when possible.
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_features > 2:
        n_components = min(10, n_samples - 1, n_features - 1)
        if n_components >= 2:
            X_reduced = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X)
        else:
            X_reduced = X.toarray()
    else:
        X_reduced = X.toarray()

    perplexity = min(5, max(2, n_samples - 1))

    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    ).fit_transform(X_reduced)

    topic_labels = sorted(set(topics))
    topic_to_id = {topic: i for i, topic in enumerate(topic_labels)}
    topic_ids = [topic_to_id[t] for t in topics]

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=topic_ids, s=80, alpha=0.85)

    handles = []
    for topic in topic_labels:
        topic_id = topic_to_id[topic]
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                label=pretty_label(topic),
                markersize=8,
                markerfacecolor=scatter.cmap(scatter.norm(topic_id)),
                markeredgecolor="none",
            )
        )

    ax.legend(handles=handles, title="Predicted Topic", loc="best")
    ax.set_title("t-SNE Visualization of Query Embeddings by Predicted Topic", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    for i, q in enumerate(queries):
        short_q = q[:28] + "..." if len(q) > 28 else q
        ax.annotate(short_q, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.75)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_summary(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Main
# =========================
def main():
    require_file(CLASSIFICATION_REPORT_PATH)
    require_file(CONFUSION_MATRIX_PATH)
    require_file(EVALUATION_RESULTS_PATH)

    report_df = pd.read_csv(CLASSIFICATION_REPORT_PATH)
    cm_raw_df = pd.read_csv(CONFUSION_MATRIX_PATH)
    eval_df = pd.read_csv(EVALUATION_RESULTS_PATH)
    summary = load_summary(SUMMARY_PATH)

    # Standardize confusion matrix
    if "expected" not in cm_raw_df.columns:
        raise ValueError("confusion_matrix.csv must contain an 'expected' column.")

    cm_df = cm_raw_df.set_index("expected")
    cm_df = cm_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # 1. Confusion matrix: counts
    save_confusion_matrix(
        cm_df,
        OUT_DIR / "confusion_matrix_counts.png",
        normalize=False,
    )

    # 2. Confusion matrix: normalized
    save_confusion_matrix(
        cm_df,
        OUT_DIR / "confusion_matrix_normalized.png",
        normalize=True,
    )

    # 3. Classification report table
    report_table = report_df.copy()
    for col in ["precision", "recall", "f1"]:
        if col in report_table.columns:
            report_table[col] = report_table[col].map(lambda x: f"{float(x):.4f}")
    if "support" in report_table.columns:
        report_table["support"] = report_table["support"].astype(int)

    report_table["topic_group"] = report_table["topic_group"].map(pretty_label)
    report_table = report_table.rename(columns={
        "topic_group": "Topic",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "support": "Support",
    })

    save_table_image(
        report_table,
        "Objective 2: Classification Report",
        OUT_DIR / "classification_report_table.png",
    )

    # 4. Topic metrics bar chart
    save_topic_metrics_bar(
        report_df,
        OUT_DIR / "topic_metrics_bar.png",
    )

    # 5. Expected vs predicted distribution
    save_topic_distribution(
        eval_df,
        OUT_DIR / "topic_distribution.png",
    )

    # 6. t-SNE query embedding visualization
    save_tsne_topic_map(
        eval_df,
        OUT_DIR / "query_tsne_topic_map.png",
    )

    # 7. Essential statistics table
    y_true = eval_df["expected_topic_group"].astype(str).tolist()
    y_pred = eval_df["predicted_topic_group"].astype(str).tolist()

    total_queries = len(eval_df)
    correct_queries = int(eval_df["topic_correct"].astype(str).str.lower().eq("true").sum())
    topic_accuracy = correct_queries / total_queries if total_queries else 0.0
    weighted_f1 = float(summary.get("topic_weighted_f1", np.nan))
    kappa = cohen_kappa_score(y_true, y_pred) if total_queries else 0.0

    stats_df = pd.DataFrame([
        ["Test Queries", total_queries],
        ["Correct Topic Predictions", correct_queries],
        ["Topic Accuracy", f"{topic_accuracy:.4f}"],
        ["Weighted F1-score", f"{weighted_f1:.4f}" if not np.isnan(weighted_f1) else "N/A"],
        ["Cohen's Kappa", f"{kappa:.4f}"],
        ["Number of Topic Groups", eval_df["expected_topic_group"].nunique()],
    ], columns=["Metric", "Value"])

    save_table_image(
        stats_df,
        "Objective 2: Essential Statistics",
        OUT_DIR / "objective2_statistics_table.png",
    )

    # 8. Save error cases
    error_df = eval_df[eval_df["expected_topic_group"] != eval_df["predicted_topic_group"]].copy()
    error_df.to_csv(OUT_DIR / "topic_misclassification_cases.csv", index=False)

    # 9. Save manifest
    manifest = {
        "input_files": {
            "classification_report": str(CLASSIFICATION_REPORT_PATH),
            "confusion_matrix": str(CONFUSION_MATRIX_PATH),
            "evaluation_results": str(EVALUATION_RESULTS_PATH),
            "summary": str(SUMMARY_PATH),
        },
        "output_dir": str(OUT_DIR),
        "generated_files": sorted([p.name for p in OUT_DIR.glob("*")]),
        "statistics": {
            "test_queries": total_queries,
            "correct_topic_predictions": correct_queries,
            "topic_accuracy": topic_accuracy,
            "topic_weighted_f1": None if np.isnan(weighted_f1) else weighted_f1,
            "cohens_kappa": kappa,
            "topic_groups": int(eval_df["expected_topic_group"].nunique()),
            "misclassification_cases": int(len(error_df)),
        },
    }

    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] Objective 2 figures generated.")
    print(f"[DONE] Output directory: {OUT_DIR}")
    print(f"[DONE] Topic accuracy: {topic_accuracy:.4f}")
    print(f"[DONE] Cohen's Kappa: {kappa:.4f}")
    print(f"[DONE] Misclassification cases: {len(error_df)}")


if __name__ == "__main__":
    main()