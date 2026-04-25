from pathlib import Path
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "obj1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_SPECS = {
    "yc": {
        "label": "YC",
        "path": DATA_DIR / "kb" / "cleaned_company_kb.jsonl",
        "text_candidates": [
            "combined_text", "text", "description", "long_description",
            "one_liner", "company_description", "content", "body",
            "summary", "name"
        ],
        "subcategory_candidates": [
            "topic", "topics", "primary_topic", "category",
            "categories", "industry", "cluster", "cluster_name", "batch"
        ],
        "update_frequency": "Snapshot / manual refresh",
    },
    "hk_registry": {
        "label": "HK Registry",
        "path": DATA_DIR / "kb" / "hk_official_kb_chunks.jsonl",
        "text_candidates": [
            "chunk_text", "text", "content", "body", "summary",
            "title", "section_title"
        ],
        "subcategory_candidates": [
            "topic", "category", "section", "section_title", "title"
        ],
        "update_frequency": "Official guidance update",
    },
    "business_guides": {
        "label": "Business Guides",
        "path": DATA_DIR / "kb" / "business_guide_kb_chunks.jsonl",
        "text_candidates": [
            "chunk_text", "text", "content", "body", "summary",
            "title", "section_title"
        ],
        "subcategory_candidates": [
            "topic", "category", "section", "section_title", "title"
        ],
        "update_frequency": "Manual / periodic refresh",
    },
    "social_media": {
        "label": "Social Media",
        "path": DATA_DIR / "sentiment" / "sentiment_cleaned.jsonl",
        "text_candidates": [
            "clean_text", "text", "content", "body", "summary"
        ],
        "subcategory_candidates": [
            "platform", "dataset_sentiment", "vader_sentiment", "country"
        ],
        "update_frequency": "Dataset snapshot",
    },
}


# =========================
# Utils
# =========================
def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def safe_to_str(x):
    if isinstance(x, list):
        return " ".join(safe_to_str(v) for v in x)
    if isinstance(x, dict):
        return " ".join(f"{k} {safe_to_str(v)}" for k, v in x.items())
    if x is None:
        return ""
    return str(x)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_first_existing_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_text_series(df: pd.DataFrame, candidates) -> pd.Series:
    available = [c for c in candidates if c in df.columns]
    if not available:
        return pd.Series([""] * len(df), index=df.index)

    cols = []
    for c in available:
        cols.append(df[c].apply(safe_to_str))

    combined = cols[0].fillna("")
    for s in cols[1:]:
        combined = combined + " " + s.fillna("")

    combined = combined.astype(str).map(normalize_text)
    return combined


def split_labels(x):
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(split_labels(v))
        return out
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            out.append(str(k).strip())
            out.extend(split_labels(v))
        return [z for z in out if z]
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    if any(sep in s for sep in ["|", ";", ","]):
        parts = re.split(r"[|;,]", s)
        return [p.strip() for p in parts if p.strip()]
    return [s]


def counter_topk_with_other(counter: Counter, topk=10):
    items = counter.most_common(topk)
    used = sum(v for _, v in items)
    total = sum(counter.values())
    if total > used:
        items.append(("Other", total - used))
    return items


def get_subcategory_counter(df: pd.DataFrame, candidates, fallback_name="All"):
    col = get_first_existing_column(df, candidates)
    if col is None:
        return Counter({fallback_name: len(df)})

    counter = Counter()
    for value in df[col].tolist():
        labels = split_labels(value)
        if not labels:
            continue
        for lb in labels:
            counter[lb] += 1

    if not counter:
        counter[fallback_name] = len(df)
    return counter


def compute_top_tfidf_terms(texts, top_n=20):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return pd.DataFrame(columns=["term", "score"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    scores = np.asarray(X.mean(axis=0)).ravel()

    order = np.argsort(scores)[::-1][:top_n]
    result = pd.DataFrame({
        "term": terms[order],
        "score": scores[order],
    })
    return result


def compute_stats(texts):
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    n_docs = len(texts)

    token_lists = [re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]*\b", t.lower()) for t in texts]
    total_tokens = sum(len(toks) for toks in token_lists)
    avg_tokens_per_doc = round(total_tokens / n_docs, 2) if n_docs else 0.0

    if n_docs == 0:
        return {
            "documents": 0,
            "total_tokens": 0,
            "avg_tokens_per_doc": 0.0,
            "unigram_vocab_size": 0,
            "bigram_vocab_size": 0,
            "sparsity": 0.0,
            "keyword_density_top20": 0.0,
        }

    uni_vec = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)
    X1 = uni_vec.fit_transform(texts)
    unigram_vocab_size = len(uni_vec.get_feature_names_out())

    bi_vec = CountVectorizer(stop_words="english", ngram_range=(2, 2), min_df=1)
    X2 = bi_vec.fit_transform(texts)
    bigram_vocab_size = len(bi_vec.get_feature_names_out())

    possible = X1.shape[0] * X1.shape[1]
    sparsity = 1.0 - (X1.nnz / possible) if possible else 0.0

    term_freq = np.asarray(X1.sum(axis=0)).ravel()
    if len(term_freq) > 0:
        top20_total = term_freq[np.argsort(term_freq)[::-1][:20]].sum()
        keyword_density_top20 = top20_total / total_tokens if total_tokens else 0.0
    else:
        keyword_density_top20 = 0.0

    return {
        "documents": n_docs,
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "unigram_vocab_size": int(unigram_vocab_size),
        "bigram_vocab_size": int(bigram_vocab_size),
        "sparsity": float(sparsity),
        "keyword_density_top20": float(keyword_density_top20),
    }


# =========================
# Plot helpers
# =========================
def save_bar_chart(df: pd.DataFrame, title: str, out_path: Path):
    if df.empty:
        return

    plot_df = df.sort_values("score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(plot_df["term"], plot_df["score"])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_ylabel("Keyword")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_table_image(df: pd.DataFrame, title: str, out_path: Path):
    n_rows, n_cols = df.shape
    fig_w = max(10, n_cols * 2.3)
    fig_h = max(2.6, n_rows * 0.55 + 1.5)

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


def save_sunburst(records_df: pd.DataFrame, html_path: Path, png_path: Path):
    if not HAS_PLOTLY or records_df.empty:
        return

    fig = px.sunburst(
        records_df,
        path=["source", "subcategory"],
        values="count",
        title="Objective 1: Knowledge Base Distribution by Source and Sub-category",
    )

    fig.write_html(str(html_path))

    # 如果装了 kaleido，则同时输出 png
    try:
        fig.write_image(str(png_path), width=1200, height=900, scale=2)
    except Exception:
        pass


# =========================
# Main
# =========================
def main():
    domain_outputs = {}
    summary_rows = []
    stats_rows = []
    sunburst_rows = []

    for key, spec in DOMAIN_SPECS.items():
        path = spec["path"]
        if not path.exists():
            print(f"[WARN] Skip missing file: {path}")
            continue

        print(f"[INFO] Loading: {path}")
        df = read_jsonl(path)

        text_series = build_text_series(df, spec["text_candidates"])
        texts = text_series[text_series.str.strip() != ""].tolist()

        # TF-IDF Top 20
        kw_df = compute_top_tfidf_terms(texts, top_n=20)
        bar_path = OUT_DIR / f"{key}_top20_tfidf.png"
        save_bar_chart(
            kw_df,
            f"Top 20 TF-IDF Keywords - {spec['label']}",
            bar_path
        )

        # Summary + stats
        stats = compute_stats(texts)
        summary_rows.append({
            "Source": spec["label"],
            "Documents": stats["documents"],
            "Total Tokens": stats["total_tokens"],
            "Avg Tokens/Doc": stats["avg_tokens_per_doc"],
            "Update Frequency": spec["update_frequency"],
        })
        stats_rows.append({
            "Source": spec["label"],
            "Keyword Density (Top20)": f"{stats['keyword_density_top20']:.4f}",
            "Unigram Vocab": stats["unigram_vocab_size"],
            "Bigram Vocab": stats["bigram_vocab_size"],
            "Data Sparsity": f"{stats['sparsity']:.4f}",
        })

        # Sunburst
        subcat_counter = get_subcategory_counter(
            df, spec["subcategory_candidates"], fallback_name="All"
        )
        for subcat, cnt in counter_topk_with_other(subcat_counter, topk=10):
            sunburst_rows.append({
                "source": spec["label"],
                "subcategory": str(subcat),
                "count": int(cnt),
            })

        domain_outputs[key] = {
            "rows": len(df),
            "bar_chart": str(bar_path),
        }

    # Tables
    summary_df = pd.DataFrame(summary_rows)
    stats_df = pd.DataFrame(stats_rows)

    if not summary_df.empty:
        summary_df.to_csv(OUT_DIR / "data_source_summary.csv", index=False)
        save_table_image(
            summary_df,
            "Objective 1: Data Source Summary",
            OUT_DIR / "data_source_summary.png"
        )

    if not stats_df.empty:
        stats_df.to_csv(OUT_DIR / "essential_statistics.csv", index=False)
        save_table_image(
            stats_df,
            "Objective 1: Essential Statistics",
            OUT_DIR / "essential_statistics.png"
        )

    # Sunburst
    sunburst_df = pd.DataFrame(sunburst_rows)
    if not sunburst_df.empty:
        sunburst_df.to_csv(OUT_DIR / "sunburst_source_data.csv", index=False)
        save_sunburst(
            sunburst_df,
            OUT_DIR / "knowledge_base_sunburst.html",
            OUT_DIR / "knowledge_base_sunburst.png"
        )

    # Save manifest
    manifest = {
        "output_dir": str(OUT_DIR),
        "generated_files": sorted([p.name for p in OUT_DIR.glob("*")]),
        "domains_processed": domain_outputs,
    }
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[DONE] Objective 1 figures generated.")
    print(f"[DONE] Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()