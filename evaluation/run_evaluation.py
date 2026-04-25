#!/usr/bin/env python3
"""
Run offline evidence checks for the six-objective chatbot project.

The default run avoids external LLM calls. It evaluates topic routing,
retrieval ranking, and answer-support coverage using retrieved context.
Pass --with-generation to call the configured Ollama model and evaluate the
generated answers against the gold answers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.retrieve import infer_topic_label, retrieve

TEST_SET_PATH = ROOT / "evaluation" / "test_queries.csv"
OUTPUT_DIR = ROOT / "data" / "evaluation"


def normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1)))


def f1_score(overlap: int, predicted_total: int, gold_total: int) -> float:
    if predicted_total == 0 or gold_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / gold_total
    return 2 * precision * recall / (precision + recall)


def rouge_n(candidate: str, gold: str, n: int) -> float:
    candidate_counts = ngrams(tokenize(candidate), n)
    gold_counts = ngrams(tokenize(gold), n)
    overlap = sum((candidate_counts & gold_counts).values())
    return f1_score(overlap, sum(candidate_counts.values()), sum(gold_counts.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, 1):
            if left == right:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, gold: str) -> float:
    candidate_tokens = tokenize(candidate)
    gold_tokens = tokenize(gold)
    return f1_score(lcs_length(candidate_tokens, gold_tokens), len(candidate_tokens), len(gold_tokens))


def classify_topic_group(topic_label: str, intent: dict[str, Any]) -> str:
    if intent.get("asks_sentiment"):
        return "social_sentiment"
    if topic_label == "HK Company Registration & Compliance" or intent.get("asks_official"):
        return "hk_compliance"
    if topic_label == "Business Operations & Planning" or intent.get("asks_business_guide"):
        return "business_operations"
    return "startup_patterns"


def field_blob(match: dict[str, Any]) -> str:
    meta = match.get("metadata", {}) or {}
    fields = [
        match.get("id", ""),
        meta.get("chunk_id", ""),
        meta.get("company_id", ""),
        meta.get("company_name", ""),
        meta.get("source_family", ""),
        meta.get("topic_label", ""),
        meta.get("url", ""),
        meta.get("text", ""),
        match.get("document", ""),
    ]
    return normalize_text(" ".join(str(field) for field in fields if field is not None))


def find_source_rank(matches: list[dict[str, Any]], expected_source: str, intent: dict[str, Any]) -> int | None:
    expected_source = normalize_text(expected_source)
    if not expected_source:
        return None
    if expected_source == "sentiment":
        return 1 if intent.get("asks_sentiment") else None

    for index, match in enumerate(matches, 1):
        source_family = normalize_text((match.get("metadata", {}) or {}).get("source_family"))
        if source_family == expected_source:
            return index
    return None


def find_relevant_rank(matches: list[dict[str, Any]], relevant_id: str) -> int | None:
    needle = normalize_text(relevant_id)
    if not needle:
        return None
    for index, match in enumerate(matches, 1):
        if needle in field_blob(match):
            return index
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def context_from_matches(matches: list[dict[str, Any]], limit: int = 3) -> str:
    parts: list[str] = []
    for match in matches[:limit]:
        meta = match.get("metadata", {}) or {}
        text = meta.get("text") or match.get("document") or ""
        parts.append(str(text)[:1200])
    return "\n\n".join(parts)


def maybe_generate_answer(query: str, matches: list[dict[str, Any]], topic_label: str, enabled: bool) -> tuple[str, str]:
    if not enabled:
        return context_from_matches(matches), "retrieved_context"

    try:
        from rag.generate import full_answer

        answer = full_answer(query, matches, topic_label)
        if answer.startswith("Warning:"):
            return context_from_matches(matches), "retrieved_context_fallback"
        return answer, "generated_answer"
    except Exception:
        return context_from_matches(matches), "retrieved_context_fallback"


def read_test_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_classification(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = sorted({row["expected_topic_group"] for row in rows} | {row["predicted_topic_group"] for row in rows})
    matrix: dict[tuple[str, str], int] = Counter((row["expected_topic_group"], row["predicted_topic_group"]) for row in rows)

    matrix_rows = []
    for expected in labels:
        item = {"expected": expected}
        for predicted in labels:
            item[predicted] = matrix.get((expected, predicted), 0)
        matrix_rows.append(item)

    report_rows = []
    total = len(rows)
    weighted_f1 = 0.0
    correct = sum(1 for row in rows if row["topic_correct"])
    for label in labels:
        tp = matrix.get((label, label), 0)
        fp = sum(matrix.get((other, label), 0) for other in labels if other != label)
        fn = sum(matrix.get((label, other), 0) for other in labels if other != label)
        support = sum(matrix.get((label, other), 0) for other in labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted_f1 += f1 * support
        report_rows.append({
            "topic_group": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })

    summary = {
        "topic_accuracy": round(correct / total, 4) if total else 0.0,
        "topic_weighted_f1": round(weighted_f1 / total, 4) if total else 0.0,
    }
    return summary, report_rows, matrix_rows


def build_summary(rows: list[dict[str, Any]], class_summary: dict[str, Any], answer_mode: str) -> dict[str, Any]:
    source_rows = [row for row in rows if row.get("expected_source_family")]
    relevant_rows = [row for row in rows if row.get("relevant_id")]

    def recall_at(rows_to_score: list[dict[str, Any]], rank_key: str, k: int) -> float:
        if not rows_to_score:
            return 0.0
        hits = sum(1 for row in rows_to_score if row.get(rank_key) and int(row[rank_key]) <= k)
        return round(hits / len(rows_to_score), 4)

    summary = {
        "test_query_count": len(rows),
        "answer_metric_mode": answer_mode,
        **class_summary,
        "source_recall_at_1": recall_at(source_rows, "source_rank", 1),
        "source_recall_at_3": recall_at(source_rows, "source_rank", 3),
        "source_recall_at_5": recall_at(source_rows, "source_rank", 5),
        "relevant_recall_at_1": recall_at(relevant_rows, "relevant_rank", 1),
        "relevant_recall_at_3": recall_at(relevant_rows, "relevant_rank", 3),
        "relevant_recall_at_5": recall_at(relevant_rows, "relevant_rank", 5),
        "source_mrr": round(sum(row["source_rr"] for row in source_rows) / len(source_rows), 4) if source_rows else 0.0,
        "relevant_mrr": round(sum(row["relevant_rr"] for row in relevant_rows) / len(relevant_rows), 4) if relevant_rows else 0.0,
        "relevant_ndcg_at_5": round(sum(row["relevant_ndcg_at_5"] for row in relevant_rows) / len(relevant_rows), 4) if relevant_rows else 0.0,
        "rouge_1": round(sum(row["rouge_1"] for row in rows) / len(rows), 4) if rows else 0.0,
        "rouge_2": round(sum(row["rouge_2"] for row in rows) / len(rows), 4) if rows else 0.0,
        "rouge_l": round(sum(row["rouge_l"] for row in rows) / len(rows), 4) if rows else 0.0,
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 2) if rows else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=TEST_SET_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-generation", action="store_true")
    args = parser.parse_args()

    rows = read_test_rows(args.test_set)
    result_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    answer_mode_counts: Counter[str] = Counter()

    for row in rows:
        query = row["query"]
        start = time.perf_counter()
        matches, intent = retrieve(query, top_k=args.top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        topic_label = infer_topic_label(intent, matches)
        predicted_topic_group = classify_topic_group(topic_label, intent)
        expected_topic_group = row["expected_topic_group"]

        source_rank = find_source_rank(matches, row["expected_source_family"], intent)
        relevant_rank = find_relevant_rank(matches, row.get("relevant_id", ""))
        answer_candidate, answer_mode = maybe_generate_answer(query, matches, topic_label, args.with_generation)
        answer_mode_counts[answer_mode] += 1

        result = {
            "query": query,
            "expected_topic_group": expected_topic_group,
            "predicted_topic_group": predicted_topic_group,
            "predicted_topic_label": topic_label,
            "topic_correct": expected_topic_group == predicted_topic_group,
            "expected_source_family": row["expected_source_family"],
            "source_rank": source_rank or "",
            "source_rr": reciprocal_rank(source_rank),
            "relevant_id": row.get("relevant_id", ""),
            "relevant_rank": relevant_rank or "",
            "relevant_rr": reciprocal_rank(relevant_rank),
            "relevant_ndcg_at_5": ndcg_at_k(relevant_rank, 5),
            "rouge_1": rouge_n(answer_candidate, row["gold_answer"], 1),
            "rouge_2": rouge_n(answer_candidate, row["gold_answer"], 2),
            "rouge_l": rouge_l(answer_candidate, row["gold_answer"]),
            "latency_ms": round(latency_ms, 2),
            "answer_mode": answer_mode,
        }
        result_rows.append(result)

        top = []
        for match in matches[:3]:
            meta = match.get("metadata", {}) or {}
            top.append({
                "source": meta.get("source_family", ""),
                "id": meta.get("company_id") or meta.get("chunk_id") or match.get("id", ""),
                "score": match.get("final_score", match.get("dense_score", "")),
                "snippet": (meta.get("text") or match.get("document") or "")[:240].replace("\n", " "),
            })
        sample_rows.append({
            "query": query,
            "expected_source_family": row["expected_source_family"],
            "top_1": json.dumps(top[0], ensure_ascii=False) if len(top) > 0 else "",
            "top_2": json.dumps(top[1], ensure_ascii=False) if len(top) > 1 else "",
            "top_3": json.dumps(top[2], ensure_ascii=False) if len(top) > 2 else "",
            "gold_answer": row["gold_answer"],
            "answer_preview": answer_candidate[:500].replace("\n", " "),
        })

    class_summary, report_rows, matrix_rows = summarize_classification(result_rows)
    answer_mode = answer_mode_counts.most_common(1)[0][0] if answer_mode_counts else "unknown"
    summary = build_summary(result_rows, class_summary, answer_mode)
    summary["answer_mode_counts"] = dict(answer_mode_counts)
    summary["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_results.csv", result_rows)
    write_csv(args.output_dir / "classification_report.csv", report_rows)
    write_csv(args.output_dir / "confusion_matrix.csv", matrix_rows)
    write_csv(args.output_dir / "topk_retrieval_samples.csv", sample_rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
