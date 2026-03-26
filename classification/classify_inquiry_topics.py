#!/usr/bin/env python3
"""
Objective 2 for Project 8:
Use classification model(s) and/or text analysis model(s) to classify the
topics of enquiry from the knowledge base.

This script builds a lightweight inquiry-topic classifier from the knowledge
base chunks produced in objective 1. It uses the knowledge-base topic labels
as weak supervision, creates query-like training samples, trains a pure-Python
TF-IDF centroid classifier, evaluates it, and exports reusable artifacts for
later chatbot / RAG integration.

Example usage
-------------
Train and evaluate:
    python classification/classify_inquiry_topics.py

Predict a single user enquiry with a trained model:
    python classification/classify_inquiry_topics.py --skip-train --query "How do I start a healthcare startup in Hong Kong?"
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    sys.modules.setdefault("classification.classify_inquiry_topics", sys.modules[__name__])


DEFAULT_KB_PATH = "project8_kb/knowledge_base_chunks.jsonl"
DEFAULT_TAXONOMY_PATH = "project8_kb/topic_taxonomy.json"
DEFAULT_OUTPUT_DIR = "classification/output"
DEFAULT_RANDOM_SEED = 42


@dataclass
class InquirySample:
    text: str
    topic_id: int
    topic_label: str
    sample_type: str
    company_id: str = ""


class TopicInquiryClassifier:
    """
    A small dependency-free text classifier.

    The classifier builds TF-IDF vectors and computes one centroid per topic
    label. Prediction is based on cosine similarity between a query vector and
    the learned topic centroids.
    """

    def __init__(
        self,
        *,
        min_df: int = 2,
        max_df_ratio: float = 0.85,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.ngram_range = ngram_range
        self.stopwords = build_stopwords()
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}
        self.classes_: List[str] = []
        self.centroids_: Dict[str, Dict[str, float]] = {}

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TopicInquiryClassifier":
        if not texts:
            raise ValueError("Cannot fit classifier with empty training data.")

        docs = [self._extract_terms(text) for text in texts]
        df_counter: Counter[str] = Counter()
        for terms in docs:
            for term in set(terms):
                df_counter[term] += 1

        n_docs = len(docs)
        max_df = max(1, int(math.floor(self.max_df_ratio * n_docs)))
        kept_terms = sorted(
            term
            for term, df in df_counter.items()
            if df >= self.min_df and df <= max_df
        )
        self.vocabulary_ = {term: idx for idx, term in enumerate(kept_terms)}
        self.idf_ = {
            term: math.log((1 + n_docs) / (1 + df_counter[term])) + 1.0
            for term in kept_terms
        }

        label_vectors: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for text, label in zip(texts, labels):
            label_vectors[label].append(self._vectorize(text))

        self.classes_ = sorted(label_vectors)
        self.centroids_ = {
            label: normalize_vector(average_vectors(vectors))
            for label, vectors in label_vectors.items()
        }
        return self

    def predict(self, texts: Sequence[str]) -> List[str]:
        probabilities = self.predict_proba(texts)
        predictions: List[str] = []
        for scores in probabilities:
            best_idx = max(range(len(scores)), key=scores.__getitem__)
            predictions.append(self.classes_[best_idx])
        return predictions

    def predict_proba(self, texts: Sequence[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for text in texts:
            vector = self._vectorize(text)
            scores = [dot_product(vector, self.centroids_[label]) for label in self.classes_]
            outputs.append(softmax(scores))
        return outputs

    def _vectorize(self, text: str) -> Dict[str, float]:
        terms = self._extract_terms(text)
        tf_counter = Counter(term for term in terms if term in self.vocabulary_)
        if not tf_counter:
            return {}

        weighted = {
            term: (1.0 + math.log(tf)) * self.idf_[term]
            for term, tf in tf_counter.items()
        }
        return normalize_vector(weighted)

    def _extract_terms(self, text: str) -> List[str]:
        tokens = tokenize(text)
        tokens = [token for token in tokens if token not in self.stopwords and len(token) > 1]
        if not tokens:
            return []

        min_n, max_n = self.ngram_range
        terms: List[str] = []
        for n in range(min_n, max_n + 1):
            if len(tokens) < n:
                continue
            for idx in range(len(tokens) - n + 1):
                terms.append(" ".join(tokens[idx : idx + n]))
        return terms


TopicInquiryClassifier.__module__ = "classification.classify_inquiry_topics"


def build_stopwords() -> set[str]:
    return {
        "a", "about", "after", "all", "an", "and", "are", "as", "at", "be", "before", "by",
        "can", "do", "for", "from", "give", "how", "i", "in", "into", "is", "it", "me",
        "my", "of", "on", "or", "our", "should", "startup", "startups", "that", "the",
        "their", "this", "to", "us", "user", "we", "what", "when", "where", "which", "who",
        "why", "with", "you", "your",
    }


def tokenize(text: str) -> List[str]:
    text = safe_text(text).lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9&/+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_jsonl(path: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(json.loads(line))
    return items


def average_vectors(vectors: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not vectors:
        return {}
    combined: Dict[str, float] = defaultdict(float)
    for vector in vectors:
        for term, value in vector.items():
            combined[term] += value
    scale = 1.0 / len(vectors)
    return {term: value * scale for term, value in combined.items()}


def normalize_vector(vector: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in vector.items()}


def dot_product(left: Dict[str, float], right: Dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total == 0:
        return [0.0 for _ in exps]
    return [value / total for value in exps]


def strip_structured_markers(text: str) -> str:
    text = safe_text(text)
    text = re.sub(r"Topic:\s*.*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:Company|Tagline|Description|Keywords):", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_company_evidence_text(chunk: Dict[str, object]) -> str:
    company_name = safe_text(chunk.get("company_name"))
    keywords = [safe_text(kw) for kw in chunk.get("keywords", []) if safe_text(kw)]
    chunk_text = strip_structured_markers(safe_text(chunk.get("text")))
    parts = [company_name, chunk_text, " ".join(keywords)]
    return " ".join(part for part in parts if part)


def synthesize_inquiry_texts(chunk: Dict[str, object], taxonomy_keywords: Sequence[str]) -> List[InquirySample]:
    topic_id = int(chunk["topic_id"])
    topic_label = safe_text(chunk["topic_label"])
    keywords = [safe_text(kw) for kw in chunk.get("keywords", []) if safe_text(kw)]
    company_id = safe_text(chunk.get("company_id"))
    company_evidence = build_company_evidence_text(chunk)
    query_hint = " ".join(list(dict.fromkeys([*taxonomy_keywords[:3], *keywords[:3]])))

    samples = [
        InquirySample(
            text=company_evidence,
            topic_id=topic_id,
            topic_label=topic_label,
            sample_type="kb_chunk",
            company_id=company_id,
        )
    ]

    inquiry_templates = [
        "how do i start a business in {hint} in hong kong",
        "what should a founder know about {hint}",
        "give me advice for a startup in {hint}",
        "what market opportunities exist in {hint}",
        "what are the risks and trends in {hint}",
    ]
    for template in inquiry_templates:
        samples.append(
            InquirySample(
                text=template.format(hint=query_hint or company_evidence),
                topic_id=topic_id,
                topic_label=topic_label,
                sample_type="topic_template",
                company_id=company_id,
            )
        )

    keyword_pool = list(dict.fromkeys([*taxonomy_keywords, *keywords[:4]]))[:6]
    if keyword_pool:
        samples.append(
            InquirySample(
                text=f"user enquiry including {' '.join(keyword_pool)}",
                topic_id=topic_id,
                topic_label=topic_label,
                sample_type="keyword_prompt",
                company_id=company_id,
            )
        )

    return samples


def build_training_samples(
    kb_chunks: Sequence[Dict[str, object]],
    taxonomy_rows: Sequence[Dict[str, object]],
) -> List[InquirySample]:
    taxonomy_lookup = {
        int(row["topic_id"]): [safe_text(kw) for kw in row.get("top_keywords", []) if safe_text(kw)]
        for row in taxonomy_rows
    }

    samples: List[InquirySample] = []
    for chunk in kb_chunks:
        topic_id = int(chunk["topic_id"])
        samples.extend(synthesize_inquiry_texts(chunk, taxonomy_lookup.get(topic_id, [])))
    return samples


def summarize_labels(labels: Sequence[str]) -> Dict[str, int]:
    counter = Counter(labels)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def stratified_split(
    items: Sequence[object],
    labels: Sequence[str],
    *,
    test_size: float,
    random_seed: int,
) -> Tuple[List[object], List[object], List[str], List[str]]:
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped_indices[label].append(idx)

    rng = random.Random(random_seed)
    test_indices: List[int] = []
    train_indices: List[int] = []

    for label, indices in grouped_indices.items():
        indices = indices[:]
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        n_test = max(1, int(round(len(indices) * test_size)))
        if n_test >= len(indices):
            n_test = len(indices) - 1
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    X_train = [items[idx] for idx in train_indices]
    X_test = [items[idx] for idx in test_indices]
    y_train = [labels[idx] for idx in train_indices]
    y_test = [labels[idx] for idx in test_indices]
    return X_train, X_test, y_train, y_test


def compute_classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    labels = sorted(set(y_true) | set(y_pred))
    report: Dict[str, object] = {}
    total_support = len(y_true)
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == label and pred != label)
        support = sum(1 for truth in y_true if truth == label)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

        report[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1-score": round(f1, 4),
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    n_labels = len(labels) or 1
    report["macro avg"] = {
        "precision": round(macro_precision / n_labels, 4),
        "recall": round(macro_recall / n_labels, 4),
        "f1-score": round(macro_f1 / n_labels, 4),
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": round(weighted_precision / total_support, 4) if total_support else 0.0,
        "recall": round(weighted_recall / total_support, 4) if total_support else 0.0,
        "f1-score": round(weighted_f1 / total_support, 4) if total_support else 0.0,
        "support": total_support,
    }
    accuracy = sum(1 for truth, pred in zip(y_true, y_pred) if truth == pred) / total_support if total_support else 0.0
    report["accuracy"] = round(accuracy, 4)
    return report


def train_and_evaluate(
    kb_chunks: Sequence[Dict[str, object]],
    taxonomy_rows: Sequence[Dict[str, object]],
    random_seed: int,
) -> Dict[str, object]:
    labels = [safe_text(chunk["topic_label"]) for chunk in kb_chunks]

    train_chunks, test_chunks, y_train_chunks, y_test = stratified_split(
        kb_chunks,
        labels,
        test_size=0.2,
        random_seed=random_seed,
    )

    train_samples = build_training_samples(train_chunks, taxonomy_rows)
    classifier = TopicInquiryClassifier()
    classifier.fit(
        [sample.text for sample in train_samples],
        [sample.topic_label for sample in train_samples],
    )

    X_test = [build_company_evidence_text(chunk) for chunk in test_chunks]
    y_pred = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidences = [max(row) if row else 0.0 for row in probabilities]
    report = compute_classification_report(y_test, y_pred)

    return {
        "classifier": classifier,
        "metrics": {
            "train_samples": len(train_samples),
            "test_samples": len(X_test),
            "n_labels": len(sorted(set(labels))),
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "avg_prediction_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "label_distribution": summarize_labels(labels),
        },
        "per_label_report": report,
        "test_examples": [
            {
                "text": text,
                "true_label": truth,
                "predicted_label": pred,
                "confidence": round(conf, 4),
            }
            for text, truth, pred, conf in zip(X_test[:30], y_test[:30], y_pred[:30], confidences[:30])
        ],
    }


def fit_full_model(samples: Sequence[InquirySample]) -> TopicInquiryClassifier:
    classifier = TopicInquiryClassifier()
    classifier.fit([sample.text for sample in samples], [sample.topic_label for sample in samples])
    return classifier


def predict_queries(
    classifier: TopicInquiryClassifier,
    queries: Sequence[str],
    taxonomy_rows: Sequence[Dict[str, object]],
    top_k: int = 3,
) -> List[Dict[str, object]]:
    probabilities = classifier.predict_proba(queries)
    taxonomy_by_label = {
        safe_text(row["topic_label"]): {
            "topic_id": int(row["topic_id"]),
            "top_keywords": [safe_text(kw) for kw in row.get("top_keywords", []) if safe_text(kw)],
        }
        for row in taxonomy_rows
    }

    results: List[Dict[str, object]] = []
    for raw_query, row in zip(queries, probabilities):
        ranking = sorted(
            zip(classifier.classes_, row),
            key=lambda item: item[1],
            reverse=True,
        )[: max(1, top_k)]

        predictions = []
        for label, score in ranking:
            topic_meta = taxonomy_by_label.get(label, {})
            predictions.append(
                {
                    "topic_label": label,
                    "topic_id": topic_meta.get("topic_id"),
                    "score": round(score, 4),
                    "top_keywords": topic_meta.get("top_keywords", [])[:6],
                }
            )

        results.append(
            {
                "query": raw_query,
                "predictions": predictions,
            }
        )
    return results


def default_demo_queries() -> List[str]:
    return [
        "How can I validate a healthcare startup idea and reach clinics or doctors?",
        "What should I know before building an AI coding agent company?",
        "How do I start an education platform for students and teachers?",
        "What are common compliance concerns for a fintech or insurance startup?",
        "How can a founder market a commerce brand through social media content?",
    ]


def ensure_output_dir(path: str) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_pickle(path: Path, obj: object) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: str) -> object:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and run Project 8 inquiry topic classification from the knowledge base."
    )
    parser.add_argument("--kb-path", default=DEFAULT_KB_PATH, help="Path to knowledge_base_chunks.jsonl")
    parser.add_argument("--taxonomy-path", default=DEFAULT_TAXONOMY_PATH, help="Path to topic_taxonomy.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for exported artifacts")
    parser.add_argument("--model-path", default="", help="Existing pickled model path for prediction-only mode")
    parser.add_argument("--query", action="append", default=[], help="User enquiry to classify. Can be repeated.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run prediction.")
    parser.add_argument("--top-k", type=int, default=3, help="How many topic predictions to return per query")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.random_seed)

    taxonomy_path = Path(args.taxonomy_path)
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")

    with open(taxonomy_path, "r", encoding="utf-8") as fh:
        taxonomy_rows = json.load(fh)

    output_dir = ensure_output_dir(args.output_dir)
    model_path = Path(args.model_path) if args.model_path else output_dir / "inquiry_topic_classifier.pkl"

    if args.skip_train:
        if not model_path.exists():
            raise SystemExit(
                f"Prediction-only mode requires an existing model file. Not found: {model_path}"
            )
        classifier = load_pickle(str(model_path))
    else:
        kb_path = Path(args.kb_path)
        if not kb_path.exists():
            raise SystemExit(f"Knowledge base chunks file not found: {kb_path}")

        kb_chunks = read_jsonl(str(kb_path))
        samples = build_training_samples(kb_chunks, taxonomy_rows)
        if not samples:
            raise SystemExit("No training samples could be built from the knowledge base.")

        training_output = train_and_evaluate(
            kb_chunks=kb_chunks,
            taxonomy_rows=taxonomy_rows,
            random_seed=args.random_seed,
        )
        classifier = fit_full_model(samples=samples)

        save_pickle(model_path, classifier)
        save_json(
            output_dir / "inquiry_topic_classifier_report.json",
            {
                "objective": "Objective 2 - classify topics of enquiry from the knowledge base",
                "input_files": {
                    "kb_path": str(kb_path),
                    "taxonomy_path": str(taxonomy_path),
                },
                "sample_summary": {
                    "n_training_samples_total": len(samples),
                    "sample_type_distribution": summarize_labels([sample.sample_type for sample in samples]),
                },
                "metrics": training_output["metrics"],
                "per_label_report": training_output["per_label_report"],
                "test_examples": training_output["test_examples"],
            },
        )
        save_json(
            output_dir / "topic_catalog.json",
            {
                "topics": [
                    {
                        "topic_id": int(row["topic_id"]),
                        "topic_label": safe_text(row["topic_label"]),
                        "top_keywords": [safe_text(kw) for kw in row.get("top_keywords", []) if safe_text(kw)],
                        "num_companies": int(row.get("num_companies", 0)),
                    }
                    for row in taxonomy_rows
                ]
            },
        )

    queries = args.query or default_demo_queries()
    predictions = predict_queries(
        classifier=classifier,
        queries=queries,
        taxonomy_rows=taxonomy_rows,
        top_k=max(1, args.top_k),
    )
    save_json(output_dir / "sample_inquiry_predictions.json", {"predictions": predictions})

    summary = {
        "model_path": str(model_path),
        "prediction_output": str(output_dir / "sample_inquiry_predictions.json"),
        "queries_scored": len(queries),
    }
    if not args.skip_train:
        summary["report_output"] = str(output_dir / "inquiry_topic_classifier_report.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
