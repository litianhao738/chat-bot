#!/usr/bin/env python3
"""Summarize follow-up suggestion impressions and clicks for Objective 5."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "analytics" / "followup_events.jsonl"
SUMMARY_PATH = ROOT / "data" / "analytics" / "followup_summary.json"
JOURNEY_PATH = ROOT / "data" / "analytics" / "followup_journeys.csv"


def read_rows() -> list[dict]:
    rows: list[dict] = []
    if not LOG_PATH.exists():
        return rows
    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_keys(rows: list[dict]) -> set[tuple]:
    return {
        (row.get("turn_id"), row.get("followup_rank"), row.get("followup_text"))
        for row in rows
    }


def main() -> None:
    rows = read_rows()
    shown = [row for row in rows if row.get("event_type") == "shown"]
    clicked = [row for row in rows if row.get("event_type") == "clicked"]

    shown_keys = unique_keys(shown)
    clicked_keys = unique_keys(clicked)
    shown_count = len(shown_keys)
    clicked_count = len(clicked_keys)

    rank_counts: dict[str, dict[str, int | float]] = {}
    for rank in sorted({int(row.get("followup_rank") or 0) for row in rows}):
        if not rank:
            continue
        rank_shown = unique_keys([row for row in shown if int(row.get("followup_rank") or 0) == rank])
        rank_clicked = unique_keys([row for row in clicked if int(row.get("followup_rank") or 0) == rank])
        rank_counts[str(rank)] = {
            "shown": len(rank_shown),
            "clicked": len(rank_clicked),
            "ctr": (len(rank_clicked) / len(rank_shown)) if rank_shown else 0.0,
        }

    journeys = Counter(
        (
            str(row.get("query", "")).strip(),
            str(row.get("followup_text", "")).strip(),
        )
        for row in clicked
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "events": len(rows),
        "suggestions_shown": shown_count,
        "suggestions_clicked": clicked_count,
        "ctr": clicked_count / shown_count if shown_count else 0.0,
        "ctr_by_rank": rank_counts,
        "log_path": str(LOG_PATH.relative_to(ROOT)),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    with JOURNEY_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["from_query", "to_followup", "clicks"])
        writer.writeheader()
        for (source, target), count in journeys.most_common():
            writer.writerow({"from_query": source, "to_followup": target, "clicks": count})

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {JOURNEY_PATH}")


if __name__ == "__main__":
    main()
