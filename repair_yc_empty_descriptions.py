#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from yc_page_extract import extract_company_fields_from_html


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

MISSING_TEXT_VALUES = {"", "N/A", "None", "null"}
PLACEHOLDER_NAMES = {"The Problem", "The Solution", "Problem", "TL;DR", "tl;dr", "TLDR", "TLDR;"}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def needs_description_repair(row: dict) -> bool:
    return str(row.get("description", "")).strip() in MISSING_TEXT_VALUES


def is_missing_text(value: object) -> bool:
    return str(value or "").strip() in MISSING_TEXT_VALUES


def looks_like_person_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text or ":" in text or "." in text:
        return False
    parts = text.split()
    return 2 <= len(parts) <= 4 and all(part[:1].isupper() for part in parts)


def fetch_and_extract(url: str, timeout: int = 30) -> dict:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return extract_company_fields_from_html(response.text, url)


def repair_row(row: dict, extracted: dict) -> tuple[dict, dict]:
    repaired = dict(row)
    changes: dict[str, str] = {}

    def update_if_missing(key: str, new_value: str | list[str]) -> None:
        if isinstance(new_value, list):
            if not repaired.get(key) and new_value:
                repaired[key] = new_value
                changes[key] = f"{len(new_value)} items"
            return

        new_value = str(new_value or "").strip()
        current_value = str(repaired.get(key, "") or "").strip()
        if is_missing_text(current_value) and new_value:
            repaired[key] = new_value
            changes[key] = new_value[:120]

    if is_missing_text(repaired.get("description", "")):
        fallback_desc = (
            str(extracted.get("description", "")).strip()
            or str(extracted.get("one_liner", "")).strip()
        )
        if fallback_desc:
            repaired["description"] = fallback_desc
            changes["description"] = fallback_desc[:120]

    current_company_name = str(repaired.get("company_name", "") or "").strip()
    extracted_company_name = str(extracted.get("company_name", "") or "").strip()
    if extracted_company_name and (
        is_missing_text(current_company_name) or current_company_name in PLACEHOLDER_NAMES
    ):
        repaired["company_name"] = extracted_company_name
        changes["company_name"] = extracted_company_name[:120]

    current_one_liner = str(repaired.get("one_liner", "") or "").strip()
    extracted_one_liner = str(extracted.get("one_liner", "") or "").strip()
    if extracted_one_liner and (
        is_missing_text(current_one_liner) or looks_like_person_name(current_one_liner)
    ):
        repaired["one_liner"] = extracted_one_liner
        changes["one_liner"] = extracted_one_liner[:120]

    update_if_missing("batch", extracted.get("batch", ""))
    update_if_missing("location", extracted.get("location", ""))
    update_if_missing("team_size", extracted.get("team_size", ""))
    update_if_missing("website", extracted.get("website", ""))
    update_if_missing("founders", extracted.get("founders", []))

    return repaired, changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair empty descriptions in yc_detailed_data.jsonl.")
    parser.add_argument("--input", default="yc_detailed_data.jsonl", help="Input JSONL path")
    parser.add_argument("--output", default="", help="Output JSONL path; default overwrites input")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent request workers")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    backup_path = input_path.with_name(f"{input_path.stem}.before_desc_repair{input_path.suffix}")

    rows = load_rows(input_path)
    targets = [(idx, row) for idx, row in enumerate(rows) if needs_description_repair(row)]

    if output_path == input_path and not backup_path.exists():
        shutil.copy2(input_path, backup_path)

    results: dict[int, dict] = {}
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {
            executor.submit(fetch_and_extract, row["url"]): (idx, row["url"])
            for idx, row in targets
        }

        for future in as_completed(future_map):
            idx, url = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                failures.append({"index": idx, "url": url, "error": str(exc)})

    repaired_count = 0
    filled_descriptions = 0
    for idx, row in enumerate(rows):
        extracted = results.get(idx)
        if not extracted:
            continue
        original_desc = str(row.get("description", "")).strip()
        repaired_row, changes = repair_row(row, extracted)
        rows[idx] = repaired_row
        if changes:
            repaired_count += 1
        if not original_desc and str(repaired_row.get("description", "")).strip():
            filled_descriptions += 1

    actual_output_path = output_path
    try:
        write_rows(output_path, rows)
    except PermissionError:
        actual_output_path = input_path.with_name(f"{input_path.stem}.repaired{input_path.suffix}")
        write_rows(actual_output_path, rows)

    summary = {
        "input_path": str(input_path),
        "output_path": str(actual_output_path),
        "backup_path": str(backup_path) if output_path == input_path else "",
        "target_rows": len(targets),
        "rows_repaired": repaired_count,
        "descriptions_filled": filled_descriptions,
        "failures": failures[:20],
        "failure_count": len(failures),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
