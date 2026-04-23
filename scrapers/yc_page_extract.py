from __future__ import annotations

import html
import json
import re
from typing import Any


SLUG_PATTERN = re.compile(r"/companies/([^/?#]+)")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slug_from_url(url: str) -> str:
    match = SLUG_PATTERN.search(url or "")
    return match.group(1).strip().lower() if match else ""


def slug_to_name(slug: str) -> str:
    pieces = [piece for piece in re.split(r"[-_]+", slug or "") if piece]
    return " ".join(piece.upper() if piece.isupper() else piece.capitalize() for piece in pieces)


def extract_title_text(page_html: str) -> str:
    match = re.search(r"<title>(.*?)</title>", page_html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()


def html_to_text_lines(page_html: str) -> list[str]:
    text = html.unescape(page_html)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return [line for line in lines if line]


def split_yc_title(page_title: str) -> tuple[str, str]:
    title = safe_text(page_title).replace(" | Y Combinator", "").strip()
    if ": " in title:
        name, one_liner = title.split(": ", 1)
        return name.strip(), one_liner.strip()
    return title, ""


def _decode_json_string(raw_value: str) -> str:
    if raw_value is None:
        return ""
    try:
        return json.loads(f'"{raw_value}"')
    except json.JSONDecodeError:
        return raw_value.replace('\\"', '"').replace("\\n", "\n").replace("\\/", "/")


def extract_json_string(blob: str, field: str) -> str:
    pattern = rf'"{re.escape(field)}":"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, blob, flags=re.DOTALL)
    return safe_text(_decode_json_string(match.group(1))) if match else ""


def extract_json_number(blob: str, field: str) -> str:
    pattern = rf'"{re.escape(field)}":(\d+)'
    match = re.search(pattern, blob)
    return match.group(1).strip() if match else ""


def extract_founders(blob: str) -> list[str]:
    match = re.search(r'"founders":\[(.*?)\](?:,|})', blob, flags=re.DOTALL)
    if not match:
        return []

    names: list[str] = []
    for name_match in re.finditer(r'"full_name":"((?:\\.|[^"\\])*)"', match.group(1)):
        name = safe_text(_decode_json_string(name_match.group(1)))
        if name and name not in names:
            names.append(name)
    return names


def pick_labeled_value(lines: list[str], label: str) -> str:
    for idx, line in enumerate(lines):
        if line == label:
            for next_line in lines[idx + 1: idx + 5]:
                if next_line and next_line != label:
                    return next_line
    return ""


def extract_launch_fallback(lines: list[str]) -> tuple[str, str]:
    if "Company Launches" not in lines:
        return "", ""

    one_liner = ""
    description = ""

    try:
        launch_idx = lines.index("Company Launches")
    except ValueError:
        return "", ""

    for candidate in lines[launch_idx + 1: launch_idx + 6]:
        if ":" in candidate:
            left, right = candidate.split(":", 1)
            if len(right.strip().split()) >= 4:
                one_liner = right.strip()
                break

    if "TLDR:" in lines[launch_idx:]:
        tldr_idx = lines.index("TLDR:")
        collected: list[str] = []
        stop_labels = {"Problem:", "Solution:", "Our ask:", "Latest News", "Footer"}
        for candidate in lines[tldr_idx + 1:]:
            if candidate in stop_labels:
                break
            if candidate.endswith(":") and len(candidate.split()) <= 3:
                break
            collected.append(candidate)
            if len(" ".join(collected)) >= 220:
                break

        description = " ".join(collected).strip()

    return one_liner, description


def extract_company_fields_from_html(page_html: str, url: str, page_title: str = "") -> dict[str, Any]:
    unescaped = html.unescape(page_html)
    text_lines = html_to_text_lines(page_html)
    slug = slug_from_url(url)

    anchor = f'"slug":"{slug}"' if slug else ""
    start = unescaped.find(anchor) if anchor else -1
    if start >= 0:
        blob = unescaped[max(0, start - 500): min(len(unescaped), start + 70000)]
    else:
        blob = unescaped

    resolved_title = page_title or extract_title_text(page_html)
    title_name, title_one_liner = split_yc_title(resolved_title)
    launch_one_liner, launch_description = extract_launch_fallback(text_lines)

    company_name = (
        extract_json_string(blob, "name")
        or title_name
        or slug_to_name(slug)
    )
    one_liner = extract_json_string(blob, "one_liner") or title_one_liner or launch_one_liner
    description = (
        extract_json_string(blob, "long_description")
        or launch_description
        or one_liner
        or title_one_liner
    )
    batch = extract_json_string(blob, "batch_name") or pick_labeled_value(text_lines, "Batch:")
    location = extract_json_string(blob, "location")
    if not location or location.startswith("/companies/"):
        location = pick_labeled_value(text_lines, "Location:")
    team_size = extract_json_number(blob, "team_size") or pick_labeled_value(text_lines, "Team Size:")
    website = extract_json_string(blob, "website")
    founders = extract_founders(blob)

    return {
        "company_name": company_name,
        "one_liner": one_liner,
        "description": description,
        "batch": batch,
        "location": location,
        "team_size": team_size,
        "website": website,
        "founders": founders,
    }
