from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "analytics" / "followup_events.jsonl"
SEED_SOURCE = "seeded_objective5_evaluation"


SESSIONS = [
    {
        "session_id": "obj5_seed_hk_compliance",
        "topic": "HK Company Registration & Compliance",
        "turns": [
            {
                "turn_id": "obj5_seed_hk_01",
                "query": "I want to incorporate a Hong Kong limited company. What should I do first?",
                "followups": [
                    "How do I check whether my preferred company name is available?",
                    "What documents are required for local company incorporation?",
                    "When do I receive the Business Registration Number after incorporation?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_hk_02",
                "query": "How do I check whether my preferred company name is available?",
                "followups": [
                    "What name restrictions should I avoid before submitting incorporation forms?",
                    "How do I reserve or change a Hong Kong company name later?",
                    "What happens if the Companies Registry rejects the proposed name?",
                ],
                "clicked_rank": 2,
            },
            {
                "turn_id": "obj5_seed_hk_03",
                "query": "How do I reserve or change a Hong Kong company name later?",
                "followups": [
                    "Which Companies Registry form is used for a company name change?",
                    "How does the new Unique Business Identifier affect my company records?",
                    "Do I need to notify banks and contracts after changing the company name?",
                ],
                "clicked_rank": 2,
            },
            {
                "turn_id": "obj5_seed_hk_04",
                "query": "How does the new Unique Business Identifier affect my company records?",
                "followups": [
                    "Where can I look up the BRN mapping for an existing company?",
                    "Does the former company number still appear in public records?",
                    "How should I update internal documents after UBI adoption?",
                ],
                "clicked_rank": None,
            },
        ],
    },
    {
        "session_id": "obj5_seed_operations",
        "topic": "Business Operations & Planning",
        "turns": [
            {
                "turn_id": "obj5_seed_ops_01",
                "query": "How should I price a small online retail product before launch?",
                "followups": [
                    "How do I calculate gross margin for my first product batch?",
                    "What demand assumptions should I test before ordering inventory?",
                    "How should I compare competitor prices without racing to the bottom?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_ops_02",
                "query": "How do I calculate gross margin for my first product batch?",
                "followups": [
                    "What costs should be included in contribution margin?",
                    "How do shipping and returns change my break-even point?",
                    "When should I adjust pricing after early customer feedback?",
                ],
                "clicked_rank": 2,
            },
            {
                "turn_id": "obj5_seed_ops_03",
                "query": "How do shipping and returns change my break-even point?",
                "followups": [
                    "How much safety stock should I keep for seasonal demand spikes?",
                    "What operational metrics should I track weekly?",
                    "How do I decide whether to outsource fulfillment?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_ops_04",
                "query": "How much safety stock should I keep for seasonal demand spikes?",
                "followups": [
                    "How do I forecast inventory when historical sales are limited?",
                    "What should I do if supplier lead times are unreliable?",
                    "How can I reduce stockouts without tying up too much cash?",
                ],
                "clicked_rank": None,
            },
        ],
    },
    {
        "session_id": "obj5_seed_yc_funding",
        "topic": "YC Startup Patterns & Funding",
        "turns": [
            {
                "turn_id": "obj5_seed_yc_01",
                "query": "What YC startup examples are useful for validating a B2B SaaS idea?",
                "followups": [
                    "Which early customer signals matter most before building the full product?",
                    "How did YC companies use MVPs to learn from customers?",
                    "When should a founder start preparing for seed fundraising?",
                ],
                "clicked_rank": 2,
            },
            {
                "turn_id": "obj5_seed_yc_02",
                "query": "How did YC companies use MVPs to learn from customers?",
                "followups": [
                    "What metrics show that an MVP is solving a real pain point?",
                    "How should I interview users after they test the MVP?",
                    "What common MVP mistakes appear in startup case studies?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_yc_03",
                "query": "What metrics show that an MVP is solving a real pain point?",
                "followups": [
                    "How do I turn early traction into a seed pitch narrative?",
                    "What retention metric should I track for a B2B SaaS product?",
                    "How many pilot customers are enough before fundraising?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_yc_04",
                "query": "How do I turn early traction into a seed pitch narrative?",
                "followups": [
                    "What should be included in a concise seed funding deck?",
                    "How do I explain market size without exaggerating it?",
                    "Which YC fundraising examples match enterprise SaaS companies?",
                ],
                "clicked_rank": None,
            },
        ],
    },
    {
        "session_id": "obj5_seed_sentiment",
        "topic": "Social Media Sentiment",
        "turns": [
            {
                "turn_id": "obj5_seed_sent_01",
                "query": "What are people saying about our product category on Reddit and Facebook?",
                "followups": [
                    "Which complaints appear most often in negative posts?",
                    "What positive phrases suggest demand for this product category?",
                    "How should I compare sentiment between Reddit and Facebook?",
                ],
                "clicked_rank": 1,
            },
            {
                "turn_id": "obj5_seed_sent_02",
                "query": "Which complaints appear most often in negative posts?",
                "followups": [
                    "How can I group complaints into product improvement themes?",
                    "Which objections should be handled in launch messaging?",
                    "Do the complaints indicate a compliance or trust issue?",
                ],
                "clicked_rank": 2,
            },
            {
                "turn_id": "obj5_seed_sent_03",
                "query": "Which objections should be handled in launch messaging?",
                "followups": [
                    "How can I turn sentiment findings into landing page copy?",
                    "What customer segments react most positively to the offer?",
                    "Which sentiment trends should I monitor after launch?",
                ],
                "clicked_rank": 3,
            },
            {
                "turn_id": "obj5_seed_sent_04",
                "query": "Which sentiment trends should I monitor after launch?",
                "followups": [
                    "How often should I refresh social sentiment analysis?",
                    "What warning signs indicate negative word of mouth is increasing?",
                    "How should sentiment insights influence the next product roadmap?",
                ],
                "clicked_rank": None,
            },
        ],
    },
]


def existing_non_seed_rows() -> list[dict]:
    rows: list[dict] = []
    if not LOG_PATH.exists():
        return rows
    with LOG_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("data_source") != SEED_SOURCE:
                rows.append(row)
    return rows


def build_seed_rows() -> list[dict]:
    rows: list[dict] = []
    current_time = datetime(2026, 4, 25, 7, 0, 0, tzinfo=timezone.utc)

    for session in SESSIONS:
        for turn in session["turns"]:
            for rank, followup in enumerate(turn["followups"], 1):
                rows.append({
                    "timestamp": current_time.isoformat(),
                    "session_id": session["session_id"],
                    "event_type": "shown",
                    "turn_id": turn["turn_id"],
                    "query": turn["query"],
                    "followup_text": followup,
                    "followup_rank": rank,
                    "topic": session["topic"],
                    "data_source": SEED_SOURCE,
                })
                current_time += timedelta(seconds=1)

            clicked_rank = turn.get("clicked_rank")
            if clicked_rank:
                rows.append({
                    "timestamp": current_time.isoformat(),
                    "session_id": session["session_id"],
                    "event_type": "clicked",
                    "turn_id": turn["turn_id"],
                    "query": turn["query"],
                    "followup_text": turn["followups"][clicked_rank - 1],
                    "followup_rank": clicked_rank,
                    "topic": session["topic"],
                    "data_source": SEED_SOURCE,
                })
                current_time += timedelta(seconds=20)
            else:
                current_time += timedelta(seconds=20)

    return rows


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = existing_non_seed_rows() + build_seed_rows()

    with LOG_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    shown = sum(1 for row in rows if row.get("event_type") == "shown")
    clicked = sum(1 for row in rows if row.get("event_type") == "clicked")
    seed_count = sum(1 for row in rows if row.get("data_source") == SEED_SOURCE)
    print(f"[DONE] Wrote {len(rows)} total follow-up events to {LOG_PATH}")
    print(f"[DONE] Seeded events: {seed_count}")
    print(f"[DONE] Total shown: {shown}")
    print(f"[DONE] Total clicked: {clicked}")


if __name__ == "__main__":
    main()
