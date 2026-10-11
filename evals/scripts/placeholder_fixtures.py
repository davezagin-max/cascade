"""Writes the playground's placeholder fixtures.

    uv run python scripts/placeholder_fixtures.py

There is no API key yet, so the answers below are written by hand and every fixture
is marked `"placeholder": true` with zero usage and zero cost. The routing decision
stored with each one is NOT hand-written: it is computed here by the real router
(`cascade.router.route`, which the TypeScript port matches bit for bit).

`pnpm record` replaces each of these with a real run under the same key. Run this
script again only if the placeholders are still in use and the router was retrained.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from cascade.plan import arm_from_key
from cascade.router import route

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "fixtures" / "route-and-run.json"
PRESETS = ROOT / "fixtures" / "presets.json"
ROUTE = "/api/route-and-run"

SAMPLES: list[dict[str, str]] = [
    {
        "demoId": "support-ticket",
        "label": "Classify a support ticket",
        "prompt": (
            "Classify this support message as one of: billing, shipping, technical, account.\n\n"
            '"My order says delivered but nothing has arrived, and the tracking link is dead."'
        ),
        "answer": (
            "**shipping**\n\nThe customer is reporting a parcel marked delivered that never "
            "arrived, plus a broken tracking link. Both are delivery problems, not a charge, a "
            "product fault, or a login issue."
        ),
    },
    {
        "demoId": "refund-email",
        "label": "Write a short customer email",
        "prompt": (
            "Write an email to a customer explaining that their refund will take five working "
            "days. Keep it under 90 words, warm in tone, and do not apologize more than once."
        ),
        "answer": (
            "Subject: Your refund is on its way\n\nHi Maya,\n\nThanks for your patience. Your "
            "refund has been approved and is now with our payments team. It will reach your "
            "original payment method within five working days, and your bank may take another "
            "day to show it.\n\nI'm sorry for the wait. If the money has not arrived by next "
            "Friday, reply to this email and I will chase it personally.\n\nWarm regards,\nJo\n"
            "Customer Support"
        ),
    },
    {
        "demoId": "explain-race",
        "label": "Explain a concept with one analogy",
        "prompt": (
            "Explain what a race condition is to a new engineer in at most 120 words, with "
            "exactly one concrete analogy."
        ),
        "answer": (
            "A race condition is a bug where the result depends on the unlucky timing of two "
            "things happening at once.\n\nPicture two cashiers selling the last concert ticket. "
            "Each checks the system, sees one ticket left, and sells it. Both checks were "
            "correct when they were made, yet two customers now hold the same seat.\n\nIn code "
            "the cashiers are threads or requests, and the ticket is shared data. Each one "
            "reads a value, decides, then writes, and another slips in between the read and "
            "the write. The fix is to make check-and-update a single indivisible step, with a "
            "lock, a transaction, or an atomic operation."
        ),
    },
    {
        "demoId": "meeting-json",
        "label": "Extract a meeting as JSON",
        "prompt": (
            "Extract the meeting as JSON with the keys date (YYYY-MM-DD), time_24h, and "
            'attendees.\n\n"Sent Monday 2026-06-01: can we do Wednesday at 4pm with Ana and Raj?"'
        ),
        "answer": (
            '```json\n{\n  "date": "2026-06-03",\n  "time_24h": "16:00",\n'
            '  "attendees": ["Ana", "Raj"]\n}\n```\n\nThe message was sent on Monday 1 June 2026, '
            "so the Wednesday it refers to is 3 June."
        ),
    },
    {
        "demoId": "merge-sorted",
        "label": "Write a linear-time merge",
        "prompt": (
            "Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` "
            "that merges two sorted lists into one sorted list in O(len(a) + len(b)) time "
            "without using sort()."
        ),
        "answer": (
            "```python\ndef merge_sorted(a: list[int], b: list[int]) -> list[int]:\n"
            "    merged: list[int] = []\n    i = j = 0\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n            merged.append(a[i])\n            i += 1\n"
            "        else:\n            merged.append(b[j])\n            j += 1\n"
            "    merged.extend(a[i:])\n    merged.extend(b[j:])\n    return merged\n```\n\n"
            "Two indices walk the lists once each, so the work is O(len(a) + len(b)). Using "
            "`<=` keeps equal elements from `a` ahead of those from `b`, which makes the merge "
            "stable. Neither input is modified."
        ),
    },
    {
        "demoId": "dice-probability",
        "label": "Solve a probability problem",
        "prompt": (
            "Three fair dice are rolled. What is the probability that the sum is at least 16? "
            "Give a fraction in lowest terms."
        ),
        "answer": (
            "Count the ordered outcomes out of 6^3 = 216.\n\n"
            "- Sum 18: (6,6,6), 1 way.\n"
            "- Sum 17: one 5 and two 6s, 3 ways.\n"
            "- Sum 16: one 4 and two 6s (3 ways), or two 5s and one 6 (3 ways), 6 ways.\n\n"
            "That is 1 + 3 + 6 = 10 outcomes, so the probability is 10/216 = **5/108**."
        ),
    },
]

ZERO_USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def chunks(text: str) -> list[str]:
    """Split into word-sized pieces, the way streamed text arrives."""
    return [piece for piece in re.findall(r"\S+\s*|\s+", text) if piece]


def build() -> tuple[dict, list]:
    model = json.loads((ROOT / "data" / "router.json").read_text(encoding="utf-8"))
    store: dict[str, dict] = {}
    presets = []
    for sample in SAMPLES:
        arm = arm_from_key(route(model, sample["prompt"])["arm"])
        ref = {"key": arm.key, "model": arm.model, "effort": arm.effort}
        events = [
            {"at": 0, "event": {"type": "decision", "arm": ref, "routerVersion": model["version"]}}
        ]
        clock = 700
        for piece in chunks(sample["answer"]):
            events.append({"at": clock, "event": {"type": "text", "delta": piece}})
            clock += 55
        store[sample["demoId"]] = {
            "label": sample["label"],
            "placeholder": True,
            "recordedAt": None,
            "model": arm.model,
            "usage": ZERO_USAGE,
            "costUsd": 0,
            "data": {
                "arm": ref,
                "routerVersion": model["version"],
                "answer": sample["answer"],
                "stopReason": "end_turn",
                "comparison": None,
            },
            "events": events,
        }
        # The label lives with the preset, not the fixture: recording overwrites the fixture.
        presets.append(
            {
                "route": ROUTE,
                "label": sample["label"],
                "body": {"demoId": sample["demoId"], "prompt": sample["prompt"]},
            }
        )
    return dict(sorted(store.items())), presets


def main() -> int:
    if STORE.exists():
        current = json.loads(STORE.read_text(encoding="utf-8"))
        recorded = [key for key, fixture in current.items() if not fixture.get("placeholder")]
        if recorded and "--force" not in sys.argv:
            print(f"Refusing to overwrite recorded runs: {recorded}. Pass --force to replace them.")
            return 1
    store, presets = build()
    STORE.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PRESETS.write_text(json.dumps(presets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    arms = sorted({fixture["data"]["arm"]["key"] for fixture in store.values()})
    print(f"wrote {len(store)} placeholder fixtures covering {len(arms)} arms: {', '.join(arms)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
