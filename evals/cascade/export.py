"""Turns a graded run into the JSON the dashboard bundles.

Outputs (paths relative to the repo root):

- `data/summary.json`: everything the dashboard renders, imported at build time.
- `data/router.json`: the trained router, loaded by the playground route and the browser.
- `data/router-vectors.json`: prompts with the Python router's exact outputs, used by
  both test suites to prove the TypeScript port matches.
- `public/run/tasks/<task>.json`: per-task detail (prompt, grader, every arm's answer
  and grade), fetched on demand by the task explorer so it stays out of the bundle.

The aggregation here is plain Python with a seeded bootstrap, so the same Parquet
file always produces the same JSON on any platform.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import BATCH_MULTIPLIER, EFFORTS, MODELS, SHORT_NAMES, pricing_snapshot
from .plan import Arm, build_arms
from .prompts import system_prompt
from .router import arm_key, baseline_arm, route
from .store import Row, round_floats, write_json
from .tasks import CATEGORIES, CATEGORY_LABELS, Task, public_grader

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_RESAMPLES = 2000

# Prompts that stress the places Python and JavaScript text handling differ.
PARITY_EXTRAS: tuple[str, ...] = (
    "",
    " ",
    "?",
    "Classify this: GREAT product, would buy again!!",
    "CLASSIFY the SENTIMENT as Positive or Negative. Return JSON.",
    "line one\r\nline two\rline three\n- item\n\t* item\n 1. item\n[2] item",
    "Café naïve résumé: extrahiere die Felder als JSON — bitte.",
    "数学の問題: 2^10 はいくつですか？",
    "Emoji test \U0001f600\U0001f680 with astral chars \U0001d54f and a KK sign, how many?",
    "What is 12,345.67 + 1,000? Then compute 3 * 4 = ? and 7^100 mod 13.",
    "def f(x):\n    return x ** 2\n\nWrite a Python function `g()` that returns f(3). O(n log n) please.",
    "Explain, in at most 100 words, why the sky is blue. Do not use jargon. Write an email to Sam.",
    "If A then B, otherwise C; before D, after E, until F, while G. Each must not, unless only.",
    'He said "yes", she said "no", and "maybe" was written on the door of Room Aspen in Leeds.',
    "a" * 5000,
    "word " * 800,
    "How many ways can 8 people be seated so that no two adjacent seats hold siblings? Prove it.",
)


def suite_hash(tasks: list[Task]) -> str:
    """Changes when any task's prompt or grader changes, so stale results are detectable."""
    payload = json.dumps([[t.id, t.prompt, public_grader(t.grader)] for t in tasks], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def bootstrap_ci(per_task: list[float], label: str) -> tuple[float, float]:
    """95% interval for a mean, resampling tasks.

    Tasks are resampled, not rows: the trials of one task are not independent, and
    treating them as if they were would make the interval too narrow.
    """
    if not per_task:
        return (0.0, 0.0)
    rng = random.Random(f"bootstrap|{label}")
    n = len(per_task)
    means = sorted(
        sum(per_task[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


def pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Keys of the points no other point dominates, cheapest first.

    A point is dominated when another costs no more and scores no less, and is strictly
    better on one of the two. Two arms with exactly the same cost and quality do not
    dominate each other, so both stay: dropping one of them by name order would be arbitrary.
    """
    frontier = [
        (cost, -quality, key)
        for key, cost, quality in points
        if not any(
            other != key and c <= cost and q >= quality and (c < cost or q > quality)
            for other, c, q in points
        )
    ]
    return [key for _cost, _quality, key in sorted(frontier)]


def _arm_order(rows: list[Row]) -> list[Arm]:
    present = {(row["model"], row["effort"]) for row in rows}
    models = tuple(model for model in MODELS if any(m == model for m, _ in present))
    efforts = tuple({effort for _, effort in present})
    return [arm for arm in build_arms(models, efforts) if (arm.model, arm.effort) in present]  # type: ignore[arg-type]


def summarize_arms(rows: list[Row]) -> list[dict[str, Any]]:
    by_arm: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_arm[arm_key(row)].append(row)
    arms: list[dict[str, Any]] = []
    for arm in _arm_order(rows):
        members = by_arm[arm.key]
        graded = [row for row in members if row["passed"] is not None]
        per_task: dict[str, list[float]] = defaultdict(list)
        for row in graded:
            per_task[row["task_id"]].append(1.0 if row["passed"] else 0.0)
        task_rates = [_mean(values) for _, values in sorted(per_task.items())]
        low, high = bootstrap_ci(task_rates, arm.key)
        latencies = [row["latency_ms"] for row in members if row["latency_ms"] is not None]
        # A request that failed at the API cost nothing and used no tokens. Counting it
        # would make the arm look cheaper than it is.
        answered = [row for row in members if row["status"] != "error"]
        cost = _mean([row["cost_usd"] for row in answered])
        by_category: dict[str, Any] = {}
        for category in CATEGORIES:
            subset = [row for row in graded if row["category"] == category]
            if subset:
                by_category[category] = {
                    "quality": _mean([1.0 if row["passed"] else 0.0 for row in subset]),
                    "mean_score": _mean([row["score"] for row in subset]),
                    "cost_per_1k": 1000 * _mean([row["cost_usd"] for row in subset]),
                    "n_tasks": len({row["task_id"] for row in subset}),
                }
        arms.append(
            {
                "key": arm.key,
                "model": arm.model,
                "model_short": SHORT_NAMES.get(arm.model, arm.model),
                "effort": arm.effort,
                "quality": _mean(task_rates),
                "ci": [low, high],
                "mean_score": _mean([row["score"] for row in graded]),
                "cost_per_request": cost,
                "cost_per_1k": 1000 * cost,
                "cost_per_1k_batch": 1000 * cost * BATCH_MULTIPLIER,
                "latency_p50_ms": percentile(latencies, 0.5),
                "latency_p95_ms": percentile(latencies, 0.95),
                "latency_n": len(latencies),
                "input_tokens_mean": _mean([row["input_tokens"] for row in answered]),
                "output_tokens_mean": _mean([row["output_tokens"] for row in answered]),
                "n_rows": len(members),
                "refusals": sum(1 for row in members if row["status"] == "refusal"),
                "truncated": sum(1 for row in members if row["status"] == "truncated"),
                "errors": sum(1 for row in members if row["status"] == "error"),
                "ungraded": len(members) - len(graded),
                "by_category": by_category,
            }
        )
    frontier = set(pareto_frontier([(a["key"], a["cost_per_request"], a["quality"]) for a in arms]))
    for arm_entry in arms:
        arm_entry["on_frontier"] = arm_entry["key"] in frontier
    return arms


def _task_cells(rows: list[Row]) -> dict[str, dict[str, float]]:
    cells: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row["passed"] is not None:
            cells[arm_key(row)].append(row)
    # Keyed in arm order, not row order, so exporting the same run twice writes the same file.
    return {
        arm.key: {
            "pass_rate": _mean([1.0 if row["passed"] else 0.0 for row in cells[arm.key]]),
            "mean_score": _mean([row["score"] for row in cells[arm.key]]),
        }
        for arm in _arm_order(rows)
        if arm.key in cells
    }


def task_detail(task: Task, rows: list[Row]) -> dict[str, Any]:
    """Everything the explorer shows for one task. Repeated answers are stored once."""
    answers: list[str] = []
    position: dict[str, int] = {}
    cells = []
    order = {arm.key: i for i, arm in enumerate(_arm_order(rows))}
    for row in sorted(rows, key=lambda r: (order[arm_key(r)], r["trial"])):
        answer = row["answer"] or ""
        if answer not in position:
            position[answer] = len(answers)
            answers.append(answer)
        cells.append(
            {
                "arm": arm_key(row),
                "trial": row["trial"],
                "mode": row["mode"],
                "status": row["status"],
                "score": row["score"],
                "passed": row["passed"],
                "answer": position[answer],
                "details": json.loads(row["grader_details"]) if row["grader_details"] else {},
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["cost_usd"],
                "latency_ms": row["latency_ms"],
            }
        )
    return {
        "id": task.id,
        "title": task.title,
        "category": task.category,
        "difficulty": task.difficulty,
        "prompt": task.prompt,
        "system": system_prompt(task),
        "grader": public_grader(task.grader),
        "reference": task.reference,
        "answers": answers,
        "cells": cells,
    }


def web_router(model: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    """The router as served to the web app: the model, without the evaluation internals."""
    tokens = {arm["key"]: arm for arm in arms}
    return {
        "version": model["version"],
        "trained_on": model["trained_on"],
        "threshold": model["threshold"],
        "threshold_logit": model["threshold_logit"],
        "baseline_arm": model["baseline_arm"],
        "features": model["features"],
        "mean": model["mean"],
        "scale": model["scale"],
        "weights": model["weights"],
        "arms": [
            {
                **arm,
                "input_tokens_mean": round(tokens[arm["key"]]["input_tokens_mean"], 2),
                "output_tokens_mean": round(tokens[arm["key"]]["output_tokens_mean"], 2),
            }
            for arm in model["arms"]
        ],
    }


def parity_vectors(model: dict[str, Any], tasks: list[Task]) -> list[dict[str, Any]]:
    """Reference outputs of the Python router. Floats are written unrounded on purpose."""
    prompts = [task.prompt for task in tasks] + list(PARITY_EXTRAS)
    vectors = []
    for prompt in prompts:
        decision = route(model, prompt)
        vectors.append(
            {
                "prompt": prompt,
                "features": decision["features"],
                "contributions": decision["contributions"],
                "score": decision["score"],
                "arm": decision["arm"],
            }
        )
    return vectors


def build_summary(
    rows: list[Row],
    tasks: list[Task],
    *,
    meta: dict[str, Any],
    router_model: dict[str, Any],
    caching: dict[str, Any] | None,
) -> dict[str, Any]:
    arms = summarize_arms(rows)
    by_task: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    cv = router_model["cv"]
    weights = sorted(
        (
            {
                "name": feature["name"],
                "label": feature["label"],
                "weight": router_model["weights"][i],
            }
            for i, feature in enumerate(router_model["features"])
        ),
        key=lambda item: -abs(item["weight"]),
    )
    summary = {
        "meta": {
            "run_id": meta["run_id"],
            "synthetic": bool(meta["synthetic"]),
            "created_at": meta["created_at"],
            "generator": dict(sorted(meta["generator"].items())) if meta.get("generator") else None,
            "suite_hash": meta.get("suite_hash"),
            "models": [m for m in MODELS if any(arm["model"] == m for arm in arms)],
            "efforts": [e for e in EFFORTS if any(arm["effort"] == e for arm in arms)],
            "trials": 1 + max(row["trial"] for row in rows),
            "n_tasks": len(by_task),
            "n_rows": len(rows),
            "sync_rows": sum(1 for row in rows if row["mode"] == "sync"),
            "batch_rows": sum(1 for row in rows if row["mode"] == "batch"),
            "answers_billed_usd": sum(row["billed_usd"] for row in rows),
            "answers_list_usd": sum(row["cost_usd"] for row in rows),
            "judge_usd": sum(row["judge_usd"] or 0.0 for row in rows),
            "baseline_arm": baseline_arm([arm["key"] for arm in router_model["arms"]]),
            "pricing": pricing_snapshot(),
        },
        "categories": [
            {
                "key": category,
                "label": CATEGORY_LABELS[category],
                "n_tasks": sum(
                    1 for task in tasks if task.category == category and task.id in by_task
                ),
            }
            for category in CATEGORIES
            if any(task.category == category and task.id in by_task for task in tasks)
        ],
        "arms": arms,
        "router": {
            "version": router_model["version"],
            "threshold": router_model["threshold"],
            "l2": router_model["l2"],
            "l2_selection": router_model.get("l2_selection"),
            "l2_by_fold": cv["l2_by_fold"],
            "log_loss": cv["log_loss"],
            "folds": cv["folds"],
            "n_tasks": cv["n_tasks"],
            "baseline_arm": cv["baseline_arm"],
            "baseline": cv["baseline"],
            "headline": cv["router"],
            "oracle": cv["oracle"],
            "sweep": cv["sweep"],
            "confusion": cv["confusion"],
            "by_category": cv["by_category"],
            "calibration": cv["calibration"],
            "weights": weights,
            "arm_bias": [{"key": arm["key"], "bias": arm["bias"]} for arm in router_model["arms"]],
        },
        "caching": caching,
        "tasks": [
            {
                "id": task.id,
                "title": task.title,
                "category": task.category,
                "difficulty": task.difficulty,
                "grader": task.grader.type,
                "cells": _task_cells(by_task[task.id]),
                "router_pick": cv["picks"].get(task.id),
                "oracle_arm": cv["oracle_arms"].get(task.id),
            }
            for task in tasks
            if task.id in by_task
        ],
    }
    return round_floats(summary, 8)


def stale_fixtures_note(out: Path, router_version: str) -> str | None:
    """A reminder when the playground's stored answers were routed by an earlier router.

    Retraining changes which arm a preset is sent to. The dashboard copes (it shows the
    current decision and says when a recorded run used an older router), but the stored
    answers should be refreshed, and nothing else would say so.
    """
    path = out / "fixtures" / "route-and-run.json"
    if not path.exists():
        return None
    try:
        fixtures = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    stale = [
        key
        for key, fixture in fixtures.items()
        if (fixture.get("data") or {}).get("routerVersion") != router_version
    ]
    if not stale:
        return None
    if all(fixture.get("placeholder") for fixture in fixtures.values()):
        remedy = (
            "`pnpm record` replaces the hand-written samples with real runs; until then "
            "`uv run python scripts/placeholder_fixtures.py` realigns them with this router"
        )
    else:
        remedy = "`pnpm record` records them again"
    return (
        f"note: {len(stale)} playground fixture(s) were routed by an earlier router "
        f"(now {router_version}). {remedy}."
    )


def export_run(
    rows: list[Row],
    tasks: list[Task],
    *,
    meta: dict[str, Any],
    router_model: dict[str, Any],
    caching: dict[str, Any] | None,
    out: Path = REPO_ROOT,
) -> dict[str, Path]:
    """Write every dashboard file. Per-task files from an earlier export are replaced."""
    summary = build_summary(rows, tasks, meta=meta, router_model=router_model, caching=caching)
    served = web_router(router_model, summary["arms"])
    data = out / "data"
    task_dir = out / "public" / "run" / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for stale in task_dir.glob("*.json"):
        stale.unlink()

    by_task: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    for task in tasks:
        if task.id in by_task:
            detail = round_floats(task_detail(task, by_task[task.id]), 8)
            write_json(task_dir / f"{task.id}.json", detail, compact=True)

    write_json(data / "summary.json", summary)
    write_json(data / "router.json", served)
    write_json(data / "router-vectors.json", parity_vectors(served, tasks), compact=True)
    return {"summary": data / "summary.json", "router": data / "router.json", "tasks": task_dir}
