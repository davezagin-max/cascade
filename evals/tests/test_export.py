import json
import math
from pathlib import Path
from typing import Any

import pytest

from cascade import store
from cascade.config import OPUS
from cascade.export import (
    REPO_ROOT,
    bootstrap_ci,
    build_summary,
    pareto_frontier,
    percentile,
    suite_hash,
    summarize_arms,
    task_detail,
    web_router,
)
from cascade.tasks import load_tasks

TASKS = load_tasks()
DATA = Path(REPO_ROOT) / "data"


def test_pareto_frontier_keeps_only_undominated_arms() -> None:
    points = [
        ("cheap-bad", 1.0, 0.50),
        ("cheap-good", 1.0, 0.60),  # same cost, better: dominates cheap-bad
        ("mid-worse", 2.0, 0.55),  # costs more than cheap-good and is worse
        ("mid", 3.0, 0.80),
        ("pricey-equal", 5.0, 0.80),  # no better than mid
        ("best", 9.0, 0.95),
    ]
    assert pareto_frontier(points) == ["cheap-good", "mid", "best"]
    assert pareto_frontier([]) == []
    # An exact tie on both axes dominates in neither direction, so both arms stay. (Efforts
    # that happen to cost and score the same must not have one dropped by name order.)
    tied = [("a-low", 1.0, 0.7), ("a-high", 1.0, 0.7), ("b", 2.0, 0.9), ("c", 2.0, 0.6)]
    assert pareto_frontier(tied) == ["a-high", "a-low", "b"]


def test_percentile_interpolates() -> None:
    assert percentile([], 0.5) is None
    assert percentile([10.0], 0.95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([4.0, 1.0, 3.0, 2.0], 1.0) == 4.0
    assert percentile(list(map(float, range(101))), 0.95) == pytest.approx(95.0)


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean() -> None:
    rates = [1.0] * 40 + [0.0] * 20
    low, high = bootstrap_ci(rates, "arm")
    assert (low, high) == bootstrap_ci(rates, "arm")
    assert low < 2 / 3 < high
    assert low > 0.5 and high < 0.82  # roughly +/- 0.12 for n = 60
    assert bootstrap_ci([1.0] * 10, "arm") == (1.0, 1.0)


def test_suite_hash_changes_with_the_suite() -> None:
    assert suite_hash(TASKS) == suite_hash(list(TASKS))
    assert suite_hash(TASKS[:-1]) != suite_hash(TASKS)


def _close(a: Any, b: Any, path: str = "") -> None:
    if isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None, path
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {a} != {b}"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            _close(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            _close(x, y, f"{path}[{index}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _committed() -> tuple[dict, list[dict], dict, dict, dict | None]:
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    run_id = summary["meta"]["run_id"]
    directory = store.run_dir(run_id)
    rows = store.read_rows(store.results_path(run_id))
    model = json.loads((directory / "router.json").read_text(encoding="utf-8"))
    caching_path = directory / "caching.json"
    caching = (
        json.loads(caching_path.read_text(encoding="utf-8")) if caching_path.exists() else None
    )
    return summary, rows, store.read_meta(run_id), model, caching


def test_committed_dashboard_data_is_exactly_what_export_produces() -> None:
    """Whatever run is committed, real or synthetic, the dashboard JSON must come from it."""
    summary, rows, meta, model, caching = _committed()
    rebuilt = build_summary(rows, TASKS, meta=meta, router_model=model, caching=caching)
    _close(json.loads(json.dumps(rebuilt)), summary)
    # Byte for byte, too: exporting a run again must not rewrite the file in a new key order.
    committed_text = (DATA / "summary.json").read_text(encoding="utf-8")
    assert json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n" == committed_text
    served = json.loads((DATA / "router.json").read_text(encoding="utf-8"))
    _close(json.loads(json.dumps(web_router(model, rebuilt["arms"]))), served)
    assert summary["meta"]["suite_hash"] == suite_hash(TASKS), (
        "results are stale: the suite changed"
    )


def test_summary_is_honest_about_where_the_data_came_from() -> None:
    summary, rows, meta, _model, caching = _committed()
    assert summary["meta"]["synthetic"] is meta["synthetic"]
    assert all(row["synthetic"] is meta["synthetic"] for row in rows)
    if caching is not None:
        assert caching["synthetic"] is meta["synthetic"]
    if meta["synthetic"]:
        assert "Not a measurement" in summary["meta"]["generator"]["note"]


def test_arm_summary_matches_the_rows() -> None:
    summary, rows, *_ = _committed()
    arms = {arm["key"]: arm for arm in summarize_arms(rows)}
    key = summary["meta"]["baseline_arm"]
    model, effort = arms[key]["model"], arms[key]["effort"]
    mine = [row for row in rows if row["model"] == model and row["effort"] == effort]
    assert arms[key]["n_rows"] == len(mine)
    answered = [row for row in mine if row["status"] != "error"]
    assert arms[key]["cost_per_1k"] == pytest.approx(
        1000 * sum(r["cost_usd"] for r in answered) / len(answered)
    )
    graded = [row for row in mine if row["passed"] is not None]
    # Quality is the mean over tasks of each task's pass rate, so a task that lost a trial
    # to an API error still counts once.
    per_task: dict[str, list[bool]] = {}
    for row in graded:
        per_task.setdefault(row["task_id"], []).append(row["passed"])
    rates = [sum(passes) / len(passes) for passes in per_task.values()]
    assert arms[key]["quality"] == pytest.approx(sum(rates) / len(rates))
    low, high = arms[key]["ci"]
    assert low <= arms[key]["quality"] <= high
    assert arms[key]["latency_n"] == sum(1 for row in mine if row["latency_ms"] is not None)
    frontier = [arm for arm in arms.values() if arm["on_frontier"]]
    assert frontier and all(0 <= arm["quality"] <= 1 for arm in arms.values())
    # The cheapest arm is always on the frontier.
    assert min(arms.values(), key=lambda arm: arm["cost_per_request"])["on_frontier"]


def test_task_detail_stores_each_distinct_answer_once() -> None:
    summary, rows, *_ = _committed()
    task = TASKS[0]
    mine = [row for row in rows if row["task_id"] == task.id]
    detail = task_detail(task, mine)
    assert len(detail["cells"]) == len(mine)
    assert len(detail["answers"]) == len({row["answer"] for row in mine})
    for cell in detail["cells"]:
        assert 0 <= cell["answer"] < len(detail["answers"])
    assert detail["prompt"] == task.prompt and detail["reference"] == task.reference
    published = json.loads(
        (Path(REPO_ROOT) / "public" / "run" / "tasks" / f"{task.id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert published["id"] == task.id and len(published["cells"]) == len(mine)
    assert {t["id"] for t in summary["tasks"]} == {row["task_id"] for row in rows}


def test_export_says_when_playground_fixtures_predate_the_router(tmp_path: Path) -> None:
    from cascade.export import stale_fixtures_note

    assert stale_fixtures_note(tmp_path, "abc") is None  # no fixtures at all
    store_path = tmp_path / "fixtures" / "route-and-run.json"
    store_path.parent.mkdir()

    def write(*fixtures: dict) -> None:
        store_path.write_text(json.dumps({f"p{i}": f for i, f in enumerate(fixtures)}))

    current = {"placeholder": True, "data": {"routerVersion": "abc"}}
    older = {"placeholder": True, "data": {"routerVersion": "old"}}
    write(current, current)
    assert stale_fixtures_note(tmp_path, "abc") is None
    write(current, older)
    note = stale_fixtures_note(tmp_path, "abc")
    assert note is not None and "1 playground fixture(s)" in note
    assert "placeholder_fixtures.py" in note and "pnpm record" in note
    write({**older, "placeholder": False}, current)
    recorded = stale_fixtures_note(tmp_path, "abc")
    assert recorded is not None and "placeholder_fixtures.py" not in recorded


def test_failed_requests_do_not_make_an_arm_look_cheaper() -> None:
    def row(task_id: str, status: str, cost: float, passed: bool | None) -> dict:
        return {
            "task_id": task_id, "category": "code", "model": OPUS, "effort": "high", "trial": 0,
            "status": status, "cost_usd": cost, "input_tokens": 100 if cost else 0,
            "output_tokens": 400 if cost else 0, "latency_ms": None, "passed": passed,
            "score": None if passed is None else float(passed),
        }  # fmt: skip

    rows = [row("a", "ok", 0.02, True), row("b", "ok", 0.04, False), row("c", "error", 0.0, None)]
    (arm,) = summarize_arms(rows)
    assert arm["cost_per_request"] == pytest.approx(0.03)  # not 0.02: the failure is left out
    assert arm["output_tokens_mean"] == 400 and arm["input_tokens_mean"] == 100
    assert (
        arm["quality"] == 0.5 and arm["errors"] == 1 and arm["ungraded"] == 1 and arm["n_rows"] == 3
    )
