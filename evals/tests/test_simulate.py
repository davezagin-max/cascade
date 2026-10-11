import json

import pytest

from cascade import store
from cascade.cli import SYNTHETIC_RUN_ID
from cascade.config import HAIKU, OPUS
from cascade.grading import grade_rows
from cascade.plan import Arm, arm_from_key, build_arms, build_plan
from cascade.simulate import (
    ABILITY,
    annotate,
    pass_probability,
    simulate_rows,
    simulated_judge,
)
from cascade.tasks import load_tasks, task_index

TASKS = load_tasks()
INDEX = task_index(TASKS)
QUIET = lambda _message: None  # noqa: E731


def simulated(seed: int, tasks=TASKS, **plan_kwargs) -> list[dict]:
    plan = build_plan(tasks, build_arms(), **plan_kwargs)
    rows = simulate_rows(plan, tasks, run_id="s", seed=seed)
    rows = grade_rows(rows, tasks, judge_many=simulated_judge(seed), log=QUIET)
    annotate(rows, plan, tasks, seed)
    return rows


def test_same_seed_same_rows_and_a_different_seed_differs() -> None:
    subset = [task for task in TASKS if task.category in ("classification", "open_ended")]
    first, second = simulated(11, subset, trials=2), simulated(11, subset, trials=2)
    assert first == second
    other = simulated(12, subset, trials=2)
    assert [row["passed"] for row in first] != [row["passed"] for row in other]


def test_rows_use_the_real_schema_and_are_flagged_synthetic(tmp_path) -> None:
    rows = simulated(3, TASKS[:3], trials=1)
    assert all(set(row) == set(store.COLUMNS) for row in rows)
    assert all(row["synthetic"] is True for row in rows)
    path = tmp_path / "r.parquet"
    store.write_rows(path, rows)
    back = store.read_rows(path)
    assert sorted(back, key=lambda r: r["custom_id"]) == sorted(rows, key=lambda r: r["custom_id"])


def test_grades_come_from_the_real_graders_on_the_stand_in_answers() -> None:
    rows = simulated(5, [INDEX["code-roman"], INDEX["ext-receipt-totals"]], trials=3)
    for row in rows:
        details = json.loads(row["grader_details"])
        stand_in = details["simulated"]["answer"]
        task = INDEX[row["task_id"]]
        assert row["answer"] == (task.reference if stand_in == "reference" else task.wrong)
        assert row["passed"] is (stand_in == "reference")
        if row["task_id"] == "code-roman":
            assert details["total"] == 10  # the hidden tests really ran
            assert details["passed"] == (10 if row["passed"] else 1)
        else:
            assert details["schema_valid"] is True
            assert row["score"] == (1.0 if row["passed"] else 0.5)


def test_pass_probability_orders_arms_and_difficulty_sensibly() -> None:
    easy, medium, hard = INDEX["rsn-crates"], INDEX["rsn-pen-notebook"], INDEX["rsn-digit-sum"]
    for seed in (1, 2, 3):
        assert pass_probability(easy, Arm(OPUS, "high"), seed) > pass_probability(
            hard, Arm(HAIKU, "low"), seed
        )
        for model in ABILITY:
            for task in (easy, medium, hard):
                low, high = (pass_probability(task, Arm(model, e), seed) for e in ("low", "high"))
                assert high >= low  # more effort never hurts in the simulator
        low, high = (pass_probability(medium, Arm(HAIKU, e), seed) for e in ("low", "high"))
        assert high > low
    assert all(0.02 <= pass_probability(task, Arm(OPUS, "low"), 9) <= 0.995 for task in TASKS)


def test_batch_rows_have_no_latency_and_cost_half() -> None:
    rows = simulated(7, TASKS[:8], trials=2, mode="batch")
    sync = [row for row in rows if row["mode"] == "sync"]
    batch = [row for row in rows if row["mode"] == "batch"]
    assert sync and batch
    assert all(row["latency_ms"] is None for row in batch)
    assert all(row["latency_ms"] > 0 for row in sync)
    assert all(row["billed_usd"] == pytest.approx(row["cost_usd"] / 2) for row in batch)
    assert all(row["billed_usd"] == pytest.approx(row["cost_usd"]) for row in sync)


def test_committed_synthetic_run_is_exactly_what_the_simulator_produces() -> None:
    """Guards against hand-edited placeholder data: it must be regenerable from the seed."""
    try:
        meta = store.read_meta(SYNTHETIC_RUN_ID)
    except FileNotFoundError:
        pytest.skip("no committed synthetic run")
    committed = store.read_rows(store.results_path(SYNTHETIC_RUN_ID))
    arms = [arm_from_key(key) for key in meta["arms"]]
    plan = build_plan(TASKS, arms, trials=meta["trials"])
    seed = meta["generator"]["seed"]
    rows = simulate_rows(plan, TASKS, run_id=SYNTHETIC_RUN_ID, seed=seed)
    rows = grade_rows(rows, TASKS, judge_many=simulated_judge(seed), log=QUIET)
    annotate(rows, plan, TASKS, seed)
    assert meta["synthetic"] is True and all(row["synthetic"] for row in committed)
    key = lambda r: (r["task_id"], r["model"], r["effort"], r["trial"])  # noqa: E731
    assert sorted(rows, key=key) == sorted(committed, key=key)
