import pytest

from cascade.cli import CONFIRM_ABOVE_USD, DEFAULT_BUDGET_USD
from cascade.config import HAIKU, OPUS, SONNET
from cascade.estimate import (
    approx_input_tokens,
    assumed_output_tokens,
    estimate_plan,
    format_estimate,
)
from cascade.plan import Arm, arm_from_key, build_arms, build_plan, latency_sample
from cascade.tasks import load_tasks

TASKS = load_tasks()


def test_arms_are_ordered_cheapest_tier_and_lowest_effort_first() -> None:
    keys = [arm.key for arm in build_arms()]
    assert keys == [
        "haiku-low", "haiku-medium", "haiku-high",
        "sonnet-low", "sonnet-medium", "sonnet-high",
        "opus-low", "opus-medium", "opus-high",
    ]  # fmt: skip
    assert arm_from_key("sonnet-xhigh") == Arm(SONNET, "xhigh")
    assert [arm.key for arm in build_arms((OPUS,), ("max", "low"))] == ["opus-low", "opus-max"]
    with pytest.raises(ValueError):
        arm_from_key("gpt-low")


def test_plan_has_one_request_per_task_arm_and_trial() -> None:
    arms = build_arms()
    plan = build_plan(TASKS, arms, trials=3, mode="batch")
    assert len(plan) == len(TASKS) * 9 * 3
    assert len({request.custom_id for request in plan}) == len(plan)


def test_batch_mode_still_times_a_stratified_sample_synchronously() -> None:
    arms = build_arms()
    plan = build_plan(TASKS, arms, trials=3, mode="batch", latency_sample_per_category=2)
    timed = latency_sample(TASKS, 2)
    assert len(timed) == 10  # two per category
    for request in plan:
        expected = "sync" if request.task_id in timed and request.trial == 0 else "batch"
        assert request.mode == expected
    # Every arm times the same tasks, so latency is comparable across arms.
    sync = [request for request in plan if request.mode == "sync"]
    assert len(sync) == 10 * 9


def test_auto_mode_uses_sync_for_a_small_run_and_batch_for_a_large_one() -> None:
    small = build_plan(TASKS[:5], build_arms((HAIKU,), ("low",)), trials=1, mode="auto")
    assert {request.mode for request in small} == {"sync"}
    large = build_plan(TASKS, build_arms(), trials=1, mode="auto")
    assert {request.mode for request in large} == {"batch", "sync"}


def test_max_tokens_leave_room_for_thinking_and_grow_with_effort() -> None:
    plan = build_plan(TASKS[:1], build_arms((OPUS,), ("low", "medium", "high", "max")), trials=1)
    limits = [request.max_tokens for request in plan]
    assert limits == sorted(limits) and limits[0] >= 4000


def test_estimate_arithmetic_for_one_arm() -> None:
    task = next(task for task in TASKS if task.id == "rsn-crates")
    arm = Arm(OPUS, "high")
    plan = build_plan([task], [arm], trials=2, mode="batch", latency_sample_per_category=0)
    estimate = estimate_plan(plan, [task])
    tokens_in = approx_input_tokens(task)
    tokens_out = assumed_output_tokens("reasoning", arm)
    assert tokens_out == round(900 * 2.2 * 1.15)
    entry = estimate.arms[0]
    assert (entry.requests, entry.batch_requests) == (2, 2)
    list_usd = 2 * (tokens_in * 4 + tokens_out * 20) / 1_000_000
    assert entry.list_usd == pytest.approx(list_usd)
    assert entry.billed_usd == pytest.approx(list_usd / 2)  # batch is half price
    assert estimate.judge_requests == 0 and estimate.total_usd == pytest.approx(list_usd / 2)


def test_sync_rows_are_estimated_at_full_price_and_extras_are_added() -> None:
    task = next(task for task in TASKS if task.id == "open-ab-paradox")
    plan = build_plan([task], [Arm(SONNET, "low")], trials=1, mode="sync")
    estimate = estimate_plan(plan, [task], extras={"caching exp.": 0.25})
    assert estimate.arms[0].billed_usd == pytest.approx(estimate.arms[0].list_usd)
    assert estimate.judge_requests == 1 and estimate.judge_usd > 0
    assert estimate.total_usd == pytest.approx(estimate.answers_usd + estimate.judge_usd + 0.25)
    assert "TOTAL" in format_estimate(estimate)


def test_the_default_run_needs_confirmation_but_fits_the_default_budget() -> None:
    """The defaults are meant to work out of the box: ask first, then run."""
    plan = build_plan(TASKS, build_arms())
    total = estimate_plan(plan, TASKS, extras={"caching exp.": 0.2}).total_usd
    assert CONFIRM_ABOVE_USD < total < DEFAULT_BUDGET_USD
