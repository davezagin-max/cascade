import re
from collections import Counter

import pytest

from cascade.plan import build_arms, custom_id
from cascade.tasks import CATEGORIES, SuiteError, _parse_task, load_tasks
from cascade.validate import validate_suite, validate_task

TASKS = load_tasks()


def test_suite_shape() -> None:
    counts = Counter(task.category for task in TASKS)
    assert 55 <= len(TASKS) <= 70
    assert set(counts) == set(CATEGORIES)
    assert all(counts[category] >= 6 for category in CATEGORIES)
    assert len({task.id for task in TASKS}) == len(TASKS)
    # A spread of difficulty within every deterministic category, so tiers can separate.
    for category in ("classification", "extraction", "reasoning", "code"):
        levels = {task.difficulty for task in TASKS if task.category == category}
        assert min(levels) == 1 and max(levels) == 5


def test_every_reference_passes_and_every_known_wrong_answer_fails() -> None:
    """The graders are tested against the real suite, hidden tests and sandbox included."""
    assert validate_suite(TASKS) == []


def test_every_task_id_fits_a_batch_custom_id() -> None:
    pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    arms = build_arms(efforts=("low", "medium", "high", "xhigh", "max"))
    for task in TASKS:
        for arm in arms:
            assert pattern.match(custom_id(task.id, arm, 99))
            assert pattern.match(f"rj__{custom_id(task.id, arm, 99)}")  # judge retry id


def test_validation_catches_a_wrong_key() -> None:
    task = next(task for task in TASKS if task.id == "rsn-crates")
    broken = _parse_task(
        {
            "id": "rsn-broken",
            "title": task.title,
            "difficulty": 1,
            "prompt": task.prompt,
            "wrong": "ANSWER: 408",  # the "wrong" answer is actually right
            "grader": {"type": "exact", "expected": "408"},
        },
        "reasoning",
    )
    assert [problem.message for problem in validate_task(broken)] == ["known-wrong answer passed"]


def test_malformed_tasks_are_rejected() -> None:
    base = {"title": "t", "difficulty": 1, "prompt": "p", "wrong": "ANSWER: x"}
    with pytest.raises(SuiteError, match="Invalid task id"):
        _parse_task(
            {**base, "id": "Bad ID", "grader": {"type": "exact", "expected": "y"}}, "reasoning"
        )
    with pytest.raises(SuiteError, match="difficulty"):
        _parse_task(
            {**base, "id": "ok-id", "difficulty": 9, "grader": {"type": "exact", "expected": "y"}},
            "reasoning",
        )
    with pytest.raises(SuiteError, match="unknown grader"):
        _parse_task({**base, "id": "ok-id", "grader": {"type": "vibes"}}, "reasoning")
