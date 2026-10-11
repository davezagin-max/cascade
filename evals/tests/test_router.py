import json
import math
import re
from pathlib import Path

import pytest

from cascade.export import REPO_ROOT
from cascade.features import ascii_lower, extract_features, load_spec, normalize
from cascade.router import (
    NONE,
    TaskStats,
    arms_by_cost,
    baseline_arm,
    choose_arm,
    cross_validate,
    fit,
    fold_of,
    logit,
    oracle_arm,
    route,
)

SPEC = load_spec()
NAMES = [feature["name"] for feature in SPEC]
DATA = Path(REPO_ROOT) / "data"


def feature(prompt: str, name: str) -> float:
    return extract_features(SPEC, prompt)[NAMES.index(name)]


def test_features_are_exact_small_fractions() -> None:
    prompt = "Classify this review as one of: good, bad."
    assert feature(prompt, "length") == len(prompt) / (len(prompt) + 400)
    assert feature(prompt, "words") == 8 / (8 + 80)
    assert feature(prompt, "label_cues") == 2 / (2 + 2)  # "classif", "one of"
    assert feature(prompt, "questions") == 0.0
    assert feature("What is 7^100? And 12,345 + 1?", "questions") == 2 / 4
    assert feature("What is 7^100? And 12,345 + 1?", "numbers") == 4 / 12
    assert feature("ab1", "digit_density") == 1 / 4


def test_text_handling_is_defined_to_match_javascript() -> None:
    assert normalize("a\r\nb\rc\nd") == "a\nb\nc\nd"
    assert ascii_lower("ABC É K") == "abc É K"  # only A-Z change
    # Length counts code points, so an astral emoji is one character.
    assert feature("\U0001f600", "length") == 1 / 401
    # Kelvin sign is not "k": an ASCII-only match, as in a JS regex without the u flag.
    assert feature("Keys", "structure_cues") == 0.0
    assert feature("keys", "structure_cues") == 1 / 4
    lines = "intro\n- one\n  * two\n 3. three\n[4] four\nnot - a bullet"
    assert feature(lines, "list_items") == 4 / 8


def test_spec_patterns_stay_inside_the_portable_regex_subset() -> None:
    for item in SPEC:
        patterns = item.get("patterns", []) + ([item["pattern"]] if "pattern" in item else [])
        for pattern in patterns:
            assert pattern.isascii(), pattern
            assert "[^" not in pattern, pattern  # negated classes differ on astral characters
            assert not re.search(r"(?<!\\)\.", pattern.replace("[.", "[")), pattern  # no bare "."
            assert "^" not in pattern.replace("\\^", "").replace("^]", "") or "+*^" in pattern, (
                pattern
            )
            assert "$" not in pattern, pattern
            assert re.compile(pattern, re.ASCII).search("") is None, f"{pattern} matches empty"
    assert len(set(NAMES)) == len(NAMES)


def test_python_router_reproduces_the_exported_vectors_exactly() -> None:
    """The same file is checked by the TypeScript test suite: exact equality on both sides."""
    model = json.loads((DATA / "router.json").read_text(encoding="utf-8"))
    vectors = json.loads((DATA / "router-vectors.json").read_text(encoding="utf-8"))
    assert len(vectors) >= 70
    for vector in vectors:
        decision = route(model, vector["prompt"])
        assert decision["features"] == vector["features"]
        assert decision["contributions"] == vector["contributions"]
        assert decision["score"] == vector["score"]
        assert decision["arm"] == vector["arm"]
    assert len({vector["arm"] for vector in vectors}) >= 3  # the vectors cover several outcomes


ARMS = [{"key": "cheap", "bias": -1.0}, {"key": "mid", "bias": 0.5}, {"key": "best", "bias": 2.0}]


def test_choose_arm_takes_the_cheapest_arm_that_clears_the_threshold() -> None:
    t = logit(0.8)  # about 1.386
    assert choose_arm(ARMS, 3.0, t)["key"] == "cheap"
    assert choose_arm(ARMS, 1.0, t)["key"] == "mid"
    assert choose_arm(ARMS, 0.0, t)["key"] == "best"
    # Meeting the threshold exactly is enough.
    assert choose_arm([{"key": "zero", "bias": 0.0}, ARMS[2]], t, t)["key"] == "zero"
    # Nothing clears it: fall back to the most able arm, not the most expensive by position.
    shuffled = [ARMS[2], ARMS[0], ARMS[1]]
    assert choose_arm(shuffled, -9.0, t)["key"] == "best"


def synthetic_stats(n: int = 60) -> list[TaskStats]:
    """Tasks whose only real signal is feature 0: larger means harder."""
    stats = []
    for i in range(n):
        hardness = i / (n - 1)
        noise = ((i * 37) % 11) / 11
        pass_rate = {
            "haiku-low": 1.0 if hardness < 0.35 else 0.0,
            "sonnet-low": 1.0 if hardness < 0.7 else 0.0,
            "opus-high": 1.0 if hardness < 0.95 else 0.0,
        }
        stats.append(
            TaskStats(
                task_id=f"t{i:02d}",
                category=("classification", "reasoning", "code")[i % 3],
                features=[hardness, noise],
                pass_rate=pass_rate,
                cost={"haiku-low": 0.001, "sonnet-low": 0.01, "opus-high": 0.1},
            )
        )
    return stats


def test_fit_learns_direction_and_arm_ordering() -> None:
    stats = synthetic_stats()
    arms = arms_by_cost(stats)
    assert arms == ["haiku-low", "sonnet-low", "opus-high"]
    assert baseline_arm(arms) == "opus-high"
    fitted = fit(stats, arms, l2=0.1)
    assert fitted.weights[0] < -1.0  # harder prompts lower the pass logit
    assert abs(fitted.weights[1]) < abs(fitted.weights[0]) / 4  # the noise feature is ignored
    assert fitted.bias["haiku-low"] < fitted.bias["sonnet-low"] < fitted.bias["opus-high"]


def test_cross_validation_routes_unseen_tasks_and_reports_the_headline() -> None:
    stats = synthetic_stats()
    spec = [{"name": "hardness"}, {"name": "noise"}]
    result = cross_validate(stats, spec, threshold=0.8, l2=0.1, folds=5)
    assert result["n_tasks"] == 60 and result["baseline_arm"] == "opus-high"
    router, oracle = result["router"], result["oracle"]
    # With a clean signal the router should save most of the cost and keep most of the quality.
    assert router["cost_saved"] > 0.5
    assert router["quality_retained"] > 0.9
    assert oracle["cost_saved"] >= router["cost_saved"] - 1e-9
    assert oracle["quality_retained"] >= 1.0
    confusion = result["confusion"]
    assert sum(map(sum, confusion["matrix"])) == 60
    assert confusion["exact"] + confusion["over_routed"] + confusion["under_routed"] == 60
    assert confusion["exact"] > 35
    saved = [point["cost_saved"] for point in result["sweep"]]
    assert saved == sorted(saved, reverse=True)  # a stricter threshold never saves more
    assert 0 < result["calibration"]["predicted"] <= 1


def test_folds_are_deterministic_stratified_and_complete() -> None:
    stats = synthetic_stats(30)
    folds = fold_of(stats, 5)
    assert folds == fold_of(list(reversed(stats)), 5)
    assert set(folds.values()) == {0, 1, 2, 3, 4}
    for fold in range(5):
        categories = {task.category for task in stats if folds[task.task_id] == fold}
        assert categories == {"classification", "reasoning", "code"}


def test_oracle_is_the_cheapest_arm_matching_the_best_result() -> None:
    arms = ["a", "b", "c"]

    def stats(rates: dict[str, float]) -> TaskStats:
        return TaskStats("t", "code", [0.0], rates, dict.fromkeys(rates, 1.0))

    assert oracle_arm(stats({"a": 1.0, "b": 1.0, "c": 1.0}), arms) == "a"
    assert oracle_arm(stats({"a": 0.33, "b": 1.0, "c": 1.0}), arms) == "b"
    assert oracle_arm(stats({"a": 0.0, "b": 0.67, "c": 0.67}), arms) == "b"
    assert oracle_arm(stats({"a": 0.0, "b": 0.0, "c": 0.0}), arms) == NONE


def test_exported_model_is_self_describing_and_consistent() -> None:
    model = json.loads((DATA / "router.json").read_text(encoding="utf-8"))
    assert (
        len(model["weights"]) == len(model["features"]) == len(model["mean"]) == len(model["scale"])
    )
    assert model["threshold_logit"] == pytest.approx(
        math.log(model["threshold"] / (1 - model["threshold"])), abs=1e-6
    )
    costs = [arm["cost_per_request"] for arm in model["arms"]]
    assert costs == sorted(costs)  # cheapest first, by measured cost
    assert model["baseline_arm"] in {arm["key"] for arm in model["arms"]}
    assert all(scale > 0 for scale in model["scale"])
