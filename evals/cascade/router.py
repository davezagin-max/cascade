"""The router: predict the cheapest arm that will get a prompt right.

Model. One logistic regression over every graded row:

    logit P(pass | prompt, arm) = bias[arm] + w . z(prompt)

`z` is the standardised feature vector of the prompt (see `features.py`), `w` is
shared by all arms, and each arm has its own bias. Read it as: every arm has an
ability, every prompt has a predicted easiness, and an arm passes when its ability
outweighs the prompt's difficulty. Sharing `w` is deliberate. The suite has about
60 prompts, so a separate weight vector per arm (9 x 18 weights from 60 examples
each) would fit noise; one shared vector is trained on every row.

Routing. Arms are ordered by measured mean cost. The router picks the first arm
whose predicted pass probability reaches the threshold, and falls back to the
most able arm when none does. The comparison is done on logits against a
precomputed `threshold_logit`, so serving needs no `exp` and the TypeScript port
makes bit-identical decisions.

Regularisation. `w` carries an L2 penalty whose strength is not a guess: it is
picked from a small grid by grouped cross-validation on held-out log loss. When
the features say little about difficulty the selection picks a strong penalty,
`w` shrinks toward zero, and the router degrades into "always use the one arm that
clears the threshold" instead of routing on noise.

Evaluation. Grouped k-fold cross-validation over tasks: every task is routed by a
model that never saw any row of that task, the features are standardised with
training-fold statistics only, and the penalty is selected inside each training
fold (nested cross-validation), so nothing about a held-out task leaks into the
model that routes it. Two comparisons are reported: the headline against sending
everything to the baseline arm (Opus at high effort), and, because that baseline
is easy to beat, the harder one against the best single arm that costs no more
than the router does.

Training uses numpy. Inference (`route`) is plain Python on purpose: it is the
reference the TypeScript implementation is tested against.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import OPUS
from .features import FeatureSpec, extract_features, load_spec
from .plan import Arm, arm_from_key
from .store import Row, round_floats
from .tasks import CATEGORIES, Task

DEFAULT_THRESHOLD = 0.9
DEFAULT_FOLDS = 5
# Candidate penalties, weakest first. Spaced by about 3x: finer steps would be fitting noise.
L2_GRID: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
INNER_FOLDS = 4
# A penalty this strong leaves only the per-arm abilities: the "no prompt features" model.
ARMS_ONLY_L2 = 1e9
SWEEP = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
BIAS_L2 = 1e-3  # keeps an arm that passed everything from running off to infinity
NONE = "none"


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def arm_key(row: Row) -> str:
    return Arm(row["model"], row["effort"]).key


# --- data ---------------------------------------------------------------------------------


@dataclass
class TaskStats:
    """What the eval measured for one task: pass rate and mean cost per arm."""

    task_id: str
    category: str
    features: list[float]
    pass_rate: dict[str, float]
    cost: dict[str, float]


def collect(rows: list[Row], tasks: list[Task], spec: list[FeatureSpec]) -> list[TaskStats]:
    graded = [row for row in rows if row["passed"] is not None]
    passes: dict[tuple[str, str], list[float]] = defaultdict(list)
    costs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in graded:
        key = (row["task_id"], arm_key(row))
        passes[key].append(1.0 if row["passed"] else 0.0)
        costs[key].append(row["cost_usd"])
    arms = sorted({arm for _, arm in passes})
    stats: list[TaskStats] = []
    for task in tasks:
        if not any((task.id, arm) in passes for arm in arms):
            continue
        stats.append(
            TaskStats(
                task_id=task.id,
                category=task.category,
                features=extract_features(spec, task.prompt),
                pass_rate={
                    arm: sum(passes[task.id, arm]) / len(passes[task.id, arm])
                    for arm in arms
                    if (task.id, arm) in passes
                },
                cost={
                    arm: sum(costs[task.id, arm]) / len(costs[task.id, arm])
                    for arm in arms
                    if (task.id, arm) in costs
                },
            )
        )
    return stats


def arms_by_cost(stats: list[TaskStats]) -> list[str]:
    """Arm keys, cheapest first, by mean measured cost per request."""
    totals: dict[str, list[float]] = defaultdict(list)
    for task in stats:
        for arm, cost in task.cost.items():
            totals[arm].append(cost)
    return sorted(totals, key=lambda arm: (sum(totals[arm]) / len(totals[arm]), arm))


def baseline_arm(arms: list[str]) -> str:
    """Opus at high effort when the matrix has it, else the most expensive arm."""
    preferred = Arm(OPUS, "high").key
    return preferred if preferred in arms else arms[-1]


def fold_of(stats: list[TaskStats], folds: int) -> dict[str, int]:
    """Deterministic folds, stratified by category so every fold sees every task type."""
    assignment: dict[str, int] = {}
    for category in CATEGORIES:
        members = sorted(
            (task.task_id for task in stats if task.category == category),
            key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest(),
        )
        for position, task_id in enumerate(members):
            assignment[task_id] = position % folds
    return assignment


# --- training -----------------------------------------------------------------------------


@dataclass
class Fit:
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: dict[str, float]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35, 35)))


def fit(stats: list[TaskStats], arms: list[str], *, l2: float) -> Fit:
    """Regularised logistic regression by Newton's method (IRLS).

    Each (task, arm) cell is one observation whose target is the cell's pass rate.
    With the same number of trials in every cell that is the per-trial likelihood
    divided by the number of trials, so `l2` is a penalty per cell, not per row.
    """
    features = np.array([task.features for task in stats], dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (features - mean) / scale
    n_features = z.shape[1]

    design: list[np.ndarray] = []
    target: list[float] = []
    for row, task in zip(z, stats, strict=True):
        for position, arm in enumerate(arms):
            if arm not in task.pass_rate:
                continue
            onehot = np.zeros(len(arms))
            onehot[position] = 1.0
            design.append(np.concatenate([row, onehot]))
            target.append(task.pass_rate[arm])
    x = np.array(design)
    y = np.array(target)
    n = len(y)

    penalty = np.concatenate([np.full(n_features, l2), np.full(len(arms), BIAS_L2)])
    beta = np.zeros(x.shape[1])

    def loss(candidate: np.ndarray) -> float:
        p = np.clip(_sigmoid(x @ candidate), 1e-12, 1 - 1e-12)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
        return float(nll + 0.5 * (penalty * candidate**2).sum())

    current = loss(beta)
    for _ in range(60):
        p = _sigmoid(x @ beta)
        gradient = x.T @ (p - y) + penalty * beta
        hessian = (x * (p * (1 - p))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        size = 1.0
        accepted = None
        while size > 1e-4:  # backtrack if a full Newton step overshoots
            candidate = beta - size * step
            value = loss(candidate)
            if value <= current:
                accepted = (candidate, value)
                break
            size /= 2
        if accepted is None:
            break
        improved = current - accepted[1]
        beta, current = accepted
        if improved < 1e-10 * max(1.0, n):
            break

    return Fit(
        mean=[float(v) for v in mean],
        scale=[float(v) for v in scale],
        weights=[float(v) for v in beta[:n_features]],
        bias={arm: float(beta[n_features + i]) for i, arm in enumerate(arms)},
    )


def _score(fitted: Fit, features: list[float]) -> float:
    """The prompt's predicted easiness under a fit: the weighted sum of its z-scores."""
    total = 0.0
    for weight, value, mean, scale in zip(
        fitted.weights, features, fitted.mean, fitted.scale, strict=True
    ):
        total += weight * ((value - mean) / scale)  # same order of operations as `score_prompt`
    return total


def _cell_loss(fitted: Fit, task: TaskStats, arms: list[str]) -> tuple[float, int]:
    """Summed log loss over one task's cells, and how many cells that was."""
    score = _score(fitted, task.features)
    total = 0.0
    cells = 0
    for arm in arms:
        if arm not in task.pass_rate:
            continue
        p = 1 / (1 + math.exp(-max(-35.0, min(35.0, fitted.bias[arm] + score))))
        p = min(max(p, 1e-12), 1 - 1e-12)
        rate = task.pass_rate[arm]
        total -= rate * math.log(p) + (1 - rate) * math.log(1 - p)
        cells += 1
    return total, cells


def held_out_loss(stats: list[TaskStats], *, l2: float, folds: int = INNER_FOLDS) -> float:
    """Mean log loss per cell on tasks the fit never saw (grouped k-fold over tasks)."""
    assignment = fold_of(stats, folds)
    total = 0.0
    cells = 0
    for fold in range(folds):
        train = [task for task in stats if assignment[task.task_id] != fold]
        held_out = [task for task in stats if assignment[task.task_id] == fold]
        if not train or not held_out:
            continue
        arms = arms_by_cost(train)
        fitted = fit(train, arms, l2=l2)
        for task in held_out:
            loss, n = _cell_loss(fitted, task, arms)
            total += loss
            cells += n
    return total / cells if cells else math.inf


def select_l2(
    stats: list[TaskStats], *, grid: tuple[float, ...] = L2_GRID, folds: int = INNER_FOLDS
) -> tuple[float, list[dict[str, float]]]:
    """The penalty with the lowest held-out log loss, and the loss at every grid value.

    A tie goes to the stronger penalty: the simpler model.
    """
    curve = [{"l2": value, "loss": held_out_loss(stats, l2=value, folds=folds)} for value in grid]
    best = min(point["loss"] for point in curve)
    chosen = max(point["l2"] for point in curve if point["loss"] <= best + 1e-12)
    return chosen, curve


# --- inference (plain Python: the reference for the TypeScript port) ------------------------


def score_prompt(model: dict[str, Any], prompt: str) -> tuple[list[float], list[float], float]:
    """Feature values, each feature's contribution to the pass logit, and their sum."""
    values = extract_features(model["features"], prompt)
    contributions: list[float] = []
    total = 0.0
    for i, value in enumerate(values):
        z = (value - model["mean"][i]) / model["scale"][i]
        contribution = model["weights"][i] * z
        contributions.append(contribution)
        total += contribution
    return values, contributions, total


def choose_arm(arms: list[dict[str, Any]], score: float, threshold_logit: float) -> dict[str, Any]:
    """The cheapest arm that clears the threshold, else the arm with the highest ability."""
    for arm in arms:  # cheapest first
        if arm["bias"] + score >= threshold_logit:
            return arm
    best = arms[0]
    for arm in arms[1:]:
        if arm["bias"] > best["bias"]:
            best = arm
    return best


def route(model: dict[str, Any], prompt: str) -> dict[str, Any]:
    values, contributions, score = score_prompt(model, prompt)
    chosen = choose_arm(model["arms"], score, model["threshold_logit"])
    return {
        "arm": chosen["key"],
        "score": score,
        "features": values,
        "contributions": contributions,
        "logits": [arm["bias"] + score for arm in model["arms"]],
    }


# --- evaluation ---------------------------------------------------------------------------


def oracle_arm(task: TaskStats, arms: list[str]) -> str:
    """With hindsight: the cheapest arm that did as well on this task as any arm did.

    Returns `none` when every arm failed every trial. No router can match this
    oracle (it sees the noise in individual trials), so it is an upper bound.
    """
    best = max(task.pass_rate.get(arm, 0.0) for arm in arms)
    if best <= 0.0:
        return NONE
    for arm in arms:  # cheapest first
        if task.pass_rate.get(arm, 0.0) >= best:
            return arm
    return NONE


def _mix_outcome(stats: list[TaskStats], chosen: dict[str, str]) -> tuple[float, float]:
    """Mean pass rate and mean cost per request when each task goes to its chosen arm."""
    quality = sum(task.pass_rate[chosen[task.task_id]] for task in stats) / len(stats)
    cost = sum(task.cost[chosen[task.task_id]] for task in stats) / len(stats)
    return quality, cost


def best_fixed_arm(
    fixed: dict[str, tuple[float, float]], arms: list[str], budget: float
) -> dict[str, Any] | None:
    """The single arm with the highest pass rate among those costing no more than `budget`.

    `fixed` maps each arm to its (pass rate, cost per request) on the same tasks. This
    is the baseline a router has to beat to be worth having: "always Opus" is easy to
    undercut, "always the best arm you can afford" is not. None when nothing is affordable.
    """
    affordable = [arm for arm in arms if fixed[arm][1] <= budget * (1 + 1e-9)]
    if not affordable:
        return None
    # `arms` is cheapest first, so on equal quality `max` keeps the cheaper arm.
    best = max(affordable, key=lambda arm: (fixed[arm][0], -arms.index(arm)))
    return {"arm": best, "quality": fixed[best][0], "cost_per_request": fixed[best][1]}


def cross_validate(
    stats: list[TaskStats],
    spec: list[FeatureSpec],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    l2: float | None = None,
    folds: int = DEFAULT_FOLDS,
) -> dict[str, Any]:
    """Out-of-fold routing for every task, at the default threshold and across the sweep.

    With `l2=None` the penalty is selected inside each training fold, so the reported
    numbers include the cost of having to choose it from data.
    """
    assignment = fold_of(stats, folds)
    all_arms = arms_by_cost(stats)
    baseline = baseline_arm(all_arms)
    thresholds = sorted({*SWEEP, threshold})
    picks: dict[float, dict[str, str]] = {t: {} for t in thresholds}
    probability: dict[str, float] = {}
    fold_l2: list[float] = []
    loss = {"router": 0.0, "arms_only": 0.0}
    cells = 0

    for fold in range(folds):
        train = [task for task in stats if assignment[task.task_id] != fold]
        held_out = [task for task in stats if assignment[task.task_id] == fold]
        if not train or not held_out:
            continue
        train_arms = arms_by_cost(train)  # cost order as seen by this fold only
        penalty = l2 if l2 is not None else select_l2(train)[0]
        fold_l2.append(penalty)
        fitted = fit(train, train_arms, l2=penalty)
        arms_only = fit(train, train_arms, l2=ARMS_ONLY_L2)
        for task in held_out:
            score = _score(fitted, task.features)
            ranked = [{"key": arm, "bias": fitted.bias[arm]} for arm in train_arms]
            for t in thresholds:
                picks[t][task.task_id] = choose_arm(ranked, score, logit(t))["key"]
            chosen_bias = fitted.bias[picks[threshold][task.task_id]]
            probability[task.task_id] = 1 / (1 + math.exp(-(chosen_bias + score)))
            task_loss, n = _cell_loss(fitted, task, train_arms)
            loss["router"] += task_loss
            loss["arms_only"] += _cell_loss(arms_only, task, train_arms)[0]
            cells += n

    routed = [task for task in stats if task.task_id in picks[threshold]]
    base_quality, base_cost = _mix_outcome(routed, {t.task_id: baseline for t in routed})
    fixed = {arm: _mix_outcome(routed, {t.task_id: arm for t in routed}) for arm in all_arms}

    def summarize(chosen: dict[str, str]) -> dict[str, Any]:
        quality, cost = _mix_outcome(routed, chosen)
        mix: dict[str, int] = defaultdict(int)
        for arm in chosen.values():
            mix[arm] += 1
        return {
            "quality": quality,
            "cost_per_request": cost,
            "cost_saved": 1 - cost / base_cost if base_cost else 0.0,
            "quality_retained": quality / base_quality if base_quality else 0.0,
            "mix": {arm: mix[arm] for arm in all_arms if mix[arm]},
            "best_fixed": best_fixed_arm(fixed, all_arms, cost),
        }

    oracle = {task.task_id: oracle_arm(task, all_arms) for task in routed}
    # When nothing solves a task, the cheapest arm is the best that can be done.
    oracle_choice = {tid: (all_arms[0] if arm == NONE else arm) for tid, arm in oracle.items()}

    labels = [*all_arms, NONE]
    matrix = [[0] * len(all_arms) for _ in labels]
    exact = over = under = 0
    for task in routed:
        predicted = picks[threshold][task.task_id]
        truth = oracle[task.task_id]
        matrix[labels.index(truth)][all_arms.index(predicted)] += 1
        truth_rank = 0 if truth == NONE else all_arms.index(truth)
        rank = all_arms.index(predicted)
        if rank == truth_rank:
            exact += 1
        elif rank > truth_rank:
            over += 1
        else:
            under += 1

    by_category: dict[str, Any] = {}
    for category in CATEGORIES:
        members = [task for task in routed if task.category == category]
        if not members:
            continue
        chosen = {task.task_id: picks[threshold][task.task_id] for task in members}
        quality, cost = _mix_outcome(members, chosen)
        b_quality, b_cost = _mix_outcome(members, {t.task_id: baseline for t in members})
        mix: dict[str, int] = defaultdict(int)
        for arm in chosen.values():
            mix[arm] += 1
        by_category[category] = {
            "n_tasks": len(members),
            "quality": quality,
            "cost_per_request": cost,
            "baseline_quality": b_quality,
            "baseline_cost_per_request": b_cost,
            "mix": {arm: mix[arm] for arm in all_arms if mix[arm]},
        }

    return {
        "folds": folds,
        "n_tasks": len(routed),
        "threshold": threshold,
        "baseline_arm": baseline,
        "baseline": {"quality": base_quality, "cost_per_request": base_cost},
        "router": summarize(picks[threshold]),
        "oracle": summarize(oracle_choice),
        "sweep": [{"threshold": t, **summarize(picks[t])} for t in thresholds],
        "confusion": {
            "predicted": all_arms,
            "oracle": labels,
            "matrix": matrix,
            "exact": exact,
            "over_routed": over,
            "under_routed": under,
        },
        "by_category": by_category,
        # Is the router's confidence honest? Mean predicted pass probability of the arms
        # it chose, next to how often those arms actually passed.
        "calibration": {
            "predicted": sum(probability.values()) / len(probability),
            "realized": _mix_outcome(routed, picks[threshold])[0],
        },
        # Do the prompt features help at all? Held-out log loss per cell with them, and
        # with the per-arm abilities alone. Lower is better; equal means no signal.
        "log_loss": {
            "router": loss["router"] / cells if cells else 0.0,
            "arms_only": loss["arms_only"] / cells if cells else 0.0,
        },
        "l2_by_fold": fold_l2,
        "picks": {task.task_id: picks[threshold][task.task_id] for task in routed},
        "oracle_arms": oracle,
        "predicted_pass": probability,
    }


# --- export -------------------------------------------------------------------------------


def _round(values: list[float], digits: int = 6) -> list[float]:
    return [round(value, digits) for value in values]


def train_router(
    rows: list[Row],
    tasks: list[Task],
    *,
    run_id: str,
    synthetic: bool,
    threshold: float = DEFAULT_THRESHOLD,
    l2: float | None = None,
    folds: int = DEFAULT_FOLDS,
) -> dict[str, Any]:
    """Cross-validate, then fit on everything and return the self-describing model JSON.

    `l2=None` (the default) selects the penalty by cross-validation; a number fixes it.
    """
    if not 0.5 <= threshold < 1:
        raise ValueError("threshold must be in [0.5, 1)")
    spec = load_spec()
    collected = collect(rows, tasks, spec)
    arms = arms_by_cost(collected)
    # A task with no graded answer from some arm (every trial of that cell failed at the
    # API) cannot be compared across arms, so it is left out of training and evaluation.
    stats = [task for task in collected if all(arm in task.pass_rate for arm in arms)]
    skipped = sorted(task.task_id for task in collected if task not in stats)
    if len(stats) < folds * 2:
        raise ValueError("Not enough graded tasks to train a router.")
    evaluation = cross_validate(stats, spec, threshold=threshold, l2=l2, folds=folds)
    selection = None
    if l2 is None:
        l2, curve = select_l2(stats)
        selection = {
            "method": f"grouped {INNER_FOLDS}-fold cross-validation on held-out log loss",
            "grid": curve,
        }
    fitted = fit(stats, arms, l2=l2)

    arm_entries = []
    for key in arms:
        arm = arm_from_key(key)
        costs = [task.cost[key] for task in stats if key in task.cost]
        passes = [task.pass_rate[key] for task in stats if key in task.pass_rate]
        arm_entries.append(
            {
                "key": key,
                "model": arm.model,
                "effort": arm.effort,
                "bias": round(fitted.bias[key], 6),
                "cost_per_request": round(sum(costs) / len(costs), 8),
                "pass_rate": round(sum(passes) / len(passes), 6),
            }
        )

    model: dict[str, Any] = {
        "trained_on": {
            "run_id": run_id,
            "synthetic": synthetic,
            "n_tasks": len(stats),
            "n_rows": sum(1 for row in rows if row["passed"] is not None),
            "skipped_tasks": skipped,
        },
        "threshold": threshold,
        "threshold_logit": round(logit(threshold), 6),
        "l2": l2,
        "l2_selection": round_floats(selection, 8),
        "baseline_arm": baseline_arm(arms),
        "features": spec,
        "mean": _round(fitted.mean),
        "scale": _round(fitted.scale),
        "weights": _round(fitted.weights),
        "arms": arm_entries,
        "cv": round_floats(evaluation, 8),
    }
    # A content hash, so a recorded playground run can tell which router routed it.
    digest = hashlib.sha256(
        json.dumps(
            {
                k: model[k]
                for k in ("features", "mean", "scale", "weights", "arms", "threshold_logit")
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {"version": digest[:12], **model}
