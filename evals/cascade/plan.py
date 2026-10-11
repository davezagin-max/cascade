"""The run matrix: which request goes to which model, at which effort, by which path.

An arm is one (model, effort) pair. A plan is one request per (task, arm, trial).
The same plan drives a real run, the cost estimate, and the simulator, so all three
agree on what a run contains.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from .config import DEFAULT_EFFORTS, EFFORTS, MODEL_BY_SHORT, MODELS, SHORT_NAMES, Effort
from .prompts import max_tokens_for
from .tasks import CATEGORIES, Task

Mode = Literal["batch", "sync"]

DEFAULT_TRIALS = 3
# Batch results carry no timing, so this many tasks per category are sent
# synchronously in every arm and timed. They are the latency sample.
DEFAULT_LATENCY_SAMPLE = 2
# Below this many requests a batch is not worth the wait.
SMALL_RUN = 60


@dataclass(frozen=True, order=True)
class Arm:
    model: str
    effort: Effort

    @property
    def key(self) -> str:
        return f"{SHORT_NAMES.get(self.model, self.model)}-{self.effort}"

    @property
    def label(self) -> str:
        return f"{SHORT_NAMES.get(self.model, self.model).capitalize()} · {self.effort}"


def arm_from_key(key: str) -> Arm:
    short, _, effort = key.partition("-")
    if short not in MODEL_BY_SHORT or effort not in EFFORTS:
        raise ValueError(f"Unknown arm: {key}")
    return Arm(MODEL_BY_SHORT[short], effort)  # type: ignore[arg-type]


def build_arms(
    models: tuple[str, ...] = MODELS, efforts: tuple[Effort, ...] = DEFAULT_EFFORTS
) -> list[Arm]:
    """Every (model, effort) pair, cheapest tier first, lowest effort first."""
    ordered_efforts = [effort for effort in EFFORTS if effort in efforts]
    return [Arm(model, effort) for model in MODELS if model in models for effort in ordered_efforts]


@dataclass(frozen=True)
class PlannedRequest:
    custom_id: str
    task_id: str
    arm: Arm
    trial: int
    mode: Mode
    max_tokens: int


def custom_id(task_id: str, arm: Arm, trial: int) -> str:
    """Stable id that ties a batch result back to its row. Fits the API's 64-character limit."""
    return f"{task_id}__{arm.key}__t{trial}"


def latency_sample(tasks: list[Task], per_category: int) -> set[str]:
    """A fixed, stratified set of task ids: the first few of each category by id hash."""
    chosen: set[str] = set()
    for category in CATEGORIES:
        members = sorted(
            (task.id for task in tasks if task.category == category),
            key=lambda task_id: hashlib.sha256(task_id.encode()).hexdigest(),
        )
        chosen.update(members[:per_category])
    return chosen


def build_plan(
    tasks: list[Task],
    arms: list[Arm],
    *,
    trials: int = DEFAULT_TRIALS,
    mode: Literal["auto", "batch", "sync"] = "auto",
    latency_sample_per_category: int = DEFAULT_LATENCY_SAMPLE,
) -> list[PlannedRequest]:
    """One request per (task, arm, trial).

    In batch mode the latency sample still goes through the synchronous path (first
    trial only), because that is the only way to time a request.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    total = len(tasks) * len(arms) * trials
    if mode == "auto":
        mode = "sync" if total <= SMALL_RUN else "batch"
    timed = latency_sample(tasks, latency_sample_per_category) if mode == "batch" else set()
    plan: list[PlannedRequest] = []
    for arm in arms:
        for task in tasks:
            for trial in range(trials):
                sync = mode == "sync" or (task.id in timed and trial == 0)
                plan.append(
                    PlannedRequest(
                        custom_id=custom_id(task.id, arm, trial),
                        task_id=task.id,
                        arm=arm,
                        trial=trial,
                        mode="sync" if sync else "batch",
                        max_tokens=max_tokens_for(arm.effort),
                    )
                )
    return plan
