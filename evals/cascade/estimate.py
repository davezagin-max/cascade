"""Token and dollar estimate for a run, made before anything is sent.

Input tokens are estimated from prompt length (or counted exactly by the API when a
key is available). Output tokens cannot be known in advance, so they come from a
small table of stated assumptions per task category, effort level, and model tier.
Output includes thinking, which is why effort moves it so much.

The assumptions are deliberately on the high side: the estimate gates spending, so
it should err toward stopping. A real run then checks its measured spend against
the budget between waves (see `runner.BudgetGuard`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from .config import HAIKU, OPUS, SONNET, Effort, Usage, cost_usd
from .plan import Arm, PlannedRequest
from .prompts import system_prompt
from .tasks import Task, task_index

CHARS_PER_TOKEN = 3.5
REQUEST_OVERHEAD_TOKENS = 12

# Output tokens (answer plus thinking) for the mid tier at medium effort.
ASSUMED_OUTPUT_TOKENS: dict[str, int] = {
    "classification": 150,
    "extraction": 320,
    "reasoning": 900,
    "code": 1100,
    "open_ended": 650,
}
EFFORT_FACTOR: dict[str, float] = {
    "low": 0.45,
    "medium": 1.0,
    "high": 2.2,
    "xhigh": 4.0,
    "max": 7.0,
}
MODEL_FACTOR: dict[str, float] = {HAIKU: 0.8, SONNET: 1.0, OPUS: 1.15}

# The judge reads the task, the rubric, and one answer, and returns short scores.
JUDGE_MODEL = OPUS
JUDGE_EFFORT: Effort = "medium"
JUDGE_INPUT_OVERHEAD = 420
JUDGE_ANSWER_TOKENS = 320
JUDGE_OUTPUT_TOKENS = 700


def approx_tokens(text: str) -> int:
    """Rough token count from length. Good to roughly 15% for English prose and code."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def approx_input_tokens(task: Task) -> int:
    return approx_tokens(system_prompt(task)) + approx_tokens(task.prompt) + REQUEST_OVERHEAD_TOKENS


def assumed_output_tokens(category: str, arm: Arm) -> int:
    return round(
        ASSUMED_OUTPUT_TOKENS[category]
        * EFFORT_FACTOR[arm.effort]
        * MODEL_FACTOR.get(arm.model, 1.0)
    )


@dataclass
class ArmEstimate:
    arm: Arm
    requests: int = 0
    batch_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    list_usd: float = 0.0
    billed_usd: float = 0.0


@dataclass
class Estimate:
    arms: list[ArmEstimate]
    judge_requests: int = 0
    judge_usd: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def requests(self) -> int:
        return sum(arm.requests for arm in self.arms)

    @property
    def answers_usd(self) -> float:
        return sum(arm.billed_usd for arm in self.arms)

    @property
    def list_usd(self) -> float:
        return sum(arm.list_usd for arm in self.arms)

    @property
    def total_usd(self) -> float:
        return self.answers_usd + self.judge_usd + sum(self.extras.values())


def estimate_plan(
    plan: list[PlannedRequest],
    tasks: list[Task],
    *,
    input_tokens: Callable[[Task, Arm], int] | None = None,
    extras: dict[str, float] | None = None,
) -> Estimate:
    """Estimated tokens and dollars for every arm, plus the judge and any extras."""
    index = task_index(tasks)
    count = input_tokens or (lambda task, _arm: approx_input_tokens(task))
    by_arm: dict[Arm, ArmEstimate] = {}
    judge_requests = 0
    judge_usd = 0.0
    for request in plan:
        task = index[request.task_id]
        entry = by_arm.setdefault(request.arm, ArmEstimate(request.arm))
        usage = Usage(count(task, request.arm), assumed_output_tokens(task.category, request.arm))
        batch = request.mode == "batch"
        entry.requests += 1
        entry.batch_requests += int(batch)
        entry.input_tokens += usage.input_tokens
        entry.output_tokens += usage.output_tokens
        entry.list_usd += cost_usd(request.arm.model, usage)
        entry.billed_usd += cost_usd(request.arm.model, usage, batch=batch)
        if task.grader.type == "judge":
            judge_requests += 1
            judge_usage = Usage(
                JUDGE_INPUT_OVERHEAD + approx_tokens(task.prompt) + JUDGE_ANSWER_TOKENS,
                JUDGE_OUTPUT_TOKENS,
            )
            judge_usd += cost_usd(JUDGE_MODEL, judge_usage, batch=batch)
    return Estimate(
        arms=list(by_arm.values()),
        judge_requests=judge_requests,
        judge_usd=judge_usd,
        extras=dict(extras or {}),
    )


def format_estimate(estimate: Estimate) -> str:
    lines = [
        f"{'arm':<16}{'requests':>9}{'batch':>7}{'input tok':>12}{'output tok':>12}{'list $':>10}{'billed $':>10}"
    ]
    for arm in estimate.arms:
        lines.append(
            f"{arm.arm.key:<16}{arm.requests:>9}{arm.batch_requests:>7}{arm.input_tokens:>12,}"
            f"{arm.output_tokens:>12,}{arm.list_usd:>10.2f}{arm.billed_usd:>10.2f}"
        )
    lines.append(
        f"{'answers':<16}{estimate.requests:>9}{'':>7}{'':>12}{'':>12}"
        f"{estimate.list_usd:>10.2f}{estimate.answers_usd:>10.2f}"
    )
    lines.append(
        f"{'judge (opus)':<16}{estimate.judge_requests:>9}{'':>41}{estimate.judge_usd:>10.2f}"
    )
    for name, dollars in estimate.extras.items():
        lines.append(f"{name:<16}{'':>50}{dollars:>10.2f}")
    lines.append(f"{'TOTAL (est.)':<16}{'':>50}{estimate.total_usd:>10.2f}")
    return "\n".join(lines)
