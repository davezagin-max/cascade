"""A deterministic stand-in for a real run.

NOTHING THIS MODULE PRODUCES IS A MEASUREMENT. It exists so the whole pipeline
(plan, grade, train the router, export, render the dashboard) can be built, tested,
and demonstrated before a single API call is paid for. Every row it writes is
flagged `synthetic`, and the dashboard shows a banner whenever that flag is set.

How it works. Each arm gets an assumed ability and each task an assumed difficulty
(the author's 1-5 guess plus hidden per-task noise). For every planned request the
simulator draws pass or fail from the resulting probability, then submits one of
the task's two hand-written stand-in answers (the reference answer on a pass, the
known-wrong answer on a fail) to the **real graders**. Scores, pass flags, and
grader details in a synthetic run are therefore genuine grader output; only the
answers, token counts, and latencies are invented.

The one exception is the open-ended tasks. Their grader is the Opus judge, which
cannot be called without a key, so `simulated_judge` draws a score per rubric
criterion instead and only the weighting is real code. Those rows say so in their
rationales, and the dashboard says so wherever it describes the synthetic run.

The numbers below are priors, not calibrated to anything. They were chosen once
and not tuned afterwards: tuning them until the router headline looked good would
make the placeholder a lie.

Determinism. All randomness comes from `random.Random` seeded with a string, and
only `random()` is used, combined with exact arithmetic and `sqrt`. There is no
`exp` or `log`, so the same seed gives the same rows on every platform.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any

from .config import HAIKU, OPUS, SONNET, Usage, cost_usd
from .estimate import (
    JUDGE_ANSWER_TOKENS,
    JUDGE_INPUT_OVERHEAD,
    JUDGE_MODEL,
    JUDGE_OUTPUT_TOKENS,
    approx_input_tokens,
    approx_tokens,
    assumed_output_tokens,
)
from .graders import Grade, grade_from_verdict
from .grading import JudgeMany
from .plan import Arm, PlannedRequest
from .runner import ApiResponse, row_from_response
from .store import Row
from .tasks import JudgeSpec, Task, task_index

DEFAULT_SEED = 7
SIMULATOR_NOTE = (
    "Deterministic simulator. Pass or fail is drawn from assumed abilities and task "
    "difficulties; the stand-in answers are graded by the real graders, except open-ended "
    "ones, whose rubric scores are drawn without calling the judge. Tokens, costs and "
    "latencies are derived from stated assumptions. Not a measurement."
)

# Assumed ability per arm, on the same 1-5 scale as task difficulty.
ABILITY: dict[str, dict[str, float]] = {
    HAIKU: {"low": 2.9, "medium": 3.3, "high": 3.6, "xhigh": 3.75, "max": 3.85},
    SONNET: {"low": 3.7, "medium": 4.2, "high": 4.6, "xhigh": 4.8, "max": 4.9},
    OPUS: {"low": 4.4, "medium": 5.0, "high": 5.5, "xhigh": 5.8, "max": 6.0},
}
# Some task types are easier than their difficulty number suggests, for every model...
CATEGORY_SHIFT: dict[str, float] = {
    "classification": 0.4,
    "extraction": 0.2,
    "reasoning": -0.1,
    "code": -0.2,
    "open_ended": 0.0,
}
# ...and the smallest tier is assumed to lose a little extra on multi-step work.
MODEL_CATEGORY_SHIFT: dict[str, dict[str, float]] = {HAIKU: {"code": -0.3, "reasoning": -0.2}}
SLOPE = 1.5
TASK_NOISE = 0.5  # hidden difficulty the author's guess (and the router) cannot see
MODEL_TASK_NOISE = 0.3  # a task one tier happens to find easier or harder than expected

# The cost estimate is deliberately high; simulated usage sits a little under it.
OUTPUT_SCALE = 0.8
TIME_TO_FIRST_TOKEN_S: dict[str, float] = {HAIKU: 0.35, SONNET: 0.6, OPUS: 0.9}
TOKENS_PER_SECOND: dict[str, float] = {HAIKU: 170.0, SONNET: 90.0, OPUS: 55.0}


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random("|".join([str(seed), *parts]))


def _bell(rng: random.Random) -> float:
    """Roughly normal noise with unit variance, from four uniforms and exact arithmetic."""
    return (rng.random() + rng.random() + rng.random() + rng.random() - 2.0) * math.sqrt(3.0)


def pass_probability(task: Task, arm: Arm, seed: int) -> float:
    ability = ABILITY[arm.model][arm.effort]
    shift = CATEGORY_SHIFT[task.category] + MODEL_CATEGORY_SHIFT.get(arm.model, {}).get(
        task.category, 0.0
    )
    hidden = TASK_NOISE * _bell(_rng(seed, "task", task.id))
    quirk = MODEL_TASK_NOISE * _bell(_rng(seed, "quirk", task.id, arm.model))
    x = SLOPE * (ability + shift - task.difficulty - hidden - quirk)
    p = 0.5 + 0.5 * x / math.sqrt(1.0 + x * x)
    return min(0.995, max(0.02, p))


def simulate_response(planned: PlannedRequest, task: Task, seed: int) -> tuple[ApiResponse, float]:
    """One synthetic response and the pass probability it was drawn from."""
    rng = _rng(seed, "row", planned.custom_id)
    probability = pass_probability(task, planned.arm, seed)
    passed = rng.random() < probability
    effort_on_task = 0.4 + 0.2 * task.difficulty  # harder tasks draw more thinking
    jitter = 0.5 + 0.5 * (rng.random() + rng.random())
    output_tokens = max(
        8,
        round(
            assumed_output_tokens(task.category, planned.arm)
            * OUTPUT_SCALE
            * effort_on_task
            * jitter
        ),
    )
    latency_ms = None
    if planned.mode == "sync":
        seconds = (
            TIME_TO_FIRST_TOKEN_S[planned.arm.model]
            + output_tokens / TOKENS_PER_SECOND[planned.arm.model]
        ) * (0.85 + 0.3 * rng.random())
        latency_ms = round(seconds * 1000.0, 1)
    response = ApiResponse(
        custom_id=planned.custom_id,
        status="ok",
        text=task.reference if passed else task.wrong,
        stop_reason="end_turn",
        usage=Usage(approx_input_tokens(task), output_tokens),
        model=planned.arm.model,
        mode=planned.mode,
        latency_ms=latency_ms,
    )
    return response, probability


def simulate_rows(
    plan: list[PlannedRequest], tasks: list[Task], *, run_id: str, seed: int = DEFAULT_SEED
) -> list[Row]:
    """Ungraded synthetic rows for a plan, in exactly the shape a real run writes."""
    index = task_index(tasks)
    rows: list[Row] = []
    for planned in plan:
        task = index[planned.task_id]
        response, _ = simulate_response(planned, task, seed)
        rows.append(row_from_response(run_id, planned, task, response, synthetic=True))
    return rows


def simulated_judge(seed: int = DEFAULT_SEED) -> JudgeMany:
    """Stands in for the Opus judge: rubric scores are drawn, then weighted by real code."""

    def judge_many(items: list[tuple[Row, Task]]) -> dict[str, tuple[Grade | None, float]]:
        results: dict[str, tuple[Grade | None, float]] = {}
        for row, task in items:
            spec = task.grader
            assert isinstance(spec, JudgeSpec)
            rng = _rng(seed, "judge", row["custom_id"])
            strong = row["answer"] == task.reference
            scores: dict[str, int] = {}
            for criterion in spec.rubric:
                draw = rng.random()
                if strong:
                    scores[criterion.name] = 2 if draw < 0.85 else 1
                else:
                    scores[criterion.name] = 0 if draw < 0.5 else (1 if draw < 0.9 else 2)
            rationales = {name: "Simulated score: no judge was called." for name in scores}
            usage = Usage(
                JUDGE_INPUT_OVERHEAD + approx_tokens(task.prompt) + JUDGE_ANSWER_TOKENS,
                round(JUDGE_OUTPUT_TOKENS * OUTPUT_SCALE),
            )
            cost = cost_usd(JUDGE_MODEL, usage, batch=row["mode"] == "batch")
            results[row["custom_id"]] = (grade_from_verdict(spec, scores, rationales), cost)
        return results

    return judge_many


def annotate(rows: list[Row], plan: list[PlannedRequest], tasks: list[Task], seed: int) -> None:
    """Record, in each graded row's details, that it is simulated and what it was drawn from."""
    index = task_index(tasks)
    by_id = {planned.custom_id: planned for planned in plan}
    for row in rows:
        planned = by_id[row["custom_id"]]
        details: dict[str, Any] = json.loads(row["grader_details"] or "{}")
        details["simulated"] = {
            "pass_probability": round(
                pass_probability(index[row["task_id"]], planned.arm, seed), 3
            ),
            "answer": "reference"
            if row["answer"] == index[row["task_id"]].reference
            else "known-wrong",
        }
        row["grader_details"] = json.dumps(details, ensure_ascii=False)
