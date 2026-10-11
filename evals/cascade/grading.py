"""Grades result rows: deterministic graders in-process, open-ended tasks through the judge.

`grade_rows` is the single grading path for real and simulated runs. The only thing
that differs is the judge passed in: `opus_judge` calls the model, and the simulator
supplies its own stand-in.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .graders import Grade, GraderUnavailable, grade_answer
from .judge import judge_request, parse_verdict
from .plan import SMALL_RUN
from .runner import ApiRequest, Executor
from .store import Row
from .tasks import Task, task_index

# For each row to judge: its grade (None when the judge's output was unusable) and what
# the judge call cost.
JudgeMany = Callable[[list[tuple[Row, Task]]], dict[str, tuple[Grade | None, float]]]
Log = Callable[[str], None]


def _apply(row: Row, grade: Grade | None, judge_usd: float = 0.0) -> None:
    if grade is None:
        row["score"] = None
        row["passed"] = None
        row["grader_details"] = json.dumps({"judge_error": "no usable verdict after one retry"})
    else:
        details = dict(grade.details)
        if row["status"] == "truncated":
            details["truncated"] = True
        row["score"] = grade.score
        row["passed"] = grade.passed
        row["grader_details"] = json.dumps(details, ensure_ascii=False)
    row["judge_usd"] = judge_usd


def grade_rows(
    rows: list[Row],
    tasks: list[Task],
    *,
    judge_many: JudgeMany | None = None,
    keep_judged: bool = False,
    log: Log = print,
) -> list[Row]:
    """Return graded copies of `rows`.

    A refusal scores zero. A request that failed (API error, expired batch item) is
    left ungraded and excluded from averages. Identical answers to the same task are
    graded once: the graders are deterministic, and arms often give the same answer.

    With `keep_judged`, open-ended rows that already carry a verdict are not sent to
    the judge again. Deterministic graders always run again; they are free.
    """
    index = task_index(tasks)
    cache: dict[tuple[str, str], Grade] = {}
    to_judge: list[tuple[Row, Task]] = []
    graded: list[Row] = []
    for original in rows:
        row = dict(original)
        graded.append(row)
        task = index.get(row["task_id"])
        if task is None:
            raise KeyError(f"Result row for unknown task {row['task_id']!r}")
        if row["status"] == "refusal":
            # Declining the task is the arm's answer, so it counts against the arm.
            _apply(row, Grade(0.0, False, {"skipped": "refusal"}))
        elif row["status"] == "error":
            # A failed request says nothing about the model. Leave it ungraded, so it is
            # excluded from every average and reported as an error instead.
            row["score"] = None
            row["passed"] = None
            earlier = json.loads(row["grader_details"] or "{}")
            row["grader_details"] = json.dumps({"skipped": "error", **earlier})
        elif task.grader.type == "judge":
            if keep_judged and row["score"] is not None:
                continue  # already judged: keep the verdict and what it cost
            to_judge.append((row, task))
        else:
            key = (task.id, row["answer"] or "")
            if key not in cache:
                cache[key] = grade_answer(task, row["answer"] or "")
            _apply(row, cache[key])

    if to_judge:
        if judge_many is None:
            raise GraderUnavailable(f"{len(to_judge)} open-ended answers need the judge")
        log(f"[judge] grading {len(to_judge)} open-ended answers")
        verdicts = judge_many(to_judge)
        for row, _task in to_judge:
            grade, judge_usd = verdicts[row["custom_id"]]
            _apply(row, grade, judge_usd)
    return graded


def judge_id(row: Row) -> str:
    return f"j__{row['custom_id']}"


def opus_judge(executor: Executor) -> JudgeMany:
    """Judge with Opus: batched when there are many answers, one synchronous retry each."""

    def judge_many(items: list[tuple[Row, Task]]) -> dict[str, tuple[Grade | None, float]]:
        mode = "batch" if len(items) > SMALL_RUN else "sync"
        requests: list[ApiRequest] = [
            judge_request(task, row["answer"] or "", judge_id(row), mode)  # type: ignore[arg-type]
            for row, task in items
        ]
        responses = executor.run(requests, label="judge")
        results: dict[str, tuple[Grade | None, float]] = {}
        retry: list[tuple[Row, Task]] = []
        spent: dict[str, float] = {}
        for row, task in items:
            response = responses[judge_id(row)]
            spent[row["custom_id"]] = response.billed_usd
            grade = parse_verdict(task, response.text) if response.status == "ok" else None
            if grade is None:
                retry.append((row, task))
            else:
                results[row["custom_id"]] = (grade, response.billed_usd)
        if retry:
            executor.log(f"[judge] retrying {len(retry)} verdicts that did not fit the rubric")
            again = executor.run(
                [
                    judge_request(task, row["answer"] or "", f"r{judge_id(row)}", "sync")
                    for row, task in retry
                ],
                label="judge-retry",
            )
            for row, task in retry:
                response = again[f"r{judge_id(row)}"]
                total = spent[row["custom_id"]] + response.billed_usd
                grade = parse_verdict(task, response.text) if response.status == "ok" else None
                results[row["custom_id"]] = (grade, total)
        return results

    return judge_many
