"""Checks the suite against its own graders.

For every deterministic task the reference answer must earn a full score and the
known-wrong answer must not pass. That catches a bad key, a bad hidden test, and a
grader that is too lenient or too strict, before any money is spent on a run.
Judge-graded tasks are checked structurally: the judge itself needs an API key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .graders import grade_answer
from .tasks import CodeSpec, ExactSpec, JsonSpec, JudgeSpec, Task


@dataclass(frozen=True)
class Problem:
    task_id: str
    message: str


def validate_task(task: Task) -> list[Problem]:
    problems: list[Problem] = []

    def flag(message: str) -> None:
        problems.append(Problem(task.id, message))

    spec = task.grader
    if not task.prompt or not task.title:
        flag("empty prompt or title")

    if isinstance(spec, JudgeSpec):
        names = [criterion.name for criterion in spec.rubric]
        if len(spec.rubric) < 2:
            flag("rubric needs at least two criteria")
        if len(set(names)) != len(names):
            flag("rubric criterion names must be unique")
        if any(criterion.weight <= 0 for criterion in spec.rubric):
            flag("rubric weights must be positive")
        if not 0 < spec.pass_mark <= 1:
            flag("pass_mark must be in (0, 1]")
        return problems

    if isinstance(spec, ExactSpec) and spec.labels and spec.expected not in spec.labels:
        flag("expected label is not in the label set")
    if isinstance(spec, JsonSpec):
        try:
            Draft202012Validator.check_schema(spec.schema)
        except SchemaError as error:
            flag(f"invalid JSON schema: {error.message}")
            return problems
    if isinstance(spec, CodeSpec) and not spec.tests:
        flag("code task has no hidden tests")

    reference = grade_answer(task, task.reference)
    if not reference.passed or reference.score != 1.0:
        flag(f"reference answer scored {reference.score:.2f}: {reference.details}")
    wrong = grade_answer(task, task.wrong)
    if wrong.passed:
        flag("known-wrong answer passed")
    return problems


def validate_suite(tasks: list[Task]) -> list[Problem]:
    return [problem for task in tasks for problem in validate_task(task)]
