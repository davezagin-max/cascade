"""The eval suite: hand-written tasks stored as TOML in `evals/tasks/`.

Every task carries two stand-in answers written the way a model would answer:

- `reference`: a correct answer. The grader must give it a full score.
- `wrong`: a plausible mistake. The grader must not pass it.

`validate_suite` checks both for every task, which is how the graders themselves are
tested against the real suite. The simulator also uses them as its stand-in answers.

`difficulty` (1 easy to 5 hard) is the author's guess. Only the simulator reads it:
the router never sees it, and a real run ignores it.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

Category = Literal["classification", "extraction", "reasoning", "code", "open_ended"]
CATEGORIES: tuple[Category, ...] = (
    "classification",
    "extraction",
    "reasoning",
    "code",
    "open_ended",
)
CATEGORY_LABELS: dict[str, str] = {
    "classification": "Classification",
    "extraction": "Extraction",
    "reasoning": "Reasoning and math",
    "code": "Code",
    "open_ended": "Open-ended",
}

# Task ids go into batch custom_ids, which allow only [a-zA-Z0-9_-] and 64 characters.
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,35}$")


@dataclass(frozen=True)
class ExactSpec:
    """Exact match on a final answer (a label, a number, a fraction, a word)."""

    expected: str
    accept: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    type: str = "exact"


@dataclass(frozen=True)
class JsonSpec:
    """A JSON object that must validate against `schema` and match `expected` field by field."""

    schema: dict[str, Any]
    expected: dict[str, Any]
    unordered: tuple[str, ...] = ()
    type: str = "json"


@dataclass(frozen=True)
class CodeSpec:
    """A Python definition checked by hidden unit tests in the sandbox."""

    entrypoint: str
    tests: tuple[str, ...]
    timeout_s: float = 6.0
    type: str = "code"


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    weight: float = 1.0


@dataclass(frozen=True)
class JudgeSpec:
    """An open-ended answer scored by the Opus judge against a written rubric."""

    rubric: tuple[Criterion, ...]
    pass_mark: float = 0.75
    type: str = "judge"


GraderSpec = ExactSpec | JsonSpec | CodeSpec | JudgeSpec


@dataclass(frozen=True)
class Task:
    id: str
    category: Category
    difficulty: int
    title: str
    prompt: str
    grader: GraderSpec
    reference: str
    wrong: str


class SuiteError(ValueError):
    """A task file is malformed."""


def _parse_grader(task_id: str, raw: dict[str, Any]) -> GraderSpec:
    kind = raw.get("type")
    if kind == "exact":
        return ExactSpec(
            expected=str(raw["expected"]),
            accept=tuple(str(a) for a in raw.get("accept", [])),
            labels=tuple(str(label) for label in raw.get("labels", [])),
        )
    if kind == "json":
        return JsonSpec(
            schema=json.loads(raw["schema"]),
            expected=json.loads(raw["expected"]),
            unordered=tuple(raw.get("unordered", [])),
        )
    if kind == "code":
        return CodeSpec(
            entrypoint=raw["entrypoint"],
            tests=tuple(raw["tests"]),
            timeout_s=float(raw.get("timeout_s", 6.0)),
        )
    if kind == "judge":
        return JudgeSpec(
            rubric=tuple(
                Criterion(c["name"], c["description"], float(c.get("weight", 1.0)))
                for c in raw["rubric"]
            ),
            pass_mark=float(raw.get("pass_mark", 0.75)),
        )
    raise SuiteError(f"{task_id}: unknown grader type {kind!r}")


def _default_reference(spec: GraderSpec) -> str | None:
    if isinstance(spec, ExactSpec):
        return f"ANSWER: {spec.expected}"
    if isinstance(spec, JsonSpec):
        return json.dumps(spec.expected, indent=2, ensure_ascii=False)
    return None


def _parse_task(raw: dict[str, Any], category: str) -> Task:
    task_id = raw.get("id", "")
    if not _TASK_ID.match(task_id):
        raise SuiteError(f"Invalid task id: {task_id!r}")
    try:
        spec = _parse_grader(task_id, raw["grader"])
        reference = raw.get("reference") or _default_reference(spec)
        wrong = raw.get("wrong")
        if wrong is None and isinstance(spec, JsonSpec) and "wrong_patch" in raw:
            # The typical mistake, written as the fields a model gets wrong.
            patched = {**spec.expected, **json.loads(raw["wrong_patch"])}
            wrong = json.dumps(patched, indent=2, ensure_ascii=False)
        if reference is None or wrong is None:
            raise SuiteError(f"{task_id}: needs both a reference and a wrong answer")
        difficulty = int(raw["difficulty"])
        if not 1 <= difficulty <= 5:
            raise SuiteError(f"{task_id}: difficulty must be 1 to 5")
        if category not in CATEGORIES:
            raise SuiteError(f"{task_id}: unknown category {category!r}")
        return Task(
            id=task_id,
            category=category,  # type: ignore[arg-type]
            difficulty=difficulty,
            title=raw["title"].strip(),
            prompt=raw["prompt"].strip(),
            grader=spec,
            reference=reference.strip(),
            wrong=wrong.strip(),
        )
    except KeyError as missing:
        raise SuiteError(f"{task_id}: missing field {missing}") from missing
    except json.JSONDecodeError as bad:
        raise SuiteError(f"{task_id}: invalid JSON in grader ({bad})") from bad


def load_tasks(directory: Path = TASKS_DIR) -> list[Task]:
    """Load every task file, in a stable order (category, then position in the file)."""
    tasks: list[Task] = []
    for category in CATEGORIES:
        path = directory / f"{category}.toml"
        if not path.exists():
            continue
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        tasks.extend(_parse_task(raw, category) for raw in data.get("task", []))
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise SuiteError(f"Duplicate task id: {task.id}")
        seen.add(task.id)
    return tasks


def task_index(tasks: list[Task]) -> dict[str, Task]:
    return {task.id: task for task in tasks}


def public_grader(spec: GraderSpec) -> dict[str, Any]:
    """The grader definition as shown in the task explorer."""
    if isinstance(spec, ExactSpec):
        return {
            "type": "exact",
            "expected": spec.expected,
            "accept": list(spec.accept),
            "labels": list(spec.labels),
        }
    if isinstance(spec, JsonSpec):
        return {
            "type": "json",
            "schema": spec.schema,
            "expected": spec.expected,
            "unordered": list(spec.unordered),
        }
    if isinstance(spec, CodeSpec):
        return {"type": "code", "entrypoint": spec.entrypoint, "tests": list(spec.tests)}
    return {
        "type": "judge",
        "pass_mark": spec.pass_mark,
        "rubric": [
            {"name": c.name, "description": c.description, "weight": c.weight} for c in spec.rubric
        ],
    }
