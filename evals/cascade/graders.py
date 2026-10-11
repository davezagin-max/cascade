"""Graders: one deterministic grader per task type, plus rubric scoring for the judge.

Every grader returns a `Grade` with a score in [0, 1], a pass flag, and details
that explain the score (which field mismatched, which hidden test failed). The
details are stored with the result and shown in the task explorer.

Grading is lenient about wrapping (code fences, a sentence before the JSON, bold
markers around the answer line) and strict about content.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from jsonschema import Draft202012Validator

from .sandbox import run_tests
from .tasks import CodeSpec, ExactSpec, JsonSpec, JudgeSpec, Task


@dataclass(frozen=True)
class Grade:
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


class GraderUnavailable(RuntimeError):
    """An open-ended task was graded without a judge."""


# --- exact match ------------------------------------------------------------------------

_ANSWER_LINE = re.compile(
    r"^[\s>*_`#-]*(?:final\s+)?answer[\s*_`]*[:=][\s*_`]*(.+?)[\s*_`.]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")
_WRAPPERS = "\"'`*_ \t"


def extract_final_answer(text: str) -> tuple[str, bool]:
    """The answer on the last `ANSWER:` line, else the last non-empty line.

    Returns the raw answer and whether the `ANSWER:` format was followed.
    """
    matches = _ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip(), True
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return (lines[-1] if lines else ""), False


def normalize_answer(text: str) -> str:
    cleaned = text.strip().strip(_WRAPPERS).rstrip(".").strip(_WRAPPERS)
    return re.sub(r"\s+", " ", cleaned).lower()


def _label(text: str) -> str:
    return re.sub(r"[\s-]+", "_", normalize_answer(text))


def parse_number(text: str) -> Fraction | None:
    """Exact value of `12`, `1,284.50`, `$80`, `60%`, or `5/12`; None if not a lone number."""
    cleaned = normalize_answer(text).replace("$", "").replace("%", "").replace(" ", "")
    cleaned = cleaned.replace(",", "")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?(?:/\d+)?", cleaned):
        return None
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def _number_in(text: str) -> Fraction | None:
    """A number when the text is one, or contains exactly one (as in `408 bottles`)."""
    direct = parse_number(text)
    if direct is not None:
        return direct
    found = _NUMBER.findall(text)
    return parse_number(found[0]) if len(found) == 1 else None


def grade_exact(spec: ExactSpec, answer: str) -> Grade:
    raw, format_ok = extract_final_answer(answer)
    targets = (spec.expected, *spec.accept)
    details: dict[str, Any] = {
        "extracted": raw[:200],
        "expected": spec.expected,
        "format_ok": format_ok,
    }
    matched = False
    for target in targets:
        wanted = parse_number(target)
        if wanted is not None:
            matched = _number_in(raw) == wanted
        elif spec.labels:
            matched = _label(raw) == _label(target)
        else:
            matched = normalize_answer(raw) == normalize_answer(target)
        if matched:
            break
    if spec.labels and not matched:
        details["valid_label"] = _label(raw) in {_label(label) for label in spec.labels}
    return Grade(1.0 if matched else 0.0, matched, details)


# --- JSON: schema validity, then field-by-field match -----------------------------------

_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> tuple[Any, str | None]:
    """Parse the JSON value in an answer, tolerating a code fence or surrounding prose."""
    candidates = [text.strip()]
    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    if start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            return value, None
        except json.JSONDecodeError as error:
            return None, f"invalid JSON: {error.msg}"
    return None, "no JSON object found"


def _text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def values_match(expected: Any, actual: Any, *, unordered: bool = False) -> bool:
    """Deep comparison: numbers to the cent, strings ignoring case and spacing."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int | float):
        if isinstance(actual, bool) or not isinstance(actual, int | float):
            return False
        return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=0.005 + 1e-9)
    if isinstance(expected, str):
        return isinstance(actual, str) and _text_key(expected) == _text_key(actual)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        if not unordered:
            return all(values_match(e, a) for e, a in zip(expected, actual, strict=True))
        remaining = list(actual)
        for item in expected:
            hit = next((i for i, other in enumerate(remaining) if values_match(item, other)), None)
            if hit is None:
                return False
            remaining.pop(hit)
        return True
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(values_match(value, actual[key]) for key, value in expected.items())
        )
    return False


def grade_json(spec: JsonSpec, answer: str) -> Grade:
    value, problem = extract_json(answer)
    if problem is not None:
        return Grade(0.0, False, {"schema_valid": False, "errors": [problem]})
    errors = sorted(Draft202012Validator(spec.schema).iter_errors(value), key=lambda e: e.path)
    if errors:
        messages = [
            f"{'/'.join(str(p) for p in error.path) or '(root)'}: {error.message[:140]}"
            for error in errors[:3]
        ]
        return Grade(0.0, False, {"schema_valid": False, "errors": messages})
    if not isinstance(value, dict):
        # Only reachable when a task's schema forgets `"type": "object"`. Fail the answer
        # instead of crashing the grading run on `value[key]`.
        return Grade(0.0, False, {"schema_valid": False, "errors": ["expected a JSON object"]})
    fields = {
        key: key in value and values_match(wanted, value[key], unordered=key in spec.unordered)
        for key, wanted in spec.expected.items()
    }
    correct = sum(fields.values())
    return Grade(
        correct / len(fields),
        correct == len(fields),
        {"schema_valid": True, "fields": fields},
    )


# --- code: hidden unit tests ------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, entrypoint: str) -> str:
    """The fenced block that defines the entrypoint, else the longest block, else the text."""
    blocks = _CODE_FENCE.findall(text)
    if not blocks:
        return text.strip()
    defining = [
        block for block in blocks if re.search(rf"^\s*(?:def|class)\s+{entrypoint}\b", block, re.M)
    ]
    return (defining[-1] if defining else max(blocks, key=len)).strip()


def grade_code(spec: CodeSpec, answer: str) -> Grade:
    code = extract_code(answer, spec.entrypoint)
    total = len(spec.tests)
    if not code:
        return Grade(0.0, False, {"passed": 0, "total": total, "load_error": "empty answer"})
    result = run_tests(code, spec.tests, timeout_s=spec.timeout_s)
    details: dict[str, Any] = {
        "passed": result.passed,
        "total": total,
        "tests": [
            {"ok": outcome.ok, **({"error": outcome.error} if outcome.error else {})}
            for outcome in result.outcomes
        ],
    }
    if result.load_error:
        details["load_error"] = result.load_error
    if result.timed_out:
        details["timed_out"] = True
    return Grade(result.passed / total, result.passed == total, details)


# --- judge: weighted rubric -------------------------------------------------------------

MAX_CRITERION_SCORE = 2


def score_rubric(spec: JudgeSpec, scores: dict[str, int]) -> float:
    """Weighted share of the rubric earned. Every criterion must be scored 0, 1, or 2."""
    names = {criterion.name for criterion in spec.rubric}
    if set(scores) != names:
        raise ValueError(f"Judge scored {sorted(scores)}, rubric has {sorted(names)}")
    if any(value not in (0, 1, 2) for value in scores.values()):
        raise ValueError("Criterion scores must be 0, 1, or 2.")
    earned = sum(criterion.weight * scores[criterion.name] for criterion in spec.rubric)
    possible = sum(criterion.weight * MAX_CRITERION_SCORE for criterion in spec.rubric)
    return earned / possible


def grade_from_verdict(
    spec: JudgeSpec, scores: dict[str, int], rationales: dict[str, str] | None = None
) -> Grade:
    score = score_rubric(spec, scores)
    details: dict[str, Any] = {"criteria": scores, "pass_mark": spec.pass_mark}
    if rationales:
        details["rationales"] = rationales
    # A hair of slack so 0.75 computed as 0.7499999 still passes a 0.75 mark.
    return Grade(score, score >= spec.pass_mark - 1e-9, details)


Judge = Callable[[Task, str], Grade]


def grade_answer(task: Task, answer: str, *, judge: Judge | None = None) -> Grade:
    """Grade one answer with the grader its task declares."""
    spec = task.grader
    if isinstance(spec, ExactSpec):
        return grade_exact(spec, answer)
    if isinstance(spec, JsonSpec):
        return grade_json(spec, answer)
    if isinstance(spec, CodeSpec):
        return grade_code(spec, answer)
    if judge is None:
        raise GraderUnavailable(f"{task.id} needs the judge")
    return judge(task, answer)
