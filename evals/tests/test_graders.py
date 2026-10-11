import pytest

from cascade.graders import (
    extract_code,
    extract_final_answer,
    extract_json,
    grade_code,
    grade_exact,
    grade_from_verdict,
    grade_json,
    score_rubric,
    values_match,
)
from cascade.tasks import CodeSpec, Criterion, ExactSpec, JsonSpec, JudgeSpec


def test_final_answer_extraction() -> None:
    assert extract_final_answer("Work...\nANSWER: 42") == ("42", True)
    assert extract_final_answer("ANSWER: 1\nwait\n**Answer:** `7`.") == ("7", True)
    assert extract_final_answer("Final answer = 3.5") == ("3.5", True)
    assert extract_final_answer("just a line\n  last line  ") == ("last line", False)
    assert extract_final_answer("") == ("", False)


@pytest.mark.parametrize(
    ("answer", "passed"),
    [
        ("ANSWER: 408", True),
        ("ANSWER: 408 bottles", True),
        ("ANSWER: 408.0", True),
        ("The total is 17 x 24.\nANSWER: 408.", True),
        ("408", True),  # no ANSWER line: the last line is used
        ("ANSWER: 398", False),
        ("ANSWER: 408 or 409", False),  # two numbers is not an answer
        ("ANSWER: four hundred", False),
    ],
)
def test_numeric_exact_match(answer: str, passed: bool) -> None:
    assert grade_exact(ExactSpec("408"), answer).passed is passed


def test_money_percent_and_fractions_compare_by_value() -> None:
    assert grade_exact(ExactSpec("100"), "ANSWER: $100.00").passed
    assert grade_exact(ExactSpec("1284.5"), "ANSWER: 1,284.50").passed
    assert grade_exact(ExactSpec("5/12"), "ANSWER: 5/12").passed
    assert grade_exact(ExactSpec("5/12"), "ANSWER: 15/36").passed  # same value
    assert not grade_exact(ExactSpec("5/12"), "ANSWER: 11/36").passed
    assert not grade_exact(ExactSpec("5/12"), "ANSWER: 0.41").passed


def test_labels_ignore_case_spacing_and_hyphens_but_not_meaning() -> None:
    spec = ExactSpec("limitation_of_liability", labels=("limitation_of_liability", "termination"))
    assert grade_exact(spec, "ANSWER: Limitation of Liability").passed
    assert grade_exact(spec, "ANSWER: limitation-of-liability").passed
    wrong = grade_exact(spec, "ANSWER: termination")
    assert not wrong.passed and wrong.details["valid_label"] is True
    invented = grade_exact(spec, "ANSWER: liability")
    assert not invented.passed and invented.details["valid_label"] is False


def test_words_match_exactly_after_normalising() -> None:
    assert grade_exact(ExactSpec("Tuesday"), "ANSWER: tuesday.").passed
    assert not grade_exact(ExactSpec("Tuesday"), "ANSWER: Monday").passed
    assert grade_exact(ExactSpec("A", accept=("talk a",)), "ANSWER: Talk A").passed


def test_json_is_found_inside_fences_and_prose() -> None:
    assert extract_json('{"a": 1}') == ({"a": 1}, None)
    assert extract_json('```json\n{"a": 1}\n```') == ({"a": 1}, None)
    assert extract_json('Here you go: {"a": {"b": [1, 2]}} hope that helps') == (
        {"a": {"b": [1, 2]}},
        None,
    )
    assert extract_json("no json here")[1] == "no JSON object found"
    assert extract_json('{"a": ')[1].startswith("invalid JSON")


def test_values_match_rules() -> None:
    assert values_match(1284.5, 1284.50)
    assert values_match(17.28, 17.283)  # to the cent
    assert not values_match(17.28, 17.3)
    assert not values_match(1, True) and not values_match(True, 1)  # bool is not a number
    assert values_match(None, None) and not values_match(None, 0)
    assert values_match("Brightline Paper Co.", "  brightline  paper co. ")
    assert not values_match("2026-03-12", "2026-03-13")
    assert not values_match([1, 2], [2, 1])
    assert values_match([{"k": 1}, {"k": 2}], [{"k": 2}, {"k": 1}], unordered=True)
    assert not values_match([1, 1, 2], [1, 2, 2], unordered=True)  # multiset, not set
    assert not values_match({"a": 1}, {"a": 1, "b": 2})


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "total", "tags"],
    "properties": {
        "name": {"type": "string"},
        "total": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
}
JSON_SPEC = JsonSpec(
    SCHEMA, {"name": "Ada", "total": 10.5, "tags": ["x", "y"]}, unordered=("tags",)
)


def test_json_grader_scores_fields_after_schema_validation() -> None:
    perfect = grade_json(JSON_SPEC, '{"name": "ada", "total": 10.5, "tags": ["y", "x"]}')
    assert perfect.passed and perfect.score == 1.0

    partial = grade_json(JSON_SPEC, '{"name": "Ada", "total": 11, "tags": ["x", "y"]}')
    assert not partial.passed
    assert partial.score == pytest.approx(2 / 3)
    assert partial.details["fields"] == {"name": True, "total": False, "tags": True}


def test_schema_invalid_json_scores_zero_even_if_values_are_right() -> None:
    wrong_type = grade_json(JSON_SPEC, '{"name": "Ada", "total": "10.5", "tags": ["x", "y"]}')
    assert wrong_type.score == 0.0 and wrong_type.details["schema_valid"] is False
    assert "total" in wrong_type.details["errors"][0]
    extra_key = grade_json(JSON_SPEC, '{"name": "Ada", "total": 10.5, "tags": [], "note": "hi"}')
    assert extra_key.score == 0.0
    assert grade_json(JSON_SPEC, "I could not find it.").score == 0.0


def test_a_json_answer_that_is_not_an_object_fails_instead_of_crashing() -> None:
    # A schema with no "type" accepts a list. The list even contains an expected key as
    # an item, which used to reach `value[key]` and raise TypeError mid-run.
    loose = JsonSpec(schema={}, expected={"name": "Ada"})
    graded = grade_json(loose, '["name", "Ada"]')
    assert graded.passed is False and graded.score == 0.0
    assert graded.details == {"schema_valid": False, "errors": ["expected a JSON object"]}


def test_code_extraction_prefers_the_block_defining_the_entrypoint() -> None:
    answer = "Usage:\n```python\nprint(add(1, 2))\n```\nCode:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(answer, "add") == "def add(a, b):\n    return a + b"
    assert extract_code("def add(a, b):\n    return a + b", "add").startswith("def add")


CODE_SPEC = CodeSpec(
    "add",
    (
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
        "assert raises(TypeError, add, 1, None)",
    ),
)


def test_code_grader_counts_hidden_tests() -> None:
    good = grade_code(CODE_SPEC, "```python\ndef add(a, b):\n    return a + b\n```")
    assert good.passed and good.details["passed"] == 3

    buggy = grade_code(CODE_SPEC, "def add(a, b):\n    return a - b")
    assert not buggy.passed
    assert buggy.score == pytest.approx(1 / 3)
    assert [test["ok"] for test in buggy.details["tests"]] == [False, False, True]

    broken = grade_code(CODE_SPEC, "def add(a, b) return a + b")
    assert broken.score == 0.0 and "SyntaxError" in broken.details["load_error"]
    assert grade_code(CODE_SPEC, "").details["load_error"] == "empty answer"


RUBRIC = JudgeSpec(
    (Criterion("facts", "d", 2.0), Criterion("tone", "d", 1.0), Criterion("length", "d", 1.0)),
    pass_mark=0.75,
)


def test_rubric_is_weighted_in_code() -> None:
    assert score_rubric(RUBRIC, {"facts": 2, "tone": 2, "length": 2}) == 1.0
    assert score_rubric(RUBRIC, {"facts": 2, "tone": 1, "length": 1}) == pytest.approx(0.75)
    assert score_rubric(RUBRIC, {"facts": 0, "tone": 2, "length": 2}) == pytest.approx(0.5)
    assert grade_from_verdict(RUBRIC, {"facts": 2, "tone": 1, "length": 1}).passed
    assert not grade_from_verdict(RUBRIC, {"facts": 1, "tone": 2, "length": 1}).passed


def test_rubric_rejects_missing_extra_or_out_of_range_scores() -> None:
    with pytest.raises(ValueError):
        score_rubric(RUBRIC, {"facts": 2, "tone": 2})
    with pytest.raises(ValueError):
        score_rubric(RUBRIC, {"facts": 2, "tone": 2, "length": 2, "bonus": 2})
    with pytest.raises(ValueError):
        score_rubric(RUBRIC, {"facts": 3, "tone": 2, "length": 2})
