import os

import pytest

from cascade.sandbox import run_tests

ADD = "def add(a, b):\n    return a + b\n"


def test_passing_and_failing_tests_are_reported_separately() -> None:
    result = run_tests(
        ADD, ["assert add(1, 2) == 3", "assert add(1, 2) == 4", "assert add(0, 0) == 0"]
    )
    assert [outcome.ok for outcome in result.outcomes] == [True, False, True]
    assert result.outcomes[1].error == "AssertionError"
    assert result.passed == 2 and not result.timed_out


def test_an_infinite_loop_fails_its_test_and_the_rest_still_run() -> None:
    code = "def spin():\n    while True:\n        pass\n\ndef ok():\n    return 1\n"
    result = run_tests(code, ["assert ok() == 1", "spin()", "assert ok() == 1"], per_test_s=0.3)
    assert [outcome.ok for outcome in result.outcomes] == [True, False, True]
    assert result.outcomes[1].error == "timeout"


def test_a_loop_that_swallows_the_timeout_is_killed_by_the_wall_clock() -> None:
    code = "def spin():\n    while True:\n        try:\n            while True:\n                pass\n        except BaseException:\n            pass\n"
    result = run_tests(code, ["spin()", "assert True"], timeout_s=1.0, per_test_s=0.2)
    assert result.timed_out
    assert [outcome.ok for outcome in result.outcomes] == [False, False]
    assert result.outcomes[1].error == "timeout"


def test_the_parent_environment_is_not_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-value")
    monkeypatch.setenv("CASCADE_CANARY", "1")
    probe = "import os\nSEEN = sorted(os.environ)\n"
    result = run_tests(
        probe,
        [
            'assert "ANTHROPIC_API_KEY" not in SEEN',
            'assert "CASCADE_CANARY" not in SEEN',
            'assert set(SEEN) <= {"PATH", "LANG", "HOME", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}, SEEN',
        ],
    )
    assert [outcome.ok for outcome in result.outcomes] == [True, True, True], result.outcomes


def test_it_runs_in_a_throwaway_directory_that_is_removed() -> None:
    probe = "import os\nCWD = os.getcwd()\nopen('scratch.txt', 'w').write(CWD)\n"
    result = run_tests(
        probe, ["assert 'cascade-sandbox-' in CWD", f"assert CWD != {os.getcwd()!r}"]
    )
    assert result.passed == 2
    assert not os.path.exists("scratch.txt")


def test_resource_limits_are_applied_inside_the_child() -> None:
    result = run_tests(ADD, ["assert add(1, 1) == 2"])
    assert {"RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_NOFILE", "RLIMIT_CORE"} <= set(result.limits)


def test_runaway_output_is_capped_not_buffered() -> None:
    code = "def noisy():\n    for _ in range(200000):\n        print('x' * 100)\n    return 1\n"
    result = run_tests(code, ["noisy()", "assert True"], timeout_s=4.0)
    # 20 MB of prints against a 1 MB file limit: the process is stopped, nothing explodes here.
    assert result.outcomes[0].ok is False


def test_load_errors_and_exits_fail_every_test() -> None:
    syntax = run_tests("def broken(:\n", ["assert True", "assert True"])
    assert syntax.passed == 0 and "SyntaxError" in (syntax.load_error or "")
    exits = run_tests("import sys\nsys.exit(0)\n", ["assert True"])
    assert exits.passed == 0 and "SystemExit" in (exits.load_error or "")


def test_a_test_cannot_pass_by_exiting() -> None:
    result = run_tests(ADD, ["import sys; sys.exit(0)", "assert add(2, 2) == 4"])
    assert [outcome.ok for outcome in result.outcomes] == [False, True]
