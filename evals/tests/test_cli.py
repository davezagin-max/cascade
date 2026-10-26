import json
from pathlib import Path

import pytest

from cascade import cli, store
from cascade.caching import HANDBOOK
from cascade.config import HAIKU, OPUS
from cascade.estimate import approx_tokens
from cascade.judge import JUDGE_SYSTEM
from cascade.tasks import load_tasks
from tests.fakes import ScriptedClient, text_message

TASKS = load_tasks()
BY_PROMPT = {task.prompt: task for task in TASKS}


def fake_api(params: dict) -> object:
    """Behaves like the API for the three kinds of request a run makes."""
    system = params["system"][0]["text"]
    prompt = params["messages"][0]["content"]
    model = params["model"]
    if system == JUDGE_SYSTEM:
        task = next(t for t in TASKS if t.grader.type == "judge" and t.prompt in prompt)
        strong = task.reference in prompt
        criteria = [
            {"name": c.name, "score": 2 if strong else 0, "rationale": "checked"}
            for c in task.grader.rubric  # type: ignore[union-attr]
        ]
        return text_message(json.dumps({"criteria": criteria}), input_tokens=900, output_tokens=300)
    if system.startswith("Handbook revision:"):
        cached = "cache_control" in params["system"][0] and HANDBOOK in system
        tokens = approx_tokens(system)
        usage = {"input_tokens": 30, "output_tokens": 60}
        if not cached:
            usage["input_tokens"] += tokens
        elif "cached" in system.split("\n")[0] and prompt.startswith("How long do I have"):
            usage["cache_creation_input_tokens"] = tokens
        else:
            usage["cache_read_input_tokens"] = tokens
        return text_message("Sixty days.", **usage)
    task = BY_PROMPT[prompt]
    # The small model gets the hard tasks wrong; the large one gets everything right.
    right = model == OPUS or task.difficulty <= 2
    return text_message(
        task.reference if right else task.wrong, model=model, input_tokens=200, output_tokens=150
    )


def parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_run_without_a_key_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_client", lambda: None)
    lines: list[str] = []
    assert cli.cmd_run(parse("run"), log=lines.append) == 2
    assert "ANTHROPIC_API_KEY is not set" in lines[0] and "cascade simulate" in lines[0]


def test_run_refuses_an_estimate_over_the_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "validate_suite", lambda _tasks: [])
    client = ScriptedClient(fake_api)
    lines: list[str] = []
    code = cli.execute_run(
        parse("run", "--budget", "2"),
        client,
        log=lines.append,
        root=tmp_path,
        out=tmp_path,  # type: ignore[arg-type]
    )
    assert code == 2
    assert any("Refusing to run" in line and "$2.00 budget" in line for line in lines)
    assert client.sync_requests == [] and client.batches == {}
    assert not (tmp_path / "run.json").exists()


def test_run_asks_before_spending_more_than_five_dollars(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "validate_suite", lambda _tasks: [])
    client = ScriptedClient(fake_api)
    asked: list[str] = []

    def decline(question: str) -> str:
        asked.append(question)
        return "n"

    lines: list[str] = []
    code = cli.execute_run(
        parse("run"),
        client,
        confirm=decline,
        log=lines.append,
        root=tmp_path,
        out=tmp_path,  # type: ignore[arg-type]
    )
    assert code == 1
    assert len(asked) == 1 and "Proceed?" in asked[0] and "$9." in asked[0]
    assert "Cancelled. Nothing was sent." in lines
    assert client.sync_requests == [] and client.batches == {}


def test_a_rejected_key_stops_the_run_with_a_plain_message(tmp_path: Path) -> None:
    import anthropic

    secret = "sk-ant-" + "x" * 24
    client = ScriptedClient(fake_api)

    def rejected(**_params: object) -> object:
        # Built without its HTTP response: only the type and the message matter here.
        error = anthropic.AuthenticationError.__new__(anthropic.AuthenticationError)
        error.args = (f"invalid x-api-key {secret}",)
        raise error

    client.messages.batches.create = rejected
    lines: list[str] = []
    code = cli.execute_run(
        parse("run", "--yes", "--no-caching", "--run-id", "bad-key"),
        client,  # type: ignore[arg-type]
        log=lines.append,
        root=tmp_path,
        out=tmp_path,
        sleep=lambda _s: None,
    )
    assert code == 2
    printed = "\n".join(lines)
    assert "Stopped: Anthropic rejected that API key." in printed
    assert "cascade run --resume bad-key" in printed
    # Whatever the SDK put in its own message, the key is never echoed.
    assert secret not in printed
    assert not (tmp_path / "data" / "summary.json").exists()


def test_a_run_refuses_to_start_on_a_broken_suite(monkeypatch, tmp_path: Path) -> None:
    from cascade.validate import Problem

    monkeypatch.setattr(cli, "validate_suite", lambda _tasks: [Problem("x", "bad key")])
    client = ScriptedClient(fake_api)
    assert (
        cli.execute_run(
            parse("run", "--yes"), client, log=lambda _m: None, root=tmp_path, out=tmp_path
        )
        == 1
    )  # type: ignore[arg-type]
    assert client.batches == {}


def test_full_run_with_a_fake_client_produces_a_real_non_synthetic_dashboard(
    tmp_path: Path,
) -> None:
    """The live path end to end: batches, the timed sample, the judge, caching, router, export."""
    client = ScriptedClient(fake_api)
    lines: list[str] = []
    root, out = tmp_path / "runs", tmp_path / "site"
    args = parse(
        "run", "--models", "haiku,opus", "--efforts", "low", "--trials", "1",
        "--mode", "batch", "--latency-sample", "1", "--run-id", "run-test", "--threshold", "0.8",
    )  # fmt: skip
    code = cli.execute_run(
        args,
        client,
        log=lines.append,
        root=root,
        out=out,
        sleep=lambda _s: None,  # type: ignore[arg-type]
    )
    assert code == 0, lines

    # One batch per model wave, plus the judge; the latency sample went out synchronously.
    answers = [b for b in client.batches.values() if not b[0]["custom_id"].startswith("j__")]
    assert sorted(len(batch) for batch in answers) == [58, 58]
    assert {batch[0]["params"]["model"] for batch in answers} == {HAIKU, OPUS}

    rows = store.read_rows(store.results_path("run-test", root))
    meta = store.read_meta("run-test", root)
    assert len(rows) == 2 * len(TASKS)
    assert meta["synthetic"] is False and meta["status"] == "complete" and meta["spent_usd"] > 0
    assert set(meta["batches"]) == {"answers-haiku", "answers-opus"}
    assert meta["matrix"] == {
        "models": ["haiku", "opus"], "efforts": ["low"], "trials": 1,
        "mode": "batch", "latency_sample": 1,
    }  # fmt: skip
    assert all(row["synthetic"] is False and row["passed"] is not None for row in rows)
    assert sum(1 for row in rows if row["mode"] == "sync") == 10
    assert all((row["latency_ms"] is not None) == (row["mode"] == "sync") for row in rows)

    # Grades came from the real graders: Opus passed everything, Haiku only the easy tasks.
    by_model = {m: [r for r in rows if r["model"] == m] for m in (HAIKU, OPUS)}
    assert all(row["passed"] for row in by_model[OPUS])
    easy = {task.id for task in TASKS if task.difficulty <= 2}
    assert all(row["passed"] is (row["task_id"] in easy) for row in by_model[HAIKU])
    judged = [row for row in rows if row["grader"] == "judge"]
    assert len(judged) == 12 and all(row["judge_usd"] > 0 for row in judged)

    summary = json.loads((out / "data" / "summary.json").read_text())
    assert summary["meta"]["synthetic"] is False and summary["meta"]["generator"] is None
    assert [arm["key"] for arm in summary["arms"]] == ["haiku-low", "opus-low"]
    assert summary["arms"][1]["quality"] == 1.0
    assert summary["caching"]["synthetic"] is False and summary["caching"]["savings_usd"] > 0
    assert summary["router"]["baseline_arm"] == "opus-low"
    assert 0 <= summary["router"]["headline"]["cost_saved"] <= 1
    assert len(list((out / "public" / "run" / "tasks").glob("*.json"))) == len(TASKS)
    router = json.loads((out / "data" / "router.json").read_text())
    assert router["trained_on"] == {
        "run_id": "run-test", "synthetic": False, "n_tasks": len(TASKS), "n_rows": len(rows),
        "skipped_tasks": [],
    }  # fmt: skip
    # The penalty was selected from the grid, and the harder baseline is reported.
    assert summary["router"]["l2_selection"]["grid"] and len(summary["router"]["l2_by_fold"]) == 5
    assert summary["router"]["headline"]["best_fixed"]["arm"] in ("haiku-low", "opus-low")
    assert any("run run-test complete" in line for line in lines)

    # Resuming a finished run sends nothing new, and keeps its own matrix whatever is passed.
    sent = (len(client.sync_requests), len(client.batches))
    resume = parse(
        "run", "--resume", "run-test", "--models", "sonnet", "--no-caching", "--no-export"
    )
    assert (
        cli.execute_run(
            resume,
            client,
            log=lambda _m: None,
            root=root,
            out=out,
            sleep=lambda _s: None,  # type: ignore[arg-type]
        )
        == 0
    )
    assert (len(client.sync_requests), len(client.batches)) == sent  # not even the judge
    rows = store.read_rows(store.results_path("run-test", root))
    assert len(rows) == 2 * len(TASKS) and {row["model"] for row in rows} == {HAIKU, OPUS}

    # Re-grading the stored answers gives the same grades without calling the judge again.
    before = len(client.sync_requests)
    assert cli.cmd_grade(parse("grade", "run-test"), log=lambda _m: None, root=root) == 0
    assert len(client.sync_requests) == before
    assert store.read_rows(store.results_path("run-test", root)) == rows

    # The other stage commands work on the saved run.
    assert (
        cli.cmd_train_router(parse("train-router", "run-test"), log=lambda _m: None, root=root) == 0
    )
    args = parse("export", "run-test", "--out", str(tmp_path / "again"))
    assert cli.cmd_export(args, log=lambda _m: None, root=root) == 0
    assert (tmp_path / "again" / "data" / "summary.json").exists()


def test_simulate_command_writes_a_labelled_synthetic_run(tmp_path: Path) -> None:
    lines: list[str] = []
    args = parse("simulate", "--trials", "1", "--seed", "3", "--run-id", "sim-test")
    assert (
        cli.cmd_simulate(args, log=lines.append, root=tmp_path / "runs", out=tmp_path / "site") == 0
    )
    assert lines[0].startswith("SYNTHETIC run sim-test") and "Not a measurement" in lines[0]
    assert "(SIMULATED)" in lines[1]
    summary = json.loads((tmp_path / "site" / "data" / "summary.json").read_text())
    assert summary["meta"]["synthetic"] is True and summary["meta"]["generator"]["seed"] == 3
    assert summary["caching"]["synthetic"] is True
    assert (
        json.loads((tmp_path / "site" / "data" / "router.json").read_text())["trained_on"][
            "synthetic"
        ]
        is True
    )


def test_tasks_and_estimate_commands() -> None:
    lines: list[str] = []
    assert cli.cmd_tasks(parse("tasks", "--category", "reasoning"), log=lines.append) == 0
    assert lines[-1] == "16 tasks"
    lines.clear()
    # A misspelt category is named as one, not reported as an empty suite.
    assert cli.cmd_tasks(parse("tasks", "--category", "reasonng"), log=lines.append) == 1
    assert lines == [
        "No category named reasonng. Categories: classification, code, extraction, "
        "open_ended, reasoning."
    ]
    lines.clear()
    assert (
        cli.cmd_estimate(parse("estimate", "--models", "haiku", "--trials", "1"), log=lines.append)
        == 0
    )
    assert "haiku-low" in lines[0] and "TOTAL (est.)" in lines[0]
    with pytest.raises(SystemExit):
        parse("estimate", "--models", "gpt")


def test_regrading_a_synthetic_run_changes_nothing(tmp_path: Path) -> None:
    """`cascade grade` is idempotent: same answers, same graders, same rows."""
    root = tmp_path / "runs"
    args = parse("simulate", "--trials", "1", "--run-id", "sim-regrade", "--no-export")
    assert cli.cmd_simulate(args, log=lambda _m: None, root=root, out=tmp_path) == 0
    before = store.read_rows(store.results_path("sim-regrade", root))
    assert cli.cmd_grade(parse("grade", "sim-regrade"), log=lambda _m: None, root=root) == 0
    assert store.read_rows(store.results_path("sim-regrade", root)) == before
    assert (
        cli.cmd_grade(parse("grade", "sim-regrade", "--rejudge"), log=lambda _m: None, root=root)
        == 0
    )
    assert store.read_rows(store.results_path("sim-regrade", root)) == before


def test_commands_on_a_missing_run_say_so_in_one_line(tmp_path: Path) -> None:
    """A run name that does not exist gets a plain message and exit 1, never a traceback."""
    root = tmp_path / "runs"
    assert (
        cli.cmd_simulate(
            parse("simulate", "--trials", "1", "--run-id", "real-one", "--no-export"),
            log=lambda _m: None,
            root=root,
            out=tmp_path,
        )
        == 0
    )
    for argv, handler in (
        (("grade", "nope"), cli.cmd_grade),
        (("train-router", "nope"), cli.cmd_train_router),
        (("export", "nope", "--out", str(tmp_path / "site")), cli.cmd_export),
        (("caching", "nope", "--simulate"), cli.cmd_caching),
    ):
        lines: list[str] = []
        assert handler(parse(*argv), log=lines.append, root=root) == 1, argv
        assert lines == [f"No run named nope in {root}. Runs found: real-one."], argv
    # Nothing was written for the missing run, so no empty folder is left behind.
    assert not (root / "nope").exists()
    assert not (tmp_path / "site").exists()
