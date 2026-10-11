"""`uv run cascade <command>`: the eval lab's command line.

    cascade tasks --validate      list the suite and check it against its own graders
    cascade estimate              tokens and dollars for a matrix, before spending anything
    cascade run                   run the matrix for real (needs ANTHROPIC_API_KEY)
    cascade simulate              write a clearly labelled synthetic run through the same pipeline
    cascade grade RUN             re-grade the stored answers of a run
    cascade train-router RUN      fit and cross-validate the router on a run
    cascade caching RUN           run (or simulate) the prompt-caching experiment
    cascade export RUN            write the JSON the dashboard bundles

`run` and `simulate` go all the way through grading, router training, and export,
so one command replaces what the dashboard shows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from . import store
from .caching import DEFAULT_CALLS, estimate_caching_usd, run_caching, simulate_caching
from .claude_client import client_for_key, count_tokens
from .config import DEFAULT_EFFORTS, EFFORTS, MODEL_BY_SHORT, MODELS, pricing_snapshot
from .estimate import Estimate, estimate_plan, format_estimate
from .export import REPO_ROOT, export_run, suite_hash
from .graders import GraderUnavailable
from .grading import JudgeMany, grade_rows, opus_judge
from .plan import (
    DEFAULT_LATENCY_SAMPLE,
    DEFAULT_TRIALS,
    Arm,
    PlannedRequest,
    build_arms,
    build_plan,
)
from .prompts import messages_for, system_prompt
from .router import DEFAULT_FOLDS, DEFAULT_L2, DEFAULT_THRESHOLD, train_router
from .runner import BudgetExceeded, BudgetGuard, Executor, run_matrix
from .simulate import DEFAULT_SEED, SIMULATOR_NOTE, annotate, simulate_rows, simulated_judge
from .tasks import Task, load_tasks
from .validate import validate_suite

CONFIRM_ABOVE_USD = 5.0
DEFAULT_BUDGET_USD = 12.0
SYNTHETIC_RUN_ID = "synthetic-v1"

Log = Callable[[str], None]


def load_env() -> None:
    """Pick up ANTHROPIC_API_KEY from a nearby .env file. The key is never printed."""
    root = Path(__file__).resolve().parents[1]
    for candidate in (root / ".env", root.parent / ".env", root.parent.parent / ".env"):
        if os.environ.get("ANTHROPIC_API_KEY") or not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "ANTHROPIC_API_KEY" and value.strip():
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")


def _client() -> anthropic.Anthropic | None:
    load_env()
    return client_for_key(os.environ.get("ANTHROPIC_API_KEY"))


def _parse_models(text: str) -> tuple[str, ...]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in MODEL_BY_SHORT]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"models must be drawn from {sorted(MODEL_BY_SHORT)}")
    return tuple(MODEL_BY_SHORT[name] for name in names)


def _parse_efforts(text: str) -> tuple[str, ...]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in EFFORTS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"efforts must be drawn from {list(EFFORTS)}")
    return tuple(names)


def _matrix_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=MODELS,
        help="comma-separated: haiku,sonnet,opus (default: all three)",
    )
    parser.add_argument(
        "--efforts",
        type=_parse_efforts,
        default=DEFAULT_EFFORTS,
        help="comma-separated: low,medium,high,xhigh,max (default: low,medium,high)",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--mode", choices=["auto", "batch", "sync"], default="auto")
    parser.add_argument("--latency-sample", type=int, default=DEFAULT_LATENCY_SAMPLE)


def _plan_from(
    args: argparse.Namespace, tasks: list[Task]
) -> tuple[list[Arm], list[PlannedRequest]]:
    arms = build_arms(args.models, args.efforts)
    plan = build_plan(
        tasks,
        arms,
        trials=args.trials,
        mode=args.mode,
        latency_sample_per_category=args.latency_sample,
    )
    return arms, plan


def _base_meta(
    run_id: str, tasks: list[Task], arms: list[Arm], plan: list[PlannedRequest], *, synthetic: bool
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "synthetic": synthetic,
        "created_at": store.now_iso(),
        "suite_hash": suite_hash(tasks),
        "n_tasks": len(tasks),
        "n_requests": len(plan),
        "arms": [arm.key for arm in arms],
        "trials": 1 + max(request.trial for request in plan),
        "pricing": pricing_snapshot(),
    }


def finish(
    run_id: str,
    rows: list[store.Row],
    tasks: list[Task],
    meta: dict[str, Any],
    *,
    caching: dict[str, Any] | None,
    threshold: float,
    export: bool,
    root: Path,
    out: Path,
    log: Log,
) -> None:
    """Shared tail of `run` and `simulate`: save, train the router, export."""
    store.write_rows(store.results_path(run_id, root), rows)
    meta = {**meta, "status": "complete"}
    store.write_meta(run_id, meta, root)
    if caching is not None:
        store.write_json(store.run_dir(run_id, root) / "caching.json", caching)
    model = train_router(
        rows, tasks, run_id=run_id, synthetic=bool(meta["synthetic"]), threshold=threshold
    )
    store.write_json(store.run_dir(run_id, root) / "router.json", model)
    headline = model["cv"]["router"]
    tag = " (SIMULATED)" if meta["synthetic"] else ""
    log(
        f"router{tag}: saves {headline['cost_saved']:.1%} of cost versus "
        f"{model['cv']['baseline_arm']} at {headline['quality_retained']:.1%} quality retained "
        f"({model['cv']['folds']}-fold cross-validation over {model['cv']['n_tasks']} tasks)"
    )
    if export:
        paths = export_run(rows, tasks, meta=meta, router_model=model, caching=caching, out=out)
        log(f"exported dashboard data to {paths['summary'].parent} and {paths['tasks']}")


# --- commands -----------------------------------------------------------------------------


def cmd_tasks(args: argparse.Namespace, log: Log = print) -> int:
    tasks = load_tasks()
    if args.category:
        tasks = [task for task in tasks if task.category == args.category]
    for task in tasks:
        log(
            f"{task.id:<28}{task.category:<16}d{task.difficulty}  {task.grader.type:<6} {task.title}"
        )
    log(f"{len(tasks)} tasks")
    if args.validate:
        problems = validate_suite(tasks)
        for problem in problems:
            log(f"PROBLEM {problem.task_id}: {problem.message}")
        if problems:
            return 1
        log("suite is valid: every reference answer passes and every known-wrong answer fails")
    return 0


def _exact_counter(client: anthropic.Anthropic) -> Callable[[Task, Arm], int]:
    cache: dict[tuple[str, str], int] = {}

    def count(task: Task, arm: Arm) -> int:
        key = (task.id, arm.model)
        if key not in cache:
            cache[key] = count_tokens(
                client, model=arm.model, system=system_prompt(task), messages=messages_for(task)
            )
        return cache[key]

    return count


def _estimate(args: argparse.Namespace, tasks: list[Task], plan: list[PlannedRequest]) -> Estimate:
    counter = None
    if getattr(args, "exact", False):
        client = _client()
        if client is None:
            raise SystemExit("--exact needs ANTHROPIC_API_KEY (token counting is free).")
        counter = _exact_counter(client)
    extras = {} if getattr(args, "no_caching", False) else {"caching exp.": estimate_caching_usd()}
    return estimate_plan(plan, tasks, input_tokens=counter, extras=extras)


def cmd_estimate(args: argparse.Namespace, log: Log = print) -> int:
    tasks = load_tasks()
    _, plan = _plan_from(args, tasks)
    estimate = _estimate(args, tasks, plan)
    log(format_estimate(estimate))
    log(
        "\nOutput tokens are assumptions (see cascade/estimate.py), set on the high side. "
        "Batch rows are billed at half price; the latency sample is sent synchronously."
    )
    return 0


def execute_run(
    args: argparse.Namespace,
    client: anthropic.Anthropic,
    *,
    confirm: Callable[[str], str] = input,
    log: Log = print,
    root: Path = store.RUNS_DIR,
    out: Path = REPO_ROOT,
    sleep: Callable[[float], None] | None = None,
) -> int:
    """The real run. Separated from argument parsing so tests can drive it with a fake client."""
    tasks = load_tasks()
    problems = validate_suite(tasks)
    if problems:
        log(f"The suite has {len(problems)} problem(s); run `cascade tasks --validate`.")
        return 1

    run_id = args.resume or args.run_id or f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    resumed: dict[str, Any] = {}
    done: list[store.Row] = []
    if args.resume:
        resumed = store.read_meta(run_id, root)
        if resumed.get("synthetic"):
            log("That run is synthetic; there is nothing to resume.")
            return 1
        if store.results_path(run_id, root).exists():
            done = store.read_rows(store.results_path(run_id, root))

    arms, plan = _plan_from(args, tasks)
    estimate = _estimate(args, tasks, plan)
    log(format_estimate(estimate))
    if estimate.total_usd > args.budget:
        log(
            f"\nRefusing to run: the estimate of ${estimate.total_usd:.2f} is over the "
            f"${args.budget:.2f} budget. Narrow the matrix or raise --budget."
        )
        return 2
    if estimate.total_usd > CONFIRM_ABOVE_USD and not args.yes:
        reply = confirm(f"\nThis run is estimated at ${estimate.total_usd:.2f}. Proceed? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            log("Cancelled. Nothing was sent.")
            return 1

    meta = {
        **_base_meta(run_id, tasks, arms, plan, synthetic=False),
        "status": "running",
        "budget_usd": args.budget,
        "estimate_usd": round(estimate.total_usd, 4),
        "batches": dict(resumed.get("batches", {})),
    }
    if resumed.get("created_at"):
        meta["created_at"] = resumed["created_at"]
    store.write_meta(run_id, meta, root)

    def remember(label: str, batch_id: str) -> None:
        meta["batches"][label] = batch_id
        store.write_meta(run_id, meta, root)

    executor = Executor(
        client,
        concurrency=args.concurrency,
        poll_seconds=args.poll_seconds,
        log=log,
        batch_ids=dict(meta["batches"]),
        on_batch_created=remember,
    )
    if sleep is not None:
        executor.sleep = sleep
    guard = BudgetGuard(args.budget)

    def checkpoint(rows: list[store.Row]) -> None:
        store.write_rows(store.results_path(run_id, root), rows)

    try:
        rows = run_matrix(
            executor, tasks, plan, run_id=run_id, guard=guard, done_rows=done, on_wave=checkpoint
        )
        guard.require(estimate.judge_usd, "the judge")
        rows = grade_rows(rows, tasks, judge_many=opus_judge(executor), log=log)
        guard.record(sum(row["judge_usd"] or 0.0 for row in rows), estimate.judge_usd)
        caching = None
        if not args.no_caching:
            guard.require(estimate_caching_usd(), "the caching experiment")
            log("[caching] running the prompt-caching experiment")
            caching = run_caching(client, nonce=run_id)
            guard.record(sum(c["total_usd"] for c in caching["conditions"]), estimate_caching_usd())
    except BudgetExceeded as stop:
        log(str(stop))
        log(f"Partial results are in {store.run_dir(run_id, root)}; resume with --resume {run_id}.")
        return 2
    except KeyboardInterrupt:
        executor.stop.set()
        log(f"\nInterrupted. Resume with: cascade run --resume {run_id}")
        return 130

    meta["spent_usd"] = round(guard.spent_usd, 4)
    finish(
        run_id,
        rows,
        tasks,
        meta,
        caching=caching,
        threshold=args.threshold,
        export=not args.no_export,
        root=root,
        out=out,
        log=log,
    )
    log(f"run {run_id} complete: {len(rows)} rows, ${guard.spent_usd:.2f} spent")
    return 0


def cmd_run(args: argparse.Namespace, log: Log = print) -> int:
    client = _client()
    if client is None:
        log(
            "ANTHROPIC_API_KEY is not set. Add it to the environment or to ~/portfolio/.env.\n"
            "To exercise the pipeline without a key, use `cascade simulate`."
        )
        return 2
    return execute_run(args, client, log=log)


def cmd_simulate(
    args: argparse.Namespace,
    log: Log = print,
    *,
    root: Path = store.RUNS_DIR,
    out: Path = REPO_ROOT,
) -> int:
    tasks = load_tasks()
    arms, plan = _plan_from(args, tasks)
    run_id = args.run_id
    rows = simulate_rows(plan, tasks, run_id=run_id, seed=args.seed)
    rows = grade_rows(rows, tasks, judge_many=simulated_judge(args.seed), log=lambda _: None)
    annotate(rows, plan, tasks, args.seed)
    meta = {
        **_base_meta(run_id, tasks, arms, plan, synthetic=True),
        "generator": {"name": "cascade simulate", "seed": args.seed, "note": SIMULATOR_NOTE},
    }
    log(
        f"SYNTHETIC run {run_id}: {len(rows)} simulated rows (seed {args.seed}). Not a measurement."
    )
    finish(
        run_id,
        rows,
        tasks,
        meta,
        caching=simulate_caching(),
        threshold=args.threshold,
        export=not args.no_export,
        root=root,
        out=out,
        log=log,
    )
    return 0


def _load_run(run_id: str, root: Path) -> tuple[list[store.Row], dict[str, Any], list[Task]]:
    return (
        store.read_rows(store.results_path(run_id, root)),
        store.read_meta(run_id, root),
        load_tasks(),
    )


def cmd_grade(args: argparse.Namespace, log: Log = print, *, root: Path = store.RUNS_DIR) -> int:
    rows, meta, tasks = _load_run(args.run, root)
    judged = {row["custom_id"]: row for row in rows if row["grader"] == "judge"}
    judge: JudgeMany
    if args.rejudge:
        if meta.get("synthetic"):
            seed = (meta.get("generator") or {}).get("seed", DEFAULT_SEED)
            judge = simulated_judge(seed)
        else:
            client = _client()
            if client is None:
                log("--rejudge needs ANTHROPIC_API_KEY.")
                return 2
            judge = opus_judge(Executor(client, log=log))
    else:
        # Keep the judge's earlier verdicts; only the deterministic graders run again.
        def judge(items: list[tuple[store.Row, Task]]) -> dict[str, Any]:
            from .graders import Grade

            kept: dict[str, Any] = {}
            for row, _task in items:
                old = judged[row["custom_id"]]
                grade = None
                if old["score"] is not None:
                    grade = Grade(
                        old["score"], bool(old["passed"]), json.loads(old["grader_details"])
                    )
                kept[row["custom_id"]] = (grade, old["judge_usd"] or 0.0)
            return kept

    try:
        graded = grade_rows(rows, tasks, judge_many=judge, log=log)
    except GraderUnavailable as missing:
        log(str(missing))
        return 2
    store.write_rows(store.results_path(args.run, root), graded)
    passed = sum(1 for row in graded if row["passed"])
    log(f"graded {len(graded)} rows: {passed} passed")
    return 0


def cmd_train_router(
    args: argparse.Namespace, log: Log = print, *, root: Path = store.RUNS_DIR
) -> int:
    rows, meta, tasks = _load_run(args.run, root)
    model = train_router(
        rows,
        tasks,
        run_id=args.run,
        synthetic=bool(meta["synthetic"]),
        threshold=args.threshold,
        l2=args.l2,
        folds=args.folds,
    )
    store.write_json(store.run_dir(args.run, root) / "router.json", model)
    cv = model["cv"]
    tag = " (SIMULATED DATA)" if meta["synthetic"] else ""
    log(f"router {model['version']}{tag}, threshold {model['threshold']}")
    log(
        f"  cost saved versus {cv['baseline_arm']}: {cv['router']['cost_saved']:.1%}   "
        f"quality retained: {cv['router']['quality_retained']:.1%}"
    )
    confusion = cv["confusion"]
    log(
        f"  predicted versus oracle arm: {confusion['exact']} exact, "
        f"{confusion['over_routed']} over-routed, {confusion['under_routed']} under-routed "
        f"of {cv['n_tasks']} tasks"
    )
    log("  threshold sweep (threshold, cost saved, quality retained):")
    for point in cv["sweep"]:
        log(
            f"    {point['threshold']:.2f}  {point['cost_saved']:>7.1%}  {point['quality_retained']:>7.1%}"
        )
    return 0


def cmd_caching(args: argparse.Namespace, log: Log = print, *, root: Path = store.RUNS_DIR) -> int:
    if args.simulate:
        report = simulate_caching(args.calls)
    else:
        client = _client()
        if client is None:
            log("ANTHROPIC_API_KEY is not set. Use --simulate to see the experiment's shape.")
            return 2
        log(f"Estimated cost: ${estimate_caching_usd(args.calls):.2f}")
        report = run_caching(
            client, nonce=f"{args.run}-{datetime.now(UTC):%H%M%S}", calls=args.calls
        )
    store.write_json(store.run_dir(args.run, root) / "caching.json", report)
    tag = " (SIMULATED)" if report["synthetic"] else ""
    log(
        f"caching{tag}: {report['calls']} calls over a {report['prefix_tokens']}-token prefix, "
        f"saves ${report['savings_usd']:.4f} ({report['savings_pct']:.1%}); "
        f"pays for itself from call {report['break_even_call']}"
    )
    return 0


def cmd_export(args: argparse.Namespace, log: Log = print, *, root: Path = store.RUNS_DIR) -> int:
    rows, meta, tasks = _load_run(args.run, root)
    directory = store.run_dir(args.run, root)
    router_path = directory / "router.json"
    if not router_path.exists():
        log("No router.json for this run. Run `cascade train-router` first.")
        return 1
    model = json.loads(router_path.read_text(encoding="utf-8"))
    caching_path = directory / "caching.json"
    caching = (
        json.loads(caching_path.read_text(encoding="utf-8")) if caching_path.exists() else None
    )
    paths = export_run(
        rows, tasks, meta=meta, router_model=model, caching=caching, out=Path(args.out)
    )
    log(f"wrote {paths['summary']}, {paths['router']} and {paths['tasks']}/")
    return 0


# --- argument parsing ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cascade", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    tasks = commands.add_parser("tasks", help="list and validate the suite")
    tasks.add_argument("--validate", action="store_true")
    tasks.add_argument("--category")
    tasks.set_defaults(handler=cmd_tasks)

    estimate = commands.add_parser("estimate", help="token and dollar estimate for a matrix")
    _matrix_flags(estimate)
    estimate.add_argument("--exact", action="store_true", help="count input tokens with the API")
    estimate.add_argument("--no-caching", action="store_true")
    estimate.set_defaults(handler=cmd_estimate)

    run = commands.add_parser("run", help="run the matrix for real")
    _matrix_flags(run)
    run.add_argument("--budget", type=float, default=DEFAULT_BUDGET_USD)
    run.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    run.add_argument("--run-id")
    run.add_argument("--resume", metavar="RUN", help="continue an interrupted run")
    run.add_argument("--exact", action="store_true", help="count input tokens with the API")
    run.add_argument("--no-caching", action="store_true")
    run.add_argument("--no-export", action="store_true")
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--poll-seconds", type=float, default=30.0)
    run.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    run.set_defaults(handler=cmd_run)

    simulate = commands.add_parser("simulate", help="write a labelled synthetic run")
    _matrix_flags(simulate)
    simulate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    simulate.add_argument("--run-id", default=SYNTHETIC_RUN_ID)
    simulate.add_argument("--no-export", action="store_true")
    simulate.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    simulate.set_defaults(handler=cmd_simulate)

    grade = commands.add_parser("grade", help="re-grade a run's stored answers")
    grade.add_argument("run")
    grade.add_argument("--rejudge", action="store_true", help="also re-run the Opus judge")
    grade.set_defaults(handler=cmd_grade)

    train = commands.add_parser("train-router", help="fit and cross-validate the router")
    train.add_argument("run")
    train.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    train.add_argument("--l2", type=float, default=DEFAULT_L2)
    train.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    train.set_defaults(handler=cmd_train_router)

    caching = commands.add_parser("caching", help="prompt-caching experiment")
    caching.add_argument("run")
    caching.add_argument("--simulate", action="store_true")
    caching.add_argument("--calls", type=int, default=DEFAULT_CALLS)
    caching.set_defaults(handler=cmd_caching)

    export = commands.add_parser("export", help="write the dashboard JSON")
    export.add_argument("run")
    export.add_argument("--out", default=str(REPO_ROOT))
    export.set_defaults(handler=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
