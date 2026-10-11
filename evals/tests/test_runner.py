import json

import pytest

from cascade.claude_client import BatchRequest, build_params, submit_batch
from cascade.config import HAIKU, OPUS, SONNET, Usage
from cascade.estimate import estimate_plan
from cascade.plan import Arm, build_arms, build_plan
from cascade.runner import (
    ApiRequest,
    BudgetExceeded,
    BudgetGuard,
    Executor,
    request_for,
    run_matrix,
    waves,
)
from cascade.tasks import load_tasks, task_index
from tests.fakes import ScriptedClient, text_message

TASKS = load_tasks()
INDEX = task_index(TASKS)
QUIET = lambda _message: None  # noqa: E731


def echo(params: dict) -> object:
    """Answers with the user prompt, so a response can be matched to its request."""
    prompt = params["messages"][0]["content"]
    return text_message(f"echo:{prompt}", model=params["model"], input_tokens=100, output_tokens=40)


def executor_for(client: ScriptedClient, **kwargs: object) -> Executor:
    return Executor(client, log=QUIET, sleep=lambda _s: None, **kwargs)  # type: ignore[arg-type]


def requests(n: int, mode: str) -> list[ApiRequest]:
    return [
        ApiRequest(
            f"id-{i}", HAIKU, "low", "system", [{"role": "user", "content": f"q{i}"}], 4000, mode
        )  # type: ignore[arg-type]
        for i in range(n)
    ]


def test_batch_results_are_matched_by_custom_id_not_by_order() -> None:
    client = ScriptedClient(echo)
    responses = executor_for(client).run(requests(6, "batch"), label="t")
    # The fake returns results in reverse order; every response must still be its own.
    assert {cid: r.text for cid, r in responses.items()} == {
        f"id-{i}": f"echo:q{i}" for i in range(6)
    }
    assert all(r.mode == "batch" and r.latency_ms is None for r in responses.values())
    assert client.polls["batch_1"] == 2  # polled until it ended
    assert client.sync_requests == []


def test_batch_requests_follow_the_api_rules() -> None:
    client = ScriptedClient(echo)
    batch = [
        ApiRequest("a", OPUS, "high", "sys", [{"role": "user", "content": "q"}], 16000, "batch"),
        ApiRequest("b", SONNET, "low", "sys", [{"role": "user", "content": "q"}], 4000, "batch"),
    ]
    executor_for(client).run(batch, label="t")
    sent = client.batches["batch_1"]
    assert [entry["custom_id"] for entry in sent] == ["a", "b"]
    for entry in sent:
        params = entry["params"]
        assert params["thinking"] == {"type": "adaptive"}
        assert "effort" in params["output_config"]
        assert "fallbacks" not in params and "betas" not in params and "tool_choice" not in params
        assert "budget_tokens" not in json.dumps(params)
    assert sent[0]["params"]["output_config"]["effort"] == "high"
    assert sent[0]["params"]["max_tokens"] == 16000


def test_failed_expired_and_missing_batch_items_become_error_rows() -> None:
    outcomes = {"id-1": "errored", "id-2": "expired", "id-3": "missing"}
    client = ScriptedClient(echo, batch_outcomes=outcomes)
    responses = executor_for(client).run(requests(5, "batch"), label="t")
    assert responses["id-0"].status == "ok"
    assert (responses["id-1"].status, responses["id-1"].error) == ("error", "overloaded_error")
    assert (responses["id-2"].status, responses["id-2"].error) == ("error", "expired")
    assert (responses["id-3"].status, responses["id-3"].error) == (
        "error",
        "missing from batch results",
    )
    assert responses["id-1"].billed_usd == 0.0


def test_sync_path_is_timed_and_never_uses_the_refusal_fallback() -> None:
    client = ScriptedClient(echo)
    request = ApiRequest(
        "s", OPUS, "medium", "sys", [{"role": "user", "content": "q"}], 8000, "sync"
    )
    response = executor_for(client).run([request], label="t")["s"]
    assert response.mode == "sync" and response.latency_ms is not None and response.latency_ms >= 0
    sent = client.sync_requests[0]
    assert "fallbacks" not in sent and "betas" not in sent
    assert sent["output_config"] == {"effort": "medium"}
    # Sync rows are billed at list price, batch rows at half.
    assert response.billed_usd == pytest.approx(response.list_usd)


def test_refusals_truncation_and_api_errors_are_statuses_not_exceptions() -> None:
    def moody(params: dict) -> object:
        prompt = params["messages"][0]["content"]
        if prompt == "q0":
            return text_message("", stop_reason="refusal")
        if prompt == "q1":
            return text_message("cut off", stop_reason="max_tokens")
        raise RuntimeError("network down")

    responses = executor_for(ScriptedClient(moody)).run(requests(3, "sync"), label="t")
    assert [responses[f"id-{i}"].status for i in range(3)] == ["refusal", "truncated", "error"]
    assert responses["id-2"].error == "Something went wrong."


def test_mixed_plan_submits_the_batch_and_runs_the_sample_synchronously() -> None:
    client = ScriptedClient(echo)
    mixed = requests(4, "batch")[:2] + requests(4, "sync")[2:]
    responses = executor_for(client).run(mixed, label="t")
    assert len(client.batches["batch_1"]) == 2 and len(client.sync_requests) == 2
    assert {r.mode for r in responses.values()} == {"batch", "sync"}


def test_resume_polls_the_recorded_batch_instead_of_creating_a_new_one() -> None:
    client = ScriptedClient(echo)
    created: list[tuple[str, str]] = []
    first = executor_for(client, on_batch_created=lambda label, bid: created.append((label, bid)))
    first.run(requests(2, "batch"), label="answers-haiku")
    assert created == [("answers-haiku", "batch_1")]

    resumed = executor_for(client, batch_ids={"answers-haiku": "batch_1"})
    responses = resumed.run(requests(2, "batch"), label="answers-haiku")
    assert list(client.batches) == ["batch_1"]  # nothing new was submitted (or billed)
    assert responses["id-0"].text == "echo:q0"


def test_submit_batch_refuses_what_a_batch_cannot_do() -> None:
    client = ScriptedClient(echo)
    good = {
        **build_params(model=OPUS, effort="low", max_tokens=1000, fallback=False),
        "messages": [],
    }
    with pytest.raises(ValueError, match="Duplicate"):
        submit_batch(client, [BatchRequest("a", good), BatchRequest("a", good)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid custom_id"):
        submit_batch(client, [BatchRequest("has space", good)])  # type: ignore[arg-type]
    with_fallback = {**build_params(model=OPUS, effort="low", max_tokens=1000), "messages": []}
    with pytest.raises(ValueError, match="fallback"):
        submit_batch(client, [BatchRequest("a", with_fallback)])  # type: ignore[arg-type]
    forced = {**good, "tool_choice": {"type": "any"}}
    with pytest.raises(ValueError, match="Forced tool choice"):
        submit_batch(client, [BatchRequest("a", forced)])  # type: ignore[arg-type]
    no_effort = {**good, "output_config": {}}
    with pytest.raises(ValueError, match="effort"):
        submit_batch(client, [BatchRequest("a", no_effort)])  # type: ignore[arg-type]
    assert client.batches == {}


def test_budget_guard_rescales_the_remaining_estimate_by_measured_spend() -> None:
    guard = BudgetGuard(10.0)
    guard.require(9.0, "wave one")  # estimate fits
    guard.record(spent_usd=3.0, estimated_usd=2.0)  # running 1.5x over
    assert guard.ratio == pytest.approx(1.5)
    assert guard.projected(4.0) == pytest.approx(9.0)
    guard.require(4.0, "wave two")
    with pytest.raises(BudgetExceeded, match="wave three"):
        guard.require(5.0, "wave three")  # 3 + 1.5 * 5 = 10.5


def test_a_cheap_wave_never_relaxes_the_cap() -> None:
    guard = BudgetGuard(10.0)
    guard.record(spent_usd=0.5, estimated_usd=2.0)
    assert guard.ratio == 1.0
    with pytest.raises(BudgetExceeded):
        guard.require(9.6, "the rest")


def test_run_matrix_goes_cheapest_tier_first_and_stops_before_overspending() -> None:
    tasks = TASKS[:4]
    plan = build_plan(tasks, build_arms(), trials=1, mode="sync")
    assert [name for name, _ in waves(plan)] == ["haiku", "sonnet", "opus"]

    def pricey(params: dict) -> object:
        # Every answer uses far more output than the estimate assumed.
        return text_message(
            "ANSWER: x", model=params["model"], input_tokens=200, output_tokens=60_000
        )

    client = ScriptedClient(pricey)
    estimate = estimate_plan(plan, tasks).answers_usd
    guard = BudgetGuard(estimate * 1.5)
    checkpoints: list[int] = []
    with pytest.raises(BudgetExceeded, match="sonnet wave"):
        run_matrix(
            executor_for(client),
            tasks,
            plan,
            run_id="r",
            guard=guard,
            on_wave=lambda rows: checkpoints.append(len(rows)),
        )
    # Only the Haiku wave ran; its rows were checkpointed; no Sonnet or Opus call was made.
    assert checkpoints == [12]
    assert {request["model"] for request in client.sync_requests} == {HAIKU}


def test_run_matrix_keeps_rows_from_a_resumed_run() -> None:
    tasks = TASKS[:2]
    plan = build_plan(tasks, [Arm(HAIKU, "low"), Arm(SONNET, "low")], trials=1, mode="sync")
    client = ScriptedClient(echo)
    rows = run_matrix(executor_for(client), tasks, plan, run_id="r", guard=BudgetGuard(5.0))
    assert len(rows) == 4 and len(client.sync_requests) == 4

    done = [row for row in rows if row["model"] == HAIKU]
    again = ScriptedClient(echo)
    resumed = run_matrix(
        executor_for(again), tasks, plan, run_id="r", guard=BudgetGuard(5.0), done_rows=done
    )
    assert len(resumed) == 4
    assert {request["model"] for request in again.sync_requests} == {
        SONNET
    }  # Haiku was not re-sent


def test_rows_record_list_price_and_billed_price_separately() -> None:
    task = TASKS[0]
    plan = build_plan(
        [task], [Arm(OPUS, "low")], trials=2, mode="batch", latency_sample_per_category=0
    )
    client = ScriptedClient(echo)
    rows = run_matrix(executor_for(client), [task], plan, run_id="r", guard=BudgetGuard(5.0))
    usage = Usage(100, 40)
    for row in rows:
        assert row["mode"] == "batch" and row["latency_ms"] is None and row["synthetic"] is False
        assert row["cost_usd"] == pytest.approx(
            (usage.input_tokens * 4 + usage.output_tokens * 20) / 1e6
        )
        assert row["billed_usd"] == pytest.approx(row["cost_usd"] / 2)
        assert row["score"] is None  # grading is a separate step


def test_the_request_sent_for_a_task_is_its_prompt_verbatim() -> None:
    task = INDEX["cls-ticket-billing"]
    planned = build_plan([task], [Arm(HAIKU, "low")], trials=1, mode="sync")[0]
    request = request_for(planned, task)
    assert request.messages == [{"role": "user", "content": task.prompt}]
    assert "ANSWER:" in request.system and "ANSWER:" not in task.prompt
