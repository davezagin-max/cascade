"""Executes a plan against the API and turns responses into result rows.

Two paths, one result shape:

- **Batch** (default for anything but a tiny run): half price. Requests go out with
  a `custom_id`, the batch is polled until it ends, and results are matched back
  by `custom_id`, never by position, because a batch returns them in any order.
- **Sync**: full price, but it is the only way to time a request, so the latency
  sample always uses it, and small runs use it for everything.

Neither path uses the refusal fallback: an eval must record what the arm under
test did, and a fallback would answer with a different model.

The run is split into waves, cheapest model tier first. After each wave the
measured spend is compared with what was estimated for that wave, and the ratio
rescales the estimate for the waves still to come. If the projection exceeds the
budget the run stops before the next wave is sent, keeping what it has.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import anthropic
from pydantic import BaseModel

from .claude_client import (
    BatchRequest,
    api_error,
    batch_results,
    batch_status,
    build_params,
    call_claude,
    read_message,
    submit_batch,
)
from .config import MODELS, SHORT_NAMES, Effort, Usage, cost_usd
from .estimate import estimate_plan
from .plan import Mode, PlannedRequest
from .prompts import messages_for, system_prompt
from .store import Row
from .tasks import Task, task_index

Log = Callable[[str], None]


@dataclass(frozen=True)
class ApiRequest:
    custom_id: str
    model: str
    effort: Effort
    system: str
    messages: list[dict[str, Any]]
    max_tokens: int
    mode: Mode
    schema: type[BaseModel] | None = None


@dataclass(frozen=True)
class ApiResponse:
    custom_id: str
    status: str  # ok | truncated | refusal | error
    text: str
    stop_reason: str | None
    usage: Usage
    model: str
    mode: Mode
    latency_ms: float | None = None
    error: str | None = None

    @property
    def list_usd(self) -> float:
        return cost_usd(self.model, self.usage)

    @property
    def billed_usd(self) -> float:
        return cost_usd(self.model, self.usage, batch=self.mode == "batch")


def _status(stop_reason: str | None, refused: bool) -> str:
    if refused:
        return "refusal"
    if stop_reason == "max_tokens":
        return "truncated"
    return "ok" if stop_reason == "end_turn" else "error"


def _failed(request: ApiRequest, reason: str) -> ApiResponse:
    return ApiResponse(
        custom_id=request.custom_id,
        status="error",
        text="",
        stop_reason=None,
        usage=Usage(),
        model=request.model,
        mode=request.mode,
        error=reason,
    )


def _send_sync(client: anthropic.Anthropic, request: ApiRequest) -> ApiResponse:
    started = time.perf_counter()
    try:
        result = call_claude(
            client,
            model=request.model,
            effort=request.effort,
            system=request.system,
            messages=request.messages,
            max_tokens=request.max_tokens,
            schema=request.schema,
            fallback=False,
        )
    except Exception as error:  # one failed request must not sink the run
        return _failed(request, api_error(error)[1])
    return ApiResponse(
        custom_id=request.custom_id,
        status=_status(result.stop_reason, result.refusal is not None),
        text=result.text,
        stop_reason=result.stop_reason,
        usage=result.usage,
        model=result.model,
        mode="sync",
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def _batch_body(request: ApiRequest) -> BatchRequest:
    params = build_params(
        model=request.model,
        effort=request.effort,
        max_tokens=request.max_tokens,
        system=request.system,
        schema=request.schema,
        fallback=False,
    )
    return BatchRequest(request.custom_id, {**params, "messages": request.messages})


@dataclass
class Executor:
    """Sends requests by their planned mode and returns responses keyed by custom_id."""

    client: anthropic.Anthropic
    concurrency: int = 4
    poll_seconds: float = 30.0
    log: Log = print
    sleep: Callable[[float], None] = time.sleep
    # Batch ids by label, saved by the caller so an interrupted run can resume polling.
    batch_ids: dict[str, str] = field(default_factory=dict)
    on_batch_created: Callable[[str, str], None] | None = None
    stop: threading.Event = field(default_factory=threading.Event)

    def run(self, requests: list[ApiRequest], *, label: str) -> dict[str, ApiResponse]:
        ids = [request.custom_id for request in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("custom_ids must be unique within a run")
        batched = [request for request in requests if request.mode == "batch"]
        direct = [request for request in requests if request.mode == "sync"]
        responses: dict[str, ApiResponse] = {}

        # Submit the batch first: it processes while the synchronous sample runs.
        batch_id = self.batch_ids.get(label)
        if batched and batch_id is None:
            batch_id = submit_batch(self.client, [_batch_body(request) for request in batched])
            self.batch_ids[label] = batch_id
            if self.on_batch_created:
                self.on_batch_created(label, batch_id)
            self.log(f"[{label}] batch {batch_id} created with {len(batched)} requests")
        elif batched:
            self.log(f"[{label}] resuming batch {batch_id}")

        if direct:
            self.log(f"[{label}] sending {len(direct)} requests synchronously")
            with ThreadPoolExecutor(max_workers=max(1, self.concurrency)) as pool:
                for response in pool.map(self._guarded_sync, direct):
                    responses[response.custom_id] = response

        if batched and batch_id is not None:
            responses.update(self._collect(batch_id, batched, label))
        return responses

    def _guarded_sync(self, request: ApiRequest) -> ApiResponse:
        if self.stop.is_set():
            return _failed(request, "cancelled before it was sent")
        return _send_sync(self.client, request)

    def _collect(
        self, batch_id: str, requests: list[ApiRequest], label: str
    ) -> dict[str, ApiResponse]:
        while True:
            status = batch_status(self.client, batch_id)
            if status.ended:
                break
            self.log(f"[{label}] batch {batch_id}: {status.processing} still processing")
            self.sleep(self.poll_seconds)

        by_id = {request.custom_id: request for request in requests}
        responses: dict[str, ApiResponse] = {}
        for item in batch_results(self.client, batch_id):
            request = by_id.get(item.custom_id)
            if request is None:
                continue  # not ours: ignore instead of guessing
            if item.outcome != "succeeded" or item.message is None:
                responses[item.custom_id] = _failed(request, item.error or item.outcome)
                continue
            read = read_message(item.message)
            responses[item.custom_id] = ApiResponse(
                custom_id=item.custom_id,
                status=_status(read.stop_reason, read.refusal is not None),
                text=read.text,
                stop_reason=read.stop_reason,
                usage=read.usage,
                model=read.model,
                mode="batch",
            )
        for custom_id, request in by_id.items():
            responses.setdefault(custom_id, _failed(request, "missing from batch results"))
        return responses


class BudgetExceeded(RuntimeError):
    """The projected spend would pass the budget cap."""


@dataclass
class BudgetGuard:
    """Tracks measured spend and rescales the estimate for the work still to come."""

    budget_usd: float
    spent_usd: float = 0.0
    estimated_for_spent: float = 0.0

    @property
    def ratio(self) -> float:
        """Measured over estimated spend so far; never below 1, so a cheap wave cannot relax the cap."""
        if self.estimated_for_spent <= 0:
            return 1.0
        return max(1.0, self.spent_usd / self.estimated_for_spent)

    def projected(self, remaining_estimate_usd: float) -> float:
        return self.spent_usd + self.ratio * remaining_estimate_usd

    def require(self, remaining_estimate_usd: float, what: str) -> None:
        projected = self.projected(remaining_estimate_usd)
        if projected > self.budget_usd:
            raise BudgetExceeded(
                f"Stopping before {what}: projected total ${projected:.2f} exceeds the "
                f"${self.budget_usd:.2f} budget (spent ${self.spent_usd:.2f} so far, running "
                f"{self.ratio:.2f}x the estimate)."
            )

    def record(self, spent_usd: float, estimated_usd: float) -> None:
        self.spent_usd += spent_usd
        self.estimated_for_spent += estimated_usd


def request_for(planned: PlannedRequest, task: Task) -> ApiRequest:
    return ApiRequest(
        custom_id=planned.custom_id,
        model=planned.arm.model,
        effort=planned.arm.effort,
        system=system_prompt(task),
        messages=messages_for(task),
        max_tokens=planned.max_tokens,
        mode=planned.mode,
    )


def row_from_response(
    run_id: str, planned: PlannedRequest, task: Task, response: ApiResponse, *, synthetic: bool
) -> Row:
    """An ungraded result row. Grading fills in score, passed, and grader details."""
    return {
        "run_id": run_id,
        "custom_id": planned.custom_id,
        "task_id": task.id,
        "category": task.category,
        "model": planned.arm.model,
        "effort": planned.arm.effort,
        "trial": planned.trial,
        "mode": response.mode,
        "status": response.status,
        "answer": response.text,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_creation_input_tokens": response.usage.cache_creation_input_tokens,
        "cache_read_input_tokens": response.usage.cache_read_input_tokens,
        "cost_usd": response.list_usd,
        "billed_usd": response.billed_usd,
        "latency_ms": response.latency_ms,
        "score": None,
        "passed": None,
        "grader": task.grader.type,
        "grader_details": None if response.error is None else json.dumps({"error": response.error}),
        "judge_usd": 0.0,
        "synthetic": synthetic,
    }


def waves(plan: list[PlannedRequest]) -> list[tuple[str, list[PlannedRequest]]]:
    """The plan split by model tier, cheapest first."""
    grouped = [
        (SHORT_NAMES[model], [request for request in plan if request.arm.model == model])
        for model in MODELS
    ]
    return [(name, requests) for name, requests in grouped if requests]


def run_matrix(
    executor: Executor,
    tasks: list[Task],
    plan: list[PlannedRequest],
    *,
    run_id: str,
    guard: BudgetGuard,
    done_rows: list[Row] | None = None,
    on_wave: Callable[[list[Row]], None] | None = None,
) -> list[Row]:
    """Run every wave of the plan.

    On a resumed run, rows that already hold an answer are kept and not re-sent.
    Rows whose request failed last time are sent again, synchronously: there are
    few of them, and the batch they belonged to has already ended.
    """
    index = task_index(tasks)
    earlier = list(done_rows or [])
    rows: list[Row] = [row for row in earlier if row["status"] != "error"]
    failed = {row["custom_id"] for row in earlier if row["status"] == "error"}
    have = {row["custom_id"] for row in rows}
    guard.record(sum(row["billed_usd"] for row in earlier), 0.0)
    pending = [
        (
            name,
            [
                replace(request, mode="sync") if request.custom_id in failed else request
                for request in requests
                if request.custom_id not in have
            ],
        )
        for name, requests in waves(plan)
    ]
    pending = [(name, requests) for name, requests in pending if requests]

    for position, (name, requests) in enumerate(pending):
        remaining = [request for _, later in pending[position:] for request in later]
        guard.require(estimate_plan(remaining, tasks).answers_usd, f"the {name} wave")
        responses = executor.run(
            [request_for(request, index[request.task_id]) for request in requests],
            label=f"answers-{name}",
        )
        wave_rows = [
            row_from_response(
                run_id,
                request,
                index[request.task_id],
                responses[request.custom_id],
                synthetic=False,
            )
            for request in requests
        ]
        rows.extend(wave_rows)
        guard.record(
            sum(row["billed_usd"] for row in wave_rows), estimate_plan(requests, tasks).answers_usd
        )
        executor.log(
            f"[answers-{name}] done: {len(wave_rows)} rows, ${guard.spent_usd:.2f} spent so far"
        )
        if on_wave:
            on_wave(rows)
    return rows
