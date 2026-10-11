"""A stand-in Anthropic client that returns queued responses and records each request."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from cascade.config import OPUS


def block(type_: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **fields)


def message(content: list[Any], stop_reason: str = "end_turn", **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(
        model=OPUS,
        content=content,
        stop_reason=stop_reason,
        stop_details=extra.pop("stop_details", None),
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=0,
        ),
        **extra,
    )


class FakeClient:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params: Any) -> Any:
        # Snapshot: the helper keeps appending to the same history list.
        self.requests.append(copy.deepcopy(params))
        return self._responses.pop(0)


class ScriptedClient:
    """A fake client whose answers come from a function of the request.

    Supports the synchronous endpoint, the Message Batches endpoints, and token
    counting. Batch results come back in reverse order, and the batch reports
    "in_progress" once before it ends, so callers must poll and must key by custom_id.
    """

    def __init__(self, responder: Any, *, batch_outcomes: dict[str, str] | None = None):
        self.responder = responder
        self.batch_outcomes = batch_outcomes or {}
        self.sync_requests: list[dict[str, Any]] = []
        self.batches: dict[str, list[dict[str, Any]]] = {}
        self.polls: dict[str, int] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(
            batches=SimpleNamespace(
                create=self._batch_create,
                retrieve=self._batch_retrieve,
                results=self._batch_results,
            ),
            count_tokens=lambda **params: SimpleNamespace(input_tokens=123),
        )

    def _create(self, **params: Any) -> Any:
        self.sync_requests.append(copy.deepcopy(params))
        return self.responder(params)

    def _batch_create(self, *, requests: list[dict[str, Any]]) -> Any:
        batch_id = f"batch_{len(self.batches) + 1}"
        self.batches[batch_id] = copy.deepcopy(requests)
        self.polls[batch_id] = 0
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def _batch_retrieve(self, batch_id: str) -> Any:
        self.polls[batch_id] += 1
        ended = self.polls[batch_id] >= 2
        total = len(self.batches[batch_id])
        return SimpleNamespace(
            id=batch_id,
            processing_status="ended" if ended else "in_progress",
            request_counts=SimpleNamespace(
                processing=0 if ended else total,
                succeeded=total if ended else 0,
                errored=0,
                canceled=0,
                expired=0,
            ),
        )

    def _batch_results(self, batch_id: str) -> Any:
        for request in reversed(self.batches[batch_id]):
            custom_id = request["custom_id"]
            outcome = self.batch_outcomes.get(custom_id, "succeeded")
            if outcome == "missing":
                continue
            if outcome == "succeeded":
                result = SimpleNamespace(
                    type="succeeded", message=self.responder(request["params"])
                )
            elif outcome == "errored":
                result = SimpleNamespace(
                    type="errored",
                    error=SimpleNamespace(error=SimpleNamespace(type="overloaded_error")),
                )
            else:
                result = SimpleNamespace(type=outcome)
            yield SimpleNamespace(custom_id=custom_id, result=result)


def text_message(
    text: str, *, model: str = OPUS, stop_reason: str = "end_turn", **usage: int
) -> Any:
    reply = message(
        [block("thinking", thinking="", signature="s"), block("text", text=text)], stop_reason
    )
    reply.model = model
    for name, value in usage.items():
        setattr(reply.usage, name, value)
    return reply
