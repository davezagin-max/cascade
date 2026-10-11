"""How a task becomes a request.

The task prompt is sent as the user turn exactly as written, so the router is
trained on the same text a real user would type. The output format each grader
needs is a short, fixed contract in the system prompt.
"""

from __future__ import annotations

from typing import Any

from .config import Effort
from .tasks import CodeSpec, ExactSpec, JsonSpec, Task

_PREAMBLE = "You are completing one task from an evaluation suite. Do the task as asked."

_CONTRACTS = {
    "exact": (
        "End your reply with a final line of the form `ANSWER: <answer>`. Put only the "
        "answer on that line: the label exactly as written in the task, the number without "
        "units, or the single word asked for."
    ),
    "json": (
        "Reply with one JSON object and nothing else: no prose and no code fence. Use "
        "exactly the keys the task names."
    ),
    "code": (
        "Reply with one Python code block that defines exactly what the task asks for. "
        "Use only the standard library. Do not include tests, examples, or print statements."
    ),
    "judge": "Reply with the requested text only, with no preamble about what you are doing.",
}

# Thinking counts toward max_tokens, so higher effort needs more room. Values stay
# under the size at which the SDK insists on streaming for a non-streaming request.
_MAX_TOKENS: dict[str, int] = {
    "low": 4_000,
    "medium": 8_000,
    "high": 16_000,
    "xhigh": 20_000,
    "max": 20_000,
}


def max_tokens_for(effort: Effort) -> int:
    return _MAX_TOKENS[effort]


def contract_for(task: Task) -> str:
    spec = task.grader
    if isinstance(spec, ExactSpec):
        return _CONTRACTS["exact"]
    if isinstance(spec, JsonSpec):
        return _CONTRACTS["json"]
    if isinstance(spec, CodeSpec):
        return _CONTRACTS["code"]
    return _CONTRACTS["judge"]


def system_prompt(task: Task) -> str:
    return f"{_PREAMBLE}\n\n{contract_for(task)}"


def messages_for(task: Task) -> list[dict[str, Any]]:
    return [{"role": "user", "content": task.prompt}]
