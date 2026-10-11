"""Prompt-caching economics: the same large prefix, with and without `cache_control`.

Three conditions, each a sequence of calls that share a system prompt and differ
only in a short question:

- `no_cache`: no cache breakpoint. Every call pays the full input price for the prefix.
- `cached`: a breakpoint on the system prompt. The first call writes the cache
  (1.25x the input price); later calls read it (0.05x) as long as each arrives
  within the five-minute TTL, which every read renews.
- `below_minimum`: a breakpoint on a prefix shorter than the 512-token minimum.
  The API accepts it and silently caches nothing, which is worth showing because
  it is the usual reason "caching is on but the bill did not change".

Calls in a condition run one after another, not in parallel: a parallel call can
arrive before the first has finished writing the cache, and then it pays full price.

The dollar figures are always computed by `cost_usd` from token counts. In a real
run the counts come from the API's `usage`; in a simulated run they come from the
length-based token estimate, and the result is flagged `synthetic`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic

from .claude_client import call_claude
from .config import (
    CACHE_MIN_TOKENS,
    CACHE_TTL_SECONDS,
    OPUS,
    PRICING,
    Effort,
    Usage,
    cost_usd,
)
from .estimate import REQUEST_OVERHEAD_TOKENS, approx_tokens

HANDBOOK = (Path(__file__).resolve().parent / "handbook.txt").read_text(encoding="utf-8")
SHORT_PREFIX = (
    "You answer customer questions for Larkspur Outfitters, an outdoor equipment shop. "
    "Standard shipping is free over 75 dollars. Unused items can be returned within 60 days. "
    "Be brief and friendly, and say so when you do not know."
)
INSTRUCTION = (
    "Answer the customer's question in at most three sentences, using only the handbook above."
)
QUESTIONS = (
    "How long do I have to return a jacket I have not worn?",
    "I ordered at 2pm Mountain Time. When does my order ship?",
    "Can I get bear spray delivered by expedited shipping?",
    "My Larkspur tent pole snapped in a storm. Is that covered?",
    "The boots I bought last week are now on sale. Can I get the difference back?",
    "Do you ship to Canada?",
    "How long does a ski base repair take in January?",
    "Can I use two promo codes on one order?",
    "I had my skis mounted and now I want to return them. Is that possible?",
    "When do Trail Club credits show up?",
)

CACHE_MODEL = OPUS
CACHE_EFFORT: Effort = "low"
DEFAULT_CALLS = 10
ASSUMED_ANSWER_TOKENS = 140


def _system(prefix: str, nonce: str) -> str:
    # The nonce makes the prefix new to the cache, so the first cached call is a
    # genuine cache write even if this experiment ran a minute ago.
    return f"Handbook revision: {nonce}\n\n{prefix}\n\n{INSTRUCTION}"


def _call_record(n: int, model: str, usage: Usage) -> dict[str, Any]:
    return {"n": n, **usage.as_dict(), "cost_usd": cost_usd(model, usage)}


def _condition(key: str, label: str, model: str, usages: list[Usage]) -> dict[str, Any]:
    calls = [_call_record(i + 1, model, usage) for i, usage in enumerate(usages)]
    return {
        "key": key,
        "label": label,
        "calls": calls,
        "input_tokens": sum(u.input_tokens for u in usages),
        "cache_creation_input_tokens": sum(u.cache_creation_input_tokens for u in usages),
        "cache_read_input_tokens": sum(u.cache_read_input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "total_usd": sum(call["cost_usd"] for call in calls),
    }


def break_even_call(no_cache: list[dict[str, Any]], cached: list[dict[str, Any]]) -> int | None:
    """The first call at which the cached run's cumulative cost drops below the uncached run's."""
    plain = with_cache = 0.0
    for a, b in zip(no_cache, cached, strict=True):
        plain += a["cost_usd"]
        with_cache += b["cost_usd"]
        if with_cache < plain:
            return a["n"]
    return None


def summarize(
    *,
    synthetic: bool,
    model: str,
    effort: str,
    usages: dict[str, list[Usage]],
    prefix_tokens: int,
    short_prefix_tokens: int,
) -> dict[str, Any]:
    """Assemble the experiment report from per-call usage. All dollars are computed here."""
    labels = {
        "no_cache": "No cache breakpoint",
        "cached": "Cache breakpoint on the prefix",
        "below_minimum": "Breakpoint on a prefix under the minimum",
    }
    conditions = [_condition(key, labels[key], model, usages[key]) for key in labels]
    plain, cached = conditions[0], conditions[1]
    rates = PRICING[model]
    saved = plain["total_usd"] - cached["total_usd"]
    plain_input = sum(cost_usd(model, Usage(call["input_tokens"], 0)) for call in plain["calls"])
    cached_input = sum(
        cost_usd(
            model,
            Usage(
                call["input_tokens"],
                0,
                call["cache_creation_input_tokens"],
                call["cache_read_input_tokens"],
            ),
        )
        for call in cached["calls"]
    )
    return {
        "synthetic": synthetic,
        "model": model,
        "effort": effort,
        "calls": len(plain["calls"]),
        "prefix_tokens": prefix_tokens,
        "short_prefix_tokens": short_prefix_tokens,
        "conditions": conditions,
        "savings_usd": saved,
        "savings_pct": saved / plain["total_usd"] if plain["total_usd"] else 0.0,
        "input_savings_pct": 1 - cached_input / plain_input if plain_input else 0.0,
        "break_even_call": break_even_call(plain["calls"], cached["calls"]),
        "rules": {
            "min_tokens": CACHE_MIN_TOKENS,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "write_multiplier": rates[2] / rates[0],
            "read_multiplier": rates[3] / rates[0],
        },
    }


def simulate_caching(calls: int = DEFAULT_CALLS) -> dict[str, Any]:
    """The experiment with token counts estimated from text length. Flagged synthetic."""
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(calls)]
    prefix = approx_tokens(_system(HANDBOOK, "simulated"))
    short = approx_tokens(_system(SHORT_PREFIX, "simulated"))
    asked = [approx_tokens(q) + REQUEST_OVERHEAD_TOKENS for q in questions]
    out = ASSUMED_ANSWER_TOKENS
    usages = {
        "no_cache": [Usage(prefix + q, out) for q in asked],
        "cached": [
            Usage(q, out, prefix, 0) if i == 0 else Usage(q, out, 0, prefix)
            for i, q in enumerate(asked)
        ],
        # Under the minimum the breakpoint is ignored: plain input tokens, no cache fields.
        "below_minimum": [Usage(short + q, out) for q in asked],
    }
    return summarize(
        synthetic=True,
        model=CACHE_MODEL,
        effort=CACHE_EFFORT,
        usages=usages,
        prefix_tokens=prefix,
        short_prefix_tokens=short,
    )


def estimate_caching_usd(calls: int = DEFAULT_CALLS) -> float:
    """What the real experiment is expected to cost."""
    return sum(condition["total_usd"] for condition in simulate_caching(calls)["conditions"])


def run_caching(
    client: anthropic.Anthropic, *, nonce: str, calls: int = DEFAULT_CALLS
) -> dict[str, Any]:
    """Run the experiment against the API. Sequential on purpose (see module docstring)."""
    questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(calls)]
    plans = {
        "no_cache": (HANDBOOK, False),
        "cached": (HANDBOOK, True),
        "below_minimum": (SHORT_PREFIX, True),
    }
    usages: dict[str, list[Usage]] = {}
    for key, (prefix, cache) in plans.items():
        usages[key] = []
        for question in questions:
            result = call_claude(
                client,
                model=CACHE_MODEL,
                effort=CACHE_EFFORT,
                system=_system(prefix, f"{nonce}-{key}"),
                cache_system=cache,
                messages=[{"role": "user", "content": question}],
                max_tokens=2_000,
                fallback=False,
            )
            usages[key].append(result.usage)
    cached_first = usages["cached"][0]
    return summarize(
        synthetic=False,
        model=CACHE_MODEL,
        effort=CACHE_EFFORT,
        usages=usages,
        prefix_tokens=cached_first.cache_creation_input_tokens
        or usages["no_cache"][0].input_tokens,
        short_prefix_tokens=usages["below_minimum"][0].input_tokens,
    )
