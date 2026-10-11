import pytest

from cascade.caching import (
    HANDBOOK,
    SHORT_PREFIX,
    break_even_call,
    estimate_caching_usd,
    run_caching,
    simulate_caching,
)
from cascade.config import CACHE_MIN_TOKENS, OPUS, Usage, cost_usd
from cascade.estimate import approx_tokens
from tests.fakes import ScriptedClient, text_message


def test_the_two_prefixes_sit_on_either_side_of_the_cache_minimum() -> None:
    assert approx_tokens(HANDBOOK) > 2 * CACHE_MIN_TOKENS
    assert approx_tokens(SHORT_PREFIX) < CACHE_MIN_TOKENS / 2


def test_simulated_experiment_follows_the_pricing_rules() -> None:
    report = simulate_caching(10)
    plain, cached, short = report["conditions"]
    prefix = report["prefix_tokens"]
    assert report["synthetic"] is True and report["calls"] == 10

    # No breakpoint: the prefix is paid at the full input rate every time.
    assert all(call["cache_read_input_tokens"] == 0 for call in plain["calls"])
    assert plain["input_tokens"] >= 10 * prefix

    # With a breakpoint: one write, then nine reads of the same prefix.
    first, *rest = cached["calls"]
    assert first["cache_creation_input_tokens"] == prefix and first["cache_read_input_tokens"] == 0
    assert all(
        c["cache_read_input_tokens"] == prefix and c["cache_creation_input_tokens"] == 0
        for c in rest
    )
    assert first["cost_usd"] > plain["calls"][0]["cost_usd"]  # the write costs 25% more
    assert rest[0]["cost_usd"] < plain["calls"][1]["cost_usd"]

    # Dollars are recomputed from tokens with the shared pricing function.
    usage = Usage(first["input_tokens"], first["output_tokens"], prefix, 0)
    assert first["cost_usd"] == pytest.approx(cost_usd(OPUS, usage))
    assert report["savings_usd"] == pytest.approx(plain["total_usd"] - cached["total_usd"])
    assert (
        report["savings_usd"] > 0 and 0 < report["savings_pct"] < report["input_savings_pct"] < 0.95
    )
    assert report["break_even_call"] == 2
    assert report["rules"] == {
        "min_tokens": 512,
        "ttl_seconds": 300,
        "write_multiplier": pytest.approx(1.25),
        "read_multiplier": pytest.approx(0.05),
    }

    # Under the minimum the breakpoint does nothing.
    assert report["short_prefix_tokens"] < CACHE_MIN_TOKENS
    assert short["cache_creation_input_tokens"] == 0 and short["cache_read_input_tokens"] == 0
    assert estimate_caching_usd(10) == pytest.approx(
        sum(c["total_usd"] for c in report["conditions"])
    )


def test_break_even_is_none_when_caching_never_pays_off() -> None:
    one_call = simulate_caching(1)
    assert one_call["break_even_call"] is None  # a single call only pays the write premium
    assert one_call["savings_usd"] < 0
    calls = [{"n": 1, "cost_usd": 1.0}, {"n": 2, "cost_usd": 1.0}]
    assert break_even_call(calls, [{"n": 1, "cost_usd": 1.2}, {"n": 2, "cost_usd": 0.7}]) == 2


def test_real_experiment_sets_the_breakpoint_only_where_intended() -> None:
    """Driven by a fake client that behaves like the API: write once, then read, above the minimum."""
    seen: set[str] = set()

    def api(params: dict) -> object:
        block = params["system"][0]
        prefix_tokens = approx_tokens(block["text"])
        cached = "cache_control" in block and prefix_tokens >= CACHE_MIN_TOKENS
        usage = {"input_tokens": 20, "output_tokens": 50}
        if not cached:
            usage["input_tokens"] += prefix_tokens
        elif block["text"] in seen:
            usage["cache_read_input_tokens"] = prefix_tokens
        else:
            seen.add(block["text"])
            usage["cache_creation_input_tokens"] = prefix_tokens
        return text_message("answer", **usage)

    client = ScriptedClient(api)
    report = run_caching(client, nonce="run-1", calls=4)  # type: ignore[arg-type]
    assert report["synthetic"] is False and len(client.sync_requests) == 12
    breakpoints = ["cache_control" in request["system"][0] for request in client.sync_requests]
    assert breakpoints == [False] * 4 + [True] * 8
    assert all("fallbacks" not in request for request in client.sync_requests)
    assert all("run-1" in request["system"][0]["text"] for request in client.sync_requests)

    plain, cached, short = report["conditions"]
    assert [call["cache_creation_input_tokens"] > 0 for call in cached["calls"]] == [
        True,
        False,
        False,
        False,
    ]
    assert [call["cache_read_input_tokens"] > 0 for call in cached["calls"]] == [
        False,
        True,
        True,
        True,
    ]
    assert short["cache_creation_input_tokens"] == 0 and short["cache_read_input_tokens"] == 0
    assert report["prefix_tokens"] == cached["calls"][0]["cache_creation_input_tokens"]
    assert report["savings_usd"] > 0 and report["break_even_call"] == 2
    assert plain["total_usd"] == pytest.approx(sum(call["cost_usd"] for call in plain["calls"]))
