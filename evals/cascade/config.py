"""Single source of truth for model IDs, effort levels, and pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OPUS = "claude-opus-5-5"  # most capable: the baseline every other arm is compared against
SONNET = "claude-sonnet-5-5"  # mid tier
HAIKU = "claude-haiku-5-5"  # cheapest tier

# `output_config.effort`. Thinking is always on for these models; effort is the depth/cost control.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "xhigh", "max")
DEFAULT_EFFORTS: tuple[Effort, ...] = ("low", "medium", "high")

# Cheapest tier first. Short names are used in arm keys, CLI flags, and batch custom_ids.
MODELS: tuple[str, ...] = (HAIKU, SONNET, OPUS)
SHORT_NAMES: dict[str, str] = {HAIKU: "haiku", SONNET: "sonnet", OPUS: "opus"}
MODEL_BY_SHORT: dict[str, str] = {short: model for model, short in SHORT_NAMES.items()}

PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
PRICING_LAST_VERIFIED = "2026-10-11"

# USD per million tokens: (input, output, 5-minute cache write, cache read)
PRICING: dict[str, tuple[float, float, float, float]] = {
    OPUS: (4.0, 20.0, 5.0, 0.20),
    SONNET: (2.0, 10.0, 2.50, 0.10),
    HAIKU: (0.10, 0.50, 0.125, 0.01),  # prompts up to 100K tokens; 5x beyond that
    # Models a refused request can be re-run on by the server-side fallback.
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.50),
}
BATCH_MULTIPLIER = 0.5  # Batch API: half price on input and output
HAIKU_LONG_PROMPT_THRESHOLD = 100_000
HAIKU_LONG_PROMPT_MULTIPLIER = 5

# Prompt caching: a prefix shorter than this is silently not cached, and a cached
# prefix is dropped five minutes after it was last written or read.
CACHE_MIN_TOKENS = 512
CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def plus(self, other: Any) -> Usage:
        """Add an API `usage` object (or another Usage); missing cache fields count as zero."""
        return Usage(
            self.input_tokens + (getattr(other, "input_tokens", 0) or 0),
            self.output_tokens + (getattr(other, "output_tokens", 0) or 0),
            self.cache_creation_input_tokens
            + (getattr(other, "cache_creation_input_tokens", 0) or 0),
            self.cache_read_input_tokens + (getattr(other, "cache_read_input_tokens", 0) or 0),
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


def cost_usd(model: str, usage: Usage, *, batch: bool = False) -> float:
    """Cost of one request. Unknown models are priced as Opus so the meter never under-reports."""
    rate_in, rate_out, rate_write, rate_read = PRICING.get(model, PRICING[OPUS])
    prompt_tokens = (
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    )
    long_prompt = (
        HAIKU_LONG_PROMPT_MULTIPLIER
        if model == HAIKU and prompt_tokens > HAIKU_LONG_PROMPT_THRESHOLD
        else 1
    )
    dollars = (
        usage.input_tokens * rate_in
        + usage.output_tokens * rate_out
        + usage.cache_creation_input_tokens * rate_write
        + usage.cache_read_input_tokens * rate_read
    ) / 1_000_000
    return dollars * long_prompt * (BATCH_MULTIPLIER if batch else 1)


def pricing_snapshot() -> dict[str, Any]:
    """The rates a run was priced with, stored beside its results."""
    return {
        "source": PRICING_SOURCE,
        "last_verified": PRICING_LAST_VERIFIED,
        "batch_multiplier": BATCH_MULTIPLIER,
        "per_mtok": {
            model: {
                "input": PRICING[model][0],
                "output": PRICING[model][1],
                "cache_write_5m": PRICING[model][2],
                "cache_read": PRICING[model][3],
            }
            for model in MODELS
        },
    }
