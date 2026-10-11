"""Prompt features for the router, interpreted from `feature_spec.json`.

The same spec is interpreted by `lib/router/features.ts` in the web app, and the two
must agree bit for bit, because the router is trained here and served there. That
shapes every choice in this file:

- Text is normalised the same way on both sides: CRLF and CR become LF, and
  lower-casing touches only A-Z (Unicode case folding differs between runtimes).
- Patterns are compiled with `re.ASCII`, so `\\b` means the same thing it does in a
  JavaScript regex without the `u` flag.
- Length is counted in code points on both sides.
- Feature values use only integer counts, one addition, and one division
  (`n / (n + k)`), all of which are exact in IEEE 754. There is no `log` or `exp`,
  whose last bit can differ between a C library and a JavaScript engine.

`tests/test_features.py` and `tests/router-parity.test.ts` check both
implementations against the same exported vectors.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parent / "feature_spec.json"
_LOWER = {code: code + 32 for code in range(ord("A"), ord("Z") + 1)}

FeatureSpec = dict[str, Any]


def load_spec(path: Path = SPEC_PATH) -> list[FeatureSpec]:
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def ascii_lower(text: str) -> str:
    return text.translate(_LOWER)


@lru_cache(maxsize=512)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.ASCII)


def _count(patterns: list[str], text: str) -> int:
    return sum(1 for pattern in patterns for _ in _compiled(pattern).finditer(text))


def _raw_count(feature: FeatureSpec, raw: str, lower: str) -> int:
    kind = feature["kind"]
    if kind == "chars":
        return len(raw)
    if kind == "count":
        return _count(feature["patterns"], lower if feature["source"] == "lower" else raw)
    if kind == "lines":
        pattern = _compiled(feature["pattern"])
        return sum(1 for line in raw.split("\n") if pattern.match(line.lstrip(" \t")))
    raise ValueError(f"Unknown feature kind: {kind}")


def extract_features(spec: list[FeatureSpec], prompt: str) -> list[float]:
    """One value per feature, each in [0, 1)."""
    raw = normalize(prompt)
    lower = ascii_lower(raw)
    chars = len(raw)
    values: list[float] = []
    for feature in spec:
        n = _raw_count(feature, raw, lower)
        if feature["transform"] == "sat":
            values.append(n / (n + feature["k"]))
        elif feature["transform"] == "ratio":
            values.append(n / (chars + 1))
        else:
            raise ValueError(f"Unknown transform: {feature['transform']}")
    return values
