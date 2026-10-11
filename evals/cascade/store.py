"""Run storage: one Parquet file of results plus a small JSON file of run metadata.

`results.parquet` has one row per (task, model, effort, trial). A simulated run
writes exactly the same columns as a real one; the only differences are the
`synthetic` flag here and in `run.json`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("custom_id", pa.string()),
        ("task_id", pa.string()),
        ("category", pa.string()),
        ("model", pa.string()),
        ("effort", pa.string()),
        ("trial", pa.int32()),
        ("mode", pa.string()),  # batch | sync: how the request was sent
        ("status", pa.string()),  # ok | truncated | refusal | error
        ("answer", pa.string()),
        ("stop_reason", pa.string()),
        ("input_tokens", pa.int64()),
        ("output_tokens", pa.int64()),
        ("cache_creation_input_tokens", pa.int64()),
        ("cache_read_input_tokens", pa.int64()),
        ("cost_usd", pa.float64()),  # at list price: what this request costs in production
        ("billed_usd", pa.float64()),  # what the run paid (batch rows are half price)
        ("latency_ms", pa.float64()),  # null for batch rows: batches carry no timing
        ("score", pa.float64()),  # null until graded
        ("passed", pa.bool_()),
        ("grader", pa.string()),
        ("grader_details", pa.string()),  # JSON
        ("judge_usd", pa.float64()),
        ("synthetic", pa.bool_()),
    ]
)
COLUMNS = [f.name for f in SCHEMA]

Row = dict[str, Any]


def write_rows(path: Path, rows: list[Row]) -> None:
    """Write rows in a stable order. Unknown keys are an error, missing ones become null."""
    for row in rows:
        extra = set(row) - set(COLUMNS)
        if extra:
            raise ValueError(f"Unknown result columns: {sorted(extra)}")
    ordered = sorted(rows, key=lambda r: (r["task_id"], r["model"], r["effort"], r["trial"]))
    table = pa.Table.from_pylist([{c: row.get(c) for c in COLUMNS} for row in ordered], SCHEMA)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_rows(path: Path) -> list[Row]:
    table = pq.read_table(path)
    if table.schema.names != COLUMNS:
        raise ValueError(f"{path} does not have the expected result columns")
    return table.to_pylist()


def run_dir(run_id: str, root: Path = RUNS_DIR) -> Path:
    return root / run_id


def results_path(run_id: str, root: Path = RUNS_DIR) -> Path:
    return run_dir(run_id, root) / "results.parquet"


def write_meta(run_id: str, meta: dict[str, Any], root: Path = RUNS_DIR) -> None:
    path = run_dir(run_id, root) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_meta(run_id: str, root: Path = RUNS_DIR) -> dict[str, Any]:
    return json.loads((run_dir(run_id, root) / "run.json").read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(value, indent=2, ensure_ascii=False)
    )
    path.write_text(text + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def round_floats(value: Any, digits: int) -> Any:
    """Round every float in a JSON-like value, so exports are stable and readable."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [round_floats(item, digits) for item in value]
    return value
