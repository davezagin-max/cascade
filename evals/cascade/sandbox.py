"""Runs model-written Python against hidden unit tests.

Model output is untrusted code, so each run gets:

- its own interpreter process (`python -I -S`: no site packages, no user paths,
  no PYTHON* environment variables), in its own session so a timeout can kill
  everything it started;
- an empty temp directory as its working directory and HOME, deleted afterwards;
- a fresh environment with only PATH and LANG, so nothing from the parent
  process (in particular no API key) is inherited;
- resource limits it applies to itself before the untrusted code is loaded: CPU
  seconds, address space, file size, open files, no core dumps, and no new
  processes. They are hard limits, so the code cannot raise them again;
- a wall-clock timeout per test and for the whole run, with stdin closed and
  stdout/stderr sent to size-capped files.

THIS IS NOT A HARDENED SANDBOX. The code still runs as the current user with
network access and read access to the filesystem, and it shares a process with
the harness, so a hostile answer could read the hidden tests or forge its own
results. It is enough to survive honest mistakes (infinite loops, runaway
memory, stray prints). A production grader would run each answer in a container
or microVM with no network, a read-only root, a seccomp filter, and a throwaway
user: gVisor, Firecracker, nsjail, or a hosted code-execution sandbox.
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

CPU_SECONDS = 10
MEMORY_BYTES = 512 * 1024 * 1024
FILE_BYTES = 1024 * 1024
OPEN_FILES = 32
PER_TEST_SECONDS = 2.0

# Runs inside the child. It limits itself first, then loads the answer, then runs
# each test in a copy of the answer's namespace and appends one JSON line per test.
_HARNESS = r"""
import json, signal, sys

def _limit():
    import resource
    spec = json.load(open("spec.json"))
    wanted = [
        ("RLIMIT_CPU", spec["cpu_seconds"]),
        ("RLIMIT_AS", spec["memory_bytes"]),
        ("RLIMIT_DATA", spec["memory_bytes"]),
        ("RLIMIT_FSIZE", spec["file_bytes"]),
        ("RLIMIT_NOFILE", spec["open_files"]),
        ("RLIMIT_CORE", 0),
        ("RLIMIT_NPROC", 0),
    ]
    applied = []
    for name, value in wanted:
        kind = getattr(resource, name, None)
        if kind is None:
            continue
        try:
            resource.setrlimit(kind, (value, value))
            applied.append(name)
        except (ValueError, OSError):
            pass  # not enforceable on this platform (for example RLIMIT_AS on macOS)
    return spec, applied

def raises(expected, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except expected:
        return True
    except BaseException:
        return False
    return False

class _TestTimeout(BaseException):
    pass

def _alarm(signum, frame):
    raise _TestTimeout()

def _describe(error):
    text = str(error).strip().splitlines()[0][:160] if str(error).strip() else ""
    return type(error).__name__ + (": " + text if text else "")

def main():
    spec, applied = _limit()
    out = open(spec["result_file"], "a", buffering=1)
    out.write(json.dumps({"limits": applied}) + "\n")
    namespace = {"__name__": "solution"}
    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.setitimer(signal.ITIMER_REAL, spec["per_test_seconds"])
        exec(compile(open("solution.py").read(), "solution.py", "exec"), namespace)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except _TestTimeout:
        out.write(json.dumps({"load_error": "timeout while loading the answer"}) + "\n")
        return
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        out.write(json.dumps({"load_error": _describe(error)}) + "\n")
        return
    namespace["raises"] = raises
    for index, test in enumerate(spec["tests"]):
        try:
            signal.setitimer(signal.ITIMER_REAL, spec["per_test_seconds"])
            exec(compile(test, "<hidden test>", "exec"), dict(namespace))
            signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(json.dumps({"test": index, "ok": True}) + "\n")
        except _TestTimeout:
            out.write(json.dumps({"test": index, "ok": False, "error": "timeout"}) + "\n")
        except BaseException as error:
            signal.setitimer(signal.ITIMER_REAL, 0)
            out.write(json.dumps({"test": index, "ok": False, "error": _describe(error)}) + "\n")

main()
"""


@dataclass(frozen=True)
class TestOutcome:
    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class SandboxResult:
    outcomes: tuple[TestOutcome, ...]
    load_error: str | None = None
    timed_out: bool = False
    limits: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)


def run_tests(
    code: str,
    tests: tuple[str, ...] | list[str],
    *,
    timeout_s: float = 6.0,
    per_test_s: float = PER_TEST_SECONDS,
) -> SandboxResult:
    """Run `tests` against `code` in a restricted child process and report each outcome."""
    tests = list(tests)
    with tempfile.TemporaryDirectory(prefix="cascade-sandbox-") as workdir:
        root = Path(workdir)
        result_name = f"results-{secrets.token_hex(8)}.jsonl"
        (root / "solution.py").write_text(code, encoding="utf-8")
        (root / "harness.py").write_text(_HARNESS, encoding="utf-8")
        (root / "spec.json").write_text(
            json.dumps(
                {
                    "tests": tests,
                    "result_file": result_name,
                    "per_test_seconds": per_test_s,
                    "cpu_seconds": CPU_SECONDS,
                    "memory_bytes": MEMORY_BYTES,
                    "file_bytes": FILE_BYTES,
                    "open_files": OPEN_FILES,
                }
            ),
            encoding="utf-8",
        )
        timed_out = False
        with (
            open(root / "stdout.txt", "wb") as stdout,
            open(root / "stderr.txt", "wb") as stderr,
        ):
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "harness.py"],
                cwd=root,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(root)},
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
            try:
                process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()

        records: list[dict] = []
        result_path = root / result_name
        if result_path.exists():
            for line in result_path.read_text(encoding="utf-8", errors="replace").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        records.append(parsed)

    limits: tuple[str, ...] = ()
    load_error: str | None = None
    by_index: dict[int, TestOutcome] = {}
    for record in records:
        if "limits" in record:
            limits = tuple(record["limits"])
        elif "load_error" in record:
            load_error = str(record["load_error"])
        elif isinstance(record.get("test"), int) and record["test"] not in by_index:
            # The harness writes one line per test; the first one for an index is kept.
            by_index[record["test"]] = TestOutcome(bool(record.get("ok")), record.get("error"))

    # A test with no line never finished: the process died or was killed first.
    missing = "timeout" if timed_out else "did not run"
    outcomes = tuple(
        by_index.get(index, TestOutcome(False, load_error or missing))
        for index in range(len(tests))
    )
    return SandboxResult(outcomes, load_error=load_error, timed_out=timed_out, limits=limits)
