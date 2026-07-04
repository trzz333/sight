"""H1 NDJSON event logger. Schema version 1.

One JSON object per line, newline terminated, UTF-8, append-only. Common fields
(schema_version, run_id, ts_utc, phase, env_id, algo, framework, seed, git_commit)
are auto-filled per event. Numpy and torch scalars are coerced; non-JSON-safe
values are stringified.
"""

from __future__ import annotations

import datetime
import json
import math
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _utc_iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_short_git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Return short git HEAD hash or None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir is not None else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            return sha or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return None


def to_jsonable(value: Any) -> Any:
    """Recursively coerce numpy/torch scalars and other types into JSON-safe Python."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    # numpy scalars / arrays
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return to_jsonable(tolist())
        except Exception:
            pass
    # torch tensors
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            return to_jsonable(detach().cpu().numpy().tolist())
        except Exception:
            pass
    return str(value)


class NDJSONLogger:
    """Append-only NDJSON writer that auto-fills H1 common fields per event."""

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        phase: str,
        env_id: str,
        algo: str,
        framework: str,
        seed: int,
        git_commit: str | None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._common = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "phase": phase,
            "env_id": env_id,
            "algo": algo,
            "framework": framework,
            "seed": int(seed),
            "git_commit": git_commit,
        }

    def log_event(self, event: str, *, step: int | None = None, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = dict(self._common)
        record["event"] = event
        record["ts_utc"] = _utc_iso_now()
        record["step"] = int(step) if step is not None else None
        for k, v in fields.items():
            record[k] = to_jsonable(v)
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        return record

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> "NDJSONLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
