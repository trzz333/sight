"""NDJSON writer for the Python agent side."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


def new_run_id(prefix: str = "run") -> str:
    """UTC timestamp + short uuid. Stable sort order by prefix."""

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"


class NDJSONLogger:
    """Append-only NDJSON. One line per event. Every record carries run_id, type, ts_unix_ns."""

    def __init__(
        self,
        run_dir: str | Path,
        side: str = "python",
        manifest_extra: dict | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.side = side
        self.run_id = self.run_dir.name
        self.path = self.run_dir / f"{side}.ndjson"
        # Append mode so re-entries into the same run_dir concatenate rather than truncate.
        self._file = self.path.open("a", encoding="utf-8")
        self._ensure_manifest(manifest_extra or {})

    def _ensure_manifest(self, extra: dict) -> None:
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            return
        payload = {
            "run_id": self.run_id,
            "created_ts_unix_ns": time.time_ns(),
            "sides": [self.side],
        }
        payload.update(extra)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log(self, event_type: str, **fields: Any) -> dict:
        record: dict[str, Any] = {
            "run_id": self.run_id,
            "ts_unix_ns": time.time_ns(),
            "type": event_type,
        }
        record.update(fields)
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
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
