"""P3 per-episode artifact loader.

Parses NDJSON artifacts written under runs\\eval\\<run_id>\\episodes\\<episode_id>\\
into pure ``Episode`` records consumed by ``metrics.aggregate``.

Spec invariants enforced here:

- Runs flagged in metadata as ignore-death are excluded per the
  SIGHT_TCP_IGNORE_DEATH exclusion invariant in docs/sight-p3-metrics.md;
  Episode.ignore_death_active=True is set so the metric layer skips them.
  This module never reads the process environment; it only inspects artifact
  content. The runner scripts/run_p3_eval.py owns the env-var refusal guard.
- Terminal classification produces exactly one of the six allowed values from
  metrics.TERMINAL_EVENTS.

Public API:

    load_ndjson(path) -> list[dict]
    extract_actions(python_events) -> tuple[str, ...]
    run_metadata_from_events(godot_events) -> dict
    classify_terminal(...) -> (terminal, terminal_ts_ns, other_reason)
    episode_from_events(...) -> Episode
    load_episode(godot_path, python_path, ...) -> Episode
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import Episode, TERMINAL_EVENTS


# Run-metadata keys that mark an ignore-death run. The literal env-var name is
# included so artifacts that mirror env into run_start are excluded; this line
# stays under the regression-guard contract via "spec invariant excluded".
RUN_META_IGNORE_DEATH_KEYS: tuple[str, ...] = (
    "ignore_death",
    "ignore_death_active",
    "tcp_ignore_death",
    "SIGHT_TCP_IGNORE_DEATH",  # spec invariant; excluded from P3 metric paths
)

_HARNESS_ABORT_STATUSES: frozenset[str] = frozenset(
    {"abort", "aborted", "error", "errored", "failed", "failure", "harness_abort"}
)

# Apply ratio below this threshold (with no death evidence) classifies as
# transport_drop. Tuned high so genuine transport-clean batches never trip it.
_TRANSPORT_DROP_APPLY_RATIO_THRESHOLD: float = 0.9

_DEATH_EVENT_TYPES: frozenset[str] = frozenset({"death", "collision"})


# --- low-level NDJSON I/O ---------------------------------------------------


def load_ndjson(path: str | Path) -> list[dict]:
    """Read newline-delimited JSON. Missing path -> []. Bad JSON -> ValueError."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}: bad json at line {i}: {e}") from e
    return out


# --- run metadata -----------------------------------------------------------


def _truthy_meta_value(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return bool(v.strip())
    return bool(v)


def run_metadata_from_events(godot_events: Sequence[Mapping[str, Any]]) -> dict:
    """Extract run metadata from the Godot run_start event.

    Returns a dict with at least ``ignore_death_active`` (bool) and
    ``run_start`` (the raw event or None).
    """
    rs: dict | None = None
    for e in godot_events:
        if e.get("type") == "run_start":
            rs = dict(e)
            break

    ignore_death_active = False
    if rs is not None:
        for key in RUN_META_IGNORE_DEATH_KEYS:
            if _truthy_meta_value(rs.get(key)):
                ignore_death_active = True
                break
        env_block = rs.get("env")
        if (
            not ignore_death_active
            and isinstance(env_block, Mapping)
        ):
            for key in RUN_META_IGNORE_DEATH_KEYS:
                if _truthy_meta_value(env_block.get(key)):
                    ignore_death_active = True
                    break

    return {"ignore_death_active": ignore_death_active, "run_start": rs}


# --- python-side actions ----------------------------------------------------


def extract_actions(python_events: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Pull the action label from each python decision event, in seq order.

    Decisions without an explicit ``seq`` retain insertion order at the tail.
    Decisions without an ``action`` field are skipped.
    """
    decisions = [e for e in python_events if e.get("type") == "decision"]
    seqd = [e for e in decisions if "seq" in e]
    unseqd = [e for e in decisions if "seq" not in e]
    seqd.sort(key=lambda e: e["seq"])
    out: list[str] = []
    for e in seqd + unseqd:
        a = e.get("action")
        if a is None:
            continue
        out.append(str(a))
    return tuple(out)


# --- timestamps -------------------------------------------------------------


_DECISION_TS_KEYS: tuple[str, ...] = (
    "decision_ts_unix_ns",
    "capture_ts_unix_ns",
    "sent_ts_unix_ns",
    "ts_unix_ns",
)


def _decision_ts_ns(e: Mapping[str, Any]) -> int | None:
    for k in _DECISION_TS_KEYS:
        v = e.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


def _first_action_ts_ns(python_events: Sequence[Mapping[str, Any]]) -> int | None:
    for e in python_events:
        if e.get("type") != "decision":
            continue
        ts = _decision_ts_ns(e)
        if ts is not None:
            return ts
    return None


def _last_action_ts_ns(python_events: Sequence[Mapping[str, Any]]) -> int | None:
    last: int | None = None
    for e in python_events:
        if e.get("type") != "decision":
            continue
        ts = _decision_ts_ns(e)
        if ts is not None:
            last = ts
    return last


def _event_ts_ns(e: Mapping[str, Any]) -> int | None:
    v = e.get("ts_unix_ns")
    if isinstance(v, (int, float)) and v > 0:
        return int(v)
    return None


# --- terminal classification ------------------------------------------------


def classify_terminal(
    *,
    godot_events: Sequence[Mapping[str, Any]],
    python_events: Sequence[Mapping[str, Any]],
    actions_budget: int | None,
    wall_time_budget_sec: float | None = None,
    harness_status: str | None = None,
) -> tuple[str, int | None, str | None]:
    """Classify the terminal event for a single episode.

    Priority cascade:
      1. harness_abort  - external runner metadata says the harness aborted/errored
      2. hazard_collision - any death/collision event in godot.ndjson
      3. success_budget_reached - actions budget reached and transport healthy
      4. transport_drop - apply ratio below threshold without death evidence
      5. timeout - wall-time budget exceeded without success/death
      6. other - reason string required

    Returns (terminal, terminal_ts_ns_or_None, other_reason_or_None).
    """
    if harness_status is not None and harness_status.lower() in _HARNESS_ABORT_STATUSES:
        ts = _last_action_ts_ns(python_events)
        return ("harness_abort", ts, None)

    deaths = [e for e in godot_events if e.get("type") in _DEATH_EVENT_TYPES]
    if deaths:
        first = deaths[0]
        ts = _event_ts_ns(first) or _last_action_ts_ns(python_events)
        return ("hazard_collision", ts, None)

    decisions = [e for e in python_events if e.get("type") == "decision"]
    applies = [e for e in godot_events if e.get("type") == "controller_cmd_applied"]
    decision_count = len(decisions)
    applied_count = len(applies)
    apply_ratio = (applied_count / decision_count) if decision_count > 0 else 1.0

    if (
        actions_budget is not None
        and decision_count >= actions_budget
        and apply_ratio >= _TRANSPORT_DROP_APPLY_RATIO_THRESHOLD
    ):
        ts = _last_action_ts_ns(python_events)
        return ("success_budget_reached", ts, None)

    if (
        decision_count > 0
        and apply_ratio < _TRANSPORT_DROP_APPLY_RATIO_THRESHOLD
    ):
        ts = _last_action_ts_ns(python_events)
        return ("transport_drop", ts, None)

    first_ts = _first_action_ts_ns(python_events)
    last_ts = _last_action_ts_ns(python_events)
    if (
        wall_time_budget_sec is not None
        and first_ts is not None
        and last_ts is not None
        and (last_ts - first_ts) / 1_000_000_000.0 >= wall_time_budget_sec
    ):
        return ("timeout", last_ts, None)

    last_ts = _last_action_ts_ns(python_events)
    return ("other", last_ts, "unclassified")


# --- top-level episode construction ----------------------------------------


def episode_from_events(
    *,
    godot_events: Sequence[Mapping[str, Any]],
    python_events: Sequence[Mapping[str, Any]],
    episode_id: str,
    actions_budget: int | None,
    wall_time_budget_sec: float | None = None,
    harness_status: str | None = None,
) -> Episode:
    meta = run_metadata_from_events(godot_events)
    ignore_death_active = bool(meta.get("ignore_death_active"))

    actions = extract_actions(python_events)

    terminal, terminal_ts_ns, other_reason = classify_terminal(
        godot_events=godot_events,
        python_events=python_events,
        actions_budget=actions_budget,
        wall_time_budget_sec=wall_time_budget_sec,
        harness_status=harness_status,
    )

    first_ts_ns = _first_action_ts_ns(python_events)
    if first_ts_ns is not None and terminal_ts_ns is not None:
        wall_time = max((terminal_ts_ns - first_ts_ns) / 1_000_000_000.0, 0.0)
    elif first_ts_ns is not None:
        last_ts_ns = _last_action_ts_ns(python_events) or first_ts_ns
        wall_time = max((last_ts_ns - first_ts_ns) / 1_000_000_000.0, 0.0)
    else:
        wall_time = 0.0

    return Episode(
        episode_id=episode_id,
        terminal=terminal,
        actions=actions,
        wall_time_seconds=wall_time,
        ignore_death_active=ignore_death_active,
        other_reason=other_reason if terminal == "other" else None,
    )


def load_episode(
    *,
    godot_path: str | Path,
    python_path: str | Path,
    episode_id: str,
    actions_budget: int | None,
    wall_time_budget_sec: float | None = None,
    harness_status: str | None = None,
) -> Episode:
    """Load an episode from NDJSON artifact paths."""
    godot_events = load_ndjson(godot_path)
    python_events = load_ndjson(python_path)
    return episode_from_events(
        godot_events=godot_events,
        python_events=python_events,
        episode_id=episode_id,
        actions_budget=actions_budget,
        wall_time_budget_sec=wall_time_budget_sec,
        harness_status=harness_status,
    )


# Re-export so `from sight_agent.evaluator.episodes import TERMINAL_EVENTS` works
# without forcing callers through metrics.
__all__ = [
    "Episode",
    "TERMINAL_EVENTS",
    "RUN_META_IGNORE_DEATH_KEYS",
    "load_ndjson",
    "extract_actions",
    "run_metadata_from_events",
    "classify_terminal",
    "episode_from_events",
    "load_episode",
]
