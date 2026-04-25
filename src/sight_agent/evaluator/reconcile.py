"""NDJSON reconciler + metrics.

Public entry points:
    load_ndjson(path) -> list[dict]
    reconcile(godot_events, python_events) -> {"joined": [...], "unmatched_python": [...],
                                               "unmatched_godot": [...]}
    evaluate(godot_events, python_events=None) -> metrics dict

The evaluator accepts two Godot event name conventions for the player tick:
`player_tick` (new contract in GPT's P2 spec) and `agent_tick` (existing Godot in-mode
output). This removes the need to break current in-Godot runs to test the reconciler.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .. import constants


PLAYER_TICK_TYPES = ("player_tick", "agent_tick")


def load_ndjson(path: str | Path) -> list[dict]:
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
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


# --- join ---------------------------------------------------------------------


def reconcile(godot_events: list[dict], python_events: list[dict]) -> dict:
    """Join python.decision.seq to godot.controller_cmd_applied.seq.

    Returns a dict with:
        joined: list of {seq, frame, python_decision, godot_applied}
        unmatched_python: list of python decision events with no applied match
        unmatched_godot:  list of godot controller_cmd_applied events with no decision match
    """

    py_decisions: dict[int, dict] = {
        e["seq"]: e for e in python_events if e.get("type") == "decision" and "seq" in e
    }
    applied: dict[int, dict] = {
        e["seq"]: e
        for e in godot_events
        if e.get("type") == "controller_cmd_applied" and "seq" in e
    }

    joined: list[dict] = []
    for seq in sorted(set(py_decisions) & set(applied)):
        ap = applied[seq]
        pd = py_decisions[seq]
        joined.append(
            {
                "seq": seq,
                "frame": ap.get("frame"),
                "python_decision": pd,
                "godot_applied": ap,
            }
        )

    unmatched_python = [pd for seq, pd in py_decisions.items() if seq not in applied]
    unmatched_godot = [ap for seq, ap in applied.items() if seq not in py_decisions]
    return {
        "joined": joined,
        "unmatched_python": unmatched_python,
        "unmatched_godot": unmatched_godot,
    }


# --- metrics ------------------------------------------------------------------


def _aabb_edge_gap(
    px: float,
    py: float,
    player_half: float,
    hx: float,
    hy: float,
    hazard_half: float,
) -> float:
    """AABB edge-to-edge gap between two axis-aligned squares. 0 on contact, negative on overlap."""

    dx = max(abs(px - hx) - (player_half + hazard_half), 0.0)
    dy = max(abs(py - hy) - (player_half + hazard_half), 0.0)
    return math.hypot(dx, dy)


def _reconstruct_hazard_positions(spawn_events: list[dict], frame: int) -> list[tuple[float, float]]:
    """Return list of (x, y) for hazards alive at `frame`. Requires spawn events to carry
    `frame` (spawn frame), `x`, `y`. If `y` is missing, assumes spawn y = -HAZARD_SIZE."""

    out: list[tuple[float, float]] = []
    for e in spawn_events:
        sf = e.get("frame")
        if sf is None or sf > frame:
            continue
        sx = float(e.get("x", 0.0))
        sy = float(e.get("y", -constants.HAZARD_SIZE))
        # Linear fall at HAZARD_SPEED px/sec, locked PHYSICS_HZ tickrate.
        y = sy + constants.HAZARD_SPEED * (frame - sf) / constants.PHYSICS_HZ
        # Drop offscreen hazards to keep the list tight.
        if y > constants.SCREEN_HEIGHT + constants.HAZARD_SIZE:
            continue
        out.append((sx, y))
    return out


def evaluate(godot_events: list[dict], python_events: list[dict] | None = None) -> dict[str, Any]:
    """Compute Sight metrics from Godot (and optional Python) NDJSON event lists."""

    python_events = python_events or []

    player_ticks = [e for e in godot_events if e.get("type") in PLAYER_TICK_TYPES]
    spawn_events = [e for e in godot_events if e.get("type") == "spawn"]
    death_events = [e for e in godot_events if e.get("type") == "death"]
    run_start = next((e for e in godot_events if e.get("type") == "run_start"), None)

    frames = [e["frame"] for e in player_ticks if "frame" in e]
    first_frame = min(frames) if frames else 0
    last_frame = max(frames) if frames else 0
    survival_frames = max(last_frame - first_frame, 0)
    survival_seconds = survival_frames / constants.PHYSICS_HZ

    # Action changes per second. Uses whichever tick type is present.
    action_changes = 0
    prev_action: int | None = None
    for t in sorted(player_ticks, key=lambda e: e.get("frame", 0)):
        a = t.get("action")
        if a is None:
            continue
        if prev_action is not None and a != prev_action:
            action_changes += 1
        prev_action = a
    action_changes_per_second = (
        action_changes / survival_seconds if survival_seconds > 0 else 0.0
    )

    # Nearest hazard gap per frame. Uses reconstructed hazard positions.
    near_miss_threshold_px = 10.0
    gaps: list[float] = []
    near_miss_frames = 0
    for t in player_ticks:
        fr = t.get("frame")
        px = t.get("player_x")
        py = t.get("player_y", constants.SCREEN_HEIGHT - constants.PLAYER_SIZE)
        if fr is None or px is None:
            continue
        hazards = _reconstruct_hazard_positions(spawn_events, int(fr))
        if not hazards:
            continue
        frame_gap = min(
            _aabb_edge_gap(
                float(px),
                float(py),
                constants.PLAYER_HALF,
                hx,
                hy,
                constants.HAZARD_HALF,
            )
            for hx, hy in hazards
        )
        gaps.append(frame_gap)
        if frame_gap <= near_miss_threshold_px:
            near_miss_frames += 1

    mean_gap = statistics.fmean(gaps) if gaps else None
    min_gap = min(gaps) if gaps else None

    # Decision latency from python decision events.
    latencies_ns: list[int] = [
        int(e["decision_ts_unix_ns"]) - int(e["capture_ts_unix_ns"])
        for e in python_events
        if e.get("type") == "decision"
        and "decision_ts_unix_ns" in e
        and "capture_ts_unix_ns" in e
    ]
    decision_latency_p50_ms: float | None = None
    decision_latency_p95_ms: float | None = None
    if latencies_ns:
        latencies_ms = sorted(ns / 1_000_000.0 for ns in latencies_ns)
        decision_latency_p50_ms = statistics.median(latencies_ms)
        # Simple p95: nearest-rank.
        k = max(int(math.ceil(0.95 * len(latencies_ms))) - 1, 0)
        decision_latency_p95_ms = latencies_ms[k]

    join = reconcile(godot_events, python_events)

    return {
        "run_id": run_start.get("run_id") if run_start else None,
        "seed": run_start.get("seed") if run_start else None,
        "survival_frames": survival_frames,
        "survival_seconds": survival_seconds,
        "hazards_spawned": len(spawn_events),
        "action_changes": action_changes,
        "action_changes_per_second": action_changes_per_second,
        "mean_nearest_hazard_gap_px": mean_gap,
        "min_nearest_hazard_gap_px": min_gap,
        "near_miss_frames": near_miss_frames,
        "death_logged": bool(death_events),
        "decision_latency_p50_ms": decision_latency_p50_ms,
        "decision_latency_p95_ms": decision_latency_p95_ms,
        "joined_count": len(join["joined"]),
        "unmatched_python_count": len(join["unmatched_python"]),
        "unmatched_godot_count": len(join["unmatched_godot"]),
    }
