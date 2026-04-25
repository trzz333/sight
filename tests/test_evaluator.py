"""Evaluator tests. Synthetic NDJSON event lists mimic a short, deterministic run."""

from __future__ import annotations

import json

from sight_agent import constants
from sight_agent.evaluator import evaluate, load_ndjson, reconcile


def _make_godot_events() -> list[dict]:
    """Synthetic Godot run. ~2 seconds of 60 Hz player_tick with one hazard."""

    run_id = "run_eval_test"
    events: list[dict] = [
        {
            "run_id": run_id,
            "type": "run_start",
            "seed": constants.RANDOM_SEED,
            "screen_width": constants.SCREEN_WIDTH,
            "screen_height": constants.SCREEN_HEIGHT,
        },
        {
            "run_id": run_id,
            "type": "spawn",
            "hazard_id": 1,
            "frame": 30,
            "x": 360.0,
            "y": float(-constants.HAZARD_SIZE),
        },
    ]
    # Player ticks frames 1..120 inclusive. Player hugs x=300 (off the spawn column).
    for f in range(1, 121):
        events.append(
            {
                "run_id": run_id,
                "type": "player_tick",
                "frame": f,
                "player_x": 300.0,
                "player_y": float(constants.SCREEN_HEIGHT - constants.PLAYER_SIZE),
                "action": 0 if f < 60 else -1,
            }
        )
    # Controller-applied command at frame 60, seq 1.
    events.append(
        {
            "run_id": run_id,
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 60,
            "action": "left",
            "move_x": -1,
        }
    )
    events.append({"run_id": run_id, "type": "death", "survival_time": 2.0})
    events.append({"run_id": run_id, "type": "run_end"})
    return events


def _make_python_events() -> list[dict]:
    """One decision with seq=1 matching the applied command; timestamps exercise latency."""

    return [
        {
            "run_id": "run_eval_test",
            "type": "decision",
            "seq": 1,
            "capture_ts_unix_ns": 1_000_000_000,
            "decision_ts_unix_ns": 1_002_500_000,  # 2.5 ms latency
            "player": {"x": 300.0, "y": 508.0},
            "hazards": [{"x": 360.0, "y": 150.0}],
            "action": "left",
            "move_x": -1,
        }
    ]


def test_load_ndjson_round_trip(tmp_path):
    path = tmp_path / "events.ndjson"
    events = _make_python_events()
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    loaded = load_ndjson(path)
    assert loaded == events


def test_reconcile_joins_on_seq():
    godot_events = _make_godot_events()
    python_events = _make_python_events()
    result = reconcile(godot_events, python_events)
    assert len(result["joined"]) == 1
    joined = result["joined"][0]
    assert joined["seq"] == 1
    assert joined["frame"] == 60
    assert joined["python_decision"]["action"] == "left"
    assert joined["godot_applied"]["move_x"] == -1
    assert result["unmatched_python"] == []
    assert result["unmatched_godot"] == []


def test_evaluate_computes_core_metrics():
    godot_events = _make_godot_events()
    python_events = _make_python_events()
    metrics = evaluate(godot_events, python_events)

    # Survival: frames 1..120 -> 119 frames, 119/60 s.
    assert metrics["survival_frames"] == 119
    assert metrics["survival_seconds"] == 119 / constants.PHYSICS_HZ

    # Hazards: exactly one spawn event.
    assert metrics["hazards_spawned"] == 1

    # At least one distance metric produced.
    assert metrics["mean_nearest_hazard_gap_px"] is not None
    assert metrics["min_nearest_hazard_gap_px"] is not None
    assert metrics["min_nearest_hazard_gap_px"] >= 0.0

    # Action changes: 0 for frames 1..59, then -1 at 60..120 -> exactly one change.
    assert metrics["action_changes"] == 1
    assert metrics["action_changes_per_second"] > 0

    # Death logged and run_id propagates.
    assert metrics["death_logged"] is True
    assert metrics["run_id"] == "run_eval_test"
    assert metrics["seed"] == constants.RANDOM_SEED

    # Join count should match the single seq=1 pair.
    assert metrics["joined_count"] == 1
    assert metrics["unmatched_python_count"] == 0
    assert metrics["unmatched_godot_count"] == 0

    # Latency p50 and p95 computed from a single sample.
    assert metrics["decision_latency_p50_ms"] == 2.5
    assert metrics["decision_latency_p95_ms"] == 2.5

    # New contract: run_id surfaced from both sides + duplicate counter clean.
    assert metrics["godot_run_id"] == "run_eval_test"
    assert metrics["python_run_id"] == "run_eval_test"
    assert metrics["run_id_mismatch"] is False
    assert metrics["duplicate_applied_seq_count"] == 0


def test_evaluate_accepts_legacy_agent_tick():
    """Current in-Godot harness emits `agent_tick`. Evaluator must accept it too."""

    run_id = "run_legacy"
    events = [
        {"run_id": run_id, "type": "run_start", "seed": constants.RANDOM_SEED},
        {"run_id": run_id, "type": "spawn", "hazard_id": 1, "frame": 10, "x": 100.0, "y": -24.0},
    ]
    for f in range(1, 61):
        events.append(
            {
                "run_id": run_id,
                "type": "agent_tick",
                "frame": f,
                "player_x": 400.0,
                "action": 0,
            }
        )
    events.append({"run_id": run_id, "type": "death", "survival_time": 1.0})
    events.append({"run_id": run_id, "type": "run_end"})

    metrics = evaluate(events)
    assert metrics["survival_frames"] == 59
    assert metrics["hazards_spawned"] == 1


# --- new instrumentation tests -----------------------------------------------


def test_reconcile_first_applied_frame_wins_on_duplicate_seq():
    """Defensive: even if two controller_cmd_applied events share a seq (legacy log,
    pre-patch tcp_controller behavior, or future regression), reconcile picks the FIRST
    applied frame and surfaces the duplicate count."""

    run_id = "run_dup_seq"
    godot_events = [
        {"run_id": run_id, "type": "run_start", "seed": constants.RANDOM_SEED},
        # First applied frame for seq=1.
        {
            "run_id": run_id,
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 30,
            "action": "left",
            "move_x": -1,
        },
        # Duplicate held-action emission on a later frame. Must NOT win the join.
        {
            "run_id": run_id,
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 31,
            "action": "left",
            "move_x": -1,
        },
        # Another duplicate, even later.
        {
            "run_id": run_id,
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 32,
            "action": "left",
            "move_x": -1,
        },
    ]
    python_events = [
        {
            "run_id": run_id,
            "type": "decision",
            "seq": 1,
            "capture_ts_unix_ns": 1_000_000_000,
            "decision_ts_unix_ns": 1_001_000_000,
            "action": "left",
            "move_x": -1,
        }
    ]

    join = reconcile(godot_events, python_events)
    assert len(join["joined"]) == 1
    assert join["joined"][0]["frame"] == 30  # FIRST applied frame, not the latest
    assert join["duplicate_applied_seq_count"] == 2

    metrics = evaluate(godot_events, python_events)
    assert metrics["duplicate_applied_seq_count"] == 2
    assert metrics["joined_count"] == 1


def test_run_id_mismatch_flag():
    """When both sides report run_id and they differ, evaluator flags it. No silent join."""

    godot_events = [
        {"run_id": "godot-run-A", "type": "run_start", "seed": constants.RANDOM_SEED},
        {
            "run_id": "godot-run-A",
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 10,
            "action": "stay",
            "move_x": 0,
        },
    ]
    python_events = [
        {
            "run_id": "python-run-B",
            "type": "decision",
            "seq": 1,
            "capture_ts_unix_ns": 0,
            "decision_ts_unix_ns": 0,
            "action": "stay",
            "move_x": 0,
        }
    ]
    metrics = evaluate(godot_events, python_events)
    assert metrics["godot_run_id"] == "godot-run-A"
    assert metrics["python_run_id"] == "python-run-B"
    assert metrics["run_id_mismatch"] is True


def test_run_id_missing_returns_none_mismatch():
    """Legacy log: Godot side carries no run_id. Mismatch flag is None, not False, so
    downstream tooling can distinguish 'cannot tell' from 'verified equal'."""

    godot_events = [
        {"type": "run_start", "seed": constants.RANDOM_SEED},
        {
            "type": "controller_cmd_applied",
            "seq": 1,
            "frame": 10,
            "action": "stay",
            "move_x": 0,
        },
    ]
    python_events = [
        {
            "run_id": "python-only",
            "type": "decision",
            "seq": 1,
            "capture_ts_unix_ns": 0,
            "decision_ts_unix_ns": 0,
            "action": "stay",
            "move_x": 0,
        }
    ]
    metrics = evaluate(godot_events, python_events)
    assert metrics["godot_run_id"] is None
    assert metrics["python_run_id"] == "python-only"
    assert metrics["run_id_mismatch"] is None
