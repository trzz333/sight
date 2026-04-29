"""Tests for the P3 episode loader and run_p3_eval scaffold.

Synthetic in-memory and on-disk fixtures only. No Godot, no harness, no live
TCP. The on-disk tests use pytest's tmp_path to exercise the full
NDJSON-to-Episode-to-aggregate plumbing without touching real artifacts.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sight_agent.evaluator.episodes import (
    Episode,
    classify_terminal,
    episode_from_events,
    extract_actions,
    load_episode,
    load_ndjson,
    run_metadata_from_events,
)
from sight_agent.evaluator.metrics import aggregate


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_P3_EVAL_PATH = REPO_ROOT / "scripts" / "run_p3_eval.py"


# --- helpers ---------------------------------------------------------------


def _decision(seq: int, action: str, ts_ns: int = 0) -> dict:
    """Synthetic Python decision event."""
    base_ts = ts_ns or (1_000_000_000 + seq * 33_000_000)
    return {
        "type": "decision",
        "seq": seq,
        "action": action,
        "move_x": 0,
        "capture_ts_unix_ns": base_ts,
        "decision_ts_unix_ns": base_ts,
        "sent_ts_unix_ns": base_ts + 50_000,
    }


def _applied(seq: int, frame: int = 0) -> dict:
    """Synthetic Godot controller_cmd_applied event."""
    return {
        "type": "controller_cmd_applied",
        "seq": seq,
        "frame": frame,
    }


def _run_start(*, ignore_death: bool = False, run_id: str = "test-run") -> dict:
    rs: dict = {
        "type": "run_start",
        "run_id": run_id,
        "seed": 42,
    }
    if ignore_death:
        rs["ignore_death"] = True
    return rs


def _death(ts_ns: int = 5_000_000_000) -> dict:
    return {
        "type": "death",
        "ts_unix_ns": ts_ns,
        "frame": 120,
    }


def _build_clean_run(actions_budget: int) -> tuple[list[dict], list[dict]]:
    """Decisions + applies for a transport-clean budget-reached run."""
    godot: list[dict] = [_run_start()]
    python: list[dict] = []
    actions = ("left", "right", "noop", "noop")
    for seq in range(actions_budget):
        python.append(_decision(seq, actions[seq % len(actions)]))
        godot.append(_applied(seq, frame=seq))
    return godot, python


# --- terminal classification ----------------------------------------------


def test_loader_classifies_success_budget_reached():
    godot, python = _build_clean_run(actions_budget=8)
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_success",
        actions_budget=8,
    )
    assert ep.terminal == "success_budget_reached"
    assert ep.ignore_death_active is False
    assert len(ep.actions) == 8
    assert ep.wall_time_seconds > 0


def test_loader_classifies_hazard_collision_from_death_event():
    godot, python = _build_clean_run(actions_budget=4)
    godot.append(_death(ts_ns=python[-1]["decision_ts_unix_ns"] + 100_000_000))
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_hazard",
        actions_budget=8,  # not reached, but death wins anyway
    )
    assert ep.terminal == "hazard_collision"
    assert ep.ignore_death_active is False


def test_loader_classifies_transport_drop_on_low_apply_ratio():
    godot: list[dict] = [_run_start()]
    python: list[dict] = []
    for seq in range(10):
        python.append(_decision(seq, "left"))
    # Apply only 2 of 10 decisions, no death, no abort.
    godot.append(_applied(0))
    godot.append(_applied(1))
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_transport_drop",
        actions_budget=20,
    )
    assert ep.terminal == "transport_drop"


def test_loader_classifies_harness_abort_from_status():
    godot, python = _build_clean_run(actions_budget=3)
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_abort",
        actions_budget=10,
        harness_status="abort",
    )
    assert ep.terminal == "harness_abort"


def test_loader_classifies_other_when_unclassified():
    """Few decisions, no death, no abort, no budget reached, no wall ceiling -> other."""
    godot: list[dict] = [_run_start()]
    python: list[dict] = [_decision(0, "noop")]
    godot.append(_applied(0))
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_other",
        actions_budget=100,
    )
    assert ep.terminal == "other"
    assert ep.other_reason == "unclassified"


def test_classify_terminal_priority_abort_beats_death():
    """Harness abort dominates death evidence in the priority cascade."""
    godot = [_run_start(), _death()]
    python = [_decision(0, "left")]
    terminal, _ts, _reason = classify_terminal(
        godot_events=godot,
        python_events=python,
        actions_budget=10,
        harness_status="abort",
    )
    assert terminal == "harness_abort"


# --- ignore-death exclusion ------------------------------------------------


def test_loader_marks_ignore_death_run_excluded():
    godot, python = _build_clean_run(actions_budget=4)
    godot[0] = _run_start(ignore_death=True)
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_excluded",
        actions_budget=4,
    )
    assert ep.ignore_death_active is True


def test_loader_recognizes_legacy_tcp_ignore_death_key():
    godot, python = _build_clean_run(actions_budget=4)
    godot[0] = {"type": "run_start", "run_id": "legacy", "tcp_ignore_death": True}
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_legacy",
        actions_budget=4,
    )
    assert ep.ignore_death_active is True


def test_loader_recognizes_env_block_with_literal_var():
    """Run metadata may mirror the env var literal under env block."""
    godot, python = _build_clean_run(actions_budget=4)
    godot[0] = {
        "type": "run_start",
        "run_id": "env-mirror",
        "env": {"SIGHT_TCP_IGNORE_DEATH": "1"},
    }
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_env_mirror",
        actions_budget=4,
    )
    assert ep.ignore_death_active is True


def test_aggregate_excludes_ignore_death_episode():
    """Round trip: loader output -> aggregate -> excluded_count behavior."""
    god_a, py_a = _build_clean_run(actions_budget=4)
    ep_win = episode_from_events(
        godot_events=god_a, python_events=py_a,
        episode_id="ep_win", actions_budget=4,
    )

    god_b, py_b = _build_clean_run(actions_budget=4)
    god_b[0] = _run_start(ignore_death=True)
    ep_skip = episode_from_events(
        godot_events=god_b, python_events=py_b,
        episode_id="ep_skip", actions_budget=4,
    )

    god_c, py_c = _build_clean_run(actions_budget=4)
    god_c.append(_death(ts_ns=py_c[-1]["decision_ts_unix_ns"] + 50_000_000))
    ep_fail = episode_from_events(
        godot_events=god_c, python_events=py_c,
        episode_id="ep_fail", actions_budget=4,
    )

    metrics = aggregate([ep_win, ep_skip, ep_fail])
    assert metrics["total_episodes"] == 2
    assert metrics["excluded_count"] == 1
    assert metrics["wins"] == 1
    assert metrics["win_rate"] == 0.5
    assert metrics["terminal_counts"]["success_budget_reached"] == 1
    assert metrics["terminal_counts"]["hazard_collision"] == 1


# --- low-level helpers -----------------------------------------------------


def test_extract_actions_orders_by_seq_when_present():
    events = [
        _decision(2, "right"),
        _decision(0, "left"),
        _decision(1, "noop"),
    ]
    assert extract_actions(events) == ("left", "noop", "right")


def test_run_metadata_returns_ignore_death_false_when_absent():
    meta = run_metadata_from_events([_run_start(ignore_death=False)])
    assert meta["ignore_death_active"] is False
    assert meta["run_start"] is not None


def test_run_metadata_handles_missing_run_start():
    meta = run_metadata_from_events([])
    assert meta["ignore_death_active"] is False
    assert meta["run_start"] is None


# --- on-disk loader round-trip --------------------------------------------


def _write_ndjson(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")


def test_load_episode_round_trip_on_disk(tmp_path: Path):
    god, py = _build_clean_run(actions_budget=5)
    ep_dir = tmp_path / "episodes" / "ep_disk"
    _write_ndjson(ep_dir / "godot.ndjson", god)
    _write_ndjson(ep_dir / "python.ndjson", py)

    ep = load_episode(
        godot_path=ep_dir / "godot.ndjson",
        python_path=ep_dir / "python.ndjson",
        episode_id="ep_disk",
        actions_budget=5,
    )
    assert ep.terminal == "success_budget_reached"
    assert ep.episode_id == "ep_disk"
    assert len(ep.actions) == 5


def test_load_ndjson_missing_file_returns_empty(tmp_path: Path):
    assert load_ndjson(tmp_path / "does_not_exist.ndjson") == []


def test_load_ndjson_raises_on_bad_json(tmp_path: Path):
    bad = tmp_path / "bad.ndjson"
    bad.write_text('{"ok":true}\nthis-is-not-json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_ndjson(bad)


# --- run_p3_eval refusal guard --------------------------------------------


def _import_run_p3_eval():
    """Load the runner script as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(
        "run_p3_eval_under_test", RUN_P3_EVAL_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_p3_eval_under_test"] = module
    spec.loader.exec_module(module)
    return module


def test_run_p3_eval_refuses_when_ignore_death_set():
    module = _import_run_p3_eval()
    with pytest.raises(SystemExit) as excinfo:
        module.refuse_if_ignore_death({"SIGHT_TCP_IGNORE_DEATH": "1"})
    assert excinfo.value.code == module.IgnoreDeathRefusal.EXIT_CODE


def test_run_p3_eval_refuses_with_whitespace_value():
    """Empty string passes; non-empty (incl. typical "1") refuses."""
    module = _import_run_p3_eval()
    # whitespace-only is treated as not set (matches cmd.exe set "VAR= " edge).
    module.refuse_if_ignore_death({"SIGHT_TCP_IGNORE_DEATH": ""})

    with pytest.raises(SystemExit):
        module.refuse_if_ignore_death({"SIGHT_TCP_IGNORE_DEATH": "true"})


def test_run_p3_eval_passes_when_var_absent():
    module = _import_run_p3_eval()
    # No raise. Other env vars should not interfere.
    module.refuse_if_ignore_death({"PATH": "/usr/bin", "FOO": "bar"})


def test_run_p3_eval_main_refuses_before_argparse(monkeypatch):
    """The guard runs before parse_args, so even bogus argv must not save the run."""
    module = _import_run_p3_eval()
    monkeypatch.setenv("SIGHT_TCP_IGNORE_DEATH", "1")
    with pytest.raises(SystemExit):
        module.main(["--bogus-flag-that-would-fail-parsing"])


def test_run_p3_eval_from_artifacts_mode_writes_summary(tmp_path: Path, monkeypatch):
    """End-to-end plumbing: synthetic artifacts -> aggregate -> summary.json on disk."""
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    module = _import_run_p3_eval()

    run_root = tmp_path / "p3-test-run"
    god, py = _build_clean_run(actions_budget=4)
    _write_ndjson(run_root / "episodes" / "ep_001" / "godot.ndjson", god)
    _write_ndjson(run_root / "episodes" / "ep_001" / "python.ndjson", py)

    rc = module.main(
        [
            "--from-artifacts",
            str(run_root),
            "--actions-budget",
            "4",
            "--out-dir",
            str(run_root),
        ]
    )
    assert rc == 0
    summary_path = run_root / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_episodes"] == 1
    assert summary["wins"] == 1
    assert summary["mode"] == "from_artifacts"
    assert summary["episode_ids"] == ["ep_001"]


def test_loader_classifies_success_budget_reached_event_authoritative():
    """Explicit Godot success_budget_reached event drives the terminal."""
    godot, python = _build_clean_run(actions_budget=8)
    godot.append({
        "type": "success_budget_reached",
        "frame": 200,
        "applied_count": 8,
        "actions_budget": 8,
        "ts_unix_ns": python[-1]["decision_ts_unix_ns"] + 50_000_000,
    })
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_success_event",
        actions_budget=8,
    )
    assert ep.terminal == "success_budget_reached"


def test_loader_collision_outranks_success_event():
    """Spec invariant: collision/death outranks success even when both appear."""
    godot, python = _build_clean_run(actions_budget=4)
    death_ts = python[-1]["decision_ts_unix_ns"] + 30_000_000
    godot.append(_death(ts_ns=death_ts))
    godot.append({
        "type": "success_budget_reached",
        "frame": 250,
        "applied_count": 4,
        "actions_budget": 4,
        "ts_unix_ns": death_ts + 10_000_000,
    })
    ep = episode_from_events(
        godot_events=godot,
        python_events=python,
        episode_id="ep_priority",
        actions_budget=4,
    )
    assert ep.terminal == "hazard_collision"


def test_classify_terminal_status_timeout_returns_timeout():
    """harness_status=='timeout' is authoritative ahead of fallback derivation."""
    godot, python = _build_clean_run(actions_budget=4)
    terminal, _ts, _reason = classify_terminal(
        godot_events=godot,
        python_events=python,
        actions_budget=4,
        harness_status="timeout",
    )
    assert terminal == "timeout"


def test_classify_terminal_godot_success_event_with_few_decisions():
    """Godot success event drives terminal even when synthetic check would not."""
    godot = [
        {"type": "run_start", "run_id": "test-run", "seed": 42},
        _applied(0),
        _applied(1),
        {
            "type": "success_budget_reached",
            "frame": 100,
            "applied_count": 2,
            "actions_budget": 2,
            "ts_unix_ns": 5_000_000_000,
        },
    ]
    python = [_decision(0, "left"), _decision(1, "right")]
    terminal, _ts, _reason = classify_terminal(
        godot_events=godot,
        python_events=python,
        actions_budget=999,  # synthetic fallback would NOT fire
    )
    assert terminal == "success_budget_reached"
