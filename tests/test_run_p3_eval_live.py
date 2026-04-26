"""Tests for the P3 live harness scaffolding in scripts/run_p3_eval.py.

No live Godot, no real subprocess. Uses synthetic fixtures, monkeypatched
subprocess.Popen and tcp_client helpers, and the existing on-disk loader
plumbing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_P3_EVAL_PATH = REPO_ROOT / "scripts" / "run_p3_eval.py"


def _import_run_p3_eval():
    """Load the runner script as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(
        "run_p3_eval_live_under_test", RUN_P3_EVAL_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- pure helpers ---------------------------------------------------------


def test_episode_id_for_index_pads_three_digits():
    m = _import_run_p3_eval()
    assert m.episode_id_for_index(1) == "ep_001"
    assert m.episode_id_for_index(10) == "ep_010"
    assert m.episode_id_for_index(123) == "ep_123"


def test_episode_id_for_index_rejects_zero_and_negative():
    m = _import_run_p3_eval()
    with pytest.raises(ValueError):
        m.episode_id_for_index(0)
    with pytest.raises(ValueError):
        m.episode_id_for_index(-1)


def test_episode_dir_path_shape(tmp_path: Path):
    m = _import_run_p3_eval()
    out = tmp_path / "p3-batch"
    d = m.episode_dir(out, "ep_007")
    assert d == out / "episodes" / "ep_007"


def test_wire_run_id_concatenates_batch_and_episode():
    m = _import_run_p3_eval()
    assert (
        m.wire_run_id("p3-20260426T120000", "ep_002")
        == "p3-20260426T120000-ep_002"
    )


def test_build_child_env_sets_tcp_mode_and_port_without_mutating_parent():
    m = _import_run_p3_eval()
    parent = {"PATH": "/usr/bin", "HOME": "/home/x"}
    env = m.build_child_env(parent, port=9999)
    assert env["SIGHT_TCP_MODE"] == "1"
    assert env["SIGHT_TCP_PORT"] == "9999"
    assert "SIGHT_TCP_MODE" not in parent
    assert "SIGHT_TCP_PORT" not in parent


def test_build_child_env_strips_ignore_death_even_if_set():
    """Defense in depth: parent guard already refuses, but the child env must
    not carry the literal regardless. Verifies the spec-invariant strip."""
    m = _import_run_p3_eval()
    parent = {"PATH": "/usr/bin", m.IGNORE_DEATH_ENV_VAR: "1"}
    env = m.build_child_env(parent, port=9999)
    assert m.IGNORE_DEATH_ENV_VAR not in env


# --- meta.json round trip ------------------------------------------------


def test_meta_json_round_trip(tmp_path: Path):
    m = _import_run_p3_eval()
    ep_dir = tmp_path / "episodes" / "ep_001"
    payload = {
        "schema_version": m.META_SCHEMA_VERSION,
        "batch_run_id": "p3-test",
        "episode_id": "ep_001",
        "harness_status": "ok",
    }
    m.write_meta_json(ep_dir, payload)
    loaded = m.read_meta_json(ep_dir)
    assert loaded == payload


def test_read_meta_json_missing_returns_none(tmp_path: Path):
    m = _import_run_p3_eval()
    ep_dir = tmp_path / "no_meta"
    ep_dir.mkdir()
    assert m.read_meta_json(ep_dir) is None


# --- from-artifacts honors meta harness_status ---------------------------


def _decision(seq, action="left"):
    base = 1_000_000_000 + seq * 33_000_000
    return {
        "type": "decision",
        "seq": seq,
        "action": action,
        "move_x": 0,
        "capture_ts_unix_ns": base,
        "decision_ts_unix_ns": base,
        "sent_ts_unix_ns": base + 50_000,
    }


def _applied(seq):
    return {"type": "controller_cmd_applied", "seq": seq, "frame": seq}


def _run_start():
    return {"type": "run_start", "run_id": "test", "seed": 42}


def _write_ndjson(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, separators=(",", ":")) + "\n")


def test_from_artifacts_with_meta_harness_abort_classifies_as_abort(
    tmp_path: Path, monkeypatch
):
    """meta.json harness_status overrides loader's classification cascade."""
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    m = _import_run_p3_eval()

    # Build a clean budget-reached run; meta forces harness_abort.
    god = [_run_start()] + [_applied(i) for i in range(4)]
    py = [_decision(i) for i in range(4)]

    run_root = tmp_path / "p3-meta"
    ep_dir = run_root / "episodes" / "ep_001"
    _write_ndjson(ep_dir / "godot.ndjson", god)
    _write_ndjson(ep_dir / "python.ndjson", py)
    m.write_meta_json(
        ep_dir,
        {"harness_status": "harness_abort", "harness_reason": "bind_fail"},
    )

    rc = m.main(
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
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "from_artifacts"
    assert summary["wins"] == 0
    assert summary["terminal_counts"]["harness_abort"] == 1
    assert summary["terminal_counts"]["success_budget_reached"] == 0


def test_from_artifacts_without_meta_preserves_old_behavior(
    tmp_path: Path, monkeypatch
):
    """When meta.json is absent, classification must match the prior slice."""
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    m = _import_run_p3_eval()

    god = [_run_start()] + [_applied(i) for i in range(4)]
    py = [_decision(i) for i in range(4)]

    run_root = tmp_path / "p3-no-meta"
    ep_dir = run_root / "episodes" / "ep_001"
    _write_ndjson(ep_dir / "godot.ndjson", god)
    _write_ndjson(ep_dir / "python.ndjson", py)
    # Intentionally no meta.json.

    rc = m.main(
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
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["wins"] == 1
    assert summary["terminal_counts"]["success_budget_reached"] == 1


# --- live-mode refusal still runs before subprocess ----------------------


def test_live_mode_refusal_runs_before_any_subprocess(monkeypatch):
    """Parent env with SIGHT_TCP_IGNORE_DEATH set must refuse before Popen."""
    m = _import_run_p3_eval()
    monkeypatch.setenv("SIGHT_TCP_IGNORE_DEATH", "1")

    import subprocess as _sub
    def _explode(*a, **kw):
        raise AssertionError("Popen invoked despite refusal guard")
    monkeypatch.setattr(_sub, "Popen", _explode)

    with pytest.raises(SystemExit) as exc:
        m.main(["--episodes", "1"])
    assert exc.value.code == m.IgnoreDeathRefusal.EXIT_CODE


# --- live preflight: missing godot exe -----------------------------------


def test_live_mode_missing_godot_exe_returns_preflight_error(
    tmp_path: Path, monkeypatch
):
    """Missing godot exe should fail preflight before any subprocess launch."""
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    m = _import_run_p3_eval()

    import subprocess as _sub
    def _explode(*a, **kw):
        raise AssertionError("Popen invoked despite preflight failure")
    monkeypatch.setattr(_sub, "Popen", _explode)

    rc = m.main(
        [
            "--episodes",
            "1",
            "--godot-exe",
            str(tmp_path / "does_not_exist.exe"),
            "--project-path",
            str(REPO_ROOT / "games" / "signal-dodge"),
            "--out-dir",
            str(tmp_path / "p3-preflight"),
        ]
    )
    assert rc == 5  # live preflight failed
    assert rc != m.IgnoreDeathRefusal.EXIT_CODE


# --- monkeypatched live: Godot never binds -> harness_abort + summary ---


class _FakeProc:
    def __init__(self, exit_code: int = 0):
        self._exit_code = exit_code
        self.returncode = exit_code
        self.pid = 12345
        self._terminated = False

    def poll(self):
        return self._exit_code

    def wait(self, timeout=None):
        return self._exit_code

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True


def test_live_mode_records_harness_abort_when_godot_never_binds(
    tmp_path: Path, monkeypatch
):
    """Fake Popen + stubbed wait_for_port_bind=False produces a clean harness
    abort record. Summary still lands and counts the abort."""
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    m = _import_run_p3_eval()

    fake_godot = tmp_path / "fake_godot.exe"
    fake_godot.write_bytes(b"\x00")
    fake_runs_dir = tmp_path / "godot_runs"
    fake_runs_dir.mkdir()
    project = REPO_ROOT / "games" / "signal-dodge"

    import subprocess as _sub
    def _fake_popen(*args, **kwargs):
        # Close any file handles the runner opened so Windows tmpdir cleanup
        # does not fight us.
        for k in ("stdout", "stderr"):
            v = kwargs.get(k)
            if v is not None:
                try:
                    v.close()
                except Exception:
                    pass
        return _FakeProc(exit_code=0)
    monkeypatch.setattr(_sub, "Popen", _fake_popen)

    from sight_agent.harness import tcp_client as _tcp
    monkeypatch.setattr(_tcp, "wait_for_port_bind", lambda h, p, t, **kw: False)

    out_dir = tmp_path / "p3-fakelive"
    rc = m.main(
        [
            "--episodes",
            "1",
            "--actions-budget",
            "4",
            "--wall-time-budget-sec",
            "5",
            "--godot-exe",
            str(fake_godot),
            "--project-path",
            str(project),
            "--godot-runs-dir",
            str(fake_runs_dir),
            "--port",
            "8765",
            "--connect-timeout-sec",
            "0.5",
            "--apply-grace-sec",
            "0.0",
            "--interval-sec",
            "0.0",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "p3-fakelive",
        ]
    )
    assert rc == 0
    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "live"
    assert summary["episode_ids"] == ["ep_001"]
    assert summary["terminal_counts"]["harness_abort"] == 1
    assert summary["game_id"] == m.GAME_ID

    meta_path = out_dir / "episodes" / "ep_001" / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["harness_status"] == "harness_abort"
    assert meta["harness_reason"] == "godot_did_not_bind_port"
    assert meta["wire_run_id"] == "p3-fakelive-ep_001"
    assert meta["actions_sent"] == 0
    assert meta["schema_version"] == m.META_SCHEMA_VERSION
