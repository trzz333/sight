"""Tests for the P3 live runner planning surface.

No Godot launches. The live-episode orchestration tests stub launch_godot,
_wait_for_port, run_python_client, and _stop_godot at the module level so the
runner's bookkeeping (env construction, artifact paths, meta.json,
harness_status precedence) can be exercised without subprocesses.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_P3_EVAL_PATH = REPO_ROOT / "scripts" / "run_p3_eval.py"


def _import_module():
    spec = importlib.util.spec_from_file_location(
        "run_p3_eval_live_under_test", RUN_P3_EVAL_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_p3_eval_live_under_test"] = module
    spec.loader.exec_module(module)
    return module


# --- argparse mode selection ----------------------------------------------


def test_parse_args_requires_mode():
    """Exactly one mode is required; missing both must error."""
    module = _import_module()
    with pytest.raises(SystemExit):
        module.parse_args([])


def test_parse_args_rejects_both_modes():
    """--live and --from-artifacts are mutually exclusive."""
    module = _import_module()
    with pytest.raises(SystemExit):
        module.parse_args(["--live", "--from-artifacts", "runs/eval/foo"])


def test_parse_args_live_flag_sets_live_true():
    module = _import_module()
    args = module.parse_args(["--live", "--episodes", "2"])
    assert args.live is True
    assert args.from_artifacts is None
    assert args.episodes == 2


def test_parse_args_from_artifacts_keeps_live_false():
    module = _import_module()
    args = module.parse_args(["--from-artifacts", "runs/eval/foo"])
    assert args.live is False
    assert args.from_artifacts == "runs/eval/foo"


def test_parse_args_exposes_live_args():
    """All live-mode args resolve with sensible defaults."""
    module = _import_module()
    args = module.parse_args(["--live"])
    assert args.actions_budget == 300
    assert args.wall_time_budget_sec == 120.0
    assert args.port == module.DEFAULT_PORT
    assert args.apply_grace_sec == module.DEFAULT_APPLY_GRACE_SEC
    assert args.run_id is None
    assert args.out_dir is None
    assert args.godot_exe is None
    assert args.project_dir is None


# --- env construction ------------------------------------------------------


def test_build_live_child_env_strips_ignore_death():
    module = _import_module()
    base = {
        "PATH": "/usr/bin",
        "SIGHT_TCP_IGNORE_DEATH": "1",
        "FOO": "bar",
    }
    env = module.build_live_child_env(
        base,
        port=8765,
        godot_log_path=Path(r"C:\tmp\godot.ndjson"),
        actions_budget=300,
        episode_id="ep000001",
    )
    # Spec invariant: P3 live runs never carry SIGHT_TCP_IGNORE_DEATH.
    assert "SIGHT_TCP_IGNORE_DEATH" not in env
    # GDScript surface vars set as expected.
    assert env["SIGHT_TCP_MODE"] == "1"
    assert env["SIGHT_TCP_PORT"] == "8765"
    assert env["SIGHT_GODOT_LOG_PATH"] == str(Path(r"C:\tmp\godot.ndjson"))
    assert env["SIGHT_P3_ACTIONS_BUDGET"] == "300"
    assert env["SIGHT_EPISODE_ID"] == "ep000001"
    # Inherited unrelated vars pass through.
    assert env["PATH"] == "/usr/bin"
    assert env["FOO"] == "bar"


def test_build_live_child_env_does_not_mutate_base():
    module = _import_module()
    base = {"SIGHT_TCP_IGNORE_DEATH": "1", "PATH": "/x"}
    module.build_live_child_env(
        base,
        port=1,
        godot_log_path=Path(r"C:\tmp\g.ndjson"),
        actions_budget=1,
        episode_id="ep000001",
    )
    assert base == {"SIGHT_TCP_IGNORE_DEATH": "1", "PATH": "/x"}


# --- episode id format -----------------------------------------------------


def test_episode_id_zero_padded_six_digits():
    module = _import_module()
    assert module.episode_id_for_index(1) == "ep000001"
    assert module.episode_id_for_index(42) == "ep000042"
    assert module.episode_id_for_index(123456) == "ep123456"


# --- godot exe resolution --------------------------------------------------


def test_resolve_godot_exe_prefers_arg(tmp_path: Path):
    module = _import_module()
    fake_exe = tmp_path / "godot.exe"
    fake_exe.write_bytes(b"")
    p = module.resolve_godot_exe(
        str(fake_exe),
        {"SIGHT_GODOT_EXE": str(tmp_path / "missing-from-env.exe")},
    )
    assert p == fake_exe


def test_resolve_godot_exe_falls_back_to_env(tmp_path: Path):
    module = _import_module()
    fake_exe = tmp_path / "godot-env.exe"
    fake_exe.write_bytes(b"")
    p = module.resolve_godot_exe(None, {"SIGHT_GODOT_EXE": str(fake_exe)})
    assert p == fake_exe


def test_resolve_godot_exe_raises_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _import_module()
    missing_arg = tmp_path / "nope-arg.exe"
    missing_env = tmp_path / "nope-env.exe"
    missing_fallback = tmp_path / "nope-fallback.exe"
    # The real fallback path exists on the maintainer's machine; force-miss it
    # so the test is deterministic regardless of host.
    monkeypatch.setattr(
        module, "DEFAULT_GODOT_EXE_FALLBACK", str(missing_fallback)
    )
    with pytest.raises(FileNotFoundError):
        module.resolve_godot_exe(
            str(missing_arg), {"SIGHT_GODOT_EXE": str(missing_env)}
        )


# --- live episode orchestration (fully stubbed; no subprocesses) -----------


class _FakeProc:
    """Minimal Popen-compatible stub for orchestration tests."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        return self.returncode


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _make_python_writer(actions_budget: int):
    """Return a fake_run_python_client that writes a synthetic NDJSON."""

    def fake_run_python_client(
        *,
        actions_budget,
        port,
        out_path,
        wire_run_id,
        stdout_path,
        stderr_path,
        timeout_sec,
        cwd,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for seq in range(actions_budget):
                f.write(
                    json.dumps(
                        {
                            "type": "decision",
                            "seq": seq,
                            "action": "noop",
                            "decision_ts_unix_ns": 1_000_000_000 + seq * 33_000_000,
                            "sent_ts_unix_ns": 1_000_000_000 + seq * 33_000_000,
                            "run_id": wire_run_id,
                        }
                    )
                    + "\n"
                )
        stdout_path.write_text("client stdout", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return _FakeCompleted(returncode=0)

    return fake_run_python_client


def test_run_one_live_episode_writes_artifacts_and_meta(tmp_path: Path, monkeypatch):
    module = _import_module()

    eval_root = tmp_path / "eval"
    diag_root = tmp_path / "diag"
    captured_envs: list[dict] = []

    def fake_launch(*, godot_exe, project_dir, env, stdout_path, stderr_path):
        captured_envs.append(dict(env))
        log_path = Path(env["SIGHT_GODOT_LOG_PATH"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({"type": "run_start", "run_id": "live", "seed": 0})
            + "\n"
            + json.dumps(
                {
                    "type": "success_budget_reached",
                    "frame": 100,
                    "applied_count": 4,
                    "actions_budget": 4,
                    "episode_id": env["SIGHT_EPISODE_ID"],
                    "ts_unix_ns": 5_000_000_000,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path.write_text("godot stdout", encoding="utf-8")
        stderr_path.write_text("godot stderr", encoding="utf-8")
        return _FakeProc(returncode=0)

    monkeypatch.setattr(module, "launch_godot", fake_launch)
    monkeypatch.setattr(
        module, "_wait_for_port", lambda host, port, proc, timeout: True
    )
    monkeypatch.setattr(module, "run_python_client", _make_python_writer(4))
    monkeypatch.setattr(module, "_stop_godot", lambda proc, kill_after_sec=5.0: 0)

    meta = module.run_one_live_episode(
        batch_run_id="batch-001",
        episode_idx=1,
        eval_root=eval_root,
        diagnostics_root=diag_root,
        godot_exe=Path(r"C:\fake\godot.exe"),
        project_dir=tmp_path,
        port=8765,
        actions_budget=4,
        wall_time_budget_sec=2.0,
        apply_grace_sec=0.0,
        base_env={"SIGHT_TCP_IGNORE_DEATH": "1", "PATH": "/x"},
    )

    assert meta.episode_id == "ep000001"
    assert meta.wire_run_id == "batch-001-ep000001"
    assert meta.harness_status == "ok"

    ep_dir = eval_root / "episodes" / "ep000001"
    assert (ep_dir / "godot.ndjson").exists()
    assert (ep_dir / "python.ndjson").exists()
    meta_path = ep_dir / "meta.json"
    assert meta_path.exists()

    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in [
        "batch_run_id",
        "episode_id",
        "wire_run_id",
        "mode",
        "actions_budget",
        "wall_time_budget_sec",
        "port",
        "godot_path",
        "python_path",
        "diagnostics_dir",
        "godot_exit_code",
        "client_exit_code",
        "harness_status",
        "started_at",
        "ended_at",
    ]:
        assert key in meta_obj, f"meta.json missing {key}"
    assert meta_obj["mode"] == "live"
    assert meta_obj["batch_run_id"] == "batch-001"
    assert meta_obj["wire_run_id"] == "batch-001-ep000001"
    assert meta_obj["actions_budget"] == 4
    assert meta_obj["port"] == 8765
    assert meta_obj["harness_status"] == "ok"

    diag_episode_dir = diag_root / "ep000001"
    for name in (
        "godot_stdout.log",
        "godot_stderr.log",
        "python_stdout.log",
        "python_stderr.log",
    ):
        assert (diag_episode_dir / name).exists(), f"diagnostics missing {name}"

    assert len(captured_envs) == 1
    env = captured_envs[0]
    # Spec invariant: P3 live runs never carry SIGHT_TCP_IGNORE_DEATH downstream.
    assert "SIGHT_TCP_IGNORE_DEATH" not in env
    assert env["SIGHT_TCP_MODE"] == "1"
    assert env["SIGHT_TCP_PORT"] == "8765"
    assert env["SIGHT_P3_ACTIONS_BUDGET"] == "4"
    assert env["SIGHT_EPISODE_ID"] == "ep000001"
    assert env["SIGHT_GODOT_LOG_PATH"] == str(ep_dir / "godot.ndjson")


def test_run_one_live_episode_marks_harness_abort_when_port_never_binds(
    tmp_path: Path, monkeypatch
):
    module = _import_module()

    def fake_launch(*, godot_exe, project_dir, env, stdout_path, stderr_path):
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return _FakeProc(returncode=99)

    monkeypatch.setattr(module, "launch_godot", fake_launch)
    monkeypatch.setattr(
        module, "_wait_for_port", lambda host, port, proc, timeout: False
    )
    monkeypatch.setattr(module, "_stop_godot", lambda proc, kill_after_sec=5.0: 99)

    eval_root = tmp_path / "eval"
    diag_root = tmp_path / "diag"
    meta = module.run_one_live_episode(
        batch_run_id="b",
        episode_idx=2,
        eval_root=eval_root,
        diagnostics_root=diag_root,
        godot_exe=Path(r"C:\fake\godot.exe"),
        project_dir=tmp_path,
        port=8765,
        actions_budget=4,
        wall_time_budget_sec=2.0,
        apply_grace_sec=0.0,
        base_env={"PATH": "/x"},
    )

    assert meta.episode_id == "ep000002"
    assert meta.harness_status == "harness_abort"
    assert meta.client_exit_code is None
    meta_obj = json.loads(
        (eval_root / "episodes" / "ep000002" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta_obj["harness_status"] == "harness_abort"


def test_run_one_live_episode_godot_terminal_wins_over_client_failure(
    tmp_path: Path, monkeypatch
):
    """Spec nuance: nonzero client exit is not harness_abort if Godot wrote a terminal."""
    module = _import_module()

    def fake_launch(*, godot_exe, project_dir, env, stdout_path, stderr_path):
        log_path = Path(env["SIGHT_GODOT_LOG_PATH"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            json.dumps({"type": "run_start", "run_id": "live", "seed": 0})
            + "\n"
            + json.dumps(
                {"type": "death", "frame": 50, "ts_unix_ns": 9_000_000_000}
            )
            + "\n",
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return _FakeProc(returncode=0)

    def fake_client_fails(
        *,
        actions_budget,
        port,
        out_path,
        wire_run_id,
        stdout_path,
        stderr_path,
        timeout_sec,
        cwd,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("client died", encoding="utf-8")
        return _FakeCompleted(returncode=1)

    monkeypatch.setattr(module, "launch_godot", fake_launch)
    monkeypatch.setattr(
        module, "_wait_for_port", lambda host, port, proc, timeout: True
    )
    monkeypatch.setattr(module, "run_python_client", fake_client_fails)
    monkeypatch.setattr(module, "_stop_godot", lambda proc, kill_after_sec=5.0: 0)

    eval_root = tmp_path / "eval"
    diag_root = tmp_path / "diag"
    meta = module.run_one_live_episode(
        batch_run_id="b",
        episode_idx=3,
        eval_root=eval_root,
        diagnostics_root=diag_root,
        godot_exe=Path(r"C:\fake\godot.exe"),
        project_dir=tmp_path,
        port=8765,
        actions_budget=4,
        wall_time_budget_sec=2.0,
        apply_grace_sec=0.0,
        base_env={"PATH": "/x"},
    )

    # Godot terminal evidence (death) wins over nonzero client exit.
    assert meta.harness_status == "ok"
    assert meta.client_exit_code == 1


def test_run_one_live_episode_harness_abort_when_no_terminal_and_client_fails(
    tmp_path: Path, monkeypatch
):
    """Inverse of the previous test: no terminal in godot.ndjson AND client fails -> abort."""
    module = _import_module()

    def fake_launch(*, godot_exe, project_dir, env, stdout_path, stderr_path):
        log_path = Path(env["SIGHT_GODOT_LOG_PATH"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # run_start only, no terminal event
        log_path.write_text(
            json.dumps({"type": "run_start", "run_id": "live", "seed": 0}) + "\n",
            encoding="utf-8",
        )
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return _FakeProc(returncode=0)

    def fake_client_fails(
        *,
        actions_budget,
        port,
        out_path,
        wire_run_id,
        stdout_path,
        stderr_path,
        timeout_sec,
        cwd,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("client crashed", encoding="utf-8")
        return _FakeCompleted(returncode=2)

    monkeypatch.setattr(module, "launch_godot", fake_launch)
    monkeypatch.setattr(
        module, "_wait_for_port", lambda host, port, proc, timeout: True
    )
    monkeypatch.setattr(module, "run_python_client", fake_client_fails)
    monkeypatch.setattr(module, "_stop_godot", lambda proc, kill_after_sec=5.0: 0)

    eval_root = tmp_path / "eval"
    diag_root = tmp_path / "diag"
    meta = module.run_one_live_episode(
        batch_run_id="b",
        episode_idx=4,
        eval_root=eval_root,
        diagnostics_root=diag_root,
        godot_exe=Path(r"C:\fake\godot.exe"),
        project_dir=tmp_path,
        port=8765,
        actions_budget=4,
        wall_time_budget_sec=2.0,
        apply_grace_sec=0.0,
        base_env={"PATH": "/x"},
    )

    assert meta.harness_status == "harness_abort"
    assert meta.client_exit_code == 2


def test_aggregate_live_run_writes_summary(tmp_path: Path):
    """Live aggregator round-trip: synthetic metas + on-disk artifacts -> summary.json."""
    module = _import_module()

    eval_root = tmp_path / "eval"
    ep_dir = eval_root / "episodes" / "ep000001"
    ep_dir.mkdir(parents=True, exist_ok=True)

    godot = [
        {"type": "run_start", "run_id": "live", "seed": 0},
        {"type": "controller_cmd_applied", "seq": 0, "frame": 0},
        {"type": "controller_cmd_applied", "seq": 1, "frame": 1},
        {"type": "controller_cmd_applied", "seq": 2, "frame": 2},
        {"type": "controller_cmd_applied", "seq": 3, "frame": 3},
        {
            "type": "success_budget_reached",
            "frame": 100,
            "applied_count": 4,
            "actions_budget": 4,
            "ts_unix_ns": 5_000_000_000,
        },
    ]
    python = [
        {
            "type": "decision",
            "seq": i,
            "action": "noop",
            "decision_ts_unix_ns": 1_000_000_000 + i * 33_000_000,
            "sent_ts_unix_ns": 1_000_000_000 + i * 33_000_000,
            "run_id": "batch-ep000001",
        }
        for i in range(4)
    ]
    with (ep_dir / "godot.ndjson").open("w", encoding="utf-8") as f:
        for e in godot:
            f.write(json.dumps(e) + "\n")
    with (ep_dir / "python.ndjson").open("w", encoding="utf-8") as f:
        for e in python:
            f.write(json.dumps(e) + "\n")

    meta = module.EpisodeMeta(
        batch_run_id="batch-001",
        episode_id="ep000001",
        wire_run_id="batch-001-ep000001",
        mode="live",
        actions_budget=4,
        wall_time_budget_sec=2.0,
        port=8765,
        godot_path=str(ep_dir / "godot.ndjson"),
        python_path=str(ep_dir / "python.ndjson"),
        diagnostics_dir=str(tmp_path / "diag" / "ep000001"),
        godot_exit_code=0,
        client_exit_code=0,
        harness_status="ok",
        started_at="2026-04-27T00:00:00+00:00",
        ended_at="2026-04-27T00:00:01+00:00",
    )

    summary_path = module.aggregate_live_run(
        eval_root=eval_root,
        metas=[meta],
        actions_budget=4,
        wall_time_budget_sec=2.0,
        batch_run_id="batch-001",
    )
    assert summary_path == eval_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "live"
    assert summary["run_id"] == "batch-001"
    assert summary["episode_ids"] == ["ep000001"]
    assert summary["total_episodes"] == 1
    assert summary["wins"] == 1
    assert summary["terminal_counts"]["success_budget_reached"] == 1


def test_main_refuses_with_ignore_death_set_even_when_live_passed(monkeypatch):
    """Refusal guard still runs first when --live is passed."""
    module = _import_module()
    monkeypatch.setenv("SIGHT_TCP_IGNORE_DEATH", "1")
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--live"])
    assert excinfo.value.code == module.IgnoreDeathRefusal.EXIT_CODE


def test_main_requires_mode_when_ignore_death_unset(monkeypatch):
    """With ignore-death unset, missing both mode flags fails argparse."""
    module = _import_module()
    monkeypatch.delenv("SIGHT_TCP_IGNORE_DEATH", raising=False)
    with pytest.raises(SystemExit):
        module.main([])
