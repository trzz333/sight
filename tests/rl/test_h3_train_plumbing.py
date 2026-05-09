"""H3 step 8: train/eval plumbing tests for ``godot:signal-dodge-v0``.

Verifies that ``sight_agent.rl.train._build_train_env`` and
``_build_eval_env`` thread the Godot kwargs and a distinct
``run_dir`` per mode through ``make_env``. ``make_env`` is monkeypatched
so no Godot subprocess is spawned and no real VecEnv is constructed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sight_agent.rl import train as train_mod
from sight_agent.rl.artifacts import TrainArtifacts


REPO_ROOT = Path(__file__).resolve().parents[2]
H3_GODOT_CFG = REPO_ROOT / "configs" / "rl" / "signal_dodge_ppo_h3.yaml"
H1_CARTPOLE_CFG = REPO_ROOT / "configs" / "rl" / "cartpole_ppo_h1.yaml"


class _SentinelVecEnv:
    """Stand-in for the VecEnv returned by ``make_env``.

    The plumbing-under-test only stores the return value; nothing in these
    tests calls reset/step on it.
    """

    def __init__(self, label: str) -> None:
        self.label = label


def _make_artifacts(run_dir: Path) -> TrainArtifacts:
    return TrainArtifacts(
        run_id="test-run",
        run_dir=run_dir,
        events_path=run_dir / "events.ndjson",
        summary_path=run_dir / "summary.json",
        config_effective_path=run_dir / "config_effective.yaml",
        model_path=run_dir / "model.zip",
    )


def _install_recording_make_env(monkeypatch) -> list[dict[str, Any]]:
    """Replace ``train.make_env`` with a recorder; return the call log."""
    calls: list[dict[str, Any]] = []

    def fake_make_env(env_id, *args, **kwargs):
        record = {"env_id": env_id, "args": args, **kwargs}
        calls.append(record)
        return _SentinelVecEnv(label=kwargs.get("mode", "?"))

    monkeypatch.setattr(train_mod, "make_env", fake_make_env)
    return calls


def test_build_train_env_passes_godot_kwargs_for_h3_config(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SIGHT_GODOT_EXE", raising=False)
    monkeypatch.delenv("SIGHT_GODOT_PROJECT", raising=False)
    monkeypatch.setenv("SIGHT_GODOT_EXE", str(tmp_path / "godot.exe"))
    calls = _install_recording_make_env(monkeypatch)

    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    artifacts = _make_artifacts(tmp_path)

    out = train_mod._build_train_env(cfg, artifacts)

    assert isinstance(out, _SentinelVecEnv)
    assert len(calls) == 1
    call = calls[0]
    assert call["env_id"] == "godot:signal-dodge-v0"
    assert call["n_envs"] == 1
    assert call["seed"] == 0
    assert call["mode"] == "train"
    assert call["godot_executable"] == str(tmp_path / "godot.exe")
    # Relative YAML project_path resolves to repo-root-relative absolute path.
    assert Path(call["project_path"]).is_absolute()
    assert Path(call["project_path"]).resolve() == (
        REPO_ROOT / "games" / "signal-dodge"
    ).resolve()
    # run_dir is exact pass-through to the env, namespaced by mode.
    assert Path(call["run_dir"]).name == "godot-train"
    assert Path(call["run_dir"]).parent == tmp_path


def test_build_eval_env_passes_distinct_run_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIGHT_GODOT_EXE", str(tmp_path / "godot.exe"))
    calls = _install_recording_make_env(monkeypatch)

    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    artifacts = _make_artifacts(tmp_path)

    train_mod._build_train_env(cfg, artifacts)
    train_mod._build_eval_env(cfg, artifacts)

    assert len(calls) == 2
    train_call, eval_call = calls

    # Both calls carry the Godot kwargs.
    assert train_call["godot_executable"] == str(tmp_path / "godot.exe")
    assert eval_call["godot_executable"] == str(tmp_path / "godot.exe")
    assert train_call["project_path"] == eval_call["project_path"]

    # Modes flip; n_envs is forced to 1 for eval.
    assert train_call["mode"] == "train"
    assert eval_call["mode"] == "eval"
    assert eval_call["n_envs"] == 1

    # Distinct run_dir per mode: <run_dir>/godot-train vs <run_dir>/godot-eval.
    assert train_call["run_dir"] != eval_call["run_dir"]
    assert Path(train_call["run_dir"]).name == "godot-train"
    assert Path(eval_call["run_dir"]).name == "godot-eval"


def test_build_train_env_skips_godot_kwargs_for_cartpole(
    monkeypatch, tmp_path: Path
) -> None:
    """Cartpole path must not leak Godot kwargs or run_dir to ``make_env``."""
    # Setting these would expose a leak if the plumbing called the resolver
    # unconditionally instead of branching on env id.
    monkeypatch.setenv("SIGHT_GODOT_EXE", str(tmp_path / "should-not-be-used.exe"))
    monkeypatch.setenv("SIGHT_GODOT_PROJECT", str(tmp_path / "should-not-be-used"))
    calls = _install_recording_make_env(monkeypatch)

    cfg = yaml.safe_load(H1_CARTPOLE_CFG.read_text(encoding="utf-8"))
    artifacts = _make_artifacts(tmp_path)

    train_mod._build_train_env(cfg, artifacts)
    train_mod._build_eval_env(cfg, artifacts)

    assert len(calls) == 2
    for call in calls:
        assert call["env_id"] == "CartPole-v1"
        assert "godot_executable" not in call
        assert "project_path" not in call
        assert "run_dir" not in call
