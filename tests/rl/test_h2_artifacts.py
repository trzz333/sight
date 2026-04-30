"""Tests for sight_agent.rl.artifacts (H2 paths + config hash)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from sight_agent.rl.artifacts import (
    build_eval_id,
    build_run_id,
    compute_config_hash,
    is_checkpoint_enabled,
    prepare_eval_artifacts,
    prepare_train_artifacts,
    write_config_effective,
)


def _h2_cfg(out_dir: str) -> dict:
    return {
        "run": {
            "phase": "H2",
            "name": "cartpole_ppo_h2",
            "seed": 0,
            "out_dir": out_dir,
        },
        "env": {"id": "CartPole-v1", "n_envs": 1},
        "algo": {
            "framework": "stable-baselines3",
            "name": "PPO",
            "policy": "MlpPolicy",
            "device": "cpu",
            "hyperparams": {},
        },
        "train": {"total_timesteps": 100},
        "eval": {"eval_freq": 50, "n_eval_episodes": 1, "deterministic": True},
        "logging": {"format": "ndjson"},
        "checkpoint": {"enabled": True, "filename": "model.zip"},
    }


def test_build_run_id_uses_override_when_present() -> None:
    rid = build_run_id("cartpole_ppo_h2", seed=0, override="abc", git_commit="deadbee")
    assert rid == "abc"


def test_build_run_id_format_when_no_override() -> None:
    rid = build_run_id("cartpole_ppo_h2", seed=0, override=None, git_commit="deadbee")
    parts = rid.split("_")
    # <ts>_<name>_seed<n>_<git>
    assert parts[0].endswith("Z") or parts[0].isdigit() or "T" in parts[0]
    assert "cartpole_ppo_h2" in rid
    assert "seed0" in rid
    assert rid.endswith("deadbee")


def test_build_eval_id_includes_seed_and_n() -> None:
    eid = build_eval_id(seed=0, n_eval_episodes=5, source_run_id="abc12345_xyz")
    assert eid.startswith("eval_")
    assert "seed0" in eid
    assert "n5" in eid


def test_prepare_train_artifacts_creates_layout(tmp_path: Path) -> None:
    cfg = _h2_cfg(str(tmp_path))
    arts = prepare_train_artifacts(cfg, run_id="run_X")
    assert arts.run_dir == tmp_path / "cartpole_ppo_h2" / "run_X"
    assert arts.run_dir.exists()
    assert arts.events_path == arts.run_dir / "events.ndjson"
    assert arts.summary_path == arts.run_dir / "summary.json"
    assert arts.config_effective_path == arts.run_dir / "config_effective.yaml"
    assert arts.model_path == arts.run_dir / "model.zip"


def test_prepare_eval_artifacts_creates_evals_subdir(tmp_path: Path) -> None:
    cfg = _h2_cfg(str(tmp_path))
    arts = prepare_train_artifacts(cfg, run_id="run_X")
    eval_arts = prepare_eval_artifacts(arts.run_dir, "eval_Y")
    assert eval_arts.eval_dir == arts.run_dir / "evals" / "eval_Y"
    assert eval_arts.eval_dir.exists()
    assert eval_arts.events_path == eval_arts.eval_dir / "events.ndjson"
    assert eval_arts.summary_path == eval_arts.eval_dir / "summary.json"


def test_write_config_effective_round_trips(tmp_path: Path) -> None:
    cfg = _h2_cfg(str(tmp_path))
    p = tmp_path / "cfg.yaml"
    write_config_effective(p, cfg)
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert loaded == cfg


def test_config_hash_is_stable_for_same_input(tmp_path: Path) -> None:
    cfg = _h2_cfg(str(tmp_path))
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_config_hash_changes_when_seed_changes(tmp_path: Path) -> None:
    cfg_a = _h2_cfg(str(tmp_path))
    cfg_b = _h2_cfg(str(tmp_path))
    cfg_b["run"]["seed"] = 1
    assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)


def test_config_hash_invariant_to_key_order(tmp_path: Path) -> None:
    cfg = _h2_cfg(str(tmp_path))
    # Reverse key order at top level by building a new dict.
    reordered = dict(reversed(list(cfg.items())))
    assert compute_config_hash(cfg) == compute_config_hash(reordered)


def test_is_checkpoint_enabled_true_when_flag_set() -> None:
    cfg = _h2_cfg("runs/rl")
    assert is_checkpoint_enabled(cfg) is True


def test_is_checkpoint_enabled_false_when_no_section() -> None:
    cfg = _h2_cfg("runs/rl")
    cfg.pop("checkpoint")
    assert is_checkpoint_enabled(cfg) is False


def test_is_checkpoint_enabled_false_when_disabled() -> None:
    cfg = _h2_cfg("runs/rl")
    cfg["checkpoint"]["enabled"] = False
    assert is_checkpoint_enabled(cfg) is False
