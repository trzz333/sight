"""H2 train + evaluate smoke tests.

These exercise the new harness end-to-end with tiny timesteps so the test
runs in seconds. The H1 production smoke test continues to live in
test_cartpole_smoke.py and is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sight_agent.rl.evaluate import run_eval
from sight_agent.rl.train import run as train_run


REPO_ROOT = Path(__file__).resolve().parents[2]
H1_CFG_PATH = REPO_ROOT / "configs" / "rl" / "cartpole_ppo_h1.yaml"
H2_CFG_PATH = REPO_ROOT / "configs" / "rl" / "cartpole_ppo_h2.yaml"


def _h2_smoke_cfg(tmp_path: Path) -> dict:
    return {
        "run": {
            "phase": "H2",
            "name": "cartpole_ppo_h2_smoke",
            "seed": 0,
            "out_dir": str(tmp_path),
            "run_id_override": "smoke_h2",
        },
        "env": {"id": "CartPole-v1", "n_envs": 1},
        "algo": {
            "framework": "stable-baselines3",
            "name": "PPO",
            "policy": "MlpPolicy",
            "device": "cpu",
            "hyperparams": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        },
        "train": {"total_timesteps": 128},
        "eval": {"eval_freq": 64, "n_eval_episodes": 1, "deterministic": True},
        "logging": {"format": "ndjson"},
        "checkpoint": {"enabled": True, "filename": "model.zip"},
    }


def _read_ndjson(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_h1_yaml_still_loads() -> None:
    """H1 config is preserved by the H2 refactor."""
    from sight_agent.rl.config import load_config

    cfg = load_config(H1_CFG_PATH)
    assert cfg["run"]["phase"] == "H1"
    assert cfg["run"]["name"] == "cartpole_ppo_h1"


def test_h2_yaml_loads() -> None:
    from sight_agent.rl.config import load_config

    cfg = load_config(H2_CFG_PATH)
    assert cfg["run"]["phase"] == "H2"
    assert cfg["run"]["name"] == "cartpole_ppo_h2"
    assert cfg["env"]["id"] == "CartPole-v1"
    assert cfg["algo"]["framework"] == "stable-baselines3"
    assert cfg["algo"]["name"] == "PPO"
    assert cfg["train"]["total_timesteps"] == 25000
    assert cfg["eval"]["eval_freq"] == 5000
    assert cfg["eval"]["n_eval_episodes"] == 5
    assert cfg["eval"]["deterministic"] is True
    # Checkpoint section is part of the H2 contract.
    assert cfg["checkpoint"]["enabled"] is True


def test_h2_train_smoke_writes_all_artifacts(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    rc = train_run(cfg, config_path="<smoke>")
    assert rc == 0

    run_dir = tmp_path / "cartpole_ppo_h2_smoke" / "smoke_h2"
    assert (run_dir / "events.ndjson").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "config_effective.yaml").exists()
    assert (run_dir / "model.zip").exists(), "H2 checkpoint must persist model.zip"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert summary["kind"] == "train"
    assert summary["status"] == "ok"
    assert summary["config_path"] == "<smoke>"
    assert isinstance(summary["config_hash"], str) and len(summary["config_hash"]) == 64
    assert summary["artifact_paths"]["model"].endswith("model.zip")
    assert "events" in summary["artifact_paths"]
    assert "summary" in summary["artifact_paths"]
    assert "config_effective" in summary["artifact_paths"]


def test_h2_train_ndjson_validates_line_by_line(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    train_run(cfg, config_path="<smoke>")
    events = _read_ndjson(
        tmp_path / "cartpole_ppo_h2_smoke" / "smoke_h2" / "events.ndjson"
    )
    assert events, "expected at least one NDJSON event"
    types = {e["event"] for e in events}
    assert "run_start" in types
    assert "run_end" in types
    for e in events:
        assert "schema_version" in e
        assert "ts_utc" in e
        assert e["phase"] == "H2"
        assert e["env_id"] == "CartPole-v1"


def test_h2_evaluate_smoke_writes_artifacts(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    train_run(cfg, config_path="<smoke>")
    train_run_dir = tmp_path / "cartpole_ppo_h2_smoke" / "smoke_h2"

    rc, summary = run_eval(
        train_run_dir=train_run_dir,
        n_eval_episodes=2,
        seed=0,
        deterministic=True,
        eval_id_override="eval_smoke",
    )
    assert rc == 0
    assert summary["status"] == "ok"
    assert summary["n_eval_episodes"] == 2
    assert summary["deterministic"] is True
    assert summary["seed"] == 0
    assert isinstance(summary["mean_reward"], float)
    assert isinstance(summary["std_reward"], float)
    assert summary["model_path"].endswith("model.zip")
    assert summary["source_train_run_id"]

    eval_dir = train_run_dir / "evals" / "eval_smoke"
    assert (eval_dir / "events.ndjson").exists()
    assert (eval_dir / "summary.json").exists()


def test_h2_evaluate_ndjson_event_types(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    train_run(cfg, config_path="<smoke>")
    train_run_dir = tmp_path / "cartpole_ppo_h2_smoke" / "smoke_h2"
    run_eval(
        train_run_dir=train_run_dir,
        n_eval_episodes=2,
        seed=0,
        deterministic=True,
        eval_id_override="eval_smoke2",
    )
    eval_events = _read_ndjson(
        train_run_dir / "evals" / "eval_smoke2" / "events.ndjson"
    )
    types = [e["event"] for e in eval_events]
    assert types[0] == "eval_start"
    assert types[-1] == "eval_end"
    assert types.count("eval_episode") == 2
    for e in eval_events:
        assert e["phase"] == "H2"
        assert e["env_id"] == "CartPole-v1"
        assert e["seed"] == 0


def test_h2_run_id_override_lands_on_disk(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    cfg["run"]["run_id_override"] = "deterministic_id"
    train_run(cfg, config_path="<smoke>")
    expected = tmp_path / "cartpole_ppo_h2_smoke" / "deterministic_id"
    assert expected.exists()
    assert (expected / "summary.json").exists()


def test_h2_config_hash_recorded_in_summary_and_run_start(tmp_path: Path) -> None:
    cfg = _h2_smoke_cfg(tmp_path)
    train_run(cfg, config_path="<smoke>")
    run_dir = tmp_path / "cartpole_ppo_h2_smoke" / "smoke_h2"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = _read_ndjson(run_dir / "events.ndjson")
    run_start = next(e for e in events if e["event"] == "run_start")
    assert summary["config_hash"] == run_start["config_hash"]
