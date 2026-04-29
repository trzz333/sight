"""Integration smoke test for the H1 CartPole PPO + NDJSON pipeline.

Runs a tiny PPO training loop (~128 timesteps) end-to-end and asserts that
events.ndjson contains run_start, at least one train_metrics, at least one
eval, and run_end events, all with schema_version == 1.

Hyperparameters are overridden ONLY for this smoke test to keep runtime under
a few seconds. The production config (configs/rl/cartpole_ppo_h1.yaml) uses
SB3 PPO defaults exactly as required by the H1 spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sight_agent.rl.train import run


@pytest.fixture
def smoke_cfg(tmp_path: Path) -> dict:
    return {
        "run": {
            "phase": "H1",
            "name": "cartpole_ppo_smoke",
            "seed": 0,
            "out_dir": str(tmp_path),
            "run_id_override": "smoke",
        },
        "env": {"id": "CartPole-v1", "n_envs": 1},
        "algo": {
            "framework": "stable-baselines3",
            "name": "PPO",
            "policy": "MlpPolicy",
            "device": "cpu",
            # Smoke-only overrides to keep the test fast. Production config
            # leaves hyperparams empty to use SB3 defaults.
            "hyperparams": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        },
        "train": {"total_timesteps": 128},
        "eval": {"eval_freq": 64, "n_eval_episodes": 1, "deterministic": True},
        "logging": {"format": "ndjson"},
    }


def _read_ndjson_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def test_h1_smoke_writes_required_events(tmp_path: Path, smoke_cfg: dict) -> None:
    rc = run(smoke_cfg, config_path="<smoke>")
    assert rc == 0

    run_dir = tmp_path / "cartpole_ppo_smoke" / "smoke"
    events_path = run_dir / "events.ndjson"
    summary_path = run_dir / "summary.json"

    assert events_path.exists(), f"missing {events_path}"
    assert summary_path.exists(), f"missing {summary_path}"

    events = _read_ndjson_events(events_path)
    assert len(events) >= 4, f"expected >=4 events, got {len(events)}"

    event_types = [e["event"] for e in events]
    assert "run_start" in event_types
    assert "run_end" in event_types
    assert "eval" in event_types
    assert "train_metrics" in event_types or "rollout" in event_types

    for e in events:
        assert e["schema_version"] == 1
        assert e["phase"] == "H1"
        assert e["env_id"] == "CartPole-v1"
        assert e["algo"] == "PPO"
        assert e["framework"] == "stable-baselines3"
        assert e["seed"] == 0
        assert "ts_utc" in e

    # No TensorBoard or W&B artifacts should be written next to events.ndjson.
    siblings = {p.name for p in run_dir.iterdir()}
    assert not any(name.startswith("events.out.tfevents") for name in siblings)
    assert "wandb" not in siblings


def test_h1_smoke_run_start_has_versions_and_effective_hparams(
    tmp_path: Path, smoke_cfg: dict
) -> None:
    rc = run(smoke_cfg, config_path="<smoke>")
    assert rc == 0

    events_path = tmp_path / "cartpole_ppo_smoke" / "smoke" / "events.ndjson"
    events = _read_ndjson_events(events_path)
    run_start = next(e for e in events if e["event"] == "run_start")

    versions = run_start["versions"]
    for key in ("python", "gymnasium", "stable_baselines3", "torch"):
        assert key in versions and isinstance(versions[key], str) and versions[key]

    hp = run_start["effective_hyperparams"]
    # Best-effort introspection. These keys exist on every recent SB3 PPO model.
    for key in ("learning_rate", "n_steps", "batch_size", "gamma", "policy_class", "device"):
        assert key in hp, f"missing effective hyperparam {key}"
    assert hp["device"] == "cpu"


def test_h1_smoke_run_end_has_artifact_paths(tmp_path: Path, smoke_cfg: dict) -> None:
    rc = run(smoke_cfg, config_path="<smoke>")
    assert rc == 0
    events = _read_ndjson_events(
        tmp_path / "cartpole_ppo_smoke" / "smoke" / "events.ndjson"
    )
    run_end = next(e for e in events if e["event"] == "run_end")
    assert run_end["status"] == "ok"
    assert run_end["total_timesteps"] >= 128
    assert "events_ndjson" in run_end["artifact_paths"]
    assert "summary_json" in run_end["artifact_paths"]
    assert run_end["elapsed_seconds"] >= 0.0
