"""Tests for sight_agent.rl.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sight_agent.rl.config import (
    ConfigError,
    apply_cli_overrides,
    load_config,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "configs" / "rl" / "cartpole_ppo_h1.yaml"


def test_default_h1_yaml_loads() -> None:
    cfg = load_config(DEFAULT_CFG)
    assert cfg["run"]["phase"] == "H1"
    assert cfg["run"]["name"] == "cartpole_ppo_h1"
    assert cfg["run"]["seed"] == 0
    assert cfg["env"]["id"] == "CartPole-v1"
    assert cfg["env"]["n_envs"] == 1
    assert cfg["algo"]["framework"] == "stable-baselines3"
    assert cfg["algo"]["name"] == "PPO"
    assert cfg["algo"]["policy"] == "MlpPolicy"
    assert cfg["algo"]["device"] == "cpu"
    assert cfg["algo"]["hyperparams"] == {}
    assert cfg["train"]["total_timesteps"] == 25000
    assert cfg["eval"]["eval_freq"] == 5000
    assert cfg["eval"]["n_eval_episodes"] == 5
    assert cfg["eval"]["deterministic"] is True
    assert cfg["logging"]["format"] == "ndjson"


def test_validate_rejects_missing_top_key(tmp_path: Path) -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg.pop("eval")
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_rejects_non_ndjson_logging(tmp_path: Path) -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg["logging"]["format"] = "tensorboard"
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_rejects_zero_timesteps() -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg["train"]["total_timesteps"] = 0
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_validate_rejects_non_dict_hyperparams() -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    cfg["algo"]["hyperparams"] = []
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_apply_cli_overrides_updates_fields() -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    overrides = {
        "seed": 42,
        "total_timesteps": 1024,
        "eval_freq": 256,
        "n_eval_episodes": 2,
        "run_id": "smoke_test",
        "out_dir": "tmp/runs",
    }
    out = apply_cli_overrides(cfg, overrides)
    assert out["run"]["seed"] == 42
    assert out["train"]["total_timesteps"] == 1024
    assert out["eval"]["eval_freq"] == 256
    assert out["eval"]["n_eval_episodes"] == 2
    assert out["run"]["run_id_override"] == "smoke_test"
    assert out["run"]["out_dir"] == "tmp/runs"


def test_apply_cli_overrides_skips_none_values() -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    out = apply_cli_overrides(cfg, {"seed": None, "total_timesteps": None})
    assert out["run"]["seed"] == 0
    assert out["train"]["total_timesteps"] == 25000


def test_apply_cli_overrides_returns_new_dict() -> None:
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    out = apply_cli_overrides(cfg, {"seed": 7})
    assert cfg["run"]["seed"] == 0
    assert out["run"]["seed"] == 7
