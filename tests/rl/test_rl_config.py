"""Tests for sight_agent.rl.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from sight_agent.rl.config import (
    ConfigError,
    apply_cli_overrides,
    load_config,
    validate_config,
)
from sight_agent.rl.godot_config import (
    is_godot_env_id,
    resolve_godot_kwargs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "configs" / "rl" / "cartpole_ppo_h1.yaml"
H3_GODOT_CFG = REPO_ROOT / "configs" / "rl" / "signal_dodge_ppo_h3.yaml"
H4_GODOT_PIXEL_CFG = REPO_ROOT / "configs" / "rl" / "signal_dodge_ppo_h4_pixel.yaml"


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


# --- H3 Godot config and resolver ---------------------------------------


def test_default_h3_godot_yaml_loads() -> None:
    cfg = load_config(H3_GODOT_CFG)
    assert cfg["run"]["phase"] == "H3"
    assert cfg["run"]["name"] == "signal_dodge_ppo_h3"
    assert cfg["run"]["seed"] == 0
    assert cfg["env"]["id"] == "godot:signal-dodge-v0"
    assert cfg["env"]["n_envs"] == 1
    assert cfg["env"]["godot_executable"] is None
    assert cfg["env"]["project_path"] == "games/signal-dodge"
    assert cfg["env"]["max_steps"] == 1800
    assert cfg["algo"]["framework"] == "stable-baselines3"
    assert cfg["algo"]["name"] == "PPO"
    assert cfg["algo"]["policy"] == "MlpPolicy"
    assert cfg["algo"]["device"] == "cpu"
    assert cfg["logging"]["format"] == "ndjson"
    assert cfg["checkpoint"]["enabled"] is True
    assert cfg["checkpoint"]["filename"] == "model.zip"


def test_is_godot_env_id_only_matches_exact_id() -> None:
    assert is_godot_env_id("godot:signal-dodge-v0") is True
    assert is_godot_env_id("CartPole-v1") is False
    assert is_godot_env_id("godot:other-v0") is False
    assert is_godot_env_id(None) is False


def test_resolve_godot_kwargs_returns_empty_for_non_godot(monkeypatch) -> None:
    monkeypatch.setenv("SIGHT_GODOT_EXE", "/should/be/ignored")
    monkeypatch.setenv("SIGHT_GODOT_PROJECT", "/should/be/ignored")
    cfg = yaml.safe_load(DEFAULT_CFG.read_text(encoding="utf-8"))
    assert resolve_godot_kwargs(cfg) == {}


def test_resolve_godot_kwargs_resolves_relative_project_path(monkeypatch) -> None:
    """Relative ``env.project_path`` resolves to the repo-root-relative path."""
    monkeypatch.delenv("SIGHT_GODOT_EXE", raising=False)
    monkeypatch.delenv("SIGHT_GODOT_PROJECT", raising=False)
    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    extra = resolve_godot_kwargs(cfg)
    # godot_executable + project_path are always present for Godot configs.
    # Additional keys (e.g. max_steps for H3, plus pixel/headless for H4)
    # depend on what the YAML carries; see the dedicated H3 omission test
    # and H4 passthrough test for those.
    assert {"godot_executable", "project_path"}.issubset(extra.keys())
    # godot_executable null in YAML and env var unset -> None passed through.
    assert extra["godot_executable"] is None
    expected = (REPO_ROOT / "games" / "signal-dodge").resolve()
    assert Path(extra["project_path"]).resolve() == expected
    assert Path(extra["project_path"]).is_absolute()


def test_resolve_godot_kwargs_uses_env_var_when_yaml_null(
    monkeypatch, tmp_path: Path
) -> None:
    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    cfg["env"]["godot_executable"] = None
    cfg["env"]["project_path"] = None
    fake_exe = tmp_path / "godot.exe"
    fake_proj = tmp_path / "fake-project"
    fake_proj.mkdir()
    monkeypatch.setenv("SIGHT_GODOT_EXE", str(fake_exe))
    monkeypatch.setenv("SIGHT_GODOT_PROJECT", str(fake_proj))
    extra = resolve_godot_kwargs(cfg)
    assert extra["godot_executable"] == str(fake_exe)
    assert Path(extra["project_path"]) == fake_proj


def test_resolve_godot_kwargs_yaml_overrides_env_var(monkeypatch, tmp_path: Path) -> None:
    """Explicit YAML values must NOT be overridden by env vars."""
    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    cfg["env"]["godot_executable"] = str(tmp_path / "yaml-godot.exe")
    cfg["env"]["project_path"] = str(tmp_path / "yaml-project")
    monkeypatch.setenv("SIGHT_GODOT_EXE", str(tmp_path / "envvar-godot.exe"))
    monkeypatch.setenv("SIGHT_GODOT_PROJECT", str(tmp_path / "envvar-project"))
    extra = resolve_godot_kwargs(cfg)
    assert extra["godot_executable"] == str(tmp_path / "yaml-godot.exe")
    assert extra["project_path"] == str(tmp_path / "yaml-project")


# --- H4 Godot pixel config and resolver passthrough ---------------------


def test_default_h4_godot_pixel_yaml_loads() -> None:
    cfg = load_config(H4_GODOT_PIXEL_CFG)
    assert cfg["run"]["phase"] == "H4"
    assert cfg["run"]["name"] == "signal_dodge_ppo_h4_pixel"
    assert cfg["run"]["seed"] == 0
    assert cfg["env"]["id"] == "godot:signal-dodge-v0"
    assert cfg["env"]["n_envs"] == 1
    assert cfg["env"]["godot_executable"] is None
    assert cfg["env"]["project_path"] == "games/signal-dodge"
    assert cfg["env"]["max_steps"] == 1800
    # Pixel mode must be windowed: YAML carries headless=false explicitly so
    # the env construction does not silently flip to headless on a future
    # default change.
    assert cfg["env"]["headless"] is False
    assert cfg["env"]["observation_mode"] == "pixel"
    assert cfg["env"]["pixel_width"] == 84
    assert cfg["env"]["pixel_height"] == 84
    assert cfg["env"]["pixel_channels"] == 1
    assert cfg["algo"]["framework"] == "stable-baselines3"
    assert cfg["algo"]["name"] == "PPO"
    assert cfg["algo"]["policy"] == "CnnPolicy"
    assert cfg["algo"]["device"] == "cpu"
    # Smoke-cheap PPO hyperparams baked into the config.
    assert cfg["algo"]["hyperparams"]["n_steps"] == 64
    assert cfg["algo"]["hyperparams"]["batch_size"] == 32
    assert cfg["algo"]["hyperparams"]["n_epochs"] == 1
    assert cfg["train"]["total_timesteps"] == 128
    assert cfg["eval"]["eval_freq"] == 64
    assert cfg["eval"]["n_eval_episodes"] == 1
    assert cfg["eval"]["deterministic"] is True
    assert cfg["logging"]["format"] == "ndjson"
    assert cfg["checkpoint"]["enabled"] is True
    assert cfg["checkpoint"]["filename"] == "model.zip"


def test_resolve_godot_kwargs_returns_h4_optional_fields(monkeypatch) -> None:
    """H4 YAML carries optional env-construction kwargs; resolver threads them."""
    monkeypatch.delenv("SIGHT_GODOT_EXE", raising=False)
    monkeypatch.delenv("SIGHT_GODOT_PROJECT", raising=False)
    cfg = yaml.safe_load(H4_GODOT_PIXEL_CFG.read_text(encoding="utf-8"))
    extra = resolve_godot_kwargs(cfg)
    assert extra["godot_executable"] is None
    assert Path(extra["project_path"]).is_absolute()
    assert extra["max_steps"] == 1800
    assert extra["headless"] is False
    assert extra["observation_mode"] == "pixel"
    assert extra["pixel_width"] == 84
    assert extra["pixel_height"] == 84
    assert extra["pixel_channels"] == 1


def test_resolve_godot_kwargs_h3_omits_pixel_fields(monkeypatch) -> None:
    """H3 YAML has max_steps but no pixel/headless fields; resolver omits the latter.

    Locks in the "no invented defaults" rule: the resolver returns exactly
    what the YAML carries plus the always-resolved path pair.
    """
    monkeypatch.delenv("SIGHT_GODOT_EXE", raising=False)
    monkeypatch.delenv("SIGHT_GODOT_PROJECT", raising=False)
    cfg = yaml.safe_load(H3_GODOT_CFG.read_text(encoding="utf-8"))
    extra = resolve_godot_kwargs(cfg)
    assert extra["max_steps"] == 1800
    for key in (
        "headless",
        "observation_mode",
        "pixel_width",
        "pixel_height",
        "pixel_channels",
    ):
        assert key not in extra, f"H3 resolver leaked {key!r}"
