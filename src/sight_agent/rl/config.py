"""H1 RL config loading and CLI override merging.

Pure: no SB3, gymnasium, or torch imports. Testable in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_KEYS = ("run", "env", "algo", "train", "eval", "logging")
REQUIRED_RUN_KEYS = ("phase", "name", "seed", "out_dir")
REQUIRED_ENV_KEYS = ("id", "n_envs")
REQUIRED_ALGO_KEYS = ("framework", "name", "policy", "device", "hyperparams")
REQUIRED_TRAIN_KEYS = ("total_timesteps",)
REQUIRED_EVAL_KEYS = ("eval_freq", "n_eval_episodes", "deterministic")
REQUIRED_LOGGING_KEYS = ("format",)


class ConfigError(ValueError):
    """Raised when a config is missing required fields or has bad values."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and validate H1-required keys."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping, got {type(cfg).__name__}")
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ConfigError if required H1 keys are missing or malformed."""
    _require_keys(cfg, REQUIRED_TOP_KEYS, "<root>")
    _require_keys(cfg["run"], REQUIRED_RUN_KEYS, "run")
    _require_keys(cfg["env"], REQUIRED_ENV_KEYS, "env")
    _require_keys(cfg["algo"], REQUIRED_ALGO_KEYS, "algo")
    _require_keys(cfg["train"], REQUIRED_TRAIN_KEYS, "train")
    _require_keys(cfg["eval"], REQUIRED_EVAL_KEYS, "eval")
    _require_keys(cfg["logging"], REQUIRED_LOGGING_KEYS, "logging")
    if not isinstance(cfg["algo"]["hyperparams"], dict):
        raise ConfigError("algo.hyperparams must be a mapping (use {} for defaults)")
    if cfg["logging"]["format"] != "ndjson":
        raise ConfigError("logging.format must be 'ndjson' for H1")
    if int(cfg["env"]["n_envs"]) < 1:
        raise ConfigError("env.n_envs must be >= 1")
    if int(cfg["train"]["total_timesteps"]) < 1:
        raise ConfigError("train.total_timesteps must be >= 1")


def _require_keys(d: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ConfigError(f"{where} missing keys: {missing}")


def apply_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply CLI overrides into a config dict (returns a shallow-copied result).

    Recognized override keys: seed, total_timesteps, eval_freq, n_eval_episodes,
    run_id, out_dir. None values are ignored.
    """
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
    seed = overrides.get("seed")
    if seed is not None:
        out["run"]["seed"] = int(seed)
    total_timesteps = overrides.get("total_timesteps")
    if total_timesteps is not None:
        out["train"]["total_timesteps"] = int(total_timesteps)
    eval_freq = overrides.get("eval_freq")
    if eval_freq is not None:
        out["eval"]["eval_freq"] = int(eval_freq)
    n_eval_episodes = overrides.get("n_eval_episodes")
    if n_eval_episodes is not None:
        out["eval"]["n_eval_episodes"] = int(n_eval_episodes)
    run_id = overrides.get("run_id")
    if run_id is not None:
        out["run"]["run_id_override"] = str(run_id)
    out_dir = overrides.get("out_dir")
    if out_dir is not None:
        out["run"]["out_dir"] = str(out_dir)
    return out
