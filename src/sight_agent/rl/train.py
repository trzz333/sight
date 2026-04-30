"""Sight RL trainer (H1 baseline + H2 reusable harness).

Run:
    python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h2.yaml

H2 changes vs H1:
- env and algo construction routed through ``factories.make_env`` /
  ``factories.make_algo`` (rejects unsupported framework/algo with clear
  errors; H3 will add a Godot env-builder branch behind the same seam).
- run-artifact paths centralized in ``artifacts``.
- summary.json gains: ``schema_version=2``, ``config_path``, ``config_hash``,
  ``artifact_paths`` (events, summary, config_effective, model), and the
  saved model path.
- model checkpoint persisted as ``model.zip`` when ``checkpoint.enabled``.
- effective config snapshotted to ``config_effective.yaml`` next to the
  artifacts.

H1 NDJSON event schema (run_start, train_metrics, eval, run_end) is
preserved unchanged. No TensorBoard, no W&B, no network services. CPU only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.logger import HumanOutputFormat, Logger

from .artifacts import (
    TrainArtifacts,
    build_run_id,
    compute_config_hash,
    is_checkpoint_enabled,
    prepare_train_artifacts,
    write_config_effective,
)
from .callbacks import NDJSONCallback, NDJSONKVWriter, introspect_effective_hyperparams
from .config import apply_cli_overrides, load_config
from .factories import make_algo, make_env, smoke_check_env
from .ndjson_logger import NDJSONLogger, get_short_git_commit


def _set_global_seeds(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds. Posture, not bit-for-bit guarantee."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _versions() -> dict[str, str]:
    import gymnasium  # noqa: WPS433
    import stable_baselines3  # noqa: WPS433

    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sight_agent.rl.train",
        description="Sight RL trainer (H1 + H2 harness, NDJSON logging).",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None, dest="total_timesteps")
    parser.add_argument("--eval-freq", type=int, default=None, dest="eval_freq")
    parser.add_argument("--n-eval-episodes", type=int, default=None, dest="n_eval_episodes")
    parser.add_argument("--run-id", type=str, default=None, dest="run_id")
    parser.add_argument("--out-dir", type=str, default=None, dest="out_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    overrides = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "run_id": args.run_id,
        "out_dir": args.out_dir,
    }
    cfg = apply_cli_overrides(cfg, overrides)
    return run(cfg, config_path=str(args.config))


def run(cfg: dict[str, Any], config_path: str = "<inline>") -> int:
    """Execute one training run from a (validated) config dict."""
    seed = int(cfg["run"]["seed"])
    _set_global_seeds(seed)

    env_id = cfg["env"]["id"]
    n_envs = int(cfg["env"]["n_envs"])
    algo_name = cfg["algo"]["name"]
    framework = cfg["algo"]["framework"]
    policy = cfg["algo"]["policy"]
    device = cfg["algo"]["device"]
    extra_hyperparams = dict(cfg["algo"]["hyperparams"]) if cfg["algo"]["hyperparams"] else {}
    total_timesteps = int(cfg["train"]["total_timesteps"])
    eval_freq = int(cfg["eval"]["eval_freq"])
    n_eval_episodes = int(cfg["eval"]["n_eval_episodes"])
    deterministic_eval = bool(cfg["eval"]["deterministic"])
    phase = cfg["run"]["phase"]
    checkpoint_enabled = is_checkpoint_enabled(cfg)

    git_commit = get_short_git_commit(_repo_root_from_here())
    run_id_override = cfg["run"].get("run_id_override")
    run_id = build_run_id(cfg["run"]["name"], seed, run_id_override, git_commit)

    artifacts: TrainArtifacts = prepare_train_artifacts(cfg, run_id)
    write_config_effective(artifacts.config_effective_path, cfg)
    config_hash = compute_config_hash(cfg)

    ndjson = NDJSONLogger(
        path=artifacts.events_path,
        run_id=run_id,
        phase=phase,
        env_id=env_id,
        algo=algo_name,
        framework=framework,
        seed=seed,
        git_commit=git_commit,
    )

    versions = _versions()
    obs_shape, action_n = smoke_check_env(env_id, seed)

    train_env = make_env(env_id, n_envs=n_envs, seed=seed, mode="train")
    eval_env = make_env(env_id, n_envs=1, seed=seed, mode="eval")

    model = make_algo(
        framework=framework,
        name=algo_name,
        policy=policy,
        device=device,
        hyperparams=extra_hyperparams,
        env=train_env,
        seed=seed,
    )

    sb3_logger = Logger(
        folder=str(artifacts.run_dir),
        output_formats=[HumanOutputFormat(sys.stdout), NDJSONKVWriter(ndjson)],
    )
    model.set_logger(sb3_logger)

    effective = introspect_effective_hyperparams(model)
    artifact_paths_for_events = {
        "events_ndjson": str(artifacts.events_path),
        "summary_json": str(artifacts.summary_path),
        "config_effective": str(artifacts.config_effective_path),
    }
    if checkpoint_enabled:
        artifact_paths_for_events["model"] = str(artifacts.model_path)

    ndjson.log_event(
        "run_start",
        step=0,
        config_path=str(config_path),
        config_hash=config_hash,
        config=cfg,
        versions=versions,
        env_smoke={"obs_shape": list(obs_shape), "action_n": action_n},
        effective_hyperparams=effective,
        artifact_paths=artifact_paths_for_events,
        provenance_note=(
            "Library defaults are runtime-introspected from installed packages; "
            "no web-verified claims."
        ),
    )

    callback = NDJSONCallback(
        ndjson=ndjson,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic_eval,
        artifact_paths=artifact_paths_for_events,
    )

    status = "ok"
    error_payload: dict[str, Any] | None = None
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=1)
    except Exception as exc:
        status = "error"
        error_payload = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        ndjson.log_event(
            "error",
            step=int(getattr(model, "num_timesteps", 0) or 0),
            status="error",
            **error_payload,
        )
        ndjson.close()
        try:
            train_env.close()
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass
        raise
    finally:
        try:
            train_env.close()
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass

    model_path_written: str | None = None
    if checkpoint_enabled:
        # SB3's ``model.save`` accepts a path with or without .zip; we keep the
        # resolved filename from artifacts.
        model.save(str(artifacts.model_path))
        model_path_written = str(artifacts.model_path)

    summary_artifact_paths = {
        "events": str(artifacts.events_path),
        "summary": str(artifacts.summary_path),
        "config_effective": str(artifacts.config_effective_path),
    }
    if model_path_written:
        summary_artifact_paths["model"] = model_path_written

    summary = {
        "schema_version": 2,
        "kind": "train",
        "run_id": run_id,
        "phase": phase,
        "env_id": env_id,
        "algo": algo_name,
        "framework": framework,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "deterministic_eval": deterministic_eval,
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "versions": versions,
        "effective_hyperparams": effective,
        "artifact_paths": summary_artifact_paths,
        # Backward-compat fields used by H1 packet/tests.
        "events_ndjson": str(artifacts.events_path),
        "status": status,
    }
    artifacts.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ndjson.close()
    print(f"Sight RL run complete ({phase}): {artifacts.events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
