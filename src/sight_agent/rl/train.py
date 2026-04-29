"""H1 CartPole PPO local baseline trainer.

Run:
    python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml

Writes NDJSON events and summary.json under runs/rl/<run.name>/<run_id>/.
No TensorBoard, no W&B, no network services. CPU only by default.
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
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, Logger

from .callbacks import NDJSONCallback, NDJSONKVWriter, introspect_effective_hyperparams
from .config import apply_cli_overrides, load_config
from .envs import make_eval_env, make_train_env, smoke_check_env
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


def _build_run_id(name: str, seed: int, override: str | None, git_commit: str | None) -> str:
    if override:
        return override
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    git_part = git_commit or "nogit"
    return f"{ts}_{name}_seed{seed}_{git_part}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sight_agent.rl.train",
        description="Sight H1 CartPole PPO local baseline (NDJSON logging).",
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
    """Execute one H1 training run from a (validated) config dict."""
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

    if algo_name != "PPO" or framework != "stable-baselines3":
        raise ValueError(
            f"H1 only supports PPO + stable-baselines3, got {framework}/{algo_name}",
        )

    git_commit = get_short_git_commit(_repo_root_from_here())
    run_id_override = cfg["run"].get("run_id_override")
    run_id = _build_run_id(cfg["run"]["name"], seed, run_id_override, git_commit)

    out_root = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    events_path = out_root / "events.ndjson"
    summary_path = out_root / "summary.json"

    ndjson = NDJSONLogger(
        path=events_path,
        run_id=run_id,
        phase=cfg["run"]["phase"],
        env_id=env_id,
        algo=algo_name,
        framework=framework,
        seed=seed,
        git_commit=git_commit,
    )

    versions = _versions()
    obs_shape, action_n = smoke_check_env(env_id, seed)

    train_env = make_train_env(env_id, n_envs=n_envs, seed=seed)
    eval_env = make_eval_env(env_id, seed=seed)

    model_kwargs: dict[str, Any] = {
        "policy": policy,
        "env": train_env,
        "seed": seed,
        "device": device,
    }
    if extra_hyperparams:
        model_kwargs.update(extra_hyperparams)
    model = PPO(**model_kwargs)

    sb3_logger = Logger(
        folder=str(out_root),
        output_formats=[HumanOutputFormat(sys.stdout), NDJSONKVWriter(ndjson)],
    )
    model.set_logger(sb3_logger)

    effective = introspect_effective_hyperparams(model)
    ndjson.log_event(
        "run_start",
        step=0,
        config_path=str(config_path),
        config=cfg,
        versions=versions,
        env_smoke={"obs_shape": list(obs_shape), "action_n": action_n},
        effective_hyperparams=effective,
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
        artifact_paths={
            "events_ndjson": str(events_path),
            "summary_json": str(summary_path),
        },
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

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "phase": cfg["run"]["phase"],
        "env_id": env_id,
        "algo": algo_name,
        "framework": framework,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "n_eval_episodes": n_eval_episodes,
        "deterministic_eval": deterministic_eval,
        "git_commit": git_commit,
        "versions": versions,
        "effective_hyperparams": effective,
        "events_ndjson": str(events_path),
        "status": status,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ndjson.close()
    print(f"H1 run complete: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
