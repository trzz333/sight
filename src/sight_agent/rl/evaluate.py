"""H2 out-of-band evaluator.

Loads a saved SB3 model from a completed train run directory and writes
a separate eval artifact set:

    <train_run_dir>/evals/<eval_id>/
        events.ndjson
        summary.json

Usage:
    python -m sight_agent.rl.evaluate --run <train_run_dir> \
        --n-eval-episodes 5 --seed 0

NDJSON events emitted: eval_start, eval_episode (one per episode), eval_end.
Common NDJSON fields are inherited from the source train summary (phase,
env_id, algo, framework). Eval events use a fresh eval run_id and the eval
seed; the source train run_id is preserved as ``source_train_run_id``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from .artifacts import (
    EvalArtifacts,
    build_eval_id,
    prepare_eval_artifacts,
)
from .factories import make_env
from .godot_config import is_godot_env_id, resolve_godot_kwargs
from .ndjson_logger import NDJSONLogger, get_short_git_commit


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_train_summary(train_run_dir: Path) -> dict[str, Any]:
    summary_path = train_run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found at {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_train_effective_config(train_run_dir: Path) -> dict[str, Any]:
    """Reload the effective YAML from a completed train run dir.

    Required when the train run was a Godot run, so eval can recover the
    ``godot_executable`` / ``project_path`` plumbing from the YAML rather
    than re-deriving them. Out-of-band evals that target Gymnasium env ids
    do not call this; the env id alone is enough for them.
    """
    import yaml  # local import: keep eval CLI free of yaml when not needed

    cfg_path = train_run_dir / "config_effective.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"config_effective.yaml not found at {cfg_path}; H3 godot eval "
            f"requires it to recover godot_executable / project_path."
        )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(
            f"config_effective.yaml at {cfg_path} did not load as a mapping"
        )
    return cfg


def _build_eval_env_for_train_run(
    env_id: str,
    seed: int,
    artifacts: EvalArtifacts,
    train_run_dir: Path,
):
    """Construct the eval VecEnv for a completed train run.

    Routes Godot ids through the factory's Godot branch with kwargs recovered
    from the train run's ``config_effective.yaml``. The eval ``run_dir`` lives
    under the eval artifacts dir so eval-side NDJSON evidence is namespaced
    away from train-side evidence; train-run dirs already carry their own
    ``godot-train`` / ``godot-eval`` subdirs from ``train.py``.
    """
    if not is_godot_env_id(env_id):
        return make_env(env_id, n_envs=1, seed=seed, mode="eval")
    train_cfg = _load_train_effective_config(train_run_dir)
    extra = resolve_godot_kwargs(train_cfg)
    if not extra:
        # Defensive: env id said Godot but the recovered config did not.
        # Fail clearly rather than guessing kwargs.
        raise ValueError(
            f"Train run {train_run_dir} reports env_id={env_id!r} but its "
            f"config_effective.yaml did not yield Godot env kwargs; cannot "
            f"build eval env without godot_executable / project_path."
        )
    return make_env(
        env_id,
        n_envs=1,
        seed=seed,
        mode="eval",
        run_dir=str(artifacts.eval_dir / "godot-eval"),
        **extra,
    )


def _resolve_model_path(train_run_dir: Path, train_summary: dict[str, Any]) -> Path:
    """Use summary.artifact_paths.model when present, fall back to model.zip."""
    paths = train_summary.get("artifact_paths") or {}
    candidate = paths.get("model")
    if isinstance(candidate, str) and candidate:
        p = Path(candidate)
        if not p.is_absolute():
            p = train_run_dir / p
        if p.exists():
            return p
    fallback = train_run_dir / "model.zip"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No model artifact found under {train_run_dir} (looked at "
        f"artifact_paths.model and model.zip)."
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sight_agent.rl.evaluate",
        description="Sight H2 out-of-band evaluator for SB3 PPO train runs.",
    )
    parser.add_argument("--run", required=True, help="Path to a completed train run dir.")
    parser.add_argument("--n-eval-episodes", type=int, default=5, dest="n_eval_episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        type=str,
        default="true",
        choices=("true", "false"),
        help="Use deterministic policy at eval time (default true).",
    )
    parser.add_argument("--eval-id", type=str, default=None, dest="eval_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rc, _summary = run_eval(
        train_run_dir=Path(args.run),
        n_eval_episodes=int(args.n_eval_episodes),
        seed=int(args.seed),
        deterministic=(args.deterministic == "true"),
        eval_id_override=args.eval_id,
    )
    return rc


def run_eval(
    train_run_dir: Path,
    n_eval_episodes: int,
    seed: int,
    deterministic: bool = True,
    eval_id_override: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run an out-of-band eval against an existing train run.

    Returns (return_code, eval_summary).
    """
    train_run_dir = Path(train_run_dir)
    if not train_run_dir.exists() or not train_run_dir.is_dir():
        raise FileNotFoundError(f"train run dir not found: {train_run_dir}")

    train_summary = _load_train_summary(train_run_dir)
    env_id = str(train_summary["env_id"])
    algo = str(train_summary.get("algo", "PPO"))
    framework = str(train_summary.get("framework", "stable-baselines3"))
    source_run_id = str(train_summary["run_id"])
    train_phase = str(train_summary.get("phase", "H2"))

    model_path = _resolve_model_path(train_run_dir, train_summary)

    eval_id = eval_id_override or build_eval_id(
        seed=seed,
        n_eval_episodes=n_eval_episodes,
        source_run_id=source_run_id,
    )
    artifacts = prepare_eval_artifacts(train_run_dir, eval_id)

    git_commit = get_short_git_commit(_repo_root_from_here())
    _set_global_seeds(seed)

    ndjson = NDJSONLogger(
        path=artifacts.events_path,
        run_id=eval_id,
        phase=train_phase,
        env_id=env_id,
        algo=algo,
        framework=framework,
        seed=seed,
        git_commit=git_commit,
    )

    eval_env = _build_eval_env_for_train_run(
        env_id=env_id,
        seed=seed,
        artifacts=artifacts,
        train_run_dir=train_run_dir,
    )
    model = PPO.load(str(model_path), device="cpu")

    t0 = time.time()
    ndjson.log_event(
        "eval_start",
        step=0,
        source_train_run_id=source_run_id,
        source_train_run_dir=str(train_run_dir),
        model_path=str(model_path),
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    status = "ok"
    error_payload: dict[str, Any] | None = None

    try:
        obs = eval_env.reset()
        ep_reward = 0.0
        ep_len = 0
        while len(episode_rewards) < n_eval_episodes:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _info = eval_env.step(action)
            ep_reward += float(np.asarray(reward).sum())
            ep_len += 1
            done_flag = bool(np.asarray(done).any())
            if done_flag:
                episode_rewards.append(float(ep_reward))
                episode_lengths.append(int(ep_len))
                ndjson.log_event(
                    "eval_episode",
                    step=int(ep_len),
                    episode_index=len(episode_rewards) - 1,
                    reward=float(ep_reward),
                    length=int(ep_len),
                )
                ep_reward = 0.0
                ep_len = 0
                # SB3 VecEnv auto-resets on done; obs already holds the next reset.
    except Exception as exc:  # pragma: no cover - defensive
        status = "error"
        error_payload = {"error_type": type(exc).__name__, "message": str(exc)}
        ndjson.log_event(
            "error",
            step=int(len(episode_rewards)),
            status="error",
            **error_payload,
        )
    finally:
        try:
            eval_env.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.time() - t0
    mean_reward = float(statistics.fmean(episode_rewards)) if episode_rewards else 0.0
    std_reward = (
        float(statistics.pstdev(episode_rewards)) if len(episode_rewards) > 1 else 0.0
    )

    ndjson.log_event(
        "eval_end",
        step=int(sum(episode_lengths)),
        status=status,
        elapsed_seconds=float(elapsed),
        mean_reward=mean_reward,
        std_reward=std_reward,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
    )
    ndjson.close()

    summary: dict[str, Any] = {
        "schema_version": 2,
        "kind": "eval",
        "eval_id": eval_id,
        "phase": train_phase,
        "env_id": env_id,
        "algo": algo,
        "framework": framework,
        "seed": seed,
        "deterministic": deterministic,
        "n_eval_episodes": n_eval_episodes,
        "git_commit": git_commit,
        "status": status,
        "elapsed_seconds": float(elapsed),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "model_path": str(model_path),
        "source_train_run_id": source_run_id,
        "source_train_run_dir": str(train_run_dir),
        "source_train_summary": train_summary,
        "artifact_paths": {
            "events": str(artifacts.events_path),
            "summary": str(artifacts.summary_path),
            "model": str(model_path),
        },
    }
    if error_payload is not None:
        summary["error"] = error_payload

    artifacts.summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"H2 eval complete: {artifacts.events_path}")
    return (0 if status == "ok" else 2), summary


if __name__ == "__main__":
    raise SystemExit(main())
