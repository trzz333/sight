"""Phase M / M2 - from-scratch on-policy PPO on Godot Signal Dodge (state obs).

Points the SB3 PPO harness proven known-good by M1 (CartPole mean 500.0)
at the real GodotSignalDodgeEnv in state-observation mode, reward_shaping
"none" (+1 per surviving step). Goal: a genuinely from-scratch RL policy
that clears the 930.27 survival bar reliably, where prior K5.5 state-PPO
collapsed.

Deltas vs the M1 CartPole config (and why):
- gamma 0.999 not 0.98: episodes run to 1800 steps; CartPole's 0.98
  gives a ~50-step effective horizon, far too short to propagate survival
  credit across a 930+ step episode. 0.999 -> ~1000-step horizon.
- ent_coef 0.01 not 0.0: nonzero entropy is the anti-collapse lever
  against the stay-pinned argmax failure mode seen in K5.1/K5.5.
- gae_lambda 0.95, n_steps 512/env, n_epochs 10, lr 3e-4, clip 0.2:
  standard SB3 PPO geometry for a long-horizon discrete task, replacing
  the CartPole-tuned (n_steps 32, gae 0.8, n_epochs 20) which is wrong
  for 1800-step rollouts.

Vectorization: the env factory caps Godot at n_envs=1 (an H3 scope
decision). This trainer constructs GodotSignalDodgeEnv directly, one
subprocess + one kernel-assigned TCP port per env, wrapped in DummyVecEnv
(smoke) or SubprocVecEnv (real, overlaps TCP round-trips across workers).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import (  # noqa: E402
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

DEFAULT_EXE = (
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64.exe"
)
DEFAULT_PROJECT = str(_REPO_ROOT / "games" / "signal-dodge")
SURVIVAL_BAR = 930.27
MAX_STEPS = 1800


def _alloc_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _build_godot_env(
    *, port: int, seed: int, run_dir: str, exe: str, project: str,
) -> Monitor:
    """Module-level so SubprocVecEnv can pickle the partial under spawn."""
    from sight_agent.rl.godot_env import GodotSignalDodgeEnv

    env = GodotSignalDodgeEnv(
        godot_executable=exe,
        project_path=project,
        run_dir=run_dir,
        max_steps=MAX_STEPS,
        headless=True,
        observation_mode="state",
        reward_shaping="none",
        tcp_port=port,
        seed=seed,
    )
    return Monitor(env)


def build_vec_env(
    *, n_envs: int, seed: int, run_root: Path, exe: str, project: str, vec: str,
):
    """One Godot subprocess + one TCP port per env. vec in {dummy,subproc}."""
    fns = []
    for i in range(n_envs):
        port = _alloc_port()
        sub = run_root / f"env{i}"
        sub.mkdir(parents=True, exist_ok=True)
        fns.append(
            partial(
                _build_godot_env,
                port=port,
                seed=seed + i,
                run_dir=str(sub),
                exe=exe,
                project=project,
            )
        )
    if vec == "subproc":
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="m2_state_ppo_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--vec", choices=("dummy", "subproc"), default="subproc")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--exe", default=DEFAULT_EXE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_root = out / "train_envs"
    train_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = build_vec_env(
        n_envs=args.n_envs, seed=args.seed, run_root=train_root,
        exe=args.exe, project=args.project, vec=args.vec,
    )
    # M2.1: normalize obs + returns. M2 defect was explained_variance~=0:
    # the critic could not fit the large high-gamma dense returns. Scaling
    # returns to ~unit variance (gamma-matched) is the SB3-standard fix.
    # Stats saved to vecnormalize.pkl; eval reloads them (training=False,
    # norm_reward=False) so the policy sees its trained obs scale.
    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, gamma=args.gamma,
        clip_obs=10.0, clip_reward=10.0,
    )
    model = PPO(
        "MlpPolicy", env, seed=args.seed, device="cpu", verbose=1,
        n_steps=args.n_steps, batch_size=args.batch_size,
        gae_lambda=args.gae_lambda, gamma=args.gamma,
        n_epochs=args.n_epochs, ent_coef=args.ent_coef,
        learning_rate=args.lr, clip_range=args.clip_range,
    )

    learn_err = None
    try:
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        learn_err = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            model.save(str(out / "model.zip"))
        except Exception as exc:  # noqa: BLE001
            learn_err = learn_err or f"save:{type(exc).__name__}: {exc}"
        try:
            env.save(str(out / "vecnormalize.pkl"))
        except Exception as exc:  # noqa: BLE001
            learn_err = learn_err or f"vecnorm:{type(exc).__name__}: {exc}"
        try:
            env.close()
        except Exception:
            pass

    logvals = dict(
        getattr(getattr(model, "logger", None), "name_to_value", {}) or {}
    )
    explained_variance = logvals.get("train/explained_variance")
    value_loss = logvals.get("train/value_loss")
    vecnorm_saved = (out / "vecnormalize.pkl").is_file()

    buf = list(getattr(model, "ep_info_buffer", []) or [])
    ep_lens = [float(e["l"]) for e in buf if "l" in e]
    ep_rews = [float(e["r"]) for e in buf if "r" in e]
    ep_len_mean = float(np.mean(ep_lens)) if ep_lens else None
    ep_rew_mean = float(np.mean(ep_rews)) if ep_rews else None
    finite = ep_len_mean is not None and np.isfinite(ep_len_mean)
    ok = (
        learn_err is None and finite
        and (out / "model.zip").is_file() and vecnorm_saved
    )

    report = {
        "phase": "M2",
        "phase_variant": "M2.1",
        "env": "godot:signal-dodge-v0",
        "observation_mode": "state",
        "reward_shaping": "none",
        "seed": args.seed,
        "n_envs": args.n_envs,
        "vec": args.vec,
        "total_timesteps": args.timesteps,
        "gamma": args.gamma,
        "ent_coef": args.ent_coef,
        "gae_lambda": args.gae_lambda,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.lr,
        "clip_range": args.clip_range,
        "vecnormalize": True,
        "explained_variance": explained_variance,
        "value_loss": value_loss,
        "vecnormalize_saved": vecnorm_saved,
        "ep_len_mean_last100": ep_len_mean,
        "ep_rew_mean_last100": ep_rew_mean,
        "n_episodes_in_buffer": len(buf),
        "elapsed_seconds": round(time.time() - t0, 2),
        "steps_per_sec": round(args.timesteps / max(time.time() - t0, 1e-9), 1),
        "learn_error": learn_err,
        "model_saved": (out / "model.zip").is_file(),
        "smoke_ok": bool(ok),
        "sb3": __import__("stable_baselines3").__version__,
        "gymnasium": __import__("gymnasium").__version__,
        "python": sys.version.split()[0],
    }
    with (out / "m2_train_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
