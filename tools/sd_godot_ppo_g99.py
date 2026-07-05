"""Discount-first Godot eval-of-record port of the g99 from-scratch recipe.

STATUS: SMOKE-VALIDATED end to end (2 envs, 2000 steps, 3 eval seeds): builds
multi-process Godot with distinct TCP ports, VecNormalize wrap + save, greedy
held-out-seed eval all exercised; headless ~70 steps/s. Real arms unrun.

Ports the recipe that cleared the replica port gate (m21 + VecNormalize +
gamma 0.99) to the REAL Godot Signal Dodge env. NO curriculum: the start-state
curriculum is not injectable into Godot without GDScript + protocol work
(godot_env.reset passes only a seed, no hazard-injection seam), so this run
isolates whether the discount fix ALONE transfers. Bar: 930.27 (Godot
constant-action baseline). Controlled contrast: Phase M2.1 ran this recipe on
Godot at gamma 0.999 / 1M steps and reached IQM 418 (below bar); the only
change here is gamma 0.999 -> 0.99 (and, for the eval of record, 5M steps).

Recipe (g99 verbatim at gamma 0.99): gamma 0.99, gae 0.95, n_steps 512,
batch 512, n_epochs 10, clip 0.2, ent_coef 0.01, lr 3e-4, 8 envs,
MlpPolicy [64,64], VecNormalize(norm_obs, norm_reward, gamma 0.99,
clip_obs 10, clip_reward 10), reward 'none' (survival).

Env construction. factories.make_env blocks n_envs>1 for the Godot env, so the
8 training envs are built directly as a DummyVecEnv of 8 GodotSignalDodgeEnv,
each with its own kernel-allocated loopback TCP port (the factory sanctions
direct construction with tcp_port=... for exactly this). headless=True for
throughput. Requires SIGHT_GODOT_EXE set (or --godot-exe).

Eval of record. Greedy on held-out seeds 5000-5029 on a single raw Godot env,
obs pushed through the saved VecNormalize stats (training=False,
norm_reward=False), exactly as tools/sd_fast_ppo.evaluate. Metric = mean
survival steps vs 930.27.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sight_agent.rl.godot_env import GodotSignalDodgeEnv  # noqa: E402

BAR = 930.27


class AnnealCurriculum(BaseCallback):
    """Linearly anneal curriculum_n_init from n0 to 0 over the first
    anneal_frac of training, then hold 0 (clean starts) for the remainder.

    Verbatim port of tools/sd_fast_ppo_curriculum.AnnealCurriculum: same
    schedule, same set_attr mechanism. The Godot env exposes a public
    ``curriculum_n_init`` attribute (godot_env.GodotSignalDodgeEnv) that its
    reset() reads and forwards on the wire, so set_attr here reaches every
    training env exactly as it does on the replica."""

    def __init__(self, n0: int, total_steps: int, anneal_frac: float = 0.7):
        super().__init__()
        self.n0 = int(n0)
        self.T = max(1.0, total_steps * anneal_frac)
        self._last = None

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.T)
        val = int(round(self.n0 * (1.0 - frac)))
        if val != self._last:
            self.training_env.set_attr("curriculum_n_init", val)
            self._last = val
        return True


def _alloc_tcp_port() -> int:
    """Kernel-assigned loopback port; mirrors factories._allocate_isolated_tcp_port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _resolve_exe(cli_exe: str | None) -> str:
    exe = cli_exe or os.environ.get("SIGHT_GODOT_EXE")
    if not exe:
        raise SystemExit("SIGHT_GODOT_EXE not set and --godot-exe not passed.")
    return exe


def _default_project() -> str:
    return str(Path(__file__).resolve().parents[1] / "games" / "signal-dodge")


def _make_godot(exe, proj, seed, run_dir, headless, max_steps):
    def _factory():
        return GodotSignalDodgeEnv(
            godot_executable=exe, project_path=proj, run_dir=run_dir,
            seed=seed, tcp_port=_alloc_tcp_port(), headless=headless,
            max_steps=max_steps,
        )
    return _factory


class _SpaceStub(gym.Env):
    """Minimal env exposing only the spaces so VecNormalize.load can attach.
    Never stepped; DummyVecEnv.__init__ does not call reset()."""
    def __init__(self, obs_space, act_space):
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (np.zeros(self.observation_space.shape, dtype=np.float32),
                0.0, True, False, {})


def evaluate(model, vecnorm_path, raw_env, n_seeds, base_seed):
    """Greedy held-out eval: obs normalized by saved training stats, survival
    steps counted on the reward-agnostic env. Mirrors sd_fast_ppo.evaluate."""
    stub = DummyVecEnv([lambda: _SpaceStub(raw_env.observation_space,
                                           raw_env.action_space)])
    vn = VecNormalize.load(str(vecnorm_path), stub)
    vn.training = False
    vn.norm_reward = False

    n_actions = int(raw_env.action_space.n)
    action_counts = np.zeros(n_actions, dtype=np.int64)
    lengths = []
    for s in range(n_seeds):
        obs, _ = raw_env.reset(seed=base_seed + s)
        steps = 0
        while True:
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(nobs, deterministic=True)
            a = int(a)
            action_counts[a] += 1
            obs, _, term, trunc, _ = raw_env.step(a)
            steps += 1
            if term or trunc:
                break
        lengths.append(steps)
    lengths = np.array(lengths, dtype=float)
    fracs = action_counts / max(action_counts.sum(), 1)
    return lengths, fracs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--eval-base-seed", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default="runs/sd_godot")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--godot-exe", type=str, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--no-headless", action="store_true")
    # Start-state curriculum (Godot port of the replica recipe). Off by default
    # so the discount-only trainer behavior is unchanged. When --curriculum is
    # passed, training resets pre-spawn n_init_max hazards above the player,
    # annealing to 0 over the first anneal_frac of training. Eval stays clean.
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--n-init-max", type=int, default=6)
    p.add_argument("--anneal-frac", type=float, default=0.7)
    # g99 recipe knobs (defaults = g99 verbatim at gamma 0.99).
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    exe = _resolve_exe(args.godot_exe)
    proj = args.project or _default_project()
    headless = not args.no_headless
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = out / f"{args.run_id}_train"
    factories = [
        _make_godot(exe, proj, args.seed, str(train_dir / f"env{i}"),
                    headless, args.max_steps)
        for i in range(args.n_envs)
    ]
    venv = DummyVecEnv(factories)
    venv.seed(args.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        clip_obs=10.0, clip_reward=10.0)

    callback = None
    if args.curriculum:
        # Pre-seed every training env so the FIRST reset already injects
        # n_init_max hazards, matching the replica (whose CurriculumSDF is
        # constructed with n_init_max). Without this, the first rollout's resets
        # would run clean because the anneal callback's first set_attr only fires
        # after the first env step. set_attr propagates through VecNormalize to
        # the underlying GodotSignalDodgeEnv instances.
        venv.set_attr("curriculum_n_init", args.n_init_max)
        callback = AnnealCurriculum(args.n_init_max, args.steps, args.anneal_frac)

    model = PPO("MlpPolicy", venv, seed=args.seed, n_steps=args.n_steps,
                batch_size=args.batch_size, n_epochs=args.n_epochs,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_range=args.clip_range, ent_coef=args.ent_coef,
                learning_rate=args.lr, policy_kwargs=dict(net_arch=[64, 64]),
                device="cpu", verbose=1)

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=callback)
    train_s = time.perf_counter() - t0

    model_path = out / f"{args.run_id}.zip"
    model.save(str(model_path))
    vecnorm_path = out / f"{args.run_id}_vecnormalize.pkl"
    venv.save(str(vecnorm_path))
    venv.close()

    try:
        logvals = model.logger.name_to_value
        ev = logvals.get("train/explained_variance")
        vl = logvals.get("train/value_loss")
    except Exception:
        ev = vl = None

    raw = _make_godot(exe, proj, 0, str(out / f"{args.run_id}_eval"),
                      headless, args.max_steps)()
    try:
        lengths, fracs = evaluate(model, vecnorm_path, raw,
                                  args.eval_seeds, args.eval_base_seed)
    finally:
        raw.close()

    summary = {
        "run_id": args.run_id,
        "recipe": ("g99-verbatim-godot+start-curriculum" if args.curriculum
                   else "g99-verbatim-godot-no-curriculum"),
        "env": "godot:signal-dodge-v0", "reward_mode": "none",
        "seed": args.seed, "steps": args.steps, "n_envs": args.n_envs,
        "headless": headless,
        "curriculum": ({"enabled": True, "n_init_max": args.n_init_max,
                        "anneal_frac": args.anneal_frac} if args.curriculum
                       else {"enabled": False}),
        "hparams": {"gamma": args.gamma, "gae_lambda": args.gae_lambda,
                    "n_steps": args.n_steps, "batch_size": args.batch_size,
                    "n_epochs": args.n_epochs, "ent_coef": args.ent_coef,
                    "clip_range": args.clip_range, "lr": args.lr,
                    "vecnormalize": "norm_obs+norm_reward, clip 10/10",
                    "net_arch": [64, 64]},
        "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "explained_variance": (round(float(ev), 4) if ev is not None else None),
        "value_loss": (round(float(vl), 4) if vl is not None else None),
        "eval": {"n_seeds": args.eval_seeds, "base_seed": args.eval_base_seed,
                 "mean_len": round(float(lengths.mean()), 2),
                 "std_len": round(float(lengths.std()), 2),
                 "min_len": float(lengths.min()), "max_len": float(lengths.max()),
                 "lengths": [int(x) for x in lengths],
                 "action_fracs": [round(float(x), 4) for x in fracs]},
        "bar": BAR,
        "diversity_ok": bool(fracs.max() < 0.97),
        "beats_bar": bool(lengths.mean() > BAR),
        "m21_godot_ref": {"gamma_0999_1M_iqm": 418, "note": "controlled contrast"},
    }
    (out / f"{args.run_id}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
