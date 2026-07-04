"""Sight MinAtar ADOPT spike: SB3 PPO from scratch on MinAtar Breakout.

Deterministic single seed. Writes NDJSON training curve (episode reward/length
per rollout) to runs/minatar/, checkpoints the model, and runs a held-out eval
(seeds disjoint from training) reporting mean episode return vs the published
from-scratch baseline (~9.4 for Breakout).

Usage:
  python tools\minatar_ppo_spike.py --steps 200000 --seed 0 --tag smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sight_agent.rl.minatar import make_env, build_extractor_cls, register

PUBLISHED_BASELINE = {"MinAtar/Breakout-v1": 9.4, "MinAtar/Breakout-v0": 9.4}
RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "minatar")


class NdjsonCurve(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.t0 = time.time()
        self.f = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if not buf:
            return
        r = np.mean([e["r"] for e in buf])
        l = np.mean([e["l"] for e in buf])
        rec = {
            "t": round(time.time() - self.t0, 1),
            "timesteps": int(self.num_timesteps),
            "ep_rew_mean": round(float(r), 4),
            "ep_len_mean": round(float(l), 2),
            "n_ep": len(buf),
        }
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def _on_training_end(self):
        if self.f:
            self.f.close()


def held_out_eval(model, game, n_ep=30, seed_base=1000):
    """Mean episode return over held-out seeds disjoint from training."""
    env = make_env(game)
    returns = []
    for i in range(n_ep):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        R = 0.0
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, _ = env.step(int(a))
            R += rew
            done = term or trunc
        returns.append(R)
    env.close()
    returns = np.array(returns)
    return {
        "n_ep": n_ep,
        "mean": round(float(returns.mean()), 3),
        "std": round(float(returns.std()), 3),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "seeds": [seed_base, seed_base + n_ep - 1],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="MinAtar/Breakout-v1")
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    register()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = f"ppo_{args.game.split('/')[-1]}_s{args.seed}_{args.tag}"
    curve_path = os.path.join(RUNS_DIR, f"{run_id}.ndjson")
    model_path = os.path.join(RUNS_DIR, f"{run_id}.zip")
    summary_path = os.path.join(RUNS_DIR, f"{run_id}_summary.json")

    def _factory():
        return Monitor(make_env(args.game))

    venv = make_vec_env(_factory, n_envs=args.n_envs, seed=args.seed)

    policy_kwargs = dict(
        features_extractor_class=build_extractor_cls(),
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[], vf=[]),
    )
    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        learning_rate=2.5e-4,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
    )

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=NdjsonCurve(curve_path))
    train_s = time.time() - t0
    model.save(model_path)

    ev = held_out_eval(model, args.game, n_ep=30, seed_base=1000)
    bar = PUBLISHED_BASELINE.get(args.game)
    summary = {
        "run_id": run_id,
        "game": args.game,
        "seed": args.seed,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "random_floor_ref": 0.333,
        "published_baseline": bar,
        "heldout_eval": ev,
        "clears_baseline": (ev["mean"] >= bar) if bar else None,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
