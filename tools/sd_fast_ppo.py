"""From-scratch PPO on the fast Signal Dodge replica, reward "none".

The MinAtar-parity budget experiment. Phase M ran from-scratch PPO on Godot
Signal Dodge at 1M steps/seed (throughput-capped) and failed (IQM 418, CI
[314,670] < 930.27). MinAtar Breakout cleared from scratch with the same SB3
stack at 5M steps. The untested axis between them is budget-at-speed. The fast
replica (tools/sd_fast_validate.py: 238k steps/s vs Godot 60) makes 5M steps a
~10 min run, so this probes whether budget alone closes the from-scratch gap.

Recipe = the MinAtar-winning PPO recipe verbatim (n_steps 128, batch 256,
n_epochs 4, gamma .99, gae .95, clip .2, ent_coef .01, vf_coef .5, lr 2.5e-4,
8 envs), MlpPolicy [64,64] (project-standard actor). Reward stays "none":
on-disk evidence (K5.5 shaped backfired; Phase M) says dense shaping rewards
wall-hugging, and "none" makes best-constant (846 Godot / ~746 replica) cap
below the dodging bar.

Eval: greedy over held-out replica seeds (disjoint from training), report mean
episode length, action fractions, and diversity gate max(frac) < 0.97. Replica
best-constant ~= 746 (right); a decisive clear here justifies porting the policy
to the Godot eval of record (bar 930.27).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sight_agent.rl.sd_fast import SignalDodgeFast


def make_env():
    return SignalDodgeFast()


def evaluate(model, n_seeds: int, base_seed: int, max_steps: int = 1800):
    env = SignalDodgeFast(max_steps=max_steps)
    lengths = []
    action_counts = np.zeros(3, dtype=np.int64)
    for s in range(n_seeds):
        obs, _ = env.reset(seed=base_seed + s)
        steps = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            action_counts[a] += 1
            obs, _, term, trunc, _ = env.step(a)
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
    p.add_argument("--out-dir", type=str, default="runs/sd_fast")
    p.add_argument("--run-id", type=str, required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([make_env for _ in range(args.n_envs)])
    venv.seed(args.seed)

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
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    train_s = time.perf_counter() - t0

    model_path = out / f"{args.run_id}.zip"
    model.save(str(model_path))

    lengths, fracs = evaluate(model, args.eval_seeds, args.eval_base_seed)

    # replica constant-action anchors (500-seed, from sd_fast_validate):
    replica_best_constant = 746.3  # const_right
    diversity_ok = bool(fracs.max() < 0.97)
    summary = {
        "run_id": args.run_id,
        "seed": args.seed,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "eval": {
            "n_seeds": args.eval_seeds,
            "base_seed": args.eval_base_seed,
            "mean_len": round(float(lengths.mean()), 2),
            "std_len": round(float(lengths.std()), 2),
            "min_len": float(lengths.min()),
            "max_len": float(lengths.max()),
            "action_fracs_LSR": [round(float(x), 4) for x in fracs],
        },
        "replica_best_constant_ref": replica_best_constant,
        "diversity_ok": diversity_ok,
        "beats_replica_best_constant": bool(lengths.mean() > replica_best_constant),
    }
    (out / f"{args.run_id}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
