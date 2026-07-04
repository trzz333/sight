"""Clean from-scratch PPO budget isolation on the fast Signal Dodge replica.

Purpose. Phase M2.1 (Godot, 1M steps, VecNormalize, healthy critic EV
0.85-0.94, diverse actions) reached IQM 418, CI [314,670], below the 930.27
bar. The untested axis is budget: Godot capped every from-scratch run at
~1M steps (59.8 steps/s). The fidelity-validated pure-Python replica
(src/sight_agent/rl/sd_fast.py, ~238k steps/s) makes 5M steps a minutes-long
run. This isolates budget as the single variable versus M2.1 by holding M2.1's
recipe fixed and changing only the step budget (1M -> 5M) and the env
(Godot -> replica, validated ~10-15% more collision-forgiving, a safe transfer
direction).

Recipe = M2.1 verbatim (read from tools/m2_state_ppo_train.py):
gamma 0.999, gae 0.95, n_steps 512, batch 512, n_epochs 10, clip 0.2,
ent_coef 0.01, lr 3e-4, vf_coef default 0.5, 8 envs, MlpPolicy [64,64],
VecNormalize(norm_obs, norm_reward, gamma 0.999, clip_obs 10, clip_reward 10).
Reward stays "none" (survival): K5.5 dense shaping backfired into wall-hugging;
"none" keeps best-constant (replica ~746 right) below the dodging bar.

Eval. Greedy over held-out replica seeds disjoint from training (5000-5029).
The policy trained on VecNormalize-normalized obs, so eval feeds obs through the
saved running stats (training=False, norm_reward=False) before predict, exactly
as M2.1's eval reloaded vecnormalize.pkl. Reports mean episode length, action
fractions, and the diversity gate max(frac) < 0.97.

Verdict logic. Clear the replica dodging bar with diverse actions -> port the
policy to a Godot 5M run for the eval of record (bar 930.27). Reproduce M2.1's
diverse sub-baseline plateau at 5x budget -> budget is refuted; the wall is
exploration/credit structure (critic blind to death-timing from the 3-hazard
obs), and the next lever moves off budget.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sight_agent.rl.sd_fast import SignalDodgeFast


def make_env():
    return SignalDodgeFast()


def evaluate(model, vecnorm_path: Path, n_seeds: int, base_seed: int,
             max_steps: int = 1800):
    """Greedy eval on held-out seeds, obs normalized by saved training stats."""
    vn = VecNormalize.load(str(vecnorm_path), DummyVecEnv([make_env]))
    vn.training = False
    vn.norm_reward = False

    raw = SignalDodgeFast(max_steps=max_steps)
    lengths = []
    action_counts = np.zeros(3, dtype=np.int64)
    for s in range(n_seeds):
        obs, _ = raw.reset(seed=base_seed + s)
        steps = 0
        while True:
            nobs = vn.normalize_obs(np.asarray(obs, dtype=np.float32))
            a, _ = model.predict(nobs, deterministic=True)
            a = int(a)
            action_counts[a] += 1
            obs, _, term, trunc, _ = raw.step(a)
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
    # M2.1 recipe knobs (defaults = M2.1 verbatim; overridable for audit).
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([make_env for _ in range(args.n_envs)])
    venv.seed(args.seed)
    venv = VecNormalize(
        venv, norm_obs=True, norm_reward=True, gamma=args.gamma,
        clip_obs=10.0, clip_reward=10.0,
    )

    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        learning_rate=args.lr,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    train_s = time.perf_counter() - t0

    model_path = out / f"{args.run_id}.zip"
    model.save(str(model_path))
    vecnorm_path = out / f"{args.run_id}_vecnormalize.pkl"
    venv.save(str(vecnorm_path))

    # Diagnostic: critic health. M2's defect was explained_variance ~= 0.
    try:
        logvals = model.logger.name_to_value
        explained_variance = logvals.get("train/explained_variance")
        value_loss = logvals.get("train/value_loss")
    except Exception:
        explained_variance = None
        value_loss = None

    lengths, fracs = evaluate(model, vecnorm_path, args.eval_seeds,
                              args.eval_base_seed)

    replica_best_constant = 746.3  # const_right, 500-seed (sd_fast_validate)
    diversity_ok = bool(fracs.max() < 0.97)
    summary = {
        "run_id": args.run_id,
        "recipe": "M2.1-verbatim",
        "seed": args.seed,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "hparams": {
            "gamma": args.gamma, "gae_lambda": args.gae_lambda,
            "n_steps": args.n_steps, "batch_size": args.batch_size,
            "n_epochs": args.n_epochs, "ent_coef": args.ent_coef,
            "clip_range": args.clip_range, "lr": args.lr,
            "vecnormalize": "norm_obs+norm_reward, clip 10/10",
            "net_arch": [64, 64],
        },
        "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "explained_variance": (
            round(float(explained_variance), 4)
            if explained_variance is not None else None
        ),
        "value_loss": (
            round(float(value_loss), 4) if value_loss is not None else None
        ),
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
