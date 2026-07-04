"""From-scratch PPO with a START-STATE CURRICULUM on the fast Signal Dodge
replica. Structurally-new lever after the from-scratch failures (CMA-ES,
CMA-MAE, elite-BC, budget 5M, NoisyNet exploration, PBRS reward geometry).

Diagnosis it targets. The m21 none arm lands run-level IQM 733.6, BELOW the
constant-action baseline (~746 replica / 845.7 Godot) and high-variance across
seeds. So PPO is not merely stuck at the passive optimum, it fails to reliably
reach even that: a +1/step survival reward credited across ~800 steps is a
long-horizon credit-assignment + deceptive-basin trap (Go-Explore, Ecoffet 2019;
curriculum learning, Bengio 2009).

Lever (found-art ADAPT, not BUILD). A reset-state curriculum: early episodes
start with several hazards pre-spawned above the player, so (a) passivity dies
quickly, removing the deceptive basin, and (b) episodes are short with fast
win/lose feedback, shortening the credit-assignment horizon. The count of
injected hazards anneals linearly to 0 over the first anneal_frac of training,
so the run ENDS on the true clean-start distribution. Everything else is the
m21 recipe verbatim. Eval is UNCHANGED: greedy on the standard clean-start
env, held-out seeds 5000-5029, via sd_fast_ppo.evaluate, so the number is
directly comparable to the m21 none arm.

sd_fast.py is left byte-identical: the curriculum lives in a subclass here, so
the base env the eval harness and the imitation number import is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sight_agent.rl.sd_fast import (  # noqa: E402
    SignalDodgeFast, HAZARD_SPAWN_MIN, HAZARD_SPAWN_MAX, HAZARD_SPAWN_Y, PLAYER_Y,
)
from sd_fast_ppo import evaluate  # noqa: E402  reuse the m21 eval verbatim

# headroom above the player for injected hazards: >COLLIDE_THRESH so no reset
# collision, and >=~30 steps of fall time so the policy can react, not insta-die.
INIT_HEADROOM = 100.0


class CurriculumSDF(SignalDodgeFast):
    """SignalDodgeFast whose reset() optionally pre-spawns curriculum_n_init
    hazards above the player. curriculum_n_init is mutated live by the anneal
    callback. When 0 (the eval/default path) behavior is byte-identical to the
    base env."""

    def __init__(self, *args, n_init_max: int = 6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.curriculum_n_init = int(n_init_max)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        k = int(self.curriculum_n_init)
        if k <= 0:
            return obs, info
        y_hi = PLAYER_Y - INIT_HEADROOM
        for _ in range(k):
            self._id_counter += 1
            x = float(self.np_random.uniform(HAZARD_SPAWN_MIN, HAZARD_SPAWN_MAX))
            y = float(self.np_random.uniform(HAZARD_SPAWN_Y, y_hi))
            self._hx.append(x)
            self._hy.append(y)
            self._hid.append(self._id_counter)
        self._prev_phi = self._potential()
        return self._obs(), info


class AnnealCurriculum(BaseCallback):
    """Linearly anneal curriculum_n_init from n0 to 0 over the first
    anneal_frac of training, then hold 0 (clean starts) for the remainder."""

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


def make_train_factory(n_init_max: int):
    def _factory():
        return CurriculumSDF(reward_mode="none", n_init_max=n_init_max)
    return _factory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--eval-base-seed", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default="runs/sd_fast")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--n-init-max", type=int, default=6)
    p.add_argument("--anneal-frac", type=float, default=0.7)
    # m21 recipe knobs (defaults = m21 verbatim)
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

    venv = DummyVecEnv([make_train_factory(args.n_init_max) for _ in range(args.n_envs)])
    venv.seed(args.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, gamma=args.gamma,
                        clip_obs=10.0, clip_reward=10.0)

    model = PPO("MlpPolicy", venv, seed=args.seed, n_steps=args.n_steps,
                batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma,
                gae_lambda=args.gae_lambda, clip_range=args.clip_range,
                ent_coef=args.ent_coef, learning_rate=args.lr,
                policy_kwargs=dict(net_arch=[64, 64]), device="cpu", verbose=0)

    cb = AnnealCurriculum(args.n_init_max, args.steps, args.anneal_frac)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=cb)
    train_s = time.perf_counter() - t0

    model_path = out / f"{args.run_id}.zip"
    model.save(str(model_path))
    vecnorm_path = out / f"{args.run_id}_vecnormalize.pkl"
    venv.save(str(vecnorm_path))

    lengths, fracs = evaluate(model, vecnorm_path, args.eval_seeds, args.eval_base_seed)
    replica_best_constant = 746.3
    summary = {
        "run_id": args.run_id, "recipe": "m21-verbatim+start-curriculum",
        "reward_mode": "none", "seed": args.seed, "steps": args.steps,
        "n_envs": args.n_envs,
        "curriculum": {"n_init_max": args.n_init_max, "anneal_frac": args.anneal_frac,
                       "headroom_px": INIT_HEADROOM},
        "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "eval": {"n_seeds": args.eval_seeds, "base_seed": args.eval_base_seed,
                 "mean_len": round(float(lengths.mean()), 2),
                 "std_len": round(float(lengths.std()), 2),
                 "min_len": float(lengths.min()), "max_len": float(lengths.max()),
                 "lengths": [int(x) for x in lengths],
                 "action_fracs_LSR": [round(float(x), 4) for x in fracs]},
        "replica_best_constant_ref": replica_best_constant,
        "diversity_ok": bool(fracs.max() < 0.97),
        "beats_replica_best_constant": bool(lengths.mean() > replica_best_constant),
    }
    (out / f"{args.run_id}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
