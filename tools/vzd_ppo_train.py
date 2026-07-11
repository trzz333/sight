"""PPO from pixels on ViZDoom defend_the_center (SB3, GPU).

RL teacher for the vzd track: trains a CnnPolicy on the bundled
VizdoomDefendCenter-v1 gymnasium env. Gray 60x80 (stride-4 downsample
of 240x320), frame-skip 4, frame-stack 4, gamma 0.99 (the K-phase
lesson). Writes progress to
<out>/log.csv, checkpoints, final model.zip + summary.json, and a
DONE sentinel for detached monitoring.

Note on --resume semantics: SB3's reset_num_timesteps=False treats
--steps as ADDITIONAL steps on top of the checkpoint, not a total
target. Resuming a 750k checkpoint with --steps 1500000 trains to
2.25M total.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_train.py --steps 1500000
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_train.py --steps 2000 --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_ID = "VizdoomDefendCenter-v1"
STRIDE = 4  # 240x320 RGB -> 60x80 gray


class GrayStride(gym.ObservationWrapper):
    """screen dict RGB (H,W,3) -> gray uint8 (H//s, W//s, 1), numpy only."""

    def __init__(self, env, stride: int = STRIDE):
        super().__init__(env)
        self._s = stride
        h, w, _ = env.observation_space["screen"].shape
        self.observation_space = gym.spaces.Box(
            0, 255, (h // stride, w // stride, 1), np.uint8)

    def observation(self, obs):
        g = obs["screen"][:: self._s, :: self._s].mean(axis=2)
        return g.astype(np.uint8)[..., None]


class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total, terminated, truncated = 0.0, False, False
        obs, info = None, {}
        for _ in range(self._skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            total += r
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


def make_env(seed: int):
    def _f():
        import vizdoom.gymnasium_wrapper  # noqa: F401  (registers envs)
        from stable_baselines3.common.monitor import Monitor
        env = gym.make(ENV_ID)
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env = Monitor(env)  # emits ep_rew_mean / ep_len_mean to the log
        env.reset(seed=seed)
        return env
    return _f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .zip to resume from; --steps stays the TOTAL target incl. checkpoint steps")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.callbacks import CheckpointCallback

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "vzd" / "ppo_defend"
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    venv = VecFrameStack(venv, 4)

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cuda")
        print("resumed from", args.resume, "at", model.num_timesteps, "steps")
    else:
        model = PPO(
            "CnnPolicy", venv, verbose=1, seed=args.seed,
            n_steps=256, batch_size=512, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1,
            ent_coef=0.01, vf_coef=0.5, n_epochs=4, device="cuda",
            tensorboard_log=None)
    print("obs space:", venv.observation_space)

    cb = None if args.smoke else CheckpointCallback(
        save_freq=max(250_000 // args.n_envs, 1), save_path=str(out),
        name_prefix="ppo_defend")

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False,
                reset_num_timesteps=not args.resume)
    train_s = time.time() - t0
    model.save(out / "model.zip")
    venv.close()

    # deterministic eval, fresh single env
    from stable_baselines3.common.vec_env import DummyVecEnv
    ev = VecFrameStack(DummyVecEnv([make_env(10_000)]), 4)
    rewards = []
    n_eval = 3 if args.smoke else 30
    obs = ev.reset()
    ep_r, done_count = 0.0, 0
    while done_count < n_eval:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = ev.step(act)
        ep_r += float(r[0])
        if done[0]:
            rewards.append(ep_r)
            ep_r = 0.0
            done_count += 1
    ev.close()

    r = np.array(rewards, float)
    from scipy.stats import trim_mean
    summary = {
        "env": ENV_ID, "steps": args.steps, "n_envs": args.n_envs,
        "seed": args.seed, "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "eval_episodes": n_eval, "mean_reward": float(r.mean()),
        "iqm_reward": float(trim_mean(r, 0.25)), "rewards": rewards,
        "gamma": 0.99, "recipe": "PPO CnnPolicy gray60x80 skip4 stack4"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text("ok")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
