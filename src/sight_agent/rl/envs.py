"""Gymnasium env builders for H1 RL training and eval.

Single env construction kept thin so SB3 owns vectorization. Seeding is set on
both the env and its action_space at reset to give a deterministic posture.
"""

from __future__ import annotations

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv


def make_train_env(env_id: str, n_envs: int, seed: int) -> VecEnv:
    """Build the training VecEnv. SB3's make_vec_env handles per-env seeding."""
    return make_vec_env(env_id, n_envs=int(n_envs), seed=int(seed))


def make_eval_env(env_id: str, seed: int) -> VecEnv:
    """Build a single-env VecEnv for evaluation, seeded distinctly from training."""
    return make_vec_env(env_id, n_envs=1, seed=int(seed) + 10_000)


def smoke_check_env(env_id: str, seed: int) -> tuple[tuple[int, ...], int]:
    """Return (obs_shape, action_n) after a single reset. Used for run_start sanity."""
    env = gym.make(env_id)
    obs, _info = env.reset(seed=int(seed))
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(seed))
    obs_shape = tuple(getattr(obs, "shape", ()))
    action_space = env.action_space
    action_n = int(getattr(action_space, "n", 0))
    env.close()
    return obs_shape, action_n
