"""H2 factory seams for env and algorithm construction.

Two responsibilities:
1. env factory: build a VecEnv for train or eval, dispatched by env_id. For H2
   only Gymnasium env ids are supported. The dispatch is a single function,
   make_env, so H3 can add a Godot state-env branch without touching train or
   evaluate.
2. algo factory: build an SB3 PPO model from a (validated) config. For H2 the
   only supported (framework, name) pair is ('stable-baselines3', 'PPO'). Any
   other pair is rejected with a clear error.

Both functions raise ValueError with explicit messages naming what is supported
when given an unsupported request.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv


SUPPORTED_FRAMEWORKS = ("stable-baselines3",)
SUPPORTED_ALGOS_BY_FRAMEWORK = {
    "stable-baselines3": ("PPO",),
}


def make_env(env_id: str, n_envs: int, seed: int, mode: str = "train") -> VecEnv:
    """Build a VecEnv for train or eval.

    Dispatches by env_id. H2 supports Gymnasium env ids only (anything that
    ``gymnasium.make`` accepts). Future H3 will add a Godot state-env branch
    here without changes to callers.

    Args:
        env_id: env identifier (e.g. ``CartPole-v1``).
        n_envs: number of parallel envs (must be >= 1).
        seed: base seed. Eval mode uses ``seed + 10_000`` so eval and train
            seedings are disjoint by construction.
        mode: ``train`` or ``eval``.
    """
    if mode not in ("train", "eval"):
        raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
    if int(n_envs) < 1:
        raise ValueError(f"n_envs must be >= 1, got {n_envs}")

    if _looks_like_gymnasium(env_id):
        if mode == "train":
            return make_vec_env(env_id, n_envs=int(n_envs), seed=int(seed))
        return make_vec_env(env_id, n_envs=1, seed=int(seed) + 10_000)

    raise ValueError(
        f"Unsupported env_id: {env_id!r}. Sight H2 supports Gymnasium env ids only. "
        f"Future phases will add additional env families via this factory."
    )


def smoke_check_env(env_id: str, seed: int) -> tuple[tuple[int, ...], int]:
    """Single-reset smoke for run_start observability."""
    env = gym.make(env_id)
    obs, _info = env.reset(seed=int(seed))
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(seed))
    obs_shape = tuple(getattr(obs, "shape", ()))
    action_n = int(getattr(env.action_space, "n", 0))
    env.close()
    return obs_shape, action_n


def make_algo(
    framework: str,
    name: str,
    policy: str,
    device: str,
    hyperparams: dict[str, Any],
    env: VecEnv,
    seed: int,
) -> Any:
    """Build an algorithm instance from a (framework, name) pair.

    Raises ValueError with a clear, enumerated message if the combination is
    not in SUPPORTED_ALGOS_BY_FRAMEWORK. H2 supports SB3 PPO only.
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"Unsupported framework: {framework!r}. Sight H2 supports only "
            f"frameworks {list(SUPPORTED_FRAMEWORKS)}."
        )
    allowed = SUPPORTED_ALGOS_BY_FRAMEWORK[framework]
    if name not in allowed:
        raise ValueError(
            f"Unsupported algo {name!r} for framework {framework!r}. "
            f"Allowed algos for this framework: {list(allowed)}."
        )

    if framework == "stable-baselines3" and name == "PPO":
        kwargs: dict[str, Any] = {
            "policy": policy,
            "env": env,
            "seed": int(seed),
            "device": device,
        }
        if hyperparams:
            kwargs.update(hyperparams)
        return PPO(**kwargs)

    # Defensive: should be unreachable given the checks above.
    raise ValueError(
        f"No factory branch matched (framework={framework!r}, name={name!r})."
    )


def _looks_like_gymnasium(env_id: str) -> bool:
    """Very loose heuristic: H2 treats every non-Godot id as Gymnasium.

    Future H3 will add an explicit ``godot:`` (or similar) prefix branch in
    make_env. Until then, anything not flagged as a future env family is sent
    to Gymnasium and Gymnasium raises if it does not recognize the id.
    """
    if not isinstance(env_id, str) or not env_id:
        return False
    lowered = env_id.lower()
    if lowered.startswith("godot:") or lowered.startswith("godot/"):
        return False
    return True
