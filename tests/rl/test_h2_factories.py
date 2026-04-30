"""Tests for sight_agent.rl.factories (H2 env + algo seams)."""

from __future__ import annotations

import pytest
from stable_baselines3.common.vec_env import VecEnv

from sight_agent.rl.factories import (
    SUPPORTED_ALGOS_BY_FRAMEWORK,
    SUPPORTED_FRAMEWORKS,
    make_algo,
    make_env,
    smoke_check_env,
)


def test_make_env_cartpole_train_returns_vecenv() -> None:
    env = make_env("CartPole-v1", n_envs=1, seed=0, mode="train")
    try:
        assert isinstance(env, VecEnv)
        obs = env.reset()
        # CartPole-v1 obs is shape (n_envs, 4); we only assert non-empty here.
        assert obs is not None
    finally:
        env.close()


def test_make_env_cartpole_eval_returns_vecenv() -> None:
    env = make_env("CartPole-v1", n_envs=1, seed=0, mode="eval")
    try:
        assert isinstance(env, VecEnv)
    finally:
        env.close()


def test_make_env_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        make_env("CartPole-v1", n_envs=1, seed=0, mode="train_eval")


def test_make_env_rejects_zero_n_envs() -> None:
    with pytest.raises(ValueError, match="n_envs"):
        make_env("CartPole-v1", n_envs=0, seed=0, mode="train")


def test_make_env_rejects_godot_prefix_with_clear_message() -> None:
    with pytest.raises(ValueError, match="Sight H2 supports Gymnasium"):
        make_env("godot:SignalDodge-v0", n_envs=1, seed=0, mode="train")


def test_smoke_check_env_returns_obs_shape_and_action_n() -> None:
    obs_shape, action_n = smoke_check_env("CartPole-v1", seed=0)
    assert obs_shape == (4,)
    assert action_n == 2


def test_make_algo_rejects_unsupported_framework() -> None:
    env = make_env("CartPole-v1", n_envs=1, seed=0, mode="train")
    try:
        with pytest.raises(ValueError, match="Unsupported framework"):
            make_algo(
                framework="cleanrl",
                name="PPO",
                policy="MlpPolicy",
                device="cpu",
                hyperparams={},
                env=env,
                seed=0,
            )
    finally:
        env.close()


def test_make_algo_rejects_unsupported_algo() -> None:
    env = make_env("CartPole-v1", n_envs=1, seed=0, mode="train")
    try:
        with pytest.raises(ValueError, match="Unsupported algo"):
            make_algo(
                framework="stable-baselines3",
                name="SAC",
                policy="MlpPolicy",
                device="cpu",
                hyperparams={},
                env=env,
                seed=0,
            )
    finally:
        env.close()


def test_make_algo_builds_sb3_ppo() -> None:
    from stable_baselines3 import PPO

    env = make_env("CartPole-v1", n_envs=1, seed=0, mode="train")
    try:
        model = make_algo(
            framework="stable-baselines3",
            name="PPO",
            policy="MlpPolicy",
            device="cpu",
            hyperparams={"n_steps": 64, "batch_size": 32, "n_epochs": 1},
            env=env,
            seed=0,
        )
        assert isinstance(model, PPO)
        assert model.n_steps == 64
        assert model.batch_size == 32
        assert model.n_epochs == 1
    finally:
        env.close()


def test_supported_constants_are_consistent() -> None:
    for fw in SUPPORTED_FRAMEWORKS:
        assert fw in SUPPORTED_ALGOS_BY_FRAMEWORK
        assert SUPPORTED_ALGOS_BY_FRAMEWORK[fw], f"no algos listed for {fw}"
