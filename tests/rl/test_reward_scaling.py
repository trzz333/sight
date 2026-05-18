"""K3.5c reward-scaling wrapper tests.

Direct unit tests for ``sight_agent.rl.reward_scaling`` covering:

- divide-by-divisor behavior on per-step rewards
- pass-through for ``reset()``
- pass-through for ``obs``, ``dones``, ``infos`` from the underlying VecEnv
- ``maybe_wrap_train_env`` no-op semantics for ``None`` and ``1.0``
- ``maybe_wrap_train_env`` wrap semantics for any other positive divisor
- ``ValueError`` on non-positive divisors
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from sight_agent.rl.reward_scaling import (
    FixedRewardScaleVecEnv,
    maybe_wrap_train_env,
)


class _ConstantRewardEnv(gym.Env):
    """Minimal env emitting a known fixed reward per step.

    Used to assert that the wrapper divides the underlying reward stream by
    the configured divisor without altering other return tuple elements.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, reward: float = 30.0) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._reward = float(reward)
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._steps = 0
        obs = np.zeros((2,), dtype=np.float32)
        return obs, {"reset": True}

    def step(self, action):
        self._steps += 1
        obs = np.full((2,), float(self._steps), dtype=np.float32)
        terminated = self._steps >= 4
        info = {"step": self._steps}
        return obs, self._reward, terminated, False, info


def _make_vec_env(reward: float = 30.0) -> DummyVecEnv:
    return DummyVecEnv([lambda: _ConstantRewardEnv(reward=reward)])


def test_divisor_30_scales_per_step_reward() -> None:
    venv = _make_vec_env(reward=30.0)
    wrapped = FixedRewardScaleVecEnv(venv, divisor=30.0)
    wrapped.reset()
    _obs, rewards, _dones, _infos = wrapped.step(np.array([0]))
    assert rewards.dtype == np.float32
    assert rewards.shape == (1,)
    assert rewards[0] == pytest.approx(1.0)
    wrapped.close()


def test_divisor_100_scales_per_step_reward() -> None:
    venv = _make_vec_env(reward=30.0)
    wrapped = FixedRewardScaleVecEnv(venv, divisor=100.0)
    wrapped.reset()
    _obs, rewards, _dones, _infos = wrapped.step(np.array([0]))
    assert rewards[0] == pytest.approx(0.3)
    wrapped.close()


def test_reset_passes_through_unchanged() -> None:
    venv = _make_vec_env()
    wrapped = FixedRewardScaleVecEnv(venv, divisor=30.0)
    obs = wrapped.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 2)
    assert np.array_equal(obs, np.zeros((1, 2), dtype=np.float32))
    wrapped.close()


def test_step_passes_obs_dones_infos_unchanged() -> None:
    venv = _make_vec_env(reward=30.0)
    wrapped = FixedRewardScaleVecEnv(venv, divisor=30.0)
    wrapped.reset()
    obs, _rewards, dones, infos = wrapped.step(np.array([0]))
    assert obs.shape == (1, 2)
    assert obs[0, 0] == pytest.approx(1.0)
    assert dones.shape == (1,)
    assert bool(dones[0]) is False
    assert len(infos) == 1
    wrapped.close()


def test_non_positive_divisor_raises() -> None:
    venv = _make_vec_env()
    with pytest.raises(ValueError, match="reward_scale_divisor must be > 0"):
        FixedRewardScaleVecEnv(venv, divisor=0.0)
    with pytest.raises(ValueError, match="reward_scale_divisor must be > 0"):
        FixedRewardScaleVecEnv(venv, divisor=-1.0)
    venv.close()


def test_maybe_wrap_none_is_noop() -> None:
    venv = _make_vec_env()
    out, applied = maybe_wrap_train_env(venv, None)
    assert out is venv
    assert applied is False
    venv.close()


def test_maybe_wrap_one_is_noop() -> None:
    venv = _make_vec_env()
    out, applied = maybe_wrap_train_env(venv, 1.0)
    assert out is venv
    assert applied is False
    venv.close()


def test_maybe_wrap_thirty_applies_division() -> None:
    venv = _make_vec_env(reward=30.0)
    out, applied = maybe_wrap_train_env(venv, 30.0)
    assert isinstance(out, FixedRewardScaleVecEnv)
    assert applied is True
    out.reset()
    _obs, rewards, _dones, _infos = out.step(np.array([0]))
    assert rewards[0] == pytest.approx(1.0)
    out.close()


def test_maybe_wrap_two_applies_division() -> None:
    venv = _make_vec_env(reward=30.0)
    out, applied = maybe_wrap_train_env(venv, 2.0)
    assert isinstance(out, FixedRewardScaleVecEnv)
    assert applied is True
    out.reset()
    _obs, rewards, _dones, _infos = out.step(np.array([0]))
    assert rewards[0] == pytest.approx(15.0)
    out.close()
