"""H3 plan section 8 default-tier smoke tests for ``GodotSignalDodgeEnv``.

This module is the acceptance-shaped overview for H3 closure. It proves
the Gym surface contract end-to-end against the shared fake transport
and fake process from ``tests/rl/h3_godot_fakes.py``: spaces, the
five-tuple ``step()`` shape, a ten-step rollout without protocol drift,
forced-collision termination, forced-timeout truncation, and
``close()`` idempotency.

The deeper lifecycle, NDJSON-evidence, early-subprocess-exit, and
process kill-fallback tests live in ``tests/rl/test_h3_godot_env.py``.
This file does not duplicate them.

No live Godot binary. No real subprocess. No ``live_godot`` marker.

Run:
    pytest tests/rl/test_h3_godot_smoke.py -v --tb=short
"""

from __future__ import annotations

import numpy as np
import pytest

from sight_agent import protocol
from sight_agent.rl.godot_env import GodotSignalDodgeEnv

from .h3_godot_fakes import (
    FakeProcess,
    FakeProcessFactoryRecorder,
    FakeTransport,
    FakeTransportFactoryRecorder,
    _reset_ok_payload,
    _step_ok_payload,
)


# --- helpers --------------------------------------------------------------


def _build_smoke_env() -> tuple[GodotSignalDodgeEnv, FakeProcess, FakeTransport]:
    """Build a ``GodotSignalDodgeEnv`` wired to fake transport + fake process.

    No subprocess is launched and no socket is opened: the env's lazy-launch
    contract means the fake factories are not invoked until ``reset()``,
    and even then they only record the call and return the pre-built fakes.
    """
    proc = FakeProcess()
    transport = FakeTransport(
        run_id="placeholder", host="127.0.0.1", port=0, recv_timeout_s=1.0
    )
    env = GodotSignalDodgeEnv(
        godot_executable=r"C:\fake\godot.exe",
        project_path=r"C:\fake\project",
        tcp_host="127.0.0.1",
        tcp_port=8765,
        run_dir=None,
        max_steps=100,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        seed=None,
        headless=True,
        transport_factory=FakeTransportFactoryRecorder(transport),
        process_factory=FakeProcessFactoryRecorder(proc),
    )
    return env, proc, transport


# --- tests ----------------------------------------------------------------


def test_spaces_match_h3_plan() -> None:
    """Observation space is ``Box(-1, 1, (10,), float32)``; action is ``Discrete(3)``."""
    import gymnasium as gym

    env, _, _ = _build_smoke_env()
    try:
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (10,)
        assert env.observation_space.dtype == np.float32
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 3
    finally:
        env.close()


def test_reset_returns_obs_in_observation_space() -> None:
    """``reset(seed=0)`` returns ``(obs, info)`` with obs inside the space."""
    env, _, transport = _build_smoke_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        assert info["seed"] == 0
    finally:
        env.close()


def test_step_returns_gym_five_tuple() -> None:
    """``step(action)`` returns ``(obs, reward, terminated, truncated, info)``."""
    env, _, transport = _build_smoke_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        transport.queue_step(_step_ok_payload(reward=1.0, frame=1))
        out = env.step(1)
        assert isinstance(out, tuple) and len(out) == 5
        obs, reward, terminated, truncated, info = out
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert terminated is False
        assert truncated is False
        assert isinstance(info, dict)
    finally:
        env.close()


def test_ten_step_rollout_runs_without_protocol_drift() -> None:
    """A ten-step survival rollout completes with no env-side errors."""
    env, _, transport = _build_smoke_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        for k in range(10):
            transport.queue_step(_step_ok_payload(seq=k, frame=k + 1, reward=1.0))
            obs, reward, terminated, truncated, _info = env.step(1)
            assert obs.shape == (10,)
            assert obs.dtype == np.float32
            assert reward == 1.0
            assert terminated is False
            assert truncated is False
        # Transport saw exactly the ten step calls we issued, in order.
        assert len(transport.step_calls) == 10
        assert [c["action"] for c in transport.step_calls] == [1] * 10
    finally:
        env.close()


def test_forced_collision_terminates_with_zero_reward() -> None:
    """Stub-injected collision yields ``terminated=True``, reward 0.0, reason collision."""
    env, _, transport = _build_smoke_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        transport.queue_step(
            _step_ok_payload(
                reward=0.0,
                terminated=True,
                truncated=False,
                terminal_reason=protocol.TERMINAL_REASON_COLLISION,
            )
        )
        _obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is True
        assert truncated is False
        assert reward == 0.0
        assert info["terminal_reason"] == protocol.TERMINAL_REASON_COLLISION
    finally:
        env.close()


def test_forced_timeout_truncates_without_terminating() -> None:
    """Stub-injected timeout yields ``truncated=True``, ``terminated=False``."""
    env, _, transport = _build_smoke_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        transport.queue_step(
            _step_ok_payload(
                reward=1.0,
                terminated=False,
                truncated=True,
                terminal_reason=protocol.TERMINAL_REASON_TIMEOUT,
            )
        )
        _obs, _reward, terminated, truncated, info = env.step(1)
        assert terminated is False
        assert truncated is True
        assert info["terminal_reason"] == protocol.TERMINAL_REASON_TIMEOUT
    finally:
        env.close()


def test_close_is_idempotent() -> None:
    """Calling ``close()`` twice is safe and does not raise."""
    env, proc, transport = _build_smoke_env()
    transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env.close()
    # Second close is a no-op; first close terminated the fake proc.
    env.close()
    assert transport.closed is True
    assert proc.terminate_calls == 1
