"""H3 plan section 8 smoke tests for ``GodotSignalDodgeEnv``.

Two tiers per ``docs/sight-h3-plan.md`` section 8.

Default tier: proves the Gym surface contract end-to-end against the
shared fake transport and fake process from ``tests/rl/h3_godot_fakes.py``
(spaces, the five-tuple ``step()`` shape, a ten-step rollout without
protocol drift, forced-collision termination, forced-timeout truncation,
and ``close()`` idempotency). No real subprocess.

Live tier: a single test marked ``@pytest.mark.live_godot`` that launches
the real Godot Signal Dodge build via ``SIGHT_GODOT_EXE``, runs
``reset(seed=0)`` plus up to 100 steps over real loopback TCP, and
verifies non-malformed protocol exchange plus a minimum
``godot.ndjson`` event-type set. The ``live_godot`` marker is excluded
from the default run by ``addopts`` in ``pyproject.toml``.

The deeper lifecycle, NDJSON-evidence, early-subprocess-exit, and
process kill-fallback tests live in ``tests/rl/test_h3_godot_env.py``.
This file does not duplicate them.

Run:
    pytest tests/rl/test_h3_godot_smoke.py -v --tb=short
    pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

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


# --- live tier (opt-in) --------------------------------------------------
#
# A single ``live_godot``-marked test launches the real Godot Signal Dodge
# build, runs ``reset(seed=0)`` plus up to 100 steps over loopback TCP, and
# asserts non-malformed protocol exchange plus a minimum ``godot.ndjson``
# event-type set. Excluded from the default run by ``addopts`` in
# ``pyproject.toml``. Opt-in command:
#     pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short


def _allocate_isolated_tcp_port() -> int:
    """Bind to ``127.0.0.1:0``, capture the kernel-assigned port, then release.

    The TOCTOU window between releasing the socket and Godot binding the
    same port is negligible on loopback inside one CI/dev box; the env's
    ``connect_timeout_s`` retry covers transient races.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _repo_root() -> Path:
    # tests/rl/test_h3_godot_smoke.py -> repo root is parents[2].
    return Path(__file__).resolve().parents[2]


@pytest.mark.live_godot
def test_live_godot_reset_and_100_step_smoke(tmp_path: Path) -> None:
    """Live Godot smoke: real subprocess, real TCP, ``reset`` plus up to 100 steps.

    Opt-in only. Requires ``SIGHT_GODOT_EXE`` pointing at a Godot 4.x binary
    that can run ``games/signal-dodge``. Failure modes:

    - missing/invalid ``SIGHT_GODOT_EXE`` -> ``pytest.fail`` (not skip;
      this gate is an acceptance command, not an optional capability probe).
    - ``GodotTransportError`` / ``GodotProtocolError`` / ``GodotRemoteError``
      from the env propagate out of ``reset`` / ``step`` and fail the test.
      That is the no-malformed-protocol assertion: the H3 transport raises
      on missing/extra fields, type mismatches, run_id/episode_id mismatches,
      protocol_version mismatches, and remote ``error`` responses.

    Asserts after close:

    - ``godot.ndjson`` exists, every line parses as a JSON object, and the
      observed ``type`` set covers the H3 plan-section-7 minimum
      (``run_start``, ``controller_connected``, ``controller_hello``,
      ``controller_reset_received``, ``episode_start``, ``h3_step``).
      ``collision`` / ``death`` / ``run_end`` are not required at this
      tier; ``collision`` and ``death`` are terminal-contingent and
      ``run_end`` is shutdown-timing-sensitive.
    - if ``python.ndjson`` exists, no event has ``type == "error"``.
    """
    godot_exe_raw = os.environ.get("SIGHT_GODOT_EXE", "")
    if not godot_exe_raw:
        pytest.fail(
            "SIGHT_GODOT_EXE is not set. The live_godot acceptance gate "
            "requires the Godot 4.x executable path. Set it at User or "
            "Machine scope, e.g.:\n"
            r'  setx SIGHT_GODOT_EXE "C:\path\to\Godot_v4.x-stable_win64.exe"'
        )
    godot_exe = Path(godot_exe_raw)
    if not godot_exe.is_file():
        pytest.fail(
            f"SIGHT_GODOT_EXE={godot_exe_raw!r} does not point to a file."
        )

    project_path = _repo_root() / "games" / "signal-dodge"
    if not (project_path / "project.godot").is_file():
        pytest.fail(
            f"Expected Godot project at {project_path}; project.godot not found."
        )

    port = _allocate_isolated_tcp_port()
    env = GodotSignalDodgeEnv(
        godot_executable=godot_exe,
        project_path=project_path,
        tcp_host="127.0.0.1",
        tcp_port=port,
        run_dir=tmp_path,
        # >100 so the env-level max-steps clamp does not truncate this rollout.
        max_steps=120,
        connect_timeout_s=20.0,
        step_timeout_s=10.0,
        seed=None,
        headless=True,
    )
    steps_taken = 0
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(1)
            steps_taken += 1
            assert obs.shape == (10,)
            assert obs.dtype == np.float32
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            if terminated or truncated:
                break
        assert steps_taken >= 1, "live env produced no steps"
    finally:
        env.close()

    godot_ndjson = tmp_path / "godot.ndjson"
    assert godot_ndjson.is_file(), f"godot.ndjson missing at {godot_ndjson}"
    raw_lines = [
        ln for ln in godot_ndjson.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert raw_lines, f"godot.ndjson at {godot_ndjson} has no events"
    types_seen: set[str] = set()
    for ln in raw_lines:
        rec = json.loads(ln)
        assert isinstance(rec, dict), f"non-object NDJSON line: {ln!r}"
        types_seen.add(str(rec.get("type", "")))
    required = {
        "run_start",
        "controller_connected",
        "controller_hello",
        "controller_reset_received",
        "episode_start",
        "h3_step",
    }
    missing = required - types_seen
    assert not missing, (
        f"godot.ndjson missing required event types {sorted(missing)}; "
        f"saw {sorted(types_seen)}"
    )

    python_ndjson = tmp_path / "python.ndjson"
    if python_ndjson.is_file():
        for ln in python_ndjson.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            assert rec.get("type") != "error", (
                f"python.ndjson contains error event: {rec!r}"
            )
