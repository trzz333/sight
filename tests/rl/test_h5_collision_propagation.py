"""H5 collision-propagation contract tests.

Locks in the env-layer contract that the GDScript bug fix in
``games/signal-dodge/scripts/main.gd`` now honors. See
``docs/h5-collision-propagation-bug.md``.

Three env-fake-level tests:

1. ``test_between_step_collision_is_consumed_on_next_step`` simulates the
   wire-level contract that the GDScript fix now produces. A hazard
   physics tick between Python step N and step N+1 causes
   ``_on_player_died`` to fire. The sticky terminal flag means step N+1
   returns ``terminated=True``, ``truncated=False``,
   ``terminal_reason="collision"``, ``reward=0.0``, with collision info
   nested under ``godot_info``.

2. ``test_reset_clears_terminal_state_and_next_step_is_non_terminal``
   covers the reset-clearing regression: after a collision-terminal
   step, ``reset()`` succeeds and the very next ``step()`` returns
   non-terminal (unless a new collision occurs in that step).

3. ``test_step_after_terminal_without_reset_raises`` covers the
   step-after-terminal guard at the env layer. After a terminal
   step_result, calling ``step()`` again before ``reset()`` raises
   ``RuntimeError`` from the env's own guard. The GDScript side also
   replies ``bad_request`` for any such request that does reach it
   (the env's RuntimeError fires first, but if a future env regression
   removed that guard, this test plus the GDScript bad_request reply
   provide defense in depth).

These do NOT directly exercise the GDScript fix. The live integration
gate for the fix is the negative-control smoke
(``python -m sight_agent.rl.h5_baseline_cli --mode negative-controls``
on the H4 pixel profile) and the pre-existing live_godot smoke. This
file is the Python-side contract anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from sight_agent import protocol
from sight_agent.rl.godot_env import GodotSignalDodgeEnv
from sight_agent.rl.godot_transport import GodotRemoteError

from .h3_godot_fakes import (
    FakeProcess,
    FakeProcessFactoryRecorder,
    FakeTransport,
    FakeTransportFactoryRecorder,
    _reset_ok_payload,
    _step_ok_payload,
)


def _build_env() -> tuple[GodotSignalDodgeEnv, FakeProcess, FakeTransport]:
    """Same fake-wired env as ``tests/rl/test_h3_godot_smoke.py``.

    Duplicated locally rather than imported because the smoke module
    treats ``_build_smoke_env`` as private and may evolve independently.
    Drift here is acceptable; the fakes are shared via
    ``tests/rl/h3_godot_fakes.py``.
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


def _collision_step_payload(seq: int, frame: int) -> dict:
    """A step_result mirroring the post-fix GDScript wire for a between-step death.

    Mirrors the contract that ``games/signal-dodge/scripts/main.gd`` now
    produces: when ``_on_player_died`` fires on a hazard physics tick
    before step N+1 arrives, the sticky ``_h3_step_terminated`` survives,
    and step N+1's reply carries ``terminated=True``, ``reward=0.0``,
    ``terminal_reason="collision"``, plus the collision info in the info
    dict.
    """
    return _step_ok_payload(
        seq=seq,
        frame=frame,
        reward=0.0,
        terminated=True,
        truncated=False,
        terminal_reason=protocol.TERMINAL_REASON_COLLISION,
        info={
            "frame": frame,
            "action": 0,  # action from this step's request (mapped left)
            "collision": {
                "frame": frame,
                "player_x": 360.0,
                "player_y": 480.0,
                "hazard_x": 360.0,
                "hazard_y": 480.0,
                "survival_time": 12.5,
            },
        },
    )


def test_between_step_collision_is_consumed_on_next_step() -> None:
    """Step N+1 reflects a between-step collision per the GDScript fix.

    Scripted timeline:
    - reset returns non-terminal post-reset state
    - step 0 returns non-terminal (no collision yet)
    - between step 0 and step 1, a hazard physics tick fires
      ``_on_player_died`` in Godot; the sticky terminal flag survives
    - step 1 returns terminated=True with terminal_reason=collision,
      reward=0.0, and the collision info nested under ``godot_info``
    """
    env, _, transport = _build_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)

        # Step 0: no collision yet. Player still alive.
        transport.queue_step(_step_ok_payload(seq=0, frame=1, reward=1.0))
        _obs, reward, terminated, truncated, _info = env.step(1)
        assert terminated is False
        assert truncated is False
        assert reward == 1.0

        # Step 1: GDScript fix means the between-step _on_player_died
        # fire-up survives the next request and propagates as terminated.
        transport.queue_step(_collision_step_payload(seq=1, frame=2))
        obs, reward, terminated, truncated, info = env.step(0)

        # Five-tuple integrity.
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10,)
        assert obs.dtype == np.float32

        # Terminal contract.
        assert terminated is True
        assert truncated is False
        assert reward == 0.0
        assert info["terminal_reason"] == protocol.TERMINAL_REASON_COLLISION

        # Collision info is forwarded under ``godot_info`` per
        # GodotSignalDodgeEnv._build_info contract.
        assert "godot_info" in info
        wire_info = info["godot_info"]
        assert isinstance(wire_info, dict)
        assert "collision" in wire_info
        collision = wire_info["collision"]
        assert collision["frame"] == 2
        assert collision["player_x"] == 360.0
        assert collision["hazard_y"] == 480.0
        assert "survival_time" in collision
    finally:
        env.close()


def test_reset_clears_terminal_state_and_next_step_is_non_terminal() -> None:
    """After collision-terminal, reset() clears state and next step is non-terminal.

    Mirrors the GDScript fix's reset-clear behavior: the sticky terminal
    flag is wiped by ``_h3_perform_soft_reset`` so the new episode starts
    clean. From the env's perspective: the second reset succeeds, the
    next step returns terminated=False, and the env's ``_episode_done``
    bookkeeping is consistent across the episode boundary.
    """
    env, _, transport = _build_env()
    try:
        # Episode 1: reset, immediately terminal collision on step 0.
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        transport.queue_step(_collision_step_payload(seq=0, frame=1))
        _obs, _reward, terminated, _truncated, _info = env.step(1)
        assert terminated is True

        # Episode 2: reset clears the terminal state on both sides.
        # Godot's _h3_perform_soft_reset wipes the sticky flags; the env
        # clears its own _episode_done. Stepping resumes.
        transport.queue_reset(_reset_ok_payload(frame=0))
        obs, info = env.reset(seed=1)
        assert obs.shape == (10,)
        assert info["seed"] == 1

        # Next step is non-terminal because the fresh episode has not
        # accumulated any new collision yet.
        transport.queue_step(_step_ok_payload(seq=0, frame=1, reward=1.0))
        _obs, reward, terminated, truncated, _info = env.step(1)
        assert terminated is False
        assert truncated is False
        assert reward == 1.0

        # Transport saw exactly the expected calls in order.
        assert len(transport.reset_calls) == 2
        assert len(transport.step_calls) == 2
    finally:
        env.close()


def test_step_after_terminal_without_reset_raises() -> None:
    """After terminal step, step() without reset raises at the env layer.

    The env guard at ``GodotSignalDodgeEnv.step`` raises ``RuntimeError``
    before the request reaches the transport. This is the Python-side
    half of the defense-in-depth pair: the GDScript side independently
    replies ``bad_request`` (which would surface as
    ``GodotRemoteError``) if a future env regression removed this
    guard. Test the env guard here; the GDScript guard is implicit in
    the live smoke.
    """
    env, _, transport = _build_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)

        # Terminal collision step.
        transport.queue_step(_collision_step_payload(seq=0, frame=1))
        _obs, _reward, terminated, _truncated, _info = env.step(1)
        assert terminated is True

        # Second step without reset must raise. The env's own
        # _episode_done gate fires here; no transport call should be
        # recorded for this attempted step.
        with pytest.raises(RuntimeError, match="step called after episode"):
            env.step(1)

        # Transport saw exactly the one step from above, not two.
        assert len(transport.step_calls) == 1
    finally:
        env.close()


def test_godot_remote_bad_request_is_propagated_to_caller() -> None:
    """If GDScript replies bad_request for step-after-terminal, env surfaces it.

    Belt-and-suspenders for the env-layer guard test above. Simulates
    the scenario where the env guard is missing or bypassed and the
    GDScript ``bad_request`` reply reaches the transport. The transport
    raises ``GodotRemoteError`` and the env propagates it unchanged per
    the error model. This locks in that no fallback ever converts a
    remote bad_request into a synthetic terminated=True.
    """
    env, _, transport = _build_env()
    try:
        transport.queue_reset(_reset_ok_payload())
        env.reset(seed=0)
        # Force the env's _episode_done back to False so this test can
        # exercise the transport-level error path. Without this hack the
        # env guard fires first (covered by the prior test).
        env._episode_done = False
        transport.queue_step_raise(
            GodotRemoteError(
                code="bad_request",
                message="step received on a done episode; reset required",
                payload={"type": "error", "code": "bad_request"},
            )
        )
        with pytest.raises(GodotRemoteError, match="bad_request"):
            env.step(1)
    finally:
        env.close()
