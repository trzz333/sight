"""Unit tests for ``sight_agent.rl.godot_env.GodotSignalDodgeEnv``.

Uses injected fake transport and fake process factories. No live Godot
binary required and no real subprocess started.

Run:
    pytest tests/rl/test_h3_godot_env.py -v --tb=short
"""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
import pytest

from sight_agent import protocol
from sight_agent.rl.godot_env import GodotSignalDodgeEnv, DEFAULT_MAX_STEPS
from sight_agent.rl.godot_transport import (
    GodotProtocolError,
    GodotRemoteError,
    GodotTransportError,
)


# --- fake/stub helpers (extracted in H3 step 9) ---------------------------
#
# The fake transport, fake process, recorder factories, and payload helpers
# are shared with ``tests/rl/test_h3_godot_smoke.py``. They live in
# ``h3_godot_fakes.py`` so the two test modules can share one fake protocol
# layer rather than re-implementing it.
from .h3_godot_fakes import (
    FakeProcess,
    FakeProcessFactoryRecorder,
    FakeTransport,
    FakeTransportFactoryRecorder,
    _reset_ok_payload,
    _step_ok_payload,
)


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def fake_proc():
    return FakeProcess()


@pytest.fixture
def fake_transport():
    return FakeTransport(run_id="placeholder", host="127.0.0.1", port=0, recv_timeout_s=1.0)


@pytest.fixture
def env(fake_proc, fake_transport):
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(fake_transport)
    e = GodotSignalDodgeEnv(
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
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    # Expose the recorders on the env for assertion access.
    e._test_proc_factory = proc_factory  # type: ignore[attr-defined]
    e._test_tx_factory = tx_factory  # type: ignore[attr-defined]
    try:
        yield e
    finally:
        e.close()


# --- tests ---------------------------------------------------------------
#
# Test 1: __init__ does not launch Godot or open TCP. Lazy launch is part
# of the contract documented in docs/sight-h3-plan.md and the step 6 spec.


def test_init_does_not_launch_godot_or_connect(env):
    assert env._test_proc_factory.calls == []  # type: ignore[attr-defined]
    assert env._test_tx_factory.calls == []  # type: ignore[attr-defined]
    assert env.godot_pid is None


# Test 2: Spaces match the H3 plan section 2 / 3.


def test_spaces_match_plan(env):
    import gymnasium as gym

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (10,)
    assert env.observation_space.dtype == np.float32
    assert float(env.observation_space.low.min()) == -1.0
    assert float(env.observation_space.high.max()) == 1.0


# Test 3: First reset() launches the subprocess, opens TCP, sends hello,
# and threads max_steps + a deterministic episode_id into the wire.


def test_reset_launches_godot_and_sends_hello(env, fake_proc, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    obs, info = env.reset(seed=42)

    # Process launched once with the SIGHT_TCP_MODE env var.
    proc_calls = env._test_proc_factory.calls  # type: ignore[attr-defined]
    assert len(proc_calls) == 1
    assert proc_calls[0]["env"]["SIGHT_TCP_MODE"] == "1"
    assert proc_calls[0]["env"]["SIGHT_TCP_PORT"] == "8765"
    # Headless flag passed through.
    assert "--headless" in proc_calls[0]["cmd"]
    # Transport built once and hello sent.
    assert len(env._test_tx_factory.calls) == 1  # type: ignore[attr-defined]
    assert fake_transport.connected is True
    assert fake_transport.hello_sent is True

    # reset() return shape.
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert info["seed"] == 42
    assert info["episode_id"] == "ep-000001"
    assert info["run_id"].startswith("sight-h3-")
    assert info["godot_pid"] == fake_proc.pid
    assert info["tcp_port"] == 8765
    # Transport saw the right reset args.
    assert fake_transport.reset_calls == [
        {"seed": 42, "max_steps": 100, "episode_id": "ep-000001",
         "curriculum_n_init": 0}
    ]


# Test 4: Multiple resets reuse the same Godot subprocess and the same
# transport (in-process soft reset per Decision 2).


def test_multiple_resets_reuse_process_and_transport(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=1)
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=2)
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=3)

    assert len(env._test_proc_factory.calls) == 1  # type: ignore[attr-defined]
    assert len(env._test_tx_factory.calls) == 1  # type: ignore[attr-defined]
    # Episode counter increments and is stable.
    assert [c["episode_id"] for c in fake_transport.reset_calls] == [
        "ep-000001",
        "ep-000002",
        "ep-000003",
    ]


# Test 5: step() returns the Gymnasium 5-tuple with correct types.


def test_step_returns_five_tuple(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step(_step_ok_payload(reward=1.0, frame=1))
    out = env.step(2)
    assert isinstance(out, tuple) and len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, np.ndarray) and obs.shape == (10,) and obs.dtype == np.float32
    assert isinstance(reward, float) and reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info["frame"] == 1
    assert info["terminal_reason"] == ""
    # numpy int actions are accepted without ValueError.
    fake_transport.queue_step(_step_ok_payload(seq=1, frame=2))
    env.step(np.int64(1))
    assert fake_transport.step_calls[-1]["action"] == 1
    assert isinstance(fake_transport.step_calls[-1]["action"], int)


# Test 6: Forced collision step produces terminated=True and reward 0.0.


def test_forced_collision_terminates(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step(
        _step_ok_payload(
            reward=0.0,
            terminated=True,
            truncated=False,
            terminal_reason=protocol.TERMINAL_REASON_COLLISION,
        )
    )
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert truncated is False
    assert reward == 0.0
    assert info["terminal_reason"] == protocol.TERMINAL_REASON_COLLISION


# Test 7: Forced timeout produces truncated=True without terminated.


def test_forced_timeout_truncates(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step(
        _step_ok_payload(
            reward=1.0,
            terminated=False,
            truncated=True,
            terminal_reason=protocol.TERMINAL_REASON_TIMEOUT,
        )
    )
    _, _, terminated, truncated, info = env.step(1)
    assert terminated is False
    assert truncated is True
    assert info["terminal_reason"] == protocol.TERMINAL_REASON_TIMEOUT


# Test 8: step() before reset() raises a clear runtime error rather than
# silently calling into the transport. Same for after-terminal step.


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError) as exc_info:
        env.step(1)
    assert "reset" in str(exc_info.value).lower()


def test_step_after_terminal_raises(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step(
        _step_ok_payload(
            terminated=True,
            terminal_reason=protocol.TERMINAL_REASON_COLLISION,
            reward=0.0,
        )
    )
    env.step(0)
    with pytest.raises(RuntimeError) as exc_info:
        env.step(0)
    assert "termin" in str(exc_info.value).lower() or "reset" in str(exc_info.value).lower()


# Test 9: close() is idempotent. Process is terminated; transport closed.


def test_close_is_idempotent_and_terminates_process(env, fake_proc, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env.close()
    assert fake_transport.closed is True
    assert fake_proc.terminate_calls == 1
    # Second and third close calls must not raise and must not re-terminate.
    env.close()
    env.close()
    assert fake_proc.terminate_calls == 1


def test_close_kills_when_terminate_hangs(env, fake_proc, fake_transport):
    fake_proc.simulate_terminate_hangs = True
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env.close()
    assert fake_proc.terminate_calls == 1
    assert fake_proc.kill_calls == 1


def test_close_before_reset_does_not_raise(env):
    # Constructor was the only call. close() must be safe.
    env.close()
    env.close()


# Test 10: Transport / protocol / remote errors propagate unchanged.
# Per docs/sight-h3-plan.md section 5: a broken transport is not a
# terminal observation.


def test_transport_error_on_reset_propagates(env, fake_transport):
    fake_transport.queue_reset_raise(GodotTransportError("simulated drop"))
    with pytest.raises(GodotTransportError):
        env.reset(seed=0)


def test_protocol_error_on_step_propagates(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step_raise(GodotProtocolError("seq mismatch"))
    with pytest.raises(GodotProtocolError):
        env.step(1)


def test_remote_error_on_reset_propagates(env, fake_transport):
    fake_transport.queue_reset_raise(
        GodotRemoteError(code="bad_request", message="x", payload={})
    )
    with pytest.raises(GodotRemoteError):
        env.reset(seed=0)


# Test 11: Connect retry loop honors connect_timeout_s. The fake transport
# raises GodotTransportError on the first N attempts and succeeds afterwards.


def test_connect_retries_until_godot_listening(fake_proc):
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.connect_attempts_to_succeed = 3
    tx.queue_reset(_reset_ok_payload())
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        connect_timeout_s=2.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        e.reset(seed=0)
        assert tx.connect_calls == 3
    finally:
        e.close()


def test_connect_timeout_raises_transport_error(fake_proc):
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    # Never succeeds in any reasonable test budget.
    tx.connect_attempts_to_succeed = 10**9
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        connect_timeout_s=0.3,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        with pytest.raises(GodotTransportError):
            e.reset(seed=0)
    finally:
        e.close()


# Test 12: SIGHT_GODOT_LOG_PATH is set when run_dir is provided, and
# parent directories are created.


def test_run_dir_sets_godot_log_path(tmp_path, fake_proc):
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.queue_reset(_reset_ok_payload())
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    run_dir = tmp_path / "runs" / "test-run"
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        run_dir=run_dir,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        e.reset(seed=0)
        env_passed = proc_factory.calls[0]["env"]
        assert "SIGHT_GODOT_LOG_PATH" in env_passed
        assert env_passed["SIGHT_GODOT_LOG_PATH"].endswith("godot.ndjson")
        # Parent directory must exist on disk.
        assert run_dir.is_dir()
    finally:
        e.close()


def test_no_run_dir_means_no_log_path_env(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env_passed = env._test_proc_factory.calls[0]["env"]  # type: ignore[attr-defined]
    assert "SIGHT_GODOT_LOG_PATH" not in env_passed


# Test 13: Constructor argument validation.


def test_constructor_rejects_zero_max_steps():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(godot_executable="x", project_path="y", max_steps=0)


def test_constructor_rejects_negative_timeouts():
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x", project_path="y", connect_timeout_s=0.0
        )
    with pytest.raises(ValueError):
        GodotSignalDodgeEnv(
            godot_executable="x", project_path="y", step_timeout_s=-1.0
        )


# Test 14: Several short scripted rollouts run without protocol drift.
# Mirrors the "10 steps without drift" assertion in plan section 8 (default
# fast smoke), but kept generic so step 6 doesn't depend on the smoke test
# name.


def test_ten_step_rollout_runs_clean(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    for i in range(10):
        fake_transport.queue_step(_step_ok_payload(seq=i, frame=i + 1, reward=1.0))
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (10,)
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert info["frame"] == i + 1


# Test 15: reset called on a closed env raises rather than silently
# re-launching.


def test_reset_after_close_raises(env, fake_transport):
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env.close()
    with pytest.raises(RuntimeError):
        env.reset(seed=0)


# Test 16: Default constructor does NOT pass --headless when headless=False.


def test_headless_false_omits_flag(fake_proc):
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.queue_reset(_reset_ok_payload())
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        headless=False,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        e.reset(seed=0)
        assert "--headless" not in proc_factory.calls[0]["cmd"]
    finally:
        e.close()


# Test 17: Default max_steps is the documented constant.


def test_default_max_steps_is_documented_constant():
    assert DEFAULT_MAX_STEPS == 1800


# --- Step 6 hardening tests: Python NDJSON evidence + early-exit detection ---


import json as _json


def _read_ndjson_events(path) -> list[dict]:
    """Read python.ndjson and return parsed event dicts."""
    events: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(_json.loads(line))
    return events


def _make_env_with_run_dir(tmp_path, fake_proc, fake_transport, **overrides):
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(fake_transport)
    kwargs = dict(
        godot_executable=r"C:\fake\godot.exe",
        project_path=r"C:\fake\project",
        tcp_host="127.0.0.1",
        tcp_port=8765,
        run_dir=tmp_path,
        max_steps=100,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        seed=None,
        headless=True,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    kwargs.update(overrides)
    return GodotSignalDodgeEnv(**kwargs)


def test_python_ndjson_written_under_run_dir(tmp_path, fake_proc, fake_transport):
    e = _make_env_with_run_dir(tmp_path, fake_proc, fake_transport)
    try:
        fake_transport.queue_reset(_reset_ok_payload())
        e.reset(seed=7)
        fake_transport.queue_step(_step_ok_payload(seq=0, frame=1, reward=1.0))
        e.step(1)
    finally:
        e.close()

    ndjson_path = tmp_path / "python.ndjson"
    assert ndjson_path.is_file(), "python.ndjson must exist under run_dir"
    events = _read_ndjson_events(ndjson_path)
    types = [ev["type"] for ev in events]
    # Order: env_start (after first connect), reset, step, close.
    assert types == ["env_start", "reset", "step", "close"], types
    # Identity decoration is present on every line.
    for ev in events:
        assert ev["run_id"].startswith("sight-h3-")
        assert ev["godot_pid"] == fake_proc.pid or ev["type"] == "close"
        assert ev["tcp_port"] == 8765
        assert "ts_unix" in ev
    reset_ev = events[1]
    assert reset_ev["episode_id"] == "ep-000001"
    assert reset_ev["seed"] == 7
    assert reset_ev["frame"] == 0
    step_ev = events[2]
    assert step_ev["frame"] == 1
    assert step_ev["reward"] == 1.0
    assert step_ev["terminated"] is False
    assert step_ev["truncated"] is False
    assert step_ev["terminal_reason"] == ""


def test_python_ndjson_omitted_when_no_run_dir(env, fake_transport, tmp_path):
    # The default ``env`` fixture has run_dir=None. Run a normal cycle and
    # verify no python.ndjson appears anywhere under tmp_path or env._run_dir.
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    fake_transport.queue_step(_step_ok_payload(seq=0))
    env.step(1)
    env.close()
    # The env never gets a run_dir, so the writer must remain None.
    assert env._ndjson is None
    # Defensive: confirm nothing leaked into tmp_path.
    leftovers = list(tmp_path.glob("**/python.ndjson"))
    assert leftovers == [], leftovers


def test_python_ndjson_logs_error_event_on_reset_failure(
    tmp_path, fake_proc, fake_transport
):
    e = _make_env_with_run_dir(tmp_path, fake_proc, fake_transport)
    try:
        fake_transport.queue_reset_raise(GodotProtocolError("simulated reset fail"))
        with pytest.raises(GodotProtocolError):
            e.reset(seed=11)
    finally:
        e.close()
    events = _read_ndjson_events(tmp_path / "python.ndjson")
    types = [ev["type"] for ev in events]
    assert "env_start" in types
    assert "error" in types
    err = next(ev for ev in events if ev["type"] == "error")
    assert err["where"] == "reset"
    assert err["kind"] == "GodotProtocolError"
    assert err["episode_id"] == "ep-000001"
    assert err["seed"] == 11
    assert "simulated reset fail" in err["message"]


def test_python_ndjson_logs_error_event_on_step_failure(
    tmp_path, fake_proc, fake_transport
):
    e = _make_env_with_run_dir(tmp_path, fake_proc, fake_transport)
    try:
        fake_transport.queue_reset(_reset_ok_payload())
        e.reset(seed=0)
        fake_transport.queue_step_raise(GodotTransportError("simulated step drop"))
        with pytest.raises(GodotTransportError):
            e.step(2)
    finally:
        e.close()
    events = _read_ndjson_events(tmp_path / "python.ndjson")
    err = next(ev for ev in events if ev["type"] == "error")
    assert err["where"] == "step"
    assert err["kind"] == "GodotTransportError"
    assert "simulated step drop" in err["message"]


def test_early_subprocess_exit_during_connect_raises_distinctly(fake_proc):
    """Godot dies before TCP listener becomes reachable -> distinct error."""
    fake_proc.set_exit_code(42)
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    # Make the fake transport perpetually fail to connect so the env's
    # check happens to land on the exited process.
    tx.connect_attempts_to_succeed = 10**9
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable=r"C:\fake\godot.exe",
        project_path=r"C:\fake\project",
        tcp_port=8765,
        connect_timeout_s=2.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        with pytest.raises(GodotTransportError) as exc_info:
            e.reset(seed=0)
        msg = str(exc_info.value)
        # Distinguishing markers: exit code + port + cmd.
        assert "exited" in msg.lower()
        assert "42" in msg, msg
        assert "8765" in msg, msg
        assert "godot.exe" in msg.lower(), msg
        # Must NOT be the connect-timeout message variant.
        assert "not reachable within" not in msg
    finally:
        e.close()


def test_early_exit_logs_error_event_with_run_dir(tmp_path, fake_proc):
    fake_proc.set_exit_code(7)
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.connect_attempts_to_succeed = 10**9
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable=r"C:\fake\godot.exe",
        project_path=r"C:\fake\project",
        tcp_port=9001,
        run_dir=tmp_path,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        with pytest.raises(GodotTransportError):
            e.reset(seed=0)
    finally:
        e.close()
    events = _read_ndjson_events(tmp_path / "python.ndjson")
    types = [ev["type"] for ev in events]
    # env_start should NOT have been logged because connect never succeeded.
    assert "env_start" not in types
    err = next(ev for ev in events if ev["type"] == "error")
    assert err["where"] == "connect"
    assert err["kind"] == "GodotTransportError"
    assert "9001" in err["message"]
    assert "exited" in err["message"].lower()



# Test: Godot 4.6.2 hangs at startup if stdout/stderr are subprocess.PIPE
# under Popen on Windows (verified by matrix test 2026-05-09 against both
# windowed and console builds). The env's process factory must be invoked
# with file-like objects (when run_dir is set) or subprocess.DEVNULL (when
# not), never subprocess.PIPE. This is a regression guard, not a Gym
# semantics test.


def test_launch_passes_devnull_for_stdio_when_no_run_dir(env, fake_transport):
    """No run_dir -> Popen called with DEVNULL for both stdout and stderr."""
    fake_transport.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    call = env._test_proc_factory.calls[0]  # type: ignore[attr-defined]
    assert call["stdout"] is subprocess.DEVNULL
    assert call["stderr"] is subprocess.DEVNULL


def test_launch_passes_file_handles_for_stdio_when_run_dir_set(
    tmp_path, fake_proc
) -> None:
    """run_dir set -> Popen called with binary-write file objects under run_dir.

    Files land at ``godot-stdout.log`` / ``godot-stderr.log`` next to
    ``godot.ndjson`` so a hang or crash leaves recoverable stdio evidence.
    """
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.queue_reset(_reset_ok_payload())
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        run_dir=tmp_path,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    try:
        e.reset(seed=0)
        call = proc_factory.calls[0]
        # Neither value may be subprocess.PIPE (the deadlock trigger) and
        # neither may be DEVNULL (we have a run_dir; capture is required).
        assert call["stdout"] is not subprocess.PIPE
        assert call["stderr"] is not subprocess.PIPE
        assert call["stdout"] is not subprocess.DEVNULL
        assert call["stderr"] is not subprocess.DEVNULL
        # The captured handles must be writable file objects pointed at
        # the documented filenames under run_dir.
        out_h = call["stdout"]
        err_h = call["stderr"]
        assert hasattr(out_h, "write")
        assert hasattr(err_h, "write")
        assert (tmp_path / "godot-stdout.log").is_file()
        assert (tmp_path / "godot-stderr.log").is_file()
    finally:
        e.close()


def test_close_releases_godot_stdio_files(tmp_path, fake_proc) -> None:
    """``close()`` must close the stdout/stderr file handles it opened."""
    tx = FakeTransport(run_id="x", host="127.0.0.1", port=0, recv_timeout_s=1.0)
    tx.queue_reset(_reset_ok_payload())
    proc_factory = FakeProcessFactoryRecorder(fake_proc)
    tx_factory = FakeTransportFactoryRecorder(tx)
    e = GodotSignalDodgeEnv(
        godot_executable="x",
        project_path="y",
        run_dir=tmp_path,
        connect_timeout_s=1.0,
        step_timeout_s=1.0,
        max_steps=10,
        transport_factory=tx_factory,
        process_factory=proc_factory,
    )
    e.reset(seed=0)
    call = proc_factory.calls[0]
    out_h = call["stdout"]
    err_h = call["stderr"]
    # Both handles are open before close.
    assert not out_h.closed
    assert not err_h.closed
    e.close()
    # Both handles must be closed after close().
    assert out_h.closed
    assert err_h.closed
    # close() is idempotent.
    e.close()
