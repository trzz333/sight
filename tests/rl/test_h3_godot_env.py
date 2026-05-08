"""Unit tests for ``sight_agent.rl.godot_env.GodotSignalDodgeEnv``.

Uses injected fake transport and fake process factories. No live Godot
binary required and no real subprocess started.

Run:
    pytest tests/rl/test_h3_godot_env.py -v --tb=short
"""

from __future__ import annotations

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


# --- fakes ----------------------------------------------------------------


class FakeTransport:
    """Duck-typed substitute for ``GodotH3Transport``.

    Tests pre-queue scripted ``reset_ok`` / ``step_result`` payloads (or
    raisable exceptions) via ``queue_reset`` / ``queue_step`` /
    ``queue_raise``. The ``reset`` and ``step`` methods then pop the next
    queued item in FIFO order.
    """

    def __init__(self, run_id: str, host: str, port: int, recv_timeout_s: float):
        self.run_id = run_id
        self.host = host
        self.port = port
        self.recv_timeout_s = recv_timeout_s
        self.connected = False
        self.hello_sent = False
        self.closed = False
        self.episode_id = ""
        self.connect_calls = 0
        self.connect_attempts_to_succeed = 1  # set >1 to simulate Godot-not-ready
        self.reset_calls: list[dict] = []
        self.step_calls: list[dict] = []
        self._reset_queue: list = []
        self._step_queue: list = []

    # --- transport API ---

    def connect(self, connect_timeout_s: float = 2.0) -> None:
        self.connect_calls += 1
        if self.connect_calls < self.connect_attempts_to_succeed:
            raise GodotTransportError("simulated: Godot listener not ready")
        self.connected = True

    def send_hello(self) -> None:
        if not self.connected:
            raise GodotTransportError("not connected")
        self.hello_sent = True

    def reset(self, seed: int, max_steps: int, episode_id: str) -> dict:
        self.reset_calls.append(
            {"seed": seed, "max_steps": max_steps, "episode_id": episode_id}
        )
        item = self._reset_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        # Backfill episode_id so the env contract sees what it requested.
        if "episode_id" not in item:
            item = {**item, "episode_id": episode_id}
        self.episode_id = episode_id
        return item

    def step(self, action: int) -> dict:
        self.step_calls.append({"action": action})
        item = self._step_queue.pop(0)
        if isinstance(item, Exception):
            raise item
        if "episode_id" not in item:
            item = {**item, "episode_id": self.episode_id}
        return item

    def close(self) -> None:
        self.closed = True

    # --- test helpers ---

    def queue_reset(self, payload: dict) -> None:
        self._reset_queue.append(payload)

    def queue_step(self, payload: dict) -> None:
        self._step_queue.append(payload)

    def queue_reset_raise(self, exc: Exception) -> None:
        self._reset_queue.append(exc)

    def queue_step_raise(self, exc: Exception) -> None:
        self._step_queue.append(exc)


class FakeProcess:
    """Duck-typed substitute for ``subprocess.Popen``.

    Tracks ``terminate`` / ``kill`` / ``wait`` calls so close() teardown
    can be asserted. ``simulate_terminate_hangs=True`` makes ``wait`` raise
    a fake ``TimeoutExpired`` so the env hits the kill fallback path.
    """

    def __init__(self, pid: int = 12345):
        self.pid = pid
        self._exit_code: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0
        self.simulate_terminate_hangs = False

    def poll(self) -> int | None:
        return self._exit_code

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.simulate_terminate_hangs:
            self._exit_code = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._exit_code = -9

    def wait(self, timeout: float | None = None) -> int:
        import subprocess as _sp

        self.wait_calls += 1
        if self.simulate_terminate_hangs and self.kill_calls == 0:
            raise _sp.TimeoutExpired(cmd="fake", timeout=timeout)
        if self._exit_code is None:
            self._exit_code = 0
        return self._exit_code


class FakeProcessFactoryRecorder:
    """Records every call to the env's process_factory and returns ``proc``."""

    def __init__(self, proc: FakeProcess):
        self.proc = proc
        self.calls: list[dict] = []

    def __call__(self, cmd: list[str], *, env: dict, stdout=None, stderr=None) -> FakeProcess:
        self.calls.append(
            {"cmd": list(cmd), "env": dict(env), "stdout": stdout, "stderr": stderr}
        )
        return self.proc


class FakeTransportFactoryRecorder:
    """Records ctor kwargs and returns ``transport``."""

    def __init__(self, transport: FakeTransport):
        self.transport = transport
        self.calls: list[dict] = []

    def __call__(self, *, run_id: str, host: str, port: int, recv_timeout_s: float) -> FakeTransport:
        # Update the transport's identifying fields so the env sees a
        # consistent view, mirroring the real factory.
        self.transport.run_id = run_id
        self.transport.host = host
        self.transport.port = port
        self.transport.recv_timeout_s = recv_timeout_s
        self.calls.append(
            {"run_id": run_id, "host": host, "port": port, "recv_timeout_s": recv_timeout_s}
        )
        return self.transport


# --- payload helpers ------------------------------------------------------


def _reset_ok_payload(frame: int = 0) -> dict:
    return {
        "type": protocol.MSG_RESET_OK,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": "ignored-by-fake",
        "frame": frame,
        "obs": [0.0] * 10,
        "terminated": False,
        "truncated": False,
        "info": {},
    }


def _step_ok_payload(
    seq: int = 0,
    frame: int = 1,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    terminal_reason: str = protocol.TERMINAL_REASON_NONE,
    obs: list | None = None,
    info: dict | None = None,
) -> dict:
    return {
        "type": protocol.MSG_STEP_RESULT,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": "ignored-by-fake",
        "seq": seq,
        "frame": frame,
        "obs": obs if obs is not None else [0.1] * 10,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": terminal_reason,
        "info": info if info is not None else {},
    }


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
        {"seed": 42, "max_steps": 100, "episode_id": "ep-000001"}
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
