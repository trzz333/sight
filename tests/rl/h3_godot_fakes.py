"""Shared fake/stub fixtures and payload helpers for H3 Godot tests.

Extracted in H3 step 9 so the default-tier smoke suite
(``tests/rl/test_h3_godot_smoke.py``) and the deeper lifecycle suite
(``tests/rl/test_h3_godot_env.py``) can share the same fake protocol
layer rather than re-implementing it. The fakes here intentionally
reproduce only what ``GodotSignalDodgeEnv`` reads from a real
transport / subprocess; they are NOT a generalized test harness.

No live Godot. No real subprocess. The factories injected via the env's
``transport_factory`` and ``process_factory`` test seams return these
fakes so the env's lifecycle, lazy-launch, and Gym surface can be
exercised entirely in-process.
"""

from __future__ import annotations

from sight_agent import protocol
from sight_agent.rl.godot_transport import GodotTransportError


__all__ = [
    "FakeTransport",
    "FakeProcess",
    "FakeProcessFactoryRecorder",
    "FakeTransportFactoryRecorder",
    "_reset_ok_payload",
    "_step_ok_payload",
]


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

    def set_exit_code(self, code: int) -> None:
        """Test helper: simulate the process having exited with ``code``.

        After this is called, ``poll()`` returns ``code`` rather than
        ``None``, which is how the env's ``_raise_if_godot_exited`` path
        is exercised without actually launching anything.
        """
        self._exit_code = code

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
