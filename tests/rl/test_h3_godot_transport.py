"""Unit tests for ``sight_agent.rl.godot_transport.GodotH3Transport``.

Local fake TCP server only; no live Godot dependency. The fake mirrors the
``games/signal-dodge/scripts/tcp_controller.gd`` H3 wire shape on the
inbound side and lets each test pre-queue scripted outbound responses so
the transport's blocking ``recv`` always finds a matching reply already in
the kernel buffer.

Run:
    pytest tests/rl/test_h3_godot_transport.py -v --tb=short
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

import pytest

from sight_agent import protocol
from sight_agent.rl.godot_transport import (
    GodotH3Transport,
    GodotProtocolError,
    GodotRemoteError,
    GodotTransportError,
)


# --- fake server -----------------------------------------------------------


class FakeGodotServer:
    """Loopback TCP fake. Single-client, scripted responses.

    The accept-and-poll loop runs in a daemon thread. Tests pre-queue
    outbound bytes via ``queue_response`` / ``queue_raw``; the loop drains
    the outbound queue every tick and parses incoming newline-delimited
    JSON lines into ``inbound`` for assertion.
    """

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.host, self.port = self._sock.getsockname()
        self._peer: socket.socket | None = None
        self._stop = threading.Event()
        self.inbound: list[dict | None] = []
        self._inbound_lock = threading.Lock()
        self._outbound: list[bytes] = []
        self._outbound_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._sock.settimeout(2.0)
        try:
            peer, _ = self._sock.accept()
        except (OSError, socket.timeout):
            return
        peer.settimeout(0.05)
        self._peer = peer
        buf = b""
        while not self._stop.is_set():
            # Drain outbound first.
            with self._outbound_lock:
                pending = self._outbound
                self._outbound = []
            for chunk in pending:
                try:
                    peer.sendall(chunk)
                except OSError:
                    return
            # Try to recv one chunk.
            try:
                chunk = peer.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                nl = buf.index(b"\n")
                line = buf[:nl]
                buf = buf[nl + 1:]
                try:
                    parsed: dict | None = json.loads(line.decode("utf-8"))
                except Exception:
                    parsed = None
                with self._inbound_lock:
                    self.inbound.append(parsed)

    def queue_response(self, payload: dict) -> None:
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with self._outbound_lock:
            self._outbound.append(line)

    def queue_raw(self, data: bytes) -> None:
        with self._outbound_lock:
            self._outbound.append(data)

    def wait_inbound(self, n: int, timeout_s: float = 2.0) -> None:
        t0 = time.time()
        while True:
            with self._inbound_lock:
                got = len(self.inbound)
            if got >= n:
                return
            if time.time() - t0 > timeout_s:
                raise TimeoutError(
                    f"timed out waiting for {n} inbound messages, got {got}"
                )
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        if self._peer is not None:
            try:
                self._peer.close()
            except OSError:
                pass
        self._thread.join(timeout=2.0)


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def server():
    s = FakeGodotServer()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def transport(server):
    t = GodotH3Transport(
        run_id="test-run",
        host=server.host,
        port=server.port,
        recv_timeout_s=2.0,
    )
    t.connect(connect_timeout_s=2.0)
    try:
        yield t
    finally:
        t.close()


# --- payload helpers -------------------------------------------------------


def _reset_ok(run_id: str, episode_id: str, frame: int = 0) -> dict:
    return {
        "type": protocol.MSG_RESET_OK,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "frame": frame,
        "obs": [0.0] * 10,
        "terminated": False,
        "truncated": False,
        "info": {},
    }


def _step_result(
    run_id: str,
    episode_id: str,
    seq: int,
    frame: int = 1,
    reward: float = 1.0,
    terminated: bool = False,
    truncated: bool = False,
    terminal_reason: str = "",
) -> dict:
    return {
        "type": protocol.MSG_STEP_RESULT,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "seq": seq,
        "frame": frame,
        "obs": [0.1] * 10,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": terminal_reason,
        "info": {},
    }


def _error(code: str, message: str) -> dict:
    return {
        "type": protocol.MSG_ERROR,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "code": code,
        "message": message,
    }



# --- tests -----------------------------------------------------------------
#
# Test 1: send_hello() uses the H3 ``protocol_version`` field, not legacy
# ``protocol``. See games/signal-dodge/scripts/tcp_controller.gd: the field
# name is the mode-locking discriminator.


def test_send_hello_uses_h3_protocol_version_field(server, transport):
    transport.send_hello()
    server.wait_inbound(1)
    msg = server.inbound[0]
    assert msg is not None
    assert msg["type"] == protocol.MSG_HELLO
    assert msg[protocol.FIELD_PROTOCOL_VERSION] == protocol.H3_PROTOCOL_VERSION
    assert msg["run_id"] == "test-run"
    # Hard guarantee: the legacy field must NOT appear, otherwise the Godot
    # mode dispatcher would lock to MODE_LEGACY and never answer reset/step.
    assert "protocol" not in msg


# Test 2: reset() sends required fields and accepts a valid reset_ok.


def test_reset_sends_required_fields_and_returns_reset_ok(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    resp = transport.reset(seed=42, max_steps=100, episode_id="ep-1")

    server.wait_inbound(1)
    sent = server.inbound[0]
    assert sent is not None
    for f in protocol.REQUIRED_FIELDS_RESET:
        assert f in sent, f"reset request missing required field {f!r}"
    assert sent["type"] == protocol.MSG_RESET
    assert sent[protocol.FIELD_PROTOCOL_VERSION] == protocol.H3_PROTOCOL_VERSION
    assert sent["run_id"] == "test-run"
    assert sent["episode_id"] == "ep-1"
    assert sent["seed"] == 42
    assert sent["max_steps"] == 100

    assert resp["type"] == protocol.MSG_RESET_OK
    assert resp["episode_id"] == "ep-1"
    assert len(resp["obs"]) == 10
    assert transport.episode_id == "ep-1"
    assert transport.seq == 0


# Test 3: step() sends seq/action and accepts a matching step_result.


def test_step_sends_seq_and_action_and_returns_step_result(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    server.queue_response(_step_result(transport.run_id, "ep-1", seq=0))
    resp = transport.step(protocol.ACTION_DISCRETE_RIGHT)

    server.wait_inbound(2)
    sent = server.inbound[1]
    assert sent is not None
    for f in protocol.REQUIRED_FIELDS_STEP:
        assert f in sent
    assert sent["type"] == protocol.MSG_STEP
    assert sent["seq"] == 0
    assert sent["action"] == protocol.ACTION_DISCRETE_RIGHT
    assert resp["type"] == protocol.MSG_STEP_RESULT
    assert resp["seq"] == 0
    assert transport.seq == 1


# Test 4: first step seq is 0; second is 1.


def test_first_step_seq_is_zero_then_one(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    server.queue_response(_step_result(transport.run_id, "ep-1", seq=0))
    transport.step(protocol.ACTION_DISCRETE_STAY)
    server.queue_response(_step_result(transport.run_id, "ep-1", seq=1))
    transport.step(protocol.ACTION_DISCRETE_LEFT)

    server.wait_inbound(3)
    step1 = server.inbound[1]
    step2 = server.inbound[2]
    assert step1 is not None and step2 is not None
    assert step1["seq"] == 0
    assert step2["seq"] == 1
    assert transport.seq == 2


# Test 5: invalid action raises before sending. Verify ValueError plus
# zero outbound bytes in the inbound buffer.


def test_invalid_action_raises_before_sending(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    server.wait_inbound(1)

    # Out-of-range int.
    with pytest.raises(ValueError):
        transport.step(99)
    # Wrong type.
    with pytest.raises(ValueError):
        transport.step("right")  # type: ignore[arg-type]
    # bool is technically int subclass; reject explicitly.
    with pytest.raises(ValueError):
        transport.step(True)  # type: ignore[arg-type]

    # Settle and confirm only the reset request reached the server.
    time.sleep(0.1)
    assert len(server.inbound) == 1, server.inbound


# Test 6: Godot ``error`` response raises GodotRemoteError with code,
# message, and raw payload preserved.


def test_godot_error_response_raises_remote_error(server, transport):
    server.queue_response(_error("bad_request", "pipeline overrun"))
    with pytest.raises(GodotRemoteError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    err = exc_info.value
    assert err.code == "bad_request"
    assert err.message == "pipeline overrun"
    assert err.payload["type"] == protocol.MSG_ERROR
    assert err.payload[protocol.FIELD_PROTOCOL_VERSION] == protocol.H3_PROTOCOL_VERSION


# Test 7: mismatched run_id, episode_id, or seq each raise GodotProtocolError.


def test_run_id_mismatch_raises_protocol_error(server, transport):
    bad = _reset_ok(run_id="evil-run", episode_id="ep-1")
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "run_id" in str(exc_info.value)


def test_episode_id_mismatch_on_reset_raises_protocol_error(server, transport):
    bad = _reset_ok(transport.run_id, episode_id="ep-other")
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "episode_id" in str(exc_info.value)


def test_episode_id_mismatch_on_step_raises_protocol_error(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    bad = _step_result(transport.run_id, episode_id="ep-other", seq=0)
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.step(protocol.ACTION_DISCRETE_STAY)
    assert "episode_id" in str(exc_info.value)


def test_seq_mismatch_raises_protocol_error(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    server.queue_response(_step_result(transport.run_id, "ep-1", seq=42))
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.step(protocol.ACTION_DISCRETE_STAY)
    assert "seq" in str(exc_info.value)


# Test 8: malformed JSON, missing required fields, or wrong type each raise.


def test_malformed_json_raises_protocol_error(server, transport):
    server.queue_raw(b"this is not json at all\n")
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "JSON" in str(exc_info.value) or "malformed" in str(exc_info.value).lower()


def test_reset_ok_missing_required_field_raises_protocol_error(server, transport):
    bad = _reset_ok(transport.run_id, "ep-1")
    del bad["obs"]
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "obs" in str(exc_info.value)


def test_response_with_wrong_type_raises_protocol_error(server, transport):
    # Server returns a step_result when reset_ok is expected.
    bad = _step_result(transport.run_id, "ep-1", seq=0)
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "type" in str(exc_info.value)


def test_invalid_terminal_reason_raises_protocol_error(server, transport):
    server.queue_response(_reset_ok(transport.run_id, "ep-1"))
    transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    bad = _step_result(
        transport.run_id, "ep-1", seq=0, terminal_reason="garbage_value"
    )
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.step(protocol.ACTION_DISCRETE_STAY)
    assert "terminal_reason" in str(exc_info.value)


def test_obs_wrong_length_raises_protocol_error(server, transport):
    bad = _reset_ok(transport.run_id, "ep-1")
    bad["obs"] = [0.0] * 5  # wrong length
    server.queue_response(bad)
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")
    assert "obs" in str(exc_info.value)


# Test 9: close() is idempotent and survives never-connected transports.


def test_close_is_idempotent(server, transport):
    transport.close()
    transport.close()  # Second call must not raise.
    transport.close()  # Third for good measure.


def test_close_on_never_connected_transport_does_not_raise():
    t = GodotH3Transport(run_id="x", host="127.0.0.1", port=1)
    # Never called connect().
    t.close()
    t.close()


# Test 10: strict request-response policy is documented in the class docstring.


def test_strict_request_response_is_documented():
    doc = GodotH3Transport.__doc__ or ""
    # The docstring must explicitly describe the no-pipelining contract that
    # the Godot ``tcp_controller`` single-pending-slot enforces.
    assert "request-response" in doc.lower()
    assert "pipelin" in doc.lower()
    # The reset and step method docs must also mark blocking behavior.
    assert GodotH3Transport.reset.__doc__ is not None
    assert "block" in GodotH3Transport.reset.__doc__.lower()
    assert GodotH3Transport.step.__doc__ is not None
    assert "block" in GodotH3Transport.step.__doc__.lower()


# --- additional coverage: peer drop, send before connect ------------------


def test_recv_after_peer_drop_raises_transport_error(server, transport):
    # Force the fake to close mid-flight by stopping its thread before any
    # request arrives. The transport's recv should raise GodotTransportError
    # rather than convert into a fake terminal observation.
    server.close()
    with pytest.raises(GodotTransportError):
        transport.reset(seed=1, max_steps=10, episode_id="ep-1")


def test_step_before_reset_raises_protocol_error(server, transport):
    with pytest.raises(GodotProtocolError) as exc_info:
        transport.step(protocol.ACTION_DISCRETE_STAY)
    assert "reset" in str(exc_info.value).lower()


def test_send_before_connect_raises_transport_error(server):
    t = GodotH3Transport(run_id="t", host=server.host, port=server.port)
    # Never called connect().
    with pytest.raises(GodotTransportError):
        t.send_hello()
