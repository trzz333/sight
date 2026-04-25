"""Controller tests. A fake in-process TCP server accepts the client, reads JSON lines, and
asserts the wire schema. No Godot dependency."""

from __future__ import annotations

import json
import socket
import threading

import pytest

from sight_agent import constants
from sight_agent.controller import TcpController


class _FakeTcpServer:
    """Accept one connection, read JSON lines until the client closes, then expose them."""

    def __init__(self) -> None:
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", 0))
        self._srv.settimeout(1.0)
        self._srv.listen(1)
        self.port = self._srv.getsockname()[1]
        self.messages: list[dict] = []
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._closed.is_set():
                try:
                    conn, _ = self._srv.accept()
                except (socket.timeout, TimeoutError):
                    continue
                except OSError:
                    # Listen socket closed during teardown. Clean exit.
                    return
                with conn:
                    buf = b""
                    while True:
                        try:
                            chunk = conn.recv(4096)
                        except OSError:
                            break
                        if not chunk:
                            break
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            if line.strip():
                                self.messages.append(json.loads(line.decode("utf-8")))
                return
        except Exception:
            # Swallow any teardown-related noise; tests assert on `messages` directly.
            return

    def close(self) -> None:
        self._closed.set()
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)


@pytest.fixture
def fake_server():
    srv = _FakeTcpServer()
    try:
        yield srv
    finally:
        srv.close()


def test_controller_sends_hello_and_actions_with_incrementing_seq(fake_server):
    ctl = TcpController(run_id="test-run-001", port=fake_server.port)
    ctl.connect()
    try:
        ctl.send_hello()
        ctl.send_action(constants.ACTION_LEFT)
        ctl.send_action(constants.ACTION_STAY)
        ctl.send_action(constants.ACTION_RIGHT)
    finally:
        ctl.close()

    # Wait for the server thread to drain the connection.
    fake_server._thread.join(timeout=2.0)

    msgs = fake_server.messages
    assert len(msgs) == 4

    hello = msgs[0]
    assert hello == {
        "type": "hello",
        "protocol": constants.PROTOCOL_VERSION,
        "run_id": "test-run-001",
        "agent": constants.AGENT_NAME_RULE_PARITY,
    }

    actions = msgs[1:]
    expected = [
        (1, constants.ACTION_LEFT, -1),
        (2, constants.ACTION_STAY, 0),
        (3, constants.ACTION_RIGHT, 1),
    ]
    for msg, (seq, action, move_x) in zip(actions, expected):
        assert msg["type"] == "action"
        assert msg["seq"] == seq
        assert msg["action"] == action
        assert msg["move_x"] == move_x
        assert isinstance(msg["ts_unix_ns"], int)


def test_controller_context_manager_sends_hello_and_closes(fake_server):
    with TcpController(run_id="ctx-run", port=fake_server.port) as ctl:
        ctl.send_action(constants.ACTION_LEFT)
    fake_server._thread.join(timeout=2.0)

    assert fake_server.messages[0]["type"] == "hello"
    assert fake_server.messages[1] == {
        "type": "action",
        "seq": 1,
        "ts_unix_ns": fake_server.messages[1]["ts_unix_ns"],
        "action": constants.ACTION_LEFT,
        "move_x": -1,
    }


def test_controller_rejects_invalid_action():
    ctl = TcpController(run_id="bad", port=1)  # no connect
    with pytest.raises(ValueError):
        ctl.send_action("jump")


def test_controller_send_before_connect_raises():
    ctl = TcpController(run_id="noconn", port=1)
    with pytest.raises(RuntimeError):
        ctl.send_hello()
