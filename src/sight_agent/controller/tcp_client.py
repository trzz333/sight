"""TCP client for the Python->Godot controller channel.

Scope:
- Binds nowhere. Connects to 127.0.0.1:8765 (override per instance).
- Newline-delimited JSON, UTF-8.
- Blocking send. No retry. Caller owns connection lifetime.
- No game state ever flows Godot->Python. Godot may send ack frames for reconciliation, but
  the perception channel is screen-scraped, not wire-fed. This is intentional; see
  docs/sight-handoff.md.
"""

from __future__ import annotations

import json
import socket
import time

from .. import constants


class TcpController:
    """JSON-line TCP client. One controller per Python agent run."""

    def __init__(
        self,
        run_id: str,
        agent: str = constants.AGENT_NAME_RULE_PARITY,
        host: str = constants.TCP_HOST,
        port: int = constants.TCP_PORT,
        protocol: int = constants.PROTOCOL_VERSION,
    ) -> None:
        self.run_id = run_id
        self.agent = agent
        self.host = host
        self.port = port
        self.protocol = protocol
        self._sock: socket.socket | None = None
        self._seq: int = 0

    # --- connection lifecycle ------------------------------------------------

    def connect(self, timeout_s: float = 2.0) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        s.connect((self.host, self.port))
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "TcpController":
        self.connect()
        self.send_hello()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- wire messages -------------------------------------------------------

    def send_hello(self) -> dict:
        msg = {
            "type": "hello",
            "protocol": self.protocol,
            "run_id": self.run_id,
            "agent": self.agent,
        }
        self._send_line(msg)
        return msg

    def send_action(self, action: str, ts_unix_ns: int | None = None) -> dict:
        """Send one action. Increments `seq` and returns the serialized message."""

        if action not in constants.ACTION_TO_MOVE_X:
            raise ValueError(
                f"invalid action {action!r}; must be one of {list(constants.ACTION_TO_MOVE_X)}"
            )
        self._seq += 1
        if ts_unix_ns is None:
            ts_unix_ns = time.time_ns()
        msg = {
            "type": "action",
            "seq": self._seq,
            "ts_unix_ns": int(ts_unix_ns),
            "action": action,
            "move_x": constants.ACTION_TO_MOVE_X[action],
        }
        self._send_line(msg)
        return msg

    # --- internals -----------------------------------------------------------

    @property
    def seq(self) -> int:
        return self._seq

    def _send_line(self, payload: dict) -> None:
        if self._sock is None:
            raise RuntimeError("controller not connected")
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        self._sock.sendall(line)
