"""Minimal TCP client helpers for the P3 live harness.

Phase B wire contract reused exactly: hello once, then action frames until a
terminal condition. Newline-delimited UTF-8 JSON. Loopback only. Std-lib only.

These functions are pure I/O helpers. They do not own a connection lifetime
beyond the call, do not own the action budget, and do not read the
environment. The caller (scripts/run_p3_eval.py) owns episode bookkeeping.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Mapping


PROTOCOL_VERSION: int = 1
DEFAULT_AGENT: str = "p3-stub"


def action_for_seq(seq: int) -> tuple[str, int]:
    """Deterministic stub policy mirroring scripts/run_phase_b.py.

    Returns (action_label, move_x). One of stay, left, right.
    """
    m = seq % 30
    if m < 10:
        return ("stay", 0)
    if m < 20:
        return ("left", -1)
    return ("right", 1)


def build_hello(
    run_id: str,
    agent: str = DEFAULT_AGENT,
    protocol: int = PROTOCOL_VERSION,
) -> dict:
    return {
        "type": "hello",
        "protocol": protocol,
        "run_id": run_id,
        "agent": agent,
    }


def build_action(seq: int, ts_unix_ns: int) -> dict:
    action, move_x = action_for_seq(seq)
    return {
        "type": "action",
        "seq": seq,
        "ts_unix_ns": ts_unix_ns,
        "action": action,
        "move_x": move_x,
    }


def build_decision(
    run_id: str,
    seq: int,
    capture_ts_unix_ns: int,
    sent_ts_unix_ns: int,
) -> dict:
    action, move_x = action_for_seq(seq)
    return {
        "type": "decision",
        "run_id": run_id,
        "seq": seq,
        "action": action,
        "move_x": move_x,
        "capture_ts_unix_ns": capture_ts_unix_ns,
        "decision_ts_unix_ns": capture_ts_unix_ns,
        "sent_ts_unix_ns": sent_ts_unix_ns,
    }


def connect_with_retry(host: str, port: int, timeout_sec: float) -> socket.socket:
    """Block until a TCP connection succeeds or the deadline passes.

    Raises ConnectionError on timeout. Returns a connected socket with a 2s
    operation timeout. Caller owns close.
    """
    deadline = time.monotonic() + timeout_sec
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            s.settimeout(2.0)
            return s
        except OSError as e:
            last_err = e
            time.sleep(0.25)
    raise ConnectionError(
        f"could not connect to {host}:{port} within {timeout_sec}s: {last_err}"
    )


def wait_for_port_bind(
    host: str,
    port: int,
    timeout_sec: float,
    *,
    poll_interval_sec: float = 0.25,
) -> bool:
    """Poll until something is listening on host:port. Returns True on success."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(poll_interval_sec)
    return False


def send_json_line(sock: socket.socket, payload: Mapping[str, Any]) -> int:
    """Send one JSON line. Returns the unix-ns timestamp at send completion."""
    line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(line)
    return time.time_ns()


__all__ = [
    "PROTOCOL_VERSION",
    "DEFAULT_AGENT",
    "action_for_seq",
    "build_hello",
    "build_action",
    "build_decision",
    "connect_with_retry",
    "wait_for_port_bind",
    "send_json_line",
]
