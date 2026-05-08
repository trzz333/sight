"""Python-side bidirectional H3 TCP transport for the Godot Signal Dodge env.

Strictly request-response, no pipelining. Send step N, block for step_result N,
then send step N+1. This is REQUIRED because the Godot ``tcp_controller.gd``
holds a single pending-request slot and rejects overruns with error code
``bad_request``. ``GodotH3Transport`` enforces the policy by performing the
matching ``recv`` inside each public method before returning, so callers
cannot pipeline by accident.

Step 5 of ``docs/sight-h3-plan.md`` Implementation Sequence. The downstream
``GodotSignalDodgeEnv`` (step 6) wraps this transport in a Gymnasium-
compatible env. This module deliberately stays free of Gymnasium and NumPy
so the transport can be tested without those dependencies.

Wire contract: see ``games/signal-dodge/scripts/tcp_controller.gd`` and
``src/sight_agent/protocol.py``. Loopback TCP, newline-delimited UTF-8 JSON.

Sequence numbering policy:
- First ``step`` after ``reset`` carries ``seq=0``.
- Subsequent steps increment by 1 per successful response.
- ``reset`` rewinds ``seq`` to 0.

Error model:
- ``GodotTransportError``: socket / OS / framing failure (closed peer,
  timeout, truncated line, send failure).
- ``GodotProtocolError``: structural protocol violation (wrong type, wrong
  protocol_version, missing required field, run_id/episode_id/seq mismatch,
  malformed JSON, out-of-range obs / terminal_reason).
- ``GodotRemoteError``: Godot replied with ``type="error"``. Carries
  ``code`` and ``message`` from the payload plus the raw payload dict.

Transport failures and protocol errors are NOT converted into terminal
observations. The env layer (step 6) is responsible for any terminal-state
mapping.
"""

from __future__ import annotations

import json
import socket
from typing import Any

from .. import constants
from .. import protocol


class GodotTransportError(RuntimeError):
    """Socket, framing, or OS-level failure on the H3 transport channel.

    Raised when the TCP peer closes mid-message, a recv times out, a send
    fails, or the underlying socket raises ``OSError``. Callers must close
    the transport and treat the env as broken; do not interpret as a
    terminal observation.
    """


class GodotProtocolError(RuntimeError):
    """Structural violation of the H3 wire protocol.

    Raised on malformed JSON, missing required fields, wrong message type,
    wrong ``protocol_version``, run_id/episode_id/seq mismatch, invalid
    ``terminal_reason`` literal, or out-of-shape ``obs``. Distinct from
    ``GodotRemoteError`` (which carries a Godot-originated ``type="error"``
    payload).
    """


class GodotRemoteError(RuntimeError):
    """Godot returned a ``type="error"`` response.

    Carries the wire ``code``, the wire ``message``, and the raw ``payload``
    dict so callers can branch on specific error codes (e.g.
    ``protocol_version_mismatch``, ``bad_request``).
    """

    def __init__(self, code: str, message: str, payload: dict) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.payload = payload


class GodotH3Transport:
    """Bidirectional H3 TCP transport. Strictly request-response, no pipelining.

    Lifecycle: ``connect()`` -> ``send_hello()`` -> ``reset(...)`` ->
    ``step(...)`` ... -> ``close()``.

    Pipelining policy (LOAD-BEARING, see module docstring): send ``step`` N,
    block for ``step_result`` N, then send ``step`` N+1. The Godot
    ``tcp_controller`` parks at most one pending request and answers a
    second-arrival with ``error`` code ``bad_request``. This class enforces
    the policy by recv-inside-method and exposes no async send variant.

    Sequence policy: first step after ``reset`` uses ``seq=0``; ``seq``
    increments by 1 after a successful ``step_result``; ``reset`` rewinds
    ``seq`` to 0.

    Threading: not thread-safe. One transport per env, one env per thread.
    """

    def __init__(
        self,
        run_id: str,
        host: str = constants.TCP_HOST,
        port: int = constants.TCP_PORT,
        recv_timeout_s: float = 5.0,
    ) -> None:
        if not isinstance(run_id, str) or run_id == "":
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id
        self.host = host
        self.port = port
        self.recv_timeout_s = recv_timeout_s
        self._sock: socket.socket | None = None
        self._buf: bytes = b""
        self._episode_id: str = ""
        self._seq: int = 0

    # --- lifecycle ---------------------------------------------------------

    def connect(self, connect_timeout_s: float = 2.0) -> None:
        if self._sock is not None:
            raise GodotTransportError("transport already connected")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(connect_timeout_s)
        try:
            s.connect((self.host, self.port))
        except OSError as e:
            s.close()
            raise GodotTransportError(
                f"connect to {self.host}:{self.port} failed: {e}"
            ) from e
        s.settimeout(self.recv_timeout_s)
        self._sock = s

    def close(self) -> None:
        """Idempotent. Safe to call multiple times and after partial init."""
        s = self._sock
        self._sock = None
        if s is None:
            return
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            s.close()
        except OSError:
            pass

    @property
    def seq(self) -> int:
        """Next ``step`` request seq. 0 immediately after reset."""
        return self._seq

    @property
    def episode_id(self) -> str:
        """Active episode_id, or empty string before the first reset."""
        return self._episode_id

    # --- public wire methods ----------------------------------------------

    def send_hello(self) -> None:
        """Send H3 hello. No response expected (Godot logs it; H3 hello has
        no ``reset_ok``-style ack, see ``tcp_controller.gd::_h3_handle_hello``).
        """
        msg = {
            "type": protocol.MSG_HELLO,
            protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
            "run_id": self.run_id,
        }
        self._send_line(msg)

    def reset(self, seed: int, max_steps: int, episode_id: str) -> dict[str, Any]:
        """Send ``reset`` and block for exactly one ``reset_ok`` (or ``error``).

        Validates the response shape, run_id, episode_id, and obs vector.
        Rewinds the local seq counter to 0 only on success.
        """
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be int, got {type(seed).__name__}")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps <= 0
        ):
            raise ValueError(f"max_steps must be positive int, got {max_steps!r}")
        if not isinstance(episode_id, str) or episode_id == "":
            raise ValueError("episode_id must be a non-empty string")

        msg = {
            "type": protocol.MSG_RESET,
            protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "episode_id": episode_id,
            "seed": seed,
            "max_steps": max_steps,
        }
        self._send_line(msg)
        resp = self._recv_message()
        self._reject_if_error(resp)
        self._validate_response_shape(
            resp,
            expected_type=protocol.MSG_RESET_OK,
            required=protocol.REQUIRED_FIELDS_RESET_OK,
        )
        self._validate_run_id(resp)
        resp_eid = resp.get("episode_id")
        if resp_eid != episode_id:
            raise GodotProtocolError(
                f"episode_id mismatch on reset_ok: expected {episode_id!r}, got {resp_eid!r}"
            )
        self._validate_obs(resp.get("obs"))
        # Commit only after full validation. Failed reset leaves prior
        # episode_id (or "") untouched so a retry sees consistent state.
        self._episode_id = episode_id
        self._seq = 0
        return resp

    def step(self, action: int) -> dict[str, Any]:
        """Send ``step`` and block for exactly one ``step_result`` (or ``error``).

        Validates action against ``protocol.VALID_DISCRETE_ACTIONS`` BEFORE
        sending so an invalid action never reaches the wire. Validates the
        response shape, run_id, episode_id, seq match, terminal_reason, and
        obs vector. Increments local seq only on success.
        """
        # Action validation runs first so step() never sends a malformed
        # action even if the caller forgot to reset(). Tests rely on this
        # ordering.
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"action must be int, got {type(action).__name__}")
        if action not in protocol.VALID_DISCRETE_ACTIONS:
            raise ValueError(
                f"invalid discrete action {action}; must be one of "
                f"{sorted(protocol.VALID_DISCRETE_ACTIONS)}"
            )
        if self._episode_id == "":
            raise GodotProtocolError("step called before reset; no active episode")

        seq = self._seq
        msg = {
            "type": protocol.MSG_STEP,
            protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "episode_id": self._episode_id,
            "seq": seq,
            "action": action,
        }
        self._send_line(msg)
        resp = self._recv_message()
        self._reject_if_error(resp)
        self._validate_response_shape(
            resp,
            expected_type=protocol.MSG_STEP_RESULT,
            required=protocol.REQUIRED_FIELDS_STEP_RESULT,
        )
        self._validate_run_id(resp)
        self._validate_episode_id(resp)
        resp_seq = resp.get("seq")
        if resp_seq != seq:
            raise GodotProtocolError(
                f"seq mismatch on step_result: expected {seq}, got {resp_seq!r}"
            )
        if resp.get("terminal_reason") not in protocol.VALID_TERMINAL_REASONS:
            raise GodotProtocolError(
                f"invalid terminal_reason: {resp.get('terminal_reason')!r}; "
                f"must be one of {sorted(protocol.VALID_TERMINAL_REASONS)}"
            )
        self._validate_obs(resp.get("obs"))
        self._seq = seq + 1
        return resp

    # --- internals ---------------------------------------------------------

    def _send_line(self, payload: dict) -> None:
        if self._sock is None:
            raise GodotTransportError("transport not connected")
        line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._sock.sendall(line)
        except OSError as e:
            raise GodotTransportError(f"send failed: {e}") from e

    def _recv_message(self) -> dict[str, Any]:
        if self._sock is None:
            raise GodotTransportError("transport not connected")
        line = self._recv_line()
        try:
            obj = json.loads(line.decode("utf-8"))
        except UnicodeDecodeError as e:
            raise GodotProtocolError(f"non-UTF-8 bytes on wire: {e}") from e
        except json.JSONDecodeError as e:
            raise GodotProtocolError(f"malformed JSON line: {e}") from e
        if not isinstance(obj, dict):
            raise GodotProtocolError(
                f"expected JSON object, got {type(obj).__name__}"
            )
        return obj

    def _recv_line(self) -> bytes:
        assert self._sock is not None
        while b"\n" not in self._buf:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout as e:
                raise GodotTransportError(
                    f"recv timed out after {self.recv_timeout_s}s"
                ) from e
            except OSError as e:
                raise GodotTransportError(f"recv failed: {e}") from e
            if not chunk:
                raise GodotTransportError("peer closed connection before newline")
            self._buf += chunk
        nl = self._buf.index(b"\n")
        line = self._buf[:nl]
        self._buf = self._buf[nl + 1:]
        return line

    def _reject_if_error(self, resp: dict) -> None:
        """If response is type=error, raise GodotRemoteError. No-op otherwise.

        A malformed error envelope (missing required fields or wrong
        protocol_version) is itself a protocol error, not a remote error.
        """
        if resp.get("type") != protocol.MSG_ERROR:
            return
        missing = protocol.REQUIRED_FIELDS_ERROR - resp.keys()
        if missing:
            raise GodotProtocolError(
                f"error response missing required fields: {sorted(missing)}"
            )
        pv = resp.get(protocol.FIELD_PROTOCOL_VERSION)
        if pv != protocol.H3_PROTOCOL_VERSION:
            raise GodotProtocolError(
                f"error response protocol_version {pv!r} != "
                f"{protocol.H3_PROTOCOL_VERSION}"
            )
        raise GodotRemoteError(
            code=str(resp["code"]),
            message=str(resp["message"]),
            payload=resp,
        )

    def _validate_response_shape(
        self,
        resp: dict,
        expected_type: str,
        required: frozenset[str],
    ) -> None:
        rtype = resp.get("type")
        if rtype != expected_type:
            raise GodotProtocolError(
                f"expected type={expected_type!r}, got type={rtype!r}"
            )
        missing = required - resp.keys()
        if missing:
            raise GodotProtocolError(
                f"{expected_type} missing required fields: {sorted(missing)}"
            )
        pv = resp.get(protocol.FIELD_PROTOCOL_VERSION)
        if pv != protocol.H3_PROTOCOL_VERSION:
            raise GodotProtocolError(
                f"{expected_type} protocol_version {pv!r} != "
                f"{protocol.H3_PROTOCOL_VERSION}"
            )

    def _validate_run_id(self, resp: dict) -> None:
        rid = resp.get("run_id")
        if rid != self.run_id:
            raise GodotProtocolError(
                f"run_id mismatch: expected {self.run_id!r}, got {rid!r}"
            )

    def _validate_episode_id(self, resp: dict) -> None:
        eid = resp.get("episode_id")
        if eid != self._episode_id:
            raise GodotProtocolError(
                f"episode_id mismatch: expected {self._episode_id!r}, got {eid!r}"
            )

    def _validate_obs(self, obs: Any) -> None:
        if not isinstance(obs, list):
            raise GodotProtocolError(
                f"obs must be JSON array, got {type(obs).__name__}"
            )
        if len(obs) != 10:
            raise GodotProtocolError(f"obs must have length 10, got {len(obs)}")
        for i, v in enumerate(obs):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise GodotProtocolError(
                    f"obs[{i}] must be numeric, got {type(v).__name__}"
                )


__all__ = [
    "GodotH3Transport",
    "GodotTransportError",
    "GodotProtocolError",
    "GodotRemoteError",
]
