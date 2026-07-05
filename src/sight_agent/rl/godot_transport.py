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
        # Active observation mode set by the most recent successful reset.
        # ``None`` until the first reset. Determines how _validate_obs
        # parses the response's ``obs`` field. State mode is the H3
        # default (length-10 numeric list); pixel/both modes are the H4
        # structured dict per docs/sight-h4-plan.md Decision 4. Set to
        # OBS_MODE_STATE on a no-mode-arg reset so legacy callers retain
        # H3 byte-compatible behavior end-to-end.
        self._observation_mode: str | None = None
        # Pixel dims locked at reset time; used by _validate_obs to assert
        # incoming pixel-obs shape matches what the env requested.
        self._pixel_width: int = 0
        self._pixel_height: int = 0
        self._pixel_channels: int = 0

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

    @property
    def observation_mode(self) -> str | None:
        """Active observation mode set by the most recent successful reset.

        ``None`` before the first reset. After a no-mode-arg reset this is
        ``OBS_MODE_STATE`` so callers can introspect the H3-default path
        explicitly rather than treating None and "state" as identical.
        """
        return self._observation_mode

    @property
    def pixel_dims(self) -> tuple[int, int, int]:
        """(channels, height, width) locked at the last reset; (0,0,0) before."""
        return (self._pixel_channels, self._pixel_height, self._pixel_width)

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

    def reset(
        self,
        seed: int,
        max_steps: int,
        episode_id: str,
        *,
        observation_mode: str | None = None,
        pixel_width: int | None = None,
        pixel_height: int | None = None,
        pixel_channels: int | None = None,
        curriculum_n_init: int = 0,
    ) -> dict[str, Any]:
        """Send ``reset`` and block for exactly one ``reset_ok`` (or ``error``).

        Validates the response shape, run_id, episode_id, and obs vector.
        Rewinds the local seq counter to 0 only on success.

        H4 extension (docs/sight-h4-plan.md sec 7): when
        ``observation_mode`` is provided, the mode and pixel dims are sent
        on the wire as optional fields and locked on the transport for
        subsequent step-response obs validation. When all four kwargs are
        ``None`` (the default), the wire payload is byte-compatible with
        H3 and the active mode is locked to OBS_MODE_STATE so step
        responses still validate as length-10 numeric lists.
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
        if (
            isinstance(curriculum_n_init, bool)
            or not isinstance(curriculum_n_init, int)
            or curriculum_n_init < 0
        ):
            raise ValueError(
                f"curriculum_n_init must be a non-negative int, "
                f"got {curriculum_n_init!r}"
            )

        # H4 mode/dim validation. The pixel-dim args are coupled to mode:
        # they are honored when mode is set and required (with defaults
        # filled by the env layer); they are rejected when mode is None
        # so callers cannot accidentally drift into a half-configured
        # state. The env layer (godot_env.GodotSignalDodgeEnv) is the
        # only authorized caller that supplies these kwargs.
        active_mode: str = protocol.OBS_MODE_STATE
        active_w: int = 0
        active_h: int = 0
        active_c: int = 0
        wire_extras: dict[str, Any] = {}
        if observation_mode is not None:
            if observation_mode not in protocol.VALID_OBSERVATION_MODES:
                raise ValueError(
                    f"observation_mode must be one of "
                    f"{sorted(protocol.VALID_OBSERVATION_MODES)}, "
                    f"got {observation_mode!r}"
                )
            for _name, _val in (
                ("pixel_width", pixel_width),
                ("pixel_height", pixel_height),
                ("pixel_channels", pixel_channels),
            ):
                if _val is None:
                    raise ValueError(
                        f"{_name} is required when observation_mode is set; "
                        f"got None"
                    )
                if (
                    isinstance(_val, bool)
                    or not isinstance(_val, int)
                    or _val <= 0
                ):
                    raise ValueError(
                        f"{_name} must be positive int, got {_val!r}"
                    )
            active_mode = observation_mode
            active_c = int(pixel_channels)  # type: ignore[arg-type]
            active_h = int(pixel_height)  # type: ignore[arg-type]
            active_w = int(pixel_width)  # type: ignore[arg-type]
            wire_extras = {
                "observation_mode": active_mode,
                "pixel_width": active_w,
                "pixel_height": active_h,
                "pixel_channels": active_c,
            }
        else:
            # H3 byte-compatible reset payload. Reject any pixel-dim kwarg
            # supplied without a mode so the caller's intent is explicit.
            for _name, _val in (
                ("pixel_width", pixel_width),
                ("pixel_height", pixel_height),
                ("pixel_channels", pixel_channels),
            ):
                if _val is not None:
                    raise ValueError(
                        f"{_name} provided without observation_mode; either "
                        f"pass all four together or none"
                    )

        msg: dict[str, Any] = {
            "type": protocol.MSG_RESET,
            protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "episode_id": episode_id,
            "seed": seed,
            "max_steps": max_steps,
        }
        msg.update(wire_extras)
        # Curriculum injection count is an optional wire field. Omit it entirely
        # when 0 so the clean-start / eval reset payload stays byte-identical to
        # the pre-curriculum H3 wire. The Godot side treats absent and 0 the same.
        if curriculum_n_init > 0:
            msg["curriculum_n_init"] = int(curriculum_n_init)
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
        # Lock active mode and dims BEFORE validating obs so the
        # validator sees what was just locked. Reset semantics: a failed
        # response (raised below) leaves the prior locks intact, which
        # is fine because the caller is expected to abandon this episode
        # rather than reuse it.
        prior_mode = self._observation_mode
        prior_dims = (self._pixel_channels, self._pixel_height, self._pixel_width)
        self._observation_mode = active_mode
        self._pixel_channels = active_c
        self._pixel_height = active_h
        self._pixel_width = active_w
        try:
            self._validate_obs(resp.get("obs"))
        except (GodotProtocolError, GodotTransportError):
            # Roll back the lock so a follow-up reset() does not see
            # half-applied state from a rejected response.
            self._observation_mode = prior_mode
            self._pixel_channels = prior_dims[0]
            self._pixel_height = prior_dims[1]
            self._pixel_width = prior_dims[2]
            raise
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
        """Dispatch obs validation on the active observation_mode.

        State mode: H3 contract. ``obs`` is a JSON array of length 10 with
        numeric (int or float) elements. Range is enforced upstream by
        Godot's clamp; the transport's job is shape and dtype.

        Pixel mode: ``obs`` is a JSON object carrying the H4 schema in
        protocol.REQUIRED_FIELDS_PIXEL_OBS. Validates mode literal, shape
        match against locked pixel dims, dtype literal, encoding literal,
        data length and per-element range, plus the metadata fields
        (pixel_source, capture_point, headless_allowed, viewport dims).

        Both mode is not yet implemented at the wire level; if the
        transport is asked to validate an obs while in OBS_MODE_BOTH, a
        protocol error is raised so callers cannot silently mis-handle
        the response.
        """
        mode = self._observation_mode
        if mode is None or mode == protocol.OBS_MODE_STATE:
            self._validate_state_obs(obs)
            return
        if mode == protocol.OBS_MODE_PIXEL:
            self._validate_pixel_obs(obs)
            return
        if mode == protocol.OBS_MODE_BOTH:
            raise GodotProtocolError(
                "observation_mode='both' wire path not yet implemented at "
                "the transport layer; use 'state' or 'pixel'"
            )
        raise GodotProtocolError(
            f"unknown active observation_mode {mode!r}; expected one of "
            f"{sorted(protocol.VALID_OBSERVATION_MODES)}"
        )

    def _validate_state_obs(self, obs: Any) -> None:
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

    def _validate_pixel_obs(self, obs: Any) -> None:
        """Validate the H4 pixel-obs payload schema.

        Per docs/sight-h4-plan.md Decision 4. The locked dims on the
        transport (set at reset time) are the source of truth for the
        expected shape. Wire-side ``shape`` is required to agree.
        """
        if not isinstance(obs, dict):
            raise GodotProtocolError(
                f"pixel obs must be JSON object, got {type(obs).__name__}"
            )
        missing = protocol.REQUIRED_FIELDS_PIXEL_OBS - obs.keys()
        if missing:
            raise GodotProtocolError(
                f"pixel obs missing required fields: {sorted(missing)}"
            )
        wire_mode = obs.get("mode")
        if wire_mode != protocol.OBS_MODE_PIXEL:
            raise GodotProtocolError(
                f"pixel obs mode must be {protocol.OBS_MODE_PIXEL!r}, "
                f"got {wire_mode!r}"
            )
        wire_dtype = obs.get("dtype")
        if wire_dtype != protocol.OBS_DTYPE_UINT8:
            raise GodotProtocolError(
                f"pixel obs dtype must be {protocol.OBS_DTYPE_UINT8!r}, "
                f"got {wire_dtype!r}"
            )
        wire_encoding = obs.get("encoding")
        if wire_encoding != protocol.OBS_ENCODING_FLAT_UINT8:
            raise GodotProtocolError(
                f"pixel obs encoding must be "
                f"{protocol.OBS_ENCODING_FLAT_UINT8!r}, got {wire_encoding!r}"
            )
        # Shape is a JSON array; under JSON loads it is a Python list of ints.
        wire_shape = obs.get("shape")
        if not isinstance(wire_shape, list) or len(wire_shape) != 3:
            raise GodotProtocolError(
                f"pixel obs shape must be a 3-element JSON array, "
                f"got {wire_shape!r}"
            )
        for i, v in enumerate(wire_shape):
            if isinstance(v, bool) or not isinstance(v, int):
                raise GodotProtocolError(
                    f"pixel obs shape[{i}] must be int, got {type(v).__name__}"
                )
        expected_shape = [self._pixel_channels, self._pixel_height, self._pixel_width]
        if list(wire_shape) != expected_shape:
            raise GodotProtocolError(
                f"pixel obs shape {list(wire_shape)} does not match locked "
                f"dims {expected_shape}"
            )
        # Data must be a flat array of ints in [0, 255], length C*H*W.
        wire_data = obs.get("data")
        if not isinstance(wire_data, list):
            raise GodotProtocolError(
                f"pixel obs data must be JSON array, got {type(wire_data).__name__}"
            )
        expected_len = (
            self._pixel_channels * self._pixel_height * self._pixel_width
        )
        if len(wire_data) != expected_len:
            raise GodotProtocolError(
                f"pixel obs data length {len(wire_data)} does not match "
                f"expected C*H*W = {expected_len}"
            )
        # Per-element range check. Skip per-pixel iteration cost when data
        # is empty (already short-circuited by the length check above).
        for i, v in enumerate(wire_data):
            if isinstance(v, bool) or not isinstance(v, int):
                raise GodotProtocolError(
                    f"pixel obs data[{i}] must be int, got {type(v).__name__}"
                )
            if v < 0 or v > 255:
                raise GodotProtocolError(
                    f"pixel obs data[{i}] out of [0,255]: {v}"
                )
        # Metadata literals. These are how reviewers audit the capture
        # path from artifacts alone, so they are validated strictly.
        if not isinstance(obs.get("pixel_source"), str):
            raise GodotProtocolError(
                f"pixel obs pixel_source must be str, got "
                f"{type(obs.get('pixel_source')).__name__}"
            )
        if not isinstance(obs.get("capture_point"), str):
            raise GodotProtocolError(
                f"pixel obs capture_point must be str, got "
                f"{type(obs.get('capture_point')).__name__}"
            )
        if not isinstance(obs.get("headless_allowed"), bool):
            raise GodotProtocolError(
                f"pixel obs headless_allowed must be bool, got "
                f"{type(obs.get('headless_allowed')).__name__}"
            )
        for _name in ("viewport_width", "viewport_height"):
            v = obs.get(_name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise GodotProtocolError(
                    f"pixel obs {_name} must be positive int, got {v!r}"
                )
        # Metadata literal pinning. Type checks above ensure these are the
        # right primitive types; here we pin the only authorized values per
        # docs/sight-h4-plan.md Decision 2 and Decision 4 plus
        # docs/sight-h4-spike.md. Any deviation is a wire contract
        # violation, not a silent fallback. Pre-H5 hardening: audits relying
        # on python.ndjson + transport survival must be able to assume these
        # values rather than re-derive them from source-code inspection.
        if obs["pixel_source"] != protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT:
            raise GodotProtocolError(
                f"pixel obs pixel_source must be "
                f"{protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT!r}, "
                f"got {obs['pixel_source']!r}"
            )
        if obs["capture_point"] != protocol.CAPTURE_POINT_FRAME_POST_DRAW:
            raise GodotProtocolError(
                f"pixel obs capture_point must be "
                f"{protocol.CAPTURE_POINT_FRAME_POST_DRAW!r}, "
                f"got {obs['capture_point']!r}"
            )
        if obs["headless_allowed"] is not False:
            raise GodotProtocolError(
                f"pixel obs headless_allowed must be False (pixel mode "
                f"requires a windowed Godot launch per "
                f"docs/sight-h4-spike.md); got {obs['headless_allowed']!r}"
            )


__all__ = [
    "GodotH3Transport",
    "GodotTransportError",
    "GodotProtocolError",
    "GodotRemoteError",
]
