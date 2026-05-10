"""H4 protocol-extension tests for ``GodotH3Transport``.

Reuses the ``FakeGodotServer`` pattern from
``tests/rl/test_h3_godot_transport.py``. No live Godot required.

Covers:
- reset() default path is byte-compatible with H3 (no new fields on wire)
- reset() with observation_mode='state' includes the four H4 fields and
  preserves H3 obs validation
- reset() with observation_mode='pixel' includes the H4 fields and
  validates the H4 pixel-obs payload schema on response
- reset() argument validation (invalid mode, bad pixel dims, partial
  pixel-dim args)
- Transport tracks active observation_mode and pixel_dims after reset
- Pixel-obs response validation: missing fields, wrong literals, shape
  mismatch, data length mismatch, out-of-range values, wrong types
- Pixel-obs metadata validation (pixel_source, capture_point,
  headless_allowed, viewport dims)
- Both mode raises at the transport (wire path not implemented)

Run:
    pytest tests/rl/test_h4_godot_transport_protocol.py -v --tb=short
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
    GodotTransportError,
)


# --- fake server (copied from H3 transport tests; module-local) ----------


class FakeGodotServer:
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
            with self._outbound_lock:
                pending = self._outbound
                self._outbound = []
            for chunk in pending:
                try:
                    peer.sendall(chunk)
                except OSError:
                    return
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


# --- payload helpers -----------------------------------------------------


def _state_reset_ok(run_id: str, episode_id: str, frame: int = 0) -> dict:
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


def _pixel_obs(
    *,
    channels: int = 1,
    height: int = 84,
    width: int = 84,
    fill: int = 0,
    pixel_source: str | None = None,
    capture_point: str | None = None,
    headless_allowed: bool = False,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    mode: str | None = None,
    dtype: str | None = None,
    encoding: str | None = None,
) -> dict:
    """Build a wire-shape pixel obs dict. Test fake only; not synthetic
    pixel emission from Godot production code."""
    n = channels * height * width
    return {
        "mode": mode if mode is not None else protocol.OBS_MODE_PIXEL,
        "shape": [channels, height, width],
        "dtype": dtype if dtype is not None else protocol.OBS_DTYPE_UINT8,
        "encoding": encoding if encoding is not None else protocol.OBS_ENCODING_FLAT_UINT8,
        "data": [fill] * n,
        "pixel_source": pixel_source if pixel_source is not None
            else protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT,
        "capture_point": capture_point if capture_point is not None
            else protocol.CAPTURE_POINT_FRAME_POST_DRAW,
        "headless_allowed": headless_allowed,
        "viewport_width": viewport_width,
        "viewport_height": viewport_height,
    }


def _pixel_reset_ok(run_id: str, episode_id: str, **pixel_kwargs) -> dict:
    return {
        "type": protocol.MSG_RESET_OK,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "frame": 0,
        "obs": _pixel_obs(**pixel_kwargs),
        "terminated": False,
        "truncated": False,
        "info": {},
    }


# --- byte-compat: no-mode reset is H3 byte-compatible --------------------


def test_default_reset_omits_h4_fields_on_wire(server, transport):
    server.queue_response(_state_reset_ok("test-run", "ep-1"))
    transport.reset(seed=0, max_steps=100, episode_id="ep-1")
    server.wait_inbound(1)
    sent = server.inbound[0]
    assert sent is not None
    for field in protocol.OPTIONAL_FIELDS_RESET_OBSERVATION_MODE:
        assert field not in sent, (
            f"H4 field {field} leaked onto wire for default reset"
        )
    # Active mode is still locked to STATE so step responses validate
    # the H3 way (length-10 list).
    assert transport.observation_mode == protocol.OBS_MODE_STATE
    assert transport.pixel_dims == (0, 0, 0)


def test_default_reset_validates_h3_state_obs(server, transport):
    """Existing H3 length-10 numeric list must continue to validate
    when observation_mode is unset (None / state)."""
    server.queue_response(_state_reset_ok("test-run", "ep-1"))
    resp = transport.reset(seed=0, max_steps=100, episode_id="ep-1")
    assert resp["obs"] == [0.0] * 10


# --- explicit state mode -------------------------------------------------


def test_state_mode_reset_includes_h4_fields_on_wire(server, transport):
    server.queue_response(_state_reset_ok("test-run", "ep-1"))
    transport.reset(
        seed=0,
        max_steps=100,
        episode_id="ep-1",
        observation_mode=protocol.OBS_MODE_STATE,
        pixel_width=84,
        pixel_height=84,
        pixel_channels=1,
    )
    server.wait_inbound(1)
    sent = server.inbound[0]
    assert sent is not None
    assert sent["observation_mode"] == "state"
    assert sent["pixel_width"] == 84
    assert sent["pixel_height"] == 84
    assert sent["pixel_channels"] == 1
    # State-mode response is the H3 length-10 list; transport accepts it.
    assert transport.observation_mode == protocol.OBS_MODE_STATE


# --- pixel mode happy path -----------------------------------------------


def test_pixel_mode_reset_round_trip(server, transport):
    server.queue_response(
        _pixel_reset_ok(
            "test-run", "ep-1",
            channels=1, height=84, width=84, fill=0,
        )
    )
    resp = transport.reset(
        seed=0,
        max_steps=100,
        episode_id="ep-1",
        observation_mode=protocol.OBS_MODE_PIXEL,
        pixel_width=84,
        pixel_height=84,
        pixel_channels=1,
    )
    server.wait_inbound(1)
    sent = server.inbound[0]
    assert sent["observation_mode"] == "pixel"
    assert sent["pixel_width"] == 84
    assert sent["pixel_height"] == 84
    assert sent["pixel_channels"] == 1
    assert transport.observation_mode == protocol.OBS_MODE_PIXEL
    assert transport.pixel_dims == (1, 84, 84)
    assert resp["obs"]["mode"] == "pixel"
    assert resp["obs"]["shape"] == [1, 84, 84]
    assert len(resp["obs"]["data"]) == 84 * 84


def test_pixel_mode_custom_dims_round_trip(server, transport):
    server.queue_response(
        _pixel_reset_ok(
            "test-run", "ep-1",
            channels=3, height=64, width=64, fill=128,
        )
    )
    transport.reset(
        seed=1,
        max_steps=50,
        episode_id="ep-1",
        observation_mode=protocol.OBS_MODE_PIXEL,
        pixel_width=64,
        pixel_height=64,
        pixel_channels=3,
    )
    assert transport.pixel_dims == (3, 64, 64)


# --- argument validation -------------------------------------------------


def test_reset_rejects_unknown_observation_mode(transport):
    with pytest.raises(ValueError, match="observation_mode"):
        transport.reset(
            seed=0, max_steps=10, episode_id="ep",
            observation_mode="rgb",
            pixel_width=84, pixel_height=84, pixel_channels=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation_mode": "pixel"},
        {"observation_mode": "pixel", "pixel_width": 84},
        {"observation_mode": "pixel", "pixel_width": 84, "pixel_height": 84},
    ],
)
def test_reset_rejects_partial_pixel_dims_when_mode_set(transport, kwargs):
    with pytest.raises(ValueError, match="required when observation_mode"):
        transport.reset(seed=0, max_steps=10, episode_id="ep", **kwargs)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("pixel_width", 0),
        ("pixel_width", -1),
        ("pixel_height", 0),
        ("pixel_channels", 0),
        ("pixel_channels", -3),
        ("pixel_width", True),  # bool should be rejected
    ],
)
def test_reset_rejects_non_positive_pixel_dim(transport, field, bad):
    args = {
        "observation_mode": "pixel",
        "pixel_width": 84,
        "pixel_height": 84,
        "pixel_channels": 1,
    }
    args[field] = bad
    with pytest.raises(ValueError, match=field):
        transport.reset(seed=0, max_steps=10, episode_id="ep", **args)


@pytest.mark.parametrize(
    "field",
    ["pixel_width", "pixel_height", "pixel_channels"],
)
def test_reset_rejects_pixel_dim_without_mode(transport, field):
    args = {field: 84}
    with pytest.raises(ValueError, match="without observation_mode"):
        transport.reset(seed=0, max_steps=10, episode_id="ep", **args)


# --- pixel-obs response validation ---------------------------------------


def _bad_pixel_reset_test(server, transport, mutator):
    """Helper: queue a tampered pixel reset_ok and assert protocol error.

    Resets the transport's locks via a successful pixel reset is not
    needed because reset() locks dims at the START of the call from its
    own arguments and validates the response with those locked dims.
    """
    payload = _pixel_reset_ok(
        "test-run", "ep-1",
        channels=1, height=84, width=84, fill=0,
    )
    mutator(payload)
    server.queue_response(payload)
    with pytest.raises(GodotProtocolError):
        transport.reset(
            seed=0, max_steps=10, episode_id="ep-1",
            observation_mode=protocol.OBS_MODE_PIXEL,
            pixel_width=84, pixel_height=84, pixel_channels=1,
        )


def test_pixel_obs_rejects_non_dict(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p.update({"obs": [0] * 10})
    )


def test_pixel_obs_rejects_missing_required_field(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].pop("pixel_source")
    )


def test_pixel_obs_rejects_wrong_mode_literal(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"mode": "state"})
    )


def test_pixel_obs_rejects_wrong_dtype_literal(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"dtype": "float32"})
    )


def test_pixel_obs_rejects_wrong_encoding_literal(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"encoding": "base64"})
    )


def test_pixel_obs_rejects_shape_mismatch(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"shape": [3, 84, 84]})
    )


def test_pixel_obs_rejects_data_length_mismatch(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"data": [0] * 100})
    )


def test_pixel_obs_rejects_out_of_range_value(server, transport):
    def mutator(p):
        p["obs"]["data"][0] = 256
    _bad_pixel_reset_test(server, transport, mutator)


def test_pixel_obs_rejects_negative_value(server, transport):
    def mutator(p):
        p["obs"]["data"][5] = -1
    _bad_pixel_reset_test(server, transport, mutator)


def test_pixel_obs_rejects_non_int_data_element(server, transport):
    def mutator(p):
        p["obs"]["data"][0] = 1.5
    _bad_pixel_reset_test(server, transport, mutator)


def test_pixel_obs_rejects_bad_headless_allowed_type(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"headless_allowed": "false"})
    )


def test_pixel_obs_rejects_bad_viewport_dim(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"viewport_width": 0})
    )


def test_pixel_obs_rejects_bad_pixel_source_type(server, transport):
    _bad_pixel_reset_test(
        server, transport, lambda p: p["obs"].update({"pixel_source": 123})
    )


# --- failed reset rolls back lock ----------------------------------------


def test_failed_pixel_reset_rolls_back_lock(server, transport):
    """A pixel reset whose response fails validation must not leave the
    transport in a half-applied state. The active mode should remain
    None (the pre-first-reset value) so a follow-up reset can recover."""
    payload = _pixel_reset_ok(
        "test-run", "ep-1",
        channels=1, height=84, width=84, fill=0,
    )
    payload["obs"].update({"shape": [99, 99, 99]})  # mismatch
    server.queue_response(payload)
    with pytest.raises(GodotProtocolError):
        transport.reset(
            seed=0, max_steps=10, episode_id="ep-1",
            observation_mode=protocol.OBS_MODE_PIXEL,
            pixel_width=84, pixel_height=84, pixel_channels=1,
        )
    assert transport.observation_mode is None
    assert transport.pixel_dims == (0, 0, 0)


# --- both mode (deferred) ------------------------------------------------


def test_both_mode_at_transport_raises_on_obs_validation(server, transport):
    """Both mode is accepted by the constructor (Step 1) and locked at
    reset time but the wire path is not yet implemented in this slice.
    The transport raises a protocol error if asked to validate an obs
    while the active mode is OBS_MODE_BOTH."""
    # Manually lock the mode to BOTH and try to validate. Cannot happen
    # via reset() through a real Godot because Godot rejects both at
    # bad_request, but the test exercises the transport's own dispatch.
    transport._observation_mode = protocol.OBS_MODE_BOTH
    transport._pixel_channels = 1
    transport._pixel_height = 84
    transport._pixel_width = 84
    with pytest.raises(GodotProtocolError, match="not yet implemented"):
        transport._validate_obs([0.0] * 10)
