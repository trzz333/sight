"""H4 Step 2 env-level protocol-routing tests.

Uses injected fake transport and process so no live Godot is required.
Covers the env's responsibility to:

- pass observation_mode and pixel dims to transport.reset() in pixel mode
- pass nothing extra (H3 byte-compat) in state mode
- dispatch _obs_to_np on observation_mode (state -> float32 (10,),
  pixel -> uint8 (C,H,W))
- raise NotImplementedError at reset for observation_mode='both'
- preserve H3 reset/step return shape unchanged in state mode

Run:
    pytest tests/rl/test_h4_godot_env_protocol.py -v --tb=short
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sight_agent import protocol
from sight_agent.rl.godot_env import GodotSignalDodgeEnv

from .h3_godot_fakes import (
    FakeProcess,
    FakeProcessFactoryRecorder,
    FakeTransport,
    FakeTransportFactoryRecorder,
    _reset_ok_payload,
    _step_ok_payload,
)


# --- fake transport extension --------------------------------------------


class _ModeRecordingFakeTransport(FakeTransport):
    """FakeTransport that records reset() kwargs so the env's routing
    can be asserted. Inherits the H3 fake's queue + state behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reset_kwargs: dict[str, Any] | None = None

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
    ) -> dict:
        self.last_reset_kwargs = {
            "seed": seed,
            "max_steps": max_steps,
            "episode_id": episode_id,
            "observation_mode": observation_mode,
            "pixel_width": pixel_width,
            "pixel_height": pixel_height,
            "pixel_channels": pixel_channels,
            "curriculum_n_init": curriculum_n_init,
        }
        # Reuse the parent's queue draining for response payloads.
        return super().reset(seed=seed, max_steps=max_steps, episode_id=episode_id,
                             curriculum_n_init=curriculum_n_init)


def _pixel_obs_payload(c: int = 1, h: int = 84, w: int = 84, fill: int = 0) -> dict:
    return {
        "mode": protocol.OBS_MODE_PIXEL,
        "shape": [c, h, w],
        "dtype": protocol.OBS_DTYPE_UINT8,
        "encoding": protocol.OBS_ENCODING_FLAT_UINT8,
        "data": [fill] * (c * h * w),
        "pixel_source": protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT,
        "capture_point": protocol.CAPTURE_POINT_FRAME_POST_DRAW,
        "headless_allowed": False,
        "viewport_width": 1280,
        "viewport_height": 720,
    }


def _pixel_reset_payload(c: int = 1, h: int = 84, w: int = 84, fill: int = 0) -> dict:
    return {
        "type": protocol.MSG_RESET_OK,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": "ignored-by-fake",
        "frame": 0,
        "obs": _pixel_obs_payload(c=c, h=h, w=w, fill=fill),
        "terminated": False,
        "truncated": False,
        "info": {},
    }


def _pixel_step_payload(
    seq: int = 0, c: int = 1, h: int = 84, w: int = 84, fill: int = 1
) -> dict:
    return {
        "type": protocol.MSG_STEP_RESULT,
        protocol.FIELD_PROTOCOL_VERSION: protocol.H3_PROTOCOL_VERSION,
        "run_id": "ignored-by-fake",
        "seq": seq,
        "frame": 1,
        "obs": _pixel_obs_payload(c=c, h=h, w=w, fill=fill),
        "reward": 1.0,
        "terminated": False,
        "truncated": False,
        "terminal_reason": protocol.TERMINAL_REASON_NONE,
        "info": {},
    }


# --- fixtures ------------------------------------------------------------


_GODOT_EXE = "/nonexistent/godot.exe"
_PROJECT = "/nonexistent/project"


def _make_env(
    *,
    observation_mode: str = "state",
    headless: bool | None = None,
    transport: FakeTransport | None = None,
    proc: FakeProcess | None = None,
    pixel_width: int = 84,
    pixel_height: int = 84,
    pixel_channels: int = 1,
    run_dir: Path | None = None,
):
    if transport is None:
        transport = _ModeRecordingFakeTransport(
            run_id="placeholder", host="127.0.0.1", port=0, recv_timeout_s=1.0
        )
    if proc is None:
        proc = FakeProcess()
    if headless is None:
        headless = observation_mode == "state"
    env = GodotSignalDodgeEnv(
        godot_executable=_GODOT_EXE,
        project_path=_PROJECT,
        tcp_port=0,
        run_dir=run_dir,
        observation_mode=observation_mode,
        headless=headless,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        pixel_channels=pixel_channels,
        transport_factory=FakeTransportFactoryRecorder(transport),
        process_factory=FakeProcessFactoryRecorder(proc),
    )
    return env, transport, proc


# --- state mode preserves H3 byte-compat behavior ------------------------


def test_state_mode_reset_passes_no_h4_kwargs():
    env, tx, _ = _make_env(observation_mode="state")
    tx.queue_reset(_reset_ok_payload())
    obs, info = env.reset(seed=0)
    assert tx.last_reset_kwargs is not None
    assert tx.last_reset_kwargs["observation_mode"] is None
    assert tx.last_reset_kwargs["pixel_width"] is None
    assert tx.last_reset_kwargs["pixel_height"] is None
    assert tx.last_reset_kwargs["pixel_channels"] is None
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    env.close()


def test_state_mode_step_returns_float32_state_obs():
    env, tx, _ = _make_env(observation_mode="state")
    tx.queue_reset(_reset_ok_payload())
    tx.queue_step(_step_ok_payload(seq=0))
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(1)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert reward == 1.0
    env.close()


# --- pixel mode end-to-end (with fake transport) -------------------------


def test_pixel_mode_reset_passes_full_kwargs():
    env, tx, _ = _make_env(observation_mode="pixel")
    tx.queue_reset(_pixel_reset_payload())
    obs, info = env.reset(seed=42)
    assert tx.last_reset_kwargs is not None
    assert tx.last_reset_kwargs["observation_mode"] == "pixel"
    assert tx.last_reset_kwargs["pixel_width"] == 84
    assert tx.last_reset_kwargs["pixel_height"] == 84
    assert tx.last_reset_kwargs["pixel_channels"] == 1
    assert tx.last_reset_kwargs["seed"] == 42
    env.close()


def test_pixel_mode_reset_returns_uint8_chw_array():
    env, tx, _ = _make_env(observation_mode="pixel")
    tx.queue_reset(_pixel_reset_payload(c=1, h=84, w=84, fill=42))
    obs, info = env.reset(seed=0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.uint8
    assert int(obs[0, 0, 0]) == 42
    assert int(obs.min()) == 42
    assert int(obs.max()) == 42
    env.close()


def test_pixel_mode_step_returns_uint8_chw_array():
    env, tx, _ = _make_env(observation_mode="pixel")
    tx.queue_reset(_pixel_reset_payload(c=1, h=84, w=84, fill=0))
    tx.queue_step(_pixel_step_payload(seq=0, c=1, h=84, w=84, fill=200))
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(2)
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.uint8
    assert int(obs.max()) == 200
    assert reward == 1.0
    env.close()


def test_pixel_mode_custom_dims_propagate_to_transport():
    env, tx, _ = _make_env(
        observation_mode="pixel",
        pixel_channels=3,
        pixel_height=64,
        pixel_width=64,
    )
    tx.queue_reset(_pixel_reset_payload(c=3, h=64, w=64, fill=1))
    obs, info = env.reset(seed=1)
    assert tx.last_reset_kwargs["pixel_channels"] == 3
    assert tx.last_reset_kwargs["pixel_height"] == 64
    assert tx.last_reset_kwargs["pixel_width"] == 64
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.uint8
    env.close()


# --- both mode is rejected at reset --------------------------------------


def test_both_mode_construct_succeeds_but_reset_raises():
    """Construction in 'both' mode must remain valid (Step 1 contract)
    so callers can still introspect intent. Reset must raise loudly so
    half-implemented paths are not silently entered."""
    env, tx, _ = _make_env(observation_mode="both", headless=False)
    # No queued response; reset should raise before touching the wire.
    with pytest.raises(NotImplementedError, match="both"):
        env.reset(seed=0)
    env.close()


# --- _obs_to_np dispatch -------------------------------------------------


def test_obs_to_np_state_mode_rejects_dict():
    env, _, _ = _make_env(observation_mode="state")
    with pytest.raises(Exception, match="state-mode obs"):
        env._obs_to_np({"mode": "pixel"})
    env.close()


def test_obs_to_np_pixel_mode_rejects_list():
    env, _, _ = _make_env(observation_mode="pixel")
    with pytest.raises(Exception, match="pixel-mode obs"):
        env._obs_to_np([0.0] * 10)
    env.close()


def test_obs_to_np_pixel_mode_returns_uint8_chw():
    env, _, _ = _make_env(observation_mode="pixel")
    payload = _pixel_obs_payload(c=1, h=84, w=84, fill=128)
    arr = env._obs_to_np(payload)
    assert arr.shape == (1, 84, 84)
    assert arr.dtype == np.uint8
    assert int(arr[0, 0, 0]) == 128
    env.close()


def test_obs_to_np_pixel_mode_rejects_data_length_mismatch():
    env, _, _ = _make_env(observation_mode="pixel")
    payload = _pixel_obs_payload(c=1, h=84, w=84, fill=0)
    payload["data"] = [0] * 100  # wrong length
    with pytest.raises(Exception, match="length"):
        env._obs_to_np(payload)
    env.close()


# --- pixel-mode obs_metadata NDJSON persistence (pre-H5 hardening) -------


def _read_python_ndjson(run_dir: Path) -> list[dict]:
    """Read and JSON-parse every line of python.ndjson under run_dir.

    Returns an empty list if the file does not exist so tests can assert
    on absence cleanly. One JSON object per line per the
    ``_NdjsonWriter`` contract in godot_env._NdjsonWriter.
    """
    path = run_dir / "python.ndjson"
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_pixel_reset_writes_python_ndjson(tmp_path):
    """Pixel-mode reset must materialize python.ndjson under run_dir so
    artifact-only audits have evidence to inspect."""
    env, tx, _ = _make_env(observation_mode="pixel", run_dir=tmp_path)
    tx.queue_reset(_pixel_reset_payload())
    env.reset(seed=0)
    env.close()
    assert (tmp_path / "python.ndjson").exists()


def test_pixel_reset_emits_exactly_one_obs_metadata_event(tmp_path):
    """Per-reset metadata persistence: exactly one ``obs_metadata`` event
    per reset call. Two resets => two events; this test pins the
    one-per-reset cadence."""
    env, tx, _ = _make_env(observation_mode="pixel", run_dir=tmp_path)
    tx.queue_reset(_pixel_reset_payload())
    env.reset(seed=0)
    env.close()
    events = _read_python_ndjson(tmp_path)
    obs_meta = [e for e in events if e.get("type") == "obs_metadata"]
    assert len(obs_meta) == 1, (
        f"expected exactly one obs_metadata event, got {len(obs_meta)}: "
        f"{[e.get('type') for e in events]}"
    )


def test_obs_metadata_event_has_listed_fields(tmp_path):
    """The obs_metadata event must carry every field the H5 audit will
    rely on. The ``_log_event`` decorator also adds run_id / godot_pid /
    tcp_port; here we assert the explicit obs-metadata fields plus
    run_id (the decorator's most load-bearing addition)."""
    env, tx, _ = _make_env(observation_mode="pixel", run_dir=tmp_path)
    tx.queue_reset(_pixel_reset_payload(c=1, h=84, w=84, fill=7))
    env.reset(seed=0)
    env.close()
    events = _read_python_ndjson(tmp_path)
    obs_meta = next(e for e in events if e.get("type") == "obs_metadata")
    # Auto-decorated identity field (sanity check that the decorator path
    # is active for this event type, not just ``reset`` events).
    assert "run_id" in obs_meta
    # Explicit metadata payload required by the H5 plan acceptance audit.
    assert obs_meta["episode_id"] == "ep-000001"
    assert obs_meta["observation_mode"] == "pixel"
    assert obs_meta["shape"] == [1, 84, 84]
    assert obs_meta["dtype"] == protocol.OBS_DTYPE_UINT8
    assert obs_meta["encoding"] == protocol.OBS_ENCODING_FLAT_UINT8
    assert obs_meta["pixel_source"] == protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT
    assert obs_meta["capture_point"] == protocol.CAPTURE_POINT_FRAME_POST_DRAW
    assert obs_meta["headless_allowed"] is False
    assert obs_meta["viewport_width"] == 1280
    assert obs_meta["viewport_height"] == 720


def test_obs_metadata_event_omits_data(tmp_path):
    """The metadata event must NOT include the pixel data array; logging
    the raw frame per reset would balloon NDJSON size and defeat the
    point of a metadata-only audit trail."""
    env, tx, _ = _make_env(observation_mode="pixel", run_dir=tmp_path)
    tx.queue_reset(_pixel_reset_payload(c=1, h=84, w=84, fill=42))
    env.reset(seed=0)
    env.close()
    events = _read_python_ndjson(tmp_path)
    obs_meta = next(e for e in events if e.get("type") == "obs_metadata")
    assert "data" not in obs_meta, (
        "obs_metadata event leaked obs.data; this defeats metadata-only "
        f"audit trail. keys: {sorted(obs_meta.keys())}"
    )


def test_state_mode_reset_emits_no_obs_metadata_event(tmp_path):
    """State mode does not produce obs metadata on the wire; the env
    must not emit an obs_metadata event for state-mode resets. Pins the
    pixel-mode gating on the new log site."""
    env, tx, _ = _make_env(observation_mode="state", run_dir=tmp_path)
    tx.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    env.close()
    events = _read_python_ndjson(tmp_path)
    obs_meta = [e for e in events if e.get("type") == "obs_metadata"]
    assert obs_meta == [], (
        f"state-mode reset emitted obs_metadata events: {obs_meta}"
    )


def test_pixel_reset_emits_two_obs_metadata_events_after_two_resets(tmp_path):
    """Cadence pin: a second reset must emit a second obs_metadata event.
    Confirms the log site is per-reset, not first-reset-only."""
    env, tx, _ = _make_env(observation_mode="pixel", run_dir=tmp_path)
    tx.queue_reset(_pixel_reset_payload())
    tx.queue_reset(_pixel_reset_payload(fill=99))
    env.reset(seed=0)
    env.reset(seed=1)
    env.close()
    events = _read_python_ndjson(tmp_path)
    obs_meta = [e for e in events if e.get("type") == "obs_metadata"]
    assert len(obs_meta) == 2, (
        f"expected two obs_metadata events across two resets, got "
        f"{len(obs_meta)}"
    )
    # Each event carries its own episode_id so downstream audits can
    # correlate metadata to a specific episode.
    assert obs_meta[0]["episode_id"] == "ep-000001"
    assert obs_meta[1]["episode_id"] == "ep-000002"


# --- start-state curriculum forwarding (env -> transport) ----------------


def test_curriculum_defaults_zero_on_reset():
    """With no curriculum set, the env forwards curriculum_n_init=0."""
    env, tx, _ = _make_env(observation_mode="state")
    tx.queue_reset(_reset_ok_payload())
    env.reset(seed=0)
    assert tx.last_reset_kwargs["curriculum_n_init"] == 0


def test_curriculum_attr_forwards_to_transport():
    """The public curriculum_n_init attribute (mutated by AnnealCurriculum via
    set_attr) is forwarded to the transport on reset."""
    env, tx, _ = _make_env(observation_mode="state")
    tx.queue_reset(_reset_ok_payload())
    env.curriculum_n_init = 4
    env.reset(seed=0)
    assert tx.last_reset_kwargs["curriculum_n_init"] == 4


def test_curriculum_options_override_wins():
    """An explicit options={curriculum_n_init: N} overrides the instance attr."""
    env, tx, _ = _make_env(observation_mode="state")
    tx.queue_reset(_reset_ok_payload())
    env.curriculum_n_init = 4
    env.reset(seed=0, options={"curriculum_n_init": 2})
    assert tx.last_reset_kwargs["curriculum_n_init"] == 2
