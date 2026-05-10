"""H4 protocol-constant tests.

Pure module-level tests over ``sight_agent.protocol``. No transport, no
env, no sockets. Locks the wire-level literals so accidental drift
between Python and Godot is caught at the unit-test gate.

Run:
    pytest tests/rl/test_h4_protocol.py -v --tb=short
"""

from __future__ import annotations

from sight_agent import protocol


# --- observation-mode literals -------------------------------------------


def test_obs_mode_literals_are_lowercase_words():
    assert protocol.OBS_MODE_STATE == "state"
    assert protocol.OBS_MODE_PIXEL == "pixel"
    assert protocol.OBS_MODE_BOTH == "both"


def test_valid_observation_modes_set():
    assert protocol.VALID_OBSERVATION_MODES == frozenset(
        {"state", "pixel", "both"}
    )


# --- optional reset-request fields ---------------------------------------


def test_optional_reset_observation_fields_set():
    assert protocol.OPTIONAL_FIELDS_RESET_OBSERVATION_MODE == frozenset(
        {"observation_mode", "pixel_width", "pixel_height", "pixel_channels"}
    )


def test_optional_fields_disjoint_from_required():
    """The H4 optional reset fields must not collide with any H3 required
    reset field. A collision would mean an H4 sender could clobber an H3
    field, which would corrupt the contract."""
    overlap = (
        protocol.OPTIONAL_FIELDS_RESET_OBSERVATION_MODE
        & protocol.REQUIRED_FIELDS_RESET
    )
    assert overlap == frozenset(), f"unexpected overlap: {overlap}"


# --- pixel-obs payload schema --------------------------------------------


def test_pixel_obs_required_fields_set():
    assert protocol.REQUIRED_FIELDS_PIXEL_OBS == frozenset(
        {
            "mode",
            "shape",
            "dtype",
            "encoding",
            "data",
            "pixel_source",
            "capture_point",
            "headless_allowed",
            "viewport_width",
            "viewport_height",
        }
    )


def test_obs_dtype_and_encoding_literals():
    assert protocol.OBS_DTYPE_UINT8 == "uint8"
    assert protocol.OBS_ENCODING_FLAT_UINT8 == "flat_uint8"


def test_pixel_source_and_capture_point_literals():
    """Per H4 plan Decision 4 and the spike-resolved synchronization
    barrier. These literals are how reviewers audit the capture path
    from artifact files alone, so they are stable identifiers, not
    free-form strings."""
    assert protocol.PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT == "godot_windowed_viewport"
    assert protocol.CAPTURE_POINT_FRAME_POST_DRAW == "RenderingServer.frame_post_draw"


# --- exports --------------------------------------------------------------


def test_all_h4_constants_exported():
    for name in (
        "OBS_MODE_STATE",
        "OBS_MODE_PIXEL",
        "OBS_MODE_BOTH",
        "VALID_OBSERVATION_MODES",
        "OPTIONAL_FIELDS_RESET_OBSERVATION_MODE",
        "REQUIRED_FIELDS_PIXEL_OBS",
        "OBS_DTYPE_UINT8",
        "OBS_ENCODING_FLAT_UINT8",
        "PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT",
        "CAPTURE_POINT_FRAME_POST_DRAW",
    ):
        assert name in protocol.__all__, f"{name} missing from protocol.__all__"
