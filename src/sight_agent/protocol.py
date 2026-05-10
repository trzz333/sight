"""Wire-protocol constants for the H3 bidirectional Sight env protocol.

Scope: defines message-type literals, field names, protocol version, terminal-
reason literals, and error codes for the Python<->Godot loopback contract used
by ``GodotSignalDodgeEnv`` (added in step 6 of docs/sight-h3-plan.md).

This module is *constants only*. No transport code, no parsing, no sockets.
Step 1 of docs/sight-h3-plan.md "Implementation sequence". Steps 2 (Godot
side) and 5 (Python transport) consume these.

Wire posture:
- newline-delimited UTF-8 JSON, one object per line
- loopback TCP only (see docs/sight-charter.md ethics armor)
- one request, one response for ``reset`` and ``step``
- ``hello`` is one-shot; the env does not block on a response

Versioning:
- ``H3_PROTOCOL_VERSION`` is bumped to 2. Version 1 (the pre-H3 unidirectional
  controller channel) used the field name ``"protocol"`` and is preserved at
  ``sight_agent.constants.PROTOCOL_VERSION`` for the legacy controller.
- The H3 protocol uses field name ``protocol_version``. The field-name change
  is intentional: a legacy hello cannot be silently mistaken for an H3 hello
  on the same Godot listener. Mixing legacy and H3 clients on one listener is
  not supported.
"""

from __future__ import annotations

from typing import Final


# Bumped from controller protocol (1) for the H3 bidirectional env contract.
H3_PROTOCOL_VERSION: Final[int] = 2

# Field name for the protocol version in every H3 message. Differs from the
# legacy controller field name ("protocol") on purpose. See module docstring.
FIELD_PROTOCOL_VERSION: Final[str] = "protocol_version"

# --- Python -> Godot message types ---------------------------------------

MSG_HELLO: Final[str] = "hello"
MSG_RESET: Final[str] = "reset"
MSG_STEP: Final[str] = "step"

# --- Godot -> Python message types ---------------------------------------

MSG_RESET_OK: Final[str] = "reset_ok"
MSG_STEP_RESULT: Final[str] = "step_result"
MSG_ERROR: Final[str] = "error"

PYTHON_TO_GODOT_MESSAGES: Final[frozenset[str]] = frozenset(
    {MSG_HELLO, MSG_RESET, MSG_STEP}
)
GODOT_TO_PYTHON_MESSAGES: Final[frozenset[str]] = frozenset(
    {MSG_RESET_OK, MSG_STEP_RESULT, MSG_ERROR}
)

# --- Required field sets per message -------------------------------------
# Required = consumers MUST fail (raise / send error) if absent.
# Extra fields are permitted for forward compatibility; consumers must not
# fail on unknown extras.

REQUIRED_FIELDS_HELLO: Final[frozenset[str]] = frozenset(
    {"type", FIELD_PROTOCOL_VERSION, "run_id"}
)
REQUIRED_FIELDS_RESET: Final[frozenset[str]] = frozenset(
    {"type", FIELD_PROTOCOL_VERSION, "run_id", "episode_id", "seed", "max_steps"}
)
# Optional H4 fields on the H3 reset request (docs/sight-h4-plan.md sec 7).
# Absent fields preserve H3 byte-compatible behavior (state mode, default
# pixel dims). Senders MAY include these; receivers MUST tolerate absence.
OPTIONAL_FIELDS_RESET_OBSERVATION_MODE: Final[frozenset[str]] = frozenset(
    {"observation_mode", "pixel_width", "pixel_height", "pixel_channels"}
)
REQUIRED_FIELDS_STEP: Final[frozenset[str]] = frozenset(
    {"type", FIELD_PROTOCOL_VERSION, "run_id", "episode_id", "seq", "action"}
)

REQUIRED_FIELDS_RESET_OK: Final[frozenset[str]] = frozenset(
    {
        "type",
        FIELD_PROTOCOL_VERSION,
        "run_id",
        "episode_id",
        "frame",
        "obs",
        "terminated",
        "truncated",
        "info",
    }
)
REQUIRED_FIELDS_STEP_RESULT: Final[frozenset[str]] = frozenset(
    {
        "type",
        FIELD_PROTOCOL_VERSION,
        "run_id",
        "episode_id",
        "seq",
        "frame",
        "obs",
        "reward",
        "terminated",
        "truncated",
        "terminal_reason",
        "info",
    }
)
REQUIRED_FIELDS_ERROR: Final[frozenset[str]] = frozenset(
    {"type", FIELD_PROTOCOL_VERSION, "code", "message"}
)

# --- Action wire encoding ------------------------------------------------
# H3 action space is gymnasium.spaces.Discrete(3). The wire carries the
# discrete integer; the Godot side re-derives move_x from the same mapping.
# String labels are kept for log readability and parity with the legacy
# controller (sight_agent.constants.ACTION_*).

ACTION_DISCRETE_LEFT: Final[int] = 0
ACTION_DISCRETE_STAY: Final[int] = 1
ACTION_DISCRETE_RIGHT: Final[int] = 2

VALID_DISCRETE_ACTIONS: Final[frozenset[int]] = frozenset(
    {ACTION_DISCRETE_LEFT, ACTION_DISCRETE_STAY, ACTION_DISCRETE_RIGHT}
)

ACTION_DISCRETE_TO_STRING: Final[dict[int, str]] = {
    ACTION_DISCRETE_LEFT: "left",
    ACTION_DISCRETE_STAY: "stay",
    ACTION_DISCRETE_RIGHT: "right",
}
ACTION_DISCRETE_TO_MOVE_X: Final[dict[int, int]] = {
    ACTION_DISCRETE_LEFT: -1,
    ACTION_DISCRETE_STAY: 0,
    ACTION_DISCRETE_RIGHT: 1,
}

# --- Terminal reason literals --------------------------------------------
# Per docs/sight-h3-plan.md section 5. ``""`` (TERMINAL_REASON_NONE) is the
# only valid value when both ``terminated=False`` and ``truncated=False``.

TERMINAL_REASON_NONE: Final[str] = ""
TERMINAL_REASON_COLLISION: Final[str] = "collision"
TERMINAL_REASON_TIMEOUT: Final[str] = "timeout"

VALID_TERMINAL_REASONS: Final[frozenset[str]] = frozenset(
    {TERMINAL_REASON_NONE, TERMINAL_REASON_COLLISION, TERMINAL_REASON_TIMEOUT}
)

# --- Error codes ---------------------------------------------------------
# Minimal set. Additional codes can be added without bumping
# H3_PROTOCOL_VERSION as long as consumers treat unknown codes as fatal-but-
# otherwise-opaque.

ERROR_CODE_PROTOCOL_VERSION_MISMATCH: Final[str] = "protocol_version_mismatch"
ERROR_CODE_RUN_ID_MISMATCH: Final[str] = "run_id_mismatch"
ERROR_CODE_EPISODE_ID_MISMATCH: Final[str] = "episode_id_mismatch"
ERROR_CODE_BAD_REQUEST: Final[str] = "bad_request"
ERROR_CODE_INTERNAL: Final[str] = "internal"

# --- H4 observation-mode literals ----------------------------------------
# Wire-level string literals. Mirror tcp_controller.gd OBS_MODE_* constants.
# State-mode wire is unchanged from H3 (length-10 numeric list). Pixel-mode
# wire is the structured dict schema in REQUIRED_FIELDS_PIXEL_OBS below.

OBS_MODE_STATE: Final[str] = "state"
OBS_MODE_PIXEL: Final[str] = "pixel"
OBS_MODE_BOTH: Final[str] = "both"

VALID_OBSERVATION_MODES: Final[frozenset[str]] = frozenset(
    {OBS_MODE_STATE, OBS_MODE_PIXEL, OBS_MODE_BOTH}
)

# --- H4 pixel-obs payload schema -----------------------------------------
# Per docs/sight-h4-plan.md Decision 4. Pixel mode reset_ok / step_result
# replace the H3 length-10 list at field "obs" with a dict carrying the
# fields below. State mode is unchanged.

OBS_DTYPE_UINT8: Final[str] = "uint8"
OBS_ENCODING_FLAT_UINT8: Final[str] = "flat_uint8"

# Source-of-pixels literal. Per H3-to-H4 closure caveats and the H4 spike
# (docs/sight-h4-spike.md), the only authorized default for H4 is option 2
# (windowed Godot viewport API). Other values indicate fallback paths and
# require explicit Jeff approval before landing.
PIXEL_SOURCE_GODOT_WINDOWED_VIEWPORT: Final[str] = "godot_windowed_viewport"

# Capture-point literal. The synchronization barrier proven by the spike.
CAPTURE_POINT_FRAME_POST_DRAW: Final[str] = "RenderingServer.frame_post_draw"

# Required keys inside an "obs" dict for pixel mode. The Python transport
# validates these on every receive when the active observation_mode is
# "pixel" (and "both" once that mode lands).
REQUIRED_FIELDS_PIXEL_OBS: Final[frozenset[str]] = frozenset(
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


__all__ = [
    "H3_PROTOCOL_VERSION",
    "FIELD_PROTOCOL_VERSION",
    "MSG_HELLO",
    "MSG_RESET",
    "MSG_STEP",
    "MSG_RESET_OK",
    "MSG_STEP_RESULT",
    "MSG_ERROR",
    "PYTHON_TO_GODOT_MESSAGES",
    "GODOT_TO_PYTHON_MESSAGES",
    "REQUIRED_FIELDS_HELLO",
    "REQUIRED_FIELDS_RESET",
    "REQUIRED_FIELDS_STEP",
    "REQUIRED_FIELDS_RESET_OK",
    "REQUIRED_FIELDS_STEP_RESULT",
    "REQUIRED_FIELDS_ERROR",
    "ACTION_DISCRETE_LEFT",
    "ACTION_DISCRETE_STAY",
    "ACTION_DISCRETE_RIGHT",
    "VALID_DISCRETE_ACTIONS",
    "ACTION_DISCRETE_TO_STRING",
    "ACTION_DISCRETE_TO_MOVE_X",
    "TERMINAL_REASON_NONE",
    "TERMINAL_REASON_COLLISION",
    "TERMINAL_REASON_TIMEOUT",
    "VALID_TERMINAL_REASONS",
    "ERROR_CODE_PROTOCOL_VERSION_MISMATCH",
    "ERROR_CODE_RUN_ID_MISMATCH",
    "ERROR_CODE_EPISODE_ID_MISMATCH",
    "ERROR_CODE_BAD_REQUEST",
    "ERROR_CODE_INTERNAL",
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
]
