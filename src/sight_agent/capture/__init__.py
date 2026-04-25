"""Region screen capture.

Two providers:
- FakeCaptureProvider: returns a preset ndarray. Used by unit tests and deterministic eval.
- MSSCaptureProvider: MSS-backed region grab for live play. Marked optional; not exercised in
  the normal test suite because StrongerJr may run headless or without the Godot window open.

P2 does not solve automatic window discovery. Caller supplies a static `Region(left, top, width=720,
height=540)` that must match the Signal Dodge viewport position. See docs/sight-handoff.md.
"""

from .region import (
    CaptureProvider,
    FakeCaptureProvider,
    MSSCaptureProvider,
    Region,
    save_calibration_frame,
)

__all__ = [
    "CaptureProvider",
    "FakeCaptureProvider",
    "MSSCaptureProvider",
    "Region",
    "save_calibration_frame",
]
