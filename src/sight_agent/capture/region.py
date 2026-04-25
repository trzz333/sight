"""Capture providers. Return BGR uint8 ndarrays at the configured region dimensions.

MSS delivers BGRA frames. This module normalizes to BGR so perception can rely on a
stable channel order regardless of provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import constants


@dataclass(frozen=True)
class Region:
    """Screen region in absolute pixels. Dimensions default to Signal Dodge viewport."""

    left: int = 0
    top: int = 0
    width: int = constants.SCREEN_WIDTH
    height: int = constants.SCREEN_HEIGHT

    def as_mss_dict(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class CaptureProvider(Protocol):
    """Minimal interface. grab() returns a BGR uint8 ndarray shaped (height, width, 3)."""

    def grab(self) -> np.ndarray: ...


class FakeCaptureProvider:
    """Returns a preset frame. Test fixture; also useful for pinning deterministic eval runs."""

    def __init__(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            raise ValueError(f"fake frame dtype must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"fake frame shape must be (H, W, 3), got {frame.shape}")
        self._frame = frame

    def grab(self) -> np.ndarray:
        # Return a copy so consumers cannot mutate the fixture in place.
        return self._frame.copy()


class MSSCaptureProvider:
    """Live MSS-backed region capture. Not exercised in the default test suite.

    Lazily imports mss so the module loads on systems without a display. Unit tests that do not
    need live capture must not instantiate this class.
    """

    def __init__(self, region: Region) -> None:
        self.region = region
        self._sct = None  # lazy

    def _ensure_open(self) -> None:
        if self._sct is None:
            import mss  # local import so pytest does not require a display
            # Prefer the class form; `mss.mss()` is deprecated in 10.x.
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()

    def grab(self) -> np.ndarray:
        self._ensure_open()
        assert self._sct is not None
        raw = np.array(self._sct.grab(self.region.as_mss_dict()))  # BGRA uint8
        if raw.ndim != 3 or raw.shape[2] != 4:
            raise RuntimeError(f"MSS returned unexpected shape {raw.shape}")
        # Drop alpha, keep BGR. mss uses BGRA layout on Windows/macOS/Linux.
        return raw[:, :, :3].copy()

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def save_calibration_frame(provider: CaptureProvider, path: str | Path) -> Path:
    """Debug helper. Grab one frame and write it as PNG so Jeff can verify region alignment.

    Uses OpenCV for the write to avoid a pillow dependency. Returns the final path.
    """

    import cv2  # local import to keep module import cheap

    frame = provider.grab()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    return out
