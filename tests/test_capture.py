"""Capture tests. Fake provider is the normal path; MSS smoke is marked optional."""

from __future__ import annotations

import numpy as np
import pytest

from sight_agent import constants
from sight_agent.capture import (
    FakeCaptureProvider,
    MSSCaptureProvider,
    Region,
    save_calibration_frame,
)


def _solid_bgr(h: int, w: int, bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = np.array(bgr, dtype=np.uint8)
    return frame


def test_fake_provider_returns_expected_shape():
    frame = _solid_bgr(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, (10, 20, 30))
    provider = FakeCaptureProvider(frame)
    grabbed = provider.grab()
    assert grabbed.shape == (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3)
    assert grabbed.dtype == np.uint8
    # BGR pixel round-trip (OpenCV ordering).
    assert tuple(int(v) for v in grabbed[0, 0]) == (10, 20, 30)


def test_fake_provider_returns_copy_not_reference():
    frame = _solid_bgr(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, (0, 0, 0))
    provider = FakeCaptureProvider(frame)
    g1 = provider.grab()
    g1[0, 0] = (255, 255, 255)
    g2 = provider.grab()
    assert tuple(int(v) for v in g2[0, 0]) == (0, 0, 0)


def test_fake_provider_rejects_wrong_shape():
    with pytest.raises(ValueError):
        FakeCaptureProvider(np.zeros((10, 10), dtype=np.uint8))


def test_fake_provider_rejects_wrong_dtype():
    with pytest.raises(ValueError):
        FakeCaptureProvider(np.zeros((10, 10, 3), dtype=np.float32))


def test_region_mss_dict():
    r = Region(left=100, top=200, width=720, height=540)
    d = r.as_mss_dict()
    assert d == {"left": 100, "top": 200, "width": 720, "height": 540}


def test_save_calibration_frame_writes_png(tmp_path):
    frame = _solid_bgr(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, (5, 5, 5))
    provider = FakeCaptureProvider(frame)
    out = tmp_path / "calib" / "frame.png"
    result = save_calibration_frame(provider, out)
    assert result.exists()
    assert result.stat().st_size > 0


@pytest.mark.live_mss
def test_mss_provider_live_smoke():
    """Optional smoke test. Skipped by default; run with `pytest -m live_mss` on StrongerJr
    with the Signal Dodge window open and positioned at (0, 0)."""

    provider = MSSCaptureProvider(Region())
    try:
        frame = provider.grab()
    finally:
        provider.close()
    assert frame.shape[0] == constants.SCREEN_HEIGHT
    assert frame.shape[1] == constants.SCREEN_WIDTH
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
