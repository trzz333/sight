"""Perception tests. Synthetic BGR frames only; no live capture."""

from __future__ import annotations

import numpy as np

from sight_agent import constants
from sight_agent.perception import perceive_frame


def _blank() -> np.ndarray:
    return np.zeros((constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3), dtype=np.uint8)


def _fill_rect(frame: np.ndarray, cx: int, cy: int, size: int, bgr: tuple[int, int, int]) -> None:
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
    frame[y0:y1, x0:x1] = np.array(bgr, dtype=np.uint8)


def test_perception_finds_player_and_hazards_within_one_px():
    frame = _blank()
    # Player: white 32x32 centered near the bottom.
    player_cx, player_cy = 360, 524
    _fill_rect(frame, player_cx, player_cy, constants.PLAYER_SIZE, (255, 255, 255))
    # Two red 24x24 hazards.
    h1_cx, h1_cy = 200, 80
    h2_cx, h2_cy = 500, 300
    _fill_rect(frame, h1_cx, h1_cy, constants.HAZARD_SIZE, (0, 0, 255))
    _fill_rect(frame, h2_cx, h2_cy, constants.HAZARD_SIZE, (0, 0, 255))

    result = perceive_frame(frame)

    assert result.player is not None
    px, py = result.player["center"]
    assert abs(px - player_cx) <= 1.0
    assert abs(py - player_cy) <= 1.0

    assert len(result.hazards) == 2
    # Sort hazards by x for stable comparison.
    centers = sorted(h["center"] for h in result.hazards)
    expected = sorted([(float(h1_cx), float(h1_cy)), (float(h2_cx), float(h2_cy))])
    for (cx, cy), (ex, ey) in zip(centers, expected):
        assert abs(cx - ex) <= 1.0
        assert abs(cy - ey) <= 1.0

    assert result.debug["player_found"] is True
    assert result.debug["hazard_count"] == 2


def test_perception_empty_frame_no_crash():
    frame = _blank()
    result = perceive_frame(frame)
    assert result.player is None
    assert result.hazards == []
    assert result.debug["player_mask_px"] == 0
    assert result.debug["hazard_mask_px"] == 0


def test_perception_pure_noise_frame_no_crash():
    rng = np.random.default_rng(0)
    frame = rng.integers(
        0, 256, size=(constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3), dtype=np.uint8
    )
    # Just prove it does not throw. It may detect nothing; that is fine.
    perceive_frame(frame)


def test_perception_policy_state_adapter():
    frame = _blank()
    _fill_rect(frame, 360, 524, constants.PLAYER_SIZE, (255, 255, 255))
    _fill_rect(frame, 200, 80, constants.HAZARD_SIZE, (0, 0, 255))
    result = perceive_frame(frame)
    state = result.as_policy_state()
    assert abs(state["player_x"] - 360.0) <= 1.0
    assert abs(state["player_y"] - 524.0) <= 1.0
    assert len(state["hazards"]) == 1
    h = state["hazards"][0]
    assert abs(h["x"] - 200.0) <= 1.0
    assert abs(h["y"] - 80.0) <= 1.0
