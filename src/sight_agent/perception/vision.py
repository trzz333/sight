"""HSV color thresholding for player (white) and hazards (red).

Rule-of-thumb thresholds chosen to match the Godot renderer output (solid Color.WHITE and
Color.RED fills on a transparent background that the gl_compatibility renderer resolves to
black-ish). Tune only if live NDJSON on StrongerJr shows systematic detection error; do not
pre-optimize.

Red spans two hue bands in OpenCV HSV (H in [0, 180)): the low band near 0 and the high band
near 180. Both are combined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import constants


# White: low saturation, high value. Very forgiving on hue.
_WHITE_LO = np.array([0, 0, 220], dtype=np.uint8)
_WHITE_HI = np.array([180, 40, 255], dtype=np.uint8)

# Red low band and high band. Saturation and value floored to ignore dark/gray pixels.
_RED_LO_1 = np.array([0, 120, 100], dtype=np.uint8)
_RED_HI_1 = np.array([10, 255, 255], dtype=np.uint8)
_RED_LO_2 = np.array([170, 120, 100], dtype=np.uint8)
_RED_HI_2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum blob pixel count to accept a connected component as a real player/hazard.
# Player is 32x32=1024 px, hazard 24x24=576 px. Keep a generous floor for renderer jitter.
_MIN_PLAYER_PX: int = 200
_MIN_HAZARD_PX: int = 120


@dataclass
class PerceptionResult:
    player: dict | None  # {"bbox": (x, y, w, h), "center": (cx, cy)} or None
    hazards: list[dict] = field(default_factory=list)  # list of {"bbox", "center"}
    debug: dict = field(default_factory=dict)  # {"player_mask_px", "hazard_mask_px", ...}

    def as_policy_state(self) -> dict:
        """Shape matches games/signal-dodge/scripts/agent.gd capture() output.

        Returned dict keys: player_x, player_y, hazards: [{"x", "y"}...]. Coordinate system
        matches the Godot viewport: +y is down, origin at top-left of the play area.
        """

        if self.player is None:
            return {"player_x": 0.0, "player_y": 0.0, "hazards": []}
        px, py = self.player["center"]
        return {
            "player_x": float(px),
            "player_y": float(py),
            "hazards": [
                {"x": float(h["center"][0]), "y": float(h["center"][1])}
                for h in self.hazards
            ],
        }


def _largest_blob(mask: np.ndarray, min_area: int) -> dict | None:
    """Return {"bbox": (x, y, w, h), "center": (cx, cy)} for the largest connected component
    in `mask` meeting `min_area`, else None.
    """

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best_idx = -1
    best_area = min_area - 1
    for i in range(1, num):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_idx = i
    if best_idx < 0:
        return None
    x, y, w, h, _ = stats[best_idx]
    cx, cy = centroids[best_idx]
    return {"bbox": (int(x), int(y), int(w), int(h)), "center": (float(cx), float(cy))}


def _all_blobs(mask: np.ndarray, min_area: int) -> list[dict]:
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[dict] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h, _ = stats[i]
        cx, cy = centroids[i]
        out.append({"bbox": (int(x), int(y), int(w), int(h)), "center": (float(cx), float(cy))})
    return out


def perceive_frame(bgr: np.ndarray) -> PerceptionResult:
    """Perceive a Signal Dodge frame. Input must be BGR uint8 with shape (H, W, 3)."""

    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected BGR (H, W, 3), got {bgr.shape}")
    if bgr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {bgr.dtype}")

    # Convert to HSV. cv2 expects BGR explicitly.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, _WHITE_LO, _WHITE_HI)
    red_mask_lo = cv2.inRange(hsv, _RED_LO_1, _RED_HI_1)
    red_mask_hi = cv2.inRange(hsv, _RED_LO_2, _RED_HI_2)
    red_mask = cv2.bitwise_or(red_mask_lo, red_mask_hi)

    player = _largest_blob(white_mask, _MIN_PLAYER_PX)
    hazards = _all_blobs(red_mask, _MIN_HAZARD_PX)

    debug = {
        "player_mask_px": int(cv2.countNonZero(white_mask)),
        "hazard_mask_px": int(cv2.countNonZero(red_mask)),
        "hazard_count": len(hazards),
        "player_found": player is not None,
    }
    return PerceptionResult(player=player, hazards=hazards, debug=debug)
