"""Rule policy. Pure port of games/signal-dodge/scripts/agent.gd.

Coordinate convention (matches Godot Signal Dodge viewport):
- Origin at top-left of the play area.
- +y is down. Player lives near the bottom of the play area.
- Hazards spawn at y = -HAZARD_SIZE (just above the top edge) and fall downward.
- A hazard with y > player.y is already past the player and can be ignored.

Policy function mapping to GDScript:
    perceive(state)                 <-> Agent.perceive in agent.gd
    decide(perceived, px, width)    <-> Agent.decide in agent.gd

Action encoding mirrors GDScript exactly: int in {-1, 0, +1}.
    -1 = move left, 0 = stay, +1 = move right.
"""

from __future__ import annotations

import math
from typing import Any

from .. import constants


def perceive(state: dict[str, Any]) -> dict[str, Any]:
    """Find the nearest aligned hazard above the player.

    GDScript reference (agent.gd::perceive):
        - iterate hazards
        - skip if h.y > player_y (already past)
        - skip if abs(h.x - player_x) > ALIGN_THRESHOLD (not in column)
        - pick smallest d = player_y - h.y
        - return {threat: true, x, y, dist} or {threat: false}
    """

    px = float(state["player_x"])
    py = float(state["player_y"])
    best_dist = math.inf
    best_x = 0.0
    best_y = 0.0
    found = False
    for h in state["hazards"]:
        hy = float(h["y"])
        hx = float(h["x"])
        if hy > py:
            continue  # already past player
        if abs(hx - px) > constants.ALIGN_THRESHOLD:
            continue  # not in player column
        d = py - hy
        if d < best_dist:
            best_dist = d
            best_x = hx
            best_y = hy
            found = True
    if found:
        return {"threat": True, "x": best_x, "y": best_y, "dist": best_dist}
    return {"threat": False}


def decide(perceived: dict[str, Any], player_x: float, screen_width: float) -> int:
    """Return action in {-1, 0, +1}. Dodge away from aligned threat; stay if safe.

    GDScript reference (agent.gd::decide):
        if not perceived.threat: return 0
        if hx < player_x: return 1
        elif hx > player_x: return -1
        else: return 1 if player_x < width/2 else -1
    """

    if not perceived.get("threat", False):
        return 0
    hx = float(perceived["x"])
    if hx < player_x:
        return 1
    if hx > player_x:
        return -1
    # Exactly centered. Dodge toward the side with more room.
    if player_x < screen_width / 2.0:
        return 1
    return -1
