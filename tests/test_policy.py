"""Policy tests. Golden synthetic states verify parity with games/signal-dodge/scripts/agent.gd.

GDScript reference lives in agent.gd. Comments below cite the exact GDScript branch each
assertion exercises.
"""

from __future__ import annotations

from sight_agent import constants
from sight_agent.policy import decide, perceive


# --- perceive() --------------------------------------------------------------


def test_perceive_no_hazards_returns_no_threat():
    state = {"player_x": 360.0, "player_y": 520.0, "hazards": []}
    assert perceive(state) == {"threat": False}


def test_perceive_hazard_below_player_ignored():
    # GDScript: `if h["y"] > py: continue  # already past player`
    state = {
        "player_x": 360.0,
        "player_y": 520.0,
        "hazards": [{"x": 360.0, "y": 530.0}],  # y > py, past player
    }
    assert perceive(state) == {"threat": False}


def test_perceive_hazard_out_of_column_ignored():
    # GDScript: `if abs(h["x"] - px) > ALIGN_THRESHOLD: continue`
    state = {
        "player_x": 100.0,
        "player_y": 520.0,
        "hazards": [{"x": 200.0, "y": 100.0}],  # dx=100 > 28
    }
    assert perceive(state) == {"threat": False}


def test_perceive_picks_nearest_aligned_above():
    # GDScript: smallest d = py - h.y wins among aligned, above-player hazards.
    state = {
        "player_x": 360.0,
        "player_y": 520.0,
        "hazards": [
            {"x": 360.0, "y": 100.0},  # far
            {"x": 350.0, "y": 400.0},  # closer, still aligned (dx=10 <= 28)
            {"x": 390.0, "y": 430.0},  # closer still, aligned (dx=30 > 28 -> skipped)
        ],
    }
    result = perceive(state)
    assert result["threat"] is True
    assert result["x"] == 350.0
    assert result["y"] == 400.0
    assert result["dist"] == 120.0  # 520 - 400


def test_perceive_alignment_threshold_boundary():
    # abs(hx - px) == ALIGN_THRESHOLD (28) should NOT skip (strict `>` in GDScript).
    state = {
        "player_x": 360.0,
        "player_y": 520.0,
        "hazards": [{"x": 360.0 + constants.ALIGN_THRESHOLD, "y": 200.0}],
    }
    result = perceive(state)
    assert result["threat"] is True


# --- decide() ----------------------------------------------------------------


def test_decide_no_threat_returns_stay():
    # GDScript: `if not perceived.get("threat", false): return 0`
    assert decide({"threat": False}, 360.0, float(constants.SCREEN_WIDTH)) == 0


def test_decide_hazard_to_left_moves_right():
    # GDScript: `if hx < player_x: return 1`
    action = decide(
        {"threat": True, "x": 300.0, "y": 100.0, "dist": 420.0},
        360.0,
        float(constants.SCREEN_WIDTH),
    )
    assert action == 1


def test_decide_hazard_to_right_moves_left():
    # GDScript: `elif hx > player_x: return -1`
    action = decide(
        {"threat": True, "x": 400.0, "y": 100.0, "dist": 420.0},
        360.0,
        float(constants.SCREEN_WIDTH),
    )
    assert action == -1


def test_decide_centered_left_half_dodges_right():
    # GDScript tie-break: `if player_x < screen_width / 2.0: return 1`
    action = decide(
        {"threat": True, "x": 200.0, "y": 100.0, "dist": 420.0},
        200.0,  # left of center
        float(constants.SCREEN_WIDTH),
    )
    assert action == 1


def test_decide_centered_right_half_dodges_left():
    # GDScript tie-break: `else: return -1`
    action = decide(
        {"threat": True, "x": 600.0, "y": 100.0, "dist": 420.0},
        600.0,  # right of center
        float(constants.SCREEN_WIDTH),
    )
    assert action == -1
