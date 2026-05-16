"""H5 amendment: bounded reward shaping computed Python-side.

Pure: no SB3, gymnasium, transport, or torch imports. The function in this
module is called once per ``GodotSignalDodgeEnv.step`` when the env's
``reward_shaping`` config is set to a recognized non-``none`` variant.

The H5 reward-amendment proposal (``docs/h5-reward-amendment-proposal.md``)
specifies exactly one bounded shaping variant: threat-weighted clearance
reward. The formula and constraints mirror section 3 of that proposal:

- The bonus is bounded in ``[0.0, alpha]``.
- The bonus is ``0.0`` when no hazards are above the player.
- Hazards strictly below the player are excluded (already passed).
- The lateral distance is absolute; no preferred lateral direction.
- The vertical weight
  ``vertical_weight_i = clamp(1.0 - vd_i / lookahead, 0, 1)``
  makes imminent threats dominate the bonus.
- Wire ``base_reward`` is NOT touched here; the caller is responsible for
  passing the Godot-supplied base reward through unchanged and only adding
  the returned ``clearance_bonus`` when the step is non-terminal.

Return value is a 3-tuple so callers can log ``threat_weight_sum`` and
``active_hazard_count_above_player`` as audit fields without re-deriving.
"""

from __future__ import annotations

from typing import Any, Mapping


__all__ = [
    "REWARD_SHAPING_NONE",
    "REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE",
    "VALID_REWARD_SHAPINGS",
    "DEFAULT_ALPHA",
    "DEFAULT_LOOKAHEAD_BAND",
    "DEFAULT_SAFE_LATERAL_DISTANCE",
    "compute_threat_weighted_clearance",
]


# String literals consumed by ``GodotSignalDodgeEnv`` and the env-config
# resolver. The default ``none`` value preserves the H4 / Phase A-E reward
# byte-identically; the only non-default value approved by the H5
# amendment is ``threat_weighted_clearance``.
REWARD_SHAPING_NONE: str = "none"
REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE: str = "threat_weighted_clearance"

VALID_REWARD_SHAPINGS: frozenset[str] = frozenset(
    {REWARD_SHAPING_NONE, REWARD_SHAPING_THREAT_WEIGHTED_CLEARANCE}
)

# Initial constants per ``docs/h5-reward-amendment-proposal.md`` section 4.
# Derived from Signal Dodge geometry, not tuned. Callers may override per
# config; the env constructor surfaces these as keyword arguments.
DEFAULT_ALPHA: float = 0.05
DEFAULT_LOOKAHEAD_BAND: float = 270.0
DEFAULT_SAFE_LATERAL_DISTANCE: float = 180.0


def compute_threat_weighted_clearance(
    reward_state: Mapping[str, Any] | None,
    *,
    alpha: float = DEFAULT_ALPHA,
    lookahead_band: float = DEFAULT_LOOKAHEAD_BAND,
    safe_lateral_distance: float = DEFAULT_SAFE_LATERAL_DISTANCE,
) -> tuple[float, float, int]:
    """Compute the threat-weighted clearance bonus for one step.

    Args:
        reward_state: dict produced Godot-side and forwarded as
            ``resp["info"]["reward_state"]``. Required schema:
                {
                    "player_x": float,
                    "player_y": float,
                    "hazards_above": [
                        {"id": int, "x": float, "y": float}, ...
                    ],
                }
            ``None``, missing keys, an empty list, or hazards strictly
            below the player all collapse the bonus to ``0.0``. The
            function does not raise on schema-soft failures; the caller's
            base reward is still returned unchanged by the caller.
        alpha: shaping coefficient, the upper bound of the bonus. Must
            be non-negative. ``0.0`` is permitted (disables the bonus
            in magnitude but still exercises the code path).
        lookahead_band: vertical range (pixels) over which an above-
            player hazard contributes. Must be > 0.
        safe_lateral_distance: lateral distance (pixels) at which a
            hazard's contribution to clearance saturates at 1.0. Must
            be > 0.

    Returns:
        ``(clearance_bonus, threat_weight_sum, active_hazard_count)``
        where ``clearance_bonus`` is in ``[0.0, alpha]``,
        ``threat_weight_sum`` is the sum of vertical weights across
        all active hazards above the player (in ``[0.0, n]``), and
        ``active_hazard_count`` is the number of hazards above the
        player that contributed (after filtering invalid entries).

    Raises:
        ValueError: if ``lookahead_band`` or ``safe_lateral_distance``
            is not > 0, or if ``alpha`` is negative. These are
            configuration errors that should fail loud rather than
            silently produce a nonsense bonus.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0.0, got {alpha!r}")
    if lookahead_band <= 0.0:
        raise ValueError(
            f"lookahead_band must be > 0.0, got {lookahead_band!r}"
        )
    if safe_lateral_distance <= 0.0:
        raise ValueError(
            f"safe_lateral_distance must be > 0.0, got "
            f"{safe_lateral_distance!r}"
        )

    if not isinstance(reward_state, Mapping):
        return 0.0, 0.0, 0
    hazards = reward_state.get("hazards_above")
    if not isinstance(hazards, list) or not hazards:
        return 0.0, 0.0, 0

    try:
        player_x = float(reward_state["player_x"])
        player_y = float(reward_state["player_y"])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0, 0

    total_weighted_clearance = 0.0
    threat_weight_sum = 0.0
    active_count = 0
    for h in hazards:
        if not isinstance(h, Mapping):
            continue
        try:
            hx = float(h["x"])
            hy = float(h["y"])
        except (KeyError, TypeError, ValueError):
            continue
        # Defensive filter: hazards strictly below the player do not
        # contribute. The Godot side already filters, but the Python
        # side must not assume that contract in case the wire payload
        # is later reused for a different game.
        if hy > player_y:
            continue
        active_count += 1
        vertical_distance = player_y - hy
        lateral_distance = abs(hx - player_x)
        vertical_weight = _clamp01(
            1.0 - vertical_distance / lookahead_band
        )
        lateral_clearance = _clamp01(
            lateral_distance / safe_lateral_distance
        )
        threat_weight_sum += vertical_weight
        total_weighted_clearance += vertical_weight * lateral_clearance

    if threat_weight_sum <= 0.0:
        return 0.0, threat_weight_sum, active_count

    bonus = float(alpha) * (total_weighted_clearance / threat_weight_sum)
    # Defensive clamp. Mathematically bonus already lies in [0, alpha]
    # because each lateral_clearance is in [0, 1] and the weighted
    # average is in [0, 1]. Floating-point error can push it slightly
    # past alpha; clamp so the caller's per-step reward is provably
    # bounded.
    if bonus < 0.0:
        bonus = 0.0
    elif bonus > float(alpha):
        bonus = float(alpha)
    return bonus, threat_weight_sum, active_count


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
