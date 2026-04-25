"""Shared constants for Sight. Mirrors games/signal-dodge constants and the TCP wire contract.

Any parameter change for Signal Dodge must happen in both this module and the matching GDScript.
Do not tune before first live NDJSON lands on StrongerJr. See docs/sight-handoff.md.
"""

from __future__ import annotations

# TCP wire contract (loopback only, see docs/sight-charter.md ethics armor).
TCP_HOST: str = "127.0.0.1"
TCP_PORT: int = 8765
PROTOCOL_VERSION: int = 1
AGENT_NAME_RULE_PARITY: str = "python-rule-parity"

# Signal Dodge parameters. Must match games/signal-dodge/scripts/*.gd.
SCREEN_WIDTH: int = 720
SCREEN_HEIGHT: int = 540
PHYSICS_HZ: int = 60
PLAYER_SIZE: int = 32      # white square, agent
HAZARD_SIZE: int = 24      # red square, falls top to bottom
PLAYER_SPEED: float = 300.0  # px/sec horizontal
HAZARD_SPEED: float = 200.0  # px/sec downward
SPAWN_INTERVAL_FRAMES: int = 30
RANDOM_SEED: int = 42

# Derived. ALIGN_THRESHOLD mirrors agent.gd exactly (player_half + hazard_half).
PLAYER_HALF: float = PLAYER_SIZE / 2.0
HAZARD_HALF: float = HAZARD_SIZE / 2.0
ALIGN_THRESHOLD: float = PLAYER_HALF + HAZARD_HALF  # 28.0

# Action enum. Matches GDScript int {-1, 0, +1} and the wire "action" string.
ACTION_LEFT: str = "left"
ACTION_STAY: str = "stay"
ACTION_RIGHT: str = "right"
ACTION_TO_MOVE_X: dict[str, int] = {
    ACTION_LEFT: -1,
    ACTION_STAY: 0,
    ACTION_RIGHT: 1,
}
MOVE_X_TO_ACTION: dict[int, str] = {v: k for k, v in ACTION_TO_MOVE_X.items()}
