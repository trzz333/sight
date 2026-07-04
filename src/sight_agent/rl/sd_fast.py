"""Fast pure-Python replica of Signal Dodge dynamics (state observation).

Purpose: Godot Signal Dodge over TCP runs ~60 steps/s single-env (measured
2026-07-04, tools/sd_throughput_probe.py). MinAtar's from-scratch PPO clear
used 5M steps at ~6000 steps/s. Phase M's from-scratch PPO failure used only
1M steps because Godot throughput made more prohibitive. This env removes the
throughput wall so the budget dimension (the one axis on which MinAtar's clear
and Phase M's failure differ and that was never tested) can be probed, and so
every future Signal Dodge lever iterates in minutes not hours.

This is the SAME game, not a new target environment. Every constant is derived
verbatim from games/signal-dodge/scripts/{main,player,hazard}.gd and the
scene collision shapes. Fidelity is validated against the measured Godot
constant-action anchor (constant_left mean episode length 845.7, K5.2) by
tools/sd_fast_validate.py. Policies train here; the eval of record still runs
on the real Godot env against the 930.27 bar.

Dynamics (per-step order mirrors main.gd _h3_perform_step):
  1. frame += 1
  2. hazards fall (SPEED_H/60 px), cull below cull line
  3. player moves (dir * SPEED_P/60 px), clamp to walls; record last_move
  4. collision: AABB overlap |dx| < 28 and |dy| < 28 (player half 16 +
     hazard half 12); terminated on any overlap
  5. spawn every 30 frames if not terminated
  6. truncate at max_steps
  7. reward "none": 1.0 per non-terminal step, 0.0 on collision terminal

Observation (10-dim, identical to main.gd _h3_build_observation) so a policy
trained here consumes byte-identical inputs on Godot.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# --- constants, derived from the .gd sources ---------------------------------
SCREEN_W = 720.0
SCREEN_H = 540.0
PLAYER_SPEED = 300.0          # player.gd SPEED
PLAYER_SIZE = 32.0            # player.gd SIZE (half = 16)
HAZARD_SPEED = 200.0          # hazard.gd SPEED
HAZARD_SIZE = 24.0           # hazard.gd SIZE (half = 12)
PHYSICS_HZ = 60.0
SPAWN_INTERVAL = 30           # main.gd SPAWN_INTERVAL_FRAMES
PLAYER_Y = SCREEN_H - PLAYER_SIZE          # player.gd _ready: 508.0
PLAYER_X_MIN = PLAYER_SIZE / 2.0           # clamp low: 16.0
PLAYER_X_MAX = SCREEN_W - PLAYER_SIZE / 2.0  # clamp high: 704.0
PLAYER_START_X = SCREEN_W / 2.0            # 360.0
HAZARD_SPAWN_MIN = HAZARD_SIZE / 2.0       # 12.0
HAZARD_SPAWN_MAX = SCREEN_W - HAZARD_SIZE / 2.0  # 708.0
HAZARD_CULL_Y = SCREEN_H + HAZARD_SIZE     # 564.0
HAZARD_SPAWN_Y = -HAZARD_SIZE              # -24.0
# AABB overlap threshold on each axis = player_half + hazard_half = 16 + 12.
COLLIDE_THRESH = PLAYER_SIZE / 2.0 + HAZARD_SIZE / 2.0  # 28.0
PLAYER_DX = PLAYER_SPEED / PHYSICS_HZ      # 5.0 px/step
HAZARD_DY = HAZARD_SPEED / PHYSICS_HZ      # 3.333.. px/step
DEFAULT_MAX_STEPS = 1800

# --- potential-based reward shaping (Ng et al. 1999), reward_mode="shaped" ---
# Phi(s) = imminence * horizontal_clearance of the nearest hazard above player.
# imminence ramps 0->1 as the nearest hazard falls within SHAPE_IMM_RANGE px
# vertically; clearance saturates at SHAPE_SAFE_DX px horizontally. Widening the
# gap to an imminent threat raises Phi (positive shaping); drifting into its
# column lowers Phi (negative shaping). PBS => optimal policy is unchanged vs
# reward "none"; only the gradient toward active dodging is sharpened.
SHAPE_IMM_RANGE = 300.0       # vertical lookahead (px); hazard beyond this = not yet a threat
SHAPE_SAFE_DX = 80.0          # horizontal clearance (px) treated as fully safe (>collide 28)

# action wire -> dir, matches main.gd _h3_map_action: 0 left(-1), 1 stay(0), 2 right(+1)
_ACTION_DIR = (-1.0, 0.0, 1.0)


class SignalDodgeFast(gym.Env):
    """Gymnasium Signal Dodge replica. Discrete(3), Box(-1,1,(10,))."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        reward_mode: str = "none",
        shape_coef: float = 0.5,
        shape_gamma: float = 0.999,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        if reward_mode not in ("none", "shaped"):
            raise ValueError(f"reward_mode must be 'none' or 'shaped', got {reward_mode!r}")
        self._reward_mode = reward_mode
        self._shape_coef = float(shape_coef)
        self._shape_gamma = float(shape_gamma)
        self._prev_phi = 0.0
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)
        self._px = PLAYER_START_X
        self._last_move = 0.0
        self._frame = 0
        # hazards as parallel arrays (x, y, spawn_id)
        self._hx: list[float] = []
        self._hy: list[float] = []
        self._hid: list[int] = []
        self._id_counter = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._px = PLAYER_START_X
        self._last_move = 0.0
        self._frame = 0
        self._hx = []
        self._hy = []
        self._hid = []
        self._id_counter = 0
        self._prev_phi = self._potential()
        return self._obs(), {}

    def step(self, action: int):
        dir_ = _ACTION_DIR[int(action)]
        # 1. frame
        self._frame += 1
        # 2. hazards fall + cull
        nhx: list[float] = []
        nhy: list[float] = []
        nhid: list[int] = []
        for x, y, i in zip(self._hx, self._hy, self._hid):
            y2 = y + HAZARD_DY
            if y2 > HAZARD_CULL_Y:
                continue
            nhx.append(x)
            nhy.append(y2)
            nhid.append(i)
        self._hx, self._hy, self._hid = nhx, nhy, nhid
        # 3. player move + clamp
        self._px += dir_ * PLAYER_DX
        if self._px < PLAYER_X_MIN:
            self._px = PLAYER_X_MIN
        elif self._px > PLAYER_X_MAX:
            self._px = PLAYER_X_MAX
        self._last_move = dir_
        # 4. collision (AABB overlap, strict)
        terminated = False
        for x, y in zip(self._hx, self._hy):
            if abs(self._px - x) < COLLIDE_THRESH and abs(PLAYER_Y - y) < COLLIDE_THRESH:
                terminated = True
                break
        # 5. spawn
        if not terminated and self._frame % SPAWN_INTERVAL == 0:
            self._id_counter += 1
            sx = float(self.np_random.uniform(HAZARD_SPAWN_MIN, HAZARD_SPAWN_MAX))
            self._hx.append(sx)
            self._hy.append(HAZARD_SPAWN_Y)
            self._hid.append(self._id_counter)
        # 6. truncate
        truncated = (not terminated) and self._frame >= self.max_steps
        # 7. reward
        if self._reward_mode == "shaped":
            base = 0.0 if terminated else 1.0
            phi_next = 0.0 if terminated else self._potential()
            reward = base + self._shape_coef * (self._shape_gamma * phi_next - self._prev_phi)
            self._prev_phi = phi_next
        else:
            reward = 0.0 if terminated else 1.0
        return self._obs(), reward, terminated, truncated, {"frame": self._frame}

    def _potential(self) -> float:
        """Phi(s): imminence-weighted horizontal clearance to nearest hazard above
        the player, in [0,1]. 0 when no hazard is within vertical lookahead."""
        if self._reward_mode != "shaped":
            return 0.0
        ranked = self._rank_hazards()
        if not ranked:
            return 0.0
        hx, hy = ranked[0]
        dy = PLAYER_Y - hy            # >= 0 (ranked hazards are at/above player)
        imm = 1.0 - dy / SHAPE_IMM_RANGE
        if imm <= 0.0:
            return 0.0
        if imm > 1.0:
            imm = 1.0
        safe = abs(hx - self._px) / SHAPE_SAFE_DX
        if safe > 1.0:
            safe = 1.0
        return imm * safe

    def _obs(self) -> np.ndarray:
        o = np.zeros(10, dtype=np.float32)
        px, py = self._px, PLAYER_Y
        o[0] = _clamp((px / SCREEN_W) * 2.0 - 1.0)
        o[1] = _clamp(self._last_move)
        ranked = self._rank_hazards()
        if len(ranked) >= 1:
            x, y = ranked[0]
            o[2] = _clamp((x - px) / SCREEN_W)
            o[3] = _clamp((y - py) / SCREEN_H)
            o[4] = 1.0
        if len(ranked) >= 2:
            x, y = ranked[1]
            o[5] = _clamp((x - px) / SCREEN_W)
            o[6] = _clamp((y - py) / SCREEN_H)
            o[7] = 1.0
        if len(ranked) >= 3:
            x, y = ranked[2]
            o[8] = _clamp((x - px) / SCREEN_W)
            o[9] = _clamp((y - py) / SCREEN_H)
        return o

    def _rank_hazards(self) -> list[tuple[float, float]]:
        # main.gd _h3_sort_hazards_by_threat: hazards at/above player only,
        # sort by (py - hy) asc [closest above], then |hx - px| asc, then id asc.
        px, py = self._px, PLAYER_Y
        cand = [
            (self._hx[k], self._hy[k], self._hid[k])
            for k in range(len(self._hx))
            if self._hy[k] <= py
        ]
        cand.sort(key=lambda h: (py - h[1], abs(h[0] - px), h[2]))
        return [(h[0], h[1]) for h in cand]


def _clamp(v: float) -> float:
    if v < -1.0:
        return -1.0
    if v > 1.0:
        return 1.0
    return v
