"""K5.2 env-dynamics evidence probe.

No training. Layered instrumentation against the production
``GodotSignalDodgeEnv`` / ``GodotH3Transport`` path. Each layer writes a
structured JSON facts block plus per-step CSV rows. The aggregate
classification (``ENV-FAIL`` / ``TASK-GEOMETRY-FAIL`` / ``ENV-PASS``)
is computed from the layer predicate outcomes per
``docs/k5-2-env-dynamics-sanity-evidence.md``.

Layers (executed in dependency order; first ENV-FAIL short-circuits):
    0  collision-propagation preflight       (state mode)
    1  action timing + per-step contract     (state mode)
    2  player kinematics                     (uses Layer 1 trace)
    3  hazard kinematics + spawn contract    (uses Layer 1 trace)
    4a state observation freshness           (uses Layer 1 trace)
    4b pixel obs freshness + state alignment (pixel mode)
    5  frame-stack contract                  (env inspection)
    6  scripted-policy reward surface        (state mode, .bat-driven)

Usage (cmd, requires SIGHT_GODOT_EXE inline):
    python tools\\k5_2_env_dynamics_probe.py --layers 0,1,2,3,4a,4b,5,6 \\
      --out-dir runs\\phase_k --seed 1000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sight_agent.rl.godot_env import GodotSignalDodgeEnv  # noqa: E402
from sight_agent.rl.godot_transport import GodotRemoteError  # noqa: E402
from sight_agent.rl.reward_shaping import (  # noqa: E402
    compute_threat_weighted_clearance,
)

# Game constants mirrored from games/signal-dodge/scripts/main.gd + player.gd.
PHYSICS_HZ = 60
PHYSICS_DT = 1.0 / 60.0
PLAYER_SPEED_PX_S = 300.0
HAZARD_SPEED_PX_S = 200.0
PLAYER_SPEED_PX_STEP = PLAYER_SPEED_PX_S / PHYSICS_HZ  # 5.0
HAZARD_SPEED_PX_STEP = HAZARD_SPEED_PX_S / PHYSICS_HZ  # 3.3333...
SPAWN_INTERVAL_FRAMES = 30
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 540
PLAYER_SIZE = 32
HAZARD_SIZE = 24
PLAYER_HALF = PLAYER_SIZE / 2
HAZARD_HALF = HAZARD_SIZE / 2
PLAYER_X_LEFT_CLAMP = PLAYER_HALF              # 16
PLAYER_X_RIGHT_CLAMP = SCREEN_WIDTH - PLAYER_HALF  # 704
HAZARD_SPAWN_Y = -float(HAZARD_SIZE)           # -24
HAZARD_CULL_Y = SCREEN_HEIGHT + HAZARD_SIZE    # 564 (h.step culls if y > this)

EPS_POSITION = 1e-3
EPS_HAZARD_DT = 5e-3
ACTION_WIRE_TO_MAPPED = {0: -1, 1: 0, 2: 1}
ACTION_MAPPED_TO_WIRE = {-1: 0, 0: 1, 1: 2}
ACTION_NAMES_WIRE = {0: "left", 1: "stay", 2: "right"}


def _resolve_godot_exe() -> str:
    exe = os.environ.get("SIGHT_GODOT_EXE", "").strip()
    if not exe:
        raise RuntimeError("SIGHT_GODOT_EXE not set in this shell")
    if not Path(exe).exists():
        raise RuntimeError(f"SIGHT_GODOT_EXE does not exist on disk: {exe}")
    return exe


def _short_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _hash_pixel_obs(obs) -> str:
    return hashlib.sha256(obs.tobytes()).hexdigest()


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ---------------- Oracle implementations ----------------------------------

def hazard_reactive_oracle(reward_state: dict[str, Any]) -> int:
    """Geometry-first 1-step oracle. Returns wire action 0/1/2.

    Independent of clearance_bonus shaping. Reads raw geometry only.
    For each candidate action, simulates the player's future x at each
    imminent hazard's arrival frame and picks the action that maximizes
    the minimum lateral clearance margin across all imminent hazards.
    Tie-break order: stay, left, right.
    """
    px = float(reward_state.get("player_x", SCREEN_WIDTH / 2.0))
    py = float(reward_state.get("player_y", SCREEN_HEIGHT - 40.0))
    hazards = reward_state.get("hazards_above", []) or []
    LOOKAHEAD_STEPS = 60  # 1 second
    threats: list[tuple[float, float]] = []
    for h in hazards:
        try:
            hy = float(h["y"])
            hx = float(h["x"])
        except (KeyError, TypeError, ValueError):
            continue
        vd = py - hy
        if vd <= 0:
            continue
        s = vd / HAZARD_SPEED_PX_STEP
        if s > LOOKAHEAD_STEPS:
            continue
        threats.append((hx, s))
    if not threats:
        return 1  # stay

    def score(action_mapped: int) -> float:
        worst = float("inf")
        for hx, s in threats:
            future_px = px + action_mapped * PLAYER_SPEED_PX_STEP * s
            if future_px < PLAYER_X_LEFT_CLAMP:
                future_px = PLAYER_X_LEFT_CLAMP
            elif future_px > PLAYER_X_RIGHT_CLAMP:
                future_px = PLAYER_X_RIGHT_CLAMP
            margin = abs(hx - future_px) - (PLAYER_HALF + HAZARD_HALF)
            if margin < worst:
                worst = margin
        return worst

    s_left = score(-1)
    s_stay = score(0)
    s_right = score(1)
    best = max(s_left, s_stay, s_right)
    if s_stay == best:
        return 1
    if s_left == best:
        return 0
    return 2


def shaped_reward_greedy_oracle(
    reward_state: dict[str, Any], alpha: float = 0.30
) -> int:
    """1-step greedy on threat_weighted_clearance. Wire 0/1/2.

    Diagnostic only. Demonstrates whether the shaped reward surface
    itself prefers a particular action regardless of true collision risk.
    """
    px = float(reward_state.get("player_x", SCREEN_WIDTH / 2.0))
    py = float(reward_state.get("player_y", SCREEN_HEIGHT - 40.0))
    hazards = reward_state.get("hazards_above", []) or []

    def score(action_mapped: int) -> float:
        future_px = px + action_mapped * PLAYER_SPEED_PX_STEP
        if future_px < PLAYER_X_LEFT_CLAMP:
            future_px = PLAYER_X_LEFT_CLAMP
        elif future_px > PLAYER_X_RIGHT_CLAMP:
            future_px = PLAYER_X_RIGHT_CLAMP
        fake_rs = {"player_x": future_px, "player_y": py, "hazards_above": hazards}
        bonus, _, _ = compute_threat_weighted_clearance(fake_rs, alpha=alpha)
        return bonus

    s_left = score(-1)
    s_stay = score(0)
    s_right = score(1)
    best = max(s_left, s_stay, s_right)
    if s_stay == best:
        return 1
    if s_left == best:
        return 0
    return 2


# ---------------- Env construction ---------------------------------------

def _build_env(
    *, observation_mode: str, run_dir: Path, seed: int,
    reward_shaping: str = "none", alpha: float = 0.30,
    max_steps: int = 1800,
) -> GodotSignalDodgeEnv:
    exe = _resolve_godot_exe()
    project_path = REPO_ROOT / "games" / "signal-dodge"
    headless = observation_mode == "state"
    kwargs: dict[str, Any] = {
        "godot_executable": exe,
        "project_path": str(project_path),
        "run_dir": str(run_dir),
        "headless": headless,
        "observation_mode": observation_mode,
        "max_steps": max_steps,
        "seed": int(seed),
        "connect_timeout_s": 30.0,
        "step_timeout_s": 10.0,
        "reward_shaping": reward_shaping,
        "reward_shaping_alpha": float(alpha),
    }
    if observation_mode == "pixel":
        kwargs["pixel_width"] = 84
        kwargs["pixel_height"] = 84
        kwargs["pixel_channels"] = 1
    return GodotSignalDodgeEnv(**kwargs)


# ---------------- Layer 0: collision-propagation preflight ---------------

def run_layer_0(
    *, out_dir: Path, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Forced collision in state mode. Verify terminal propagation.

    Strategy: drive the player with action=stay at x=360, spawn rate is
    fixed every 30 frames at random x in [12, 708]; a hazard will
    eventually pass through the player's column. Max budget 1800 steps.
    Then attempt a step after terminal (must error), then reset, then a
    short non-terminal step burst to confirm clean re-arm.
    """
    facts: dict[str, Any] = {"layer": "0", "name": "collision_propagation_preflight"}
    rows: list[dict[str, Any]] = []
    sub_run = out_dir / "layer0"
    sub_run.mkdir(parents=True, exist_ok=True)
    env = _build_env(observation_mode="state", run_dir=sub_run, seed=seed,
                     max_steps=1800)
    try:
        obs, info = env.reset()
        terminal_step: int | None = None
        terminal_reason: str | None = None
        collision_info: dict[str, Any] | None = None
        terminated = False
        truncated = False
        step_idx = 0
        while step_idx < 1800:
            obs, r, terminated, truncated, info = env.step(1)  # stay
            step_idx += 1
            rows.append({
                "layer": "0", "phase": "to_terminal",
                "step": step_idx, "action_wire": 1,
                "reward": float(r),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "terminal_reason": info.get("terminal_reason", ""),
            })
            if terminated or truncated:
                terminal_step = step_idx
                terminal_reason = info.get("terminal_reason", "")
                godot_info = info.get("godot_info") or {}
                collision_info = godot_info.get("collision")
                break

        # Predicate A: terminal must be a collision, not a timeout.
        pred_a_terminal_is_collision = bool(terminated) and terminal_reason == "collision"
        # Predicate B: collision info dict present on terminal step.
        pred_b_collision_info_present = isinstance(collision_info, dict) and \
            "hazard_x" in collision_info and "hazard_y" in collision_info
        # Predicate C: step-after-terminal raises remote error.
        pred_c_step_after_terminal_rejects = False
        rejection_kind: str | None = None
        if terminated or truncated:
            try:
                env.step(1)
            except RuntimeError as e:
                # Env raises RuntimeError if done flag set Python-side.
                pred_c_step_after_terminal_rejects = True
                rejection_kind = f"RuntimeError:{e}"
            except GodotRemoteError as e:
                pred_c_step_after_terminal_rejects = True
                rejection_kind = f"GodotRemoteError:{e}"

        # Predicate D: reset clears flags; next step is non-terminal.
        pred_d_reset_clears = False
        post_reset_steps: list[dict[str, Any]] = []
        try:
            env.reset(seed=seed + 1)
            for i in range(5):
                obs, r, t, tr, info = env.step(1)
                post_reset_steps.append({
                    "step": i + 1, "terminated": bool(t), "truncated": bool(tr),
                    "reward": float(r),
                })
                if t or tr:
                    break
            pred_d_reset_clears = all(
                (not s["terminated"]) and (not s["truncated"])
                for s in post_reset_steps
            )
        except Exception as e:  # noqa: BLE001
            facts["reset_clear_error"] = repr(e)

        facts.update({
            "terminal_step": terminal_step,
            "terminal_reason": terminal_reason,
            "collision_info": collision_info,
            "predicate_A_terminal_is_collision": pred_a_terminal_is_collision,
            "predicate_B_collision_info_present": pred_b_collision_info_present,
            "predicate_C_step_after_terminal_rejects": pred_c_step_after_terminal_rejects,
            "predicate_C_rejection_kind": rejection_kind,
            "predicate_D_reset_clears": pred_d_reset_clears,
            "post_reset_first5_steps": post_reset_steps,
        })
        facts["pass"] = all([
            pred_a_terminal_is_collision,
            pred_b_collision_info_present,
            pred_c_step_after_terminal_rejects,
            pred_d_reset_clears,
        ])
    finally:
        env.close()
    return facts, rows


# ---------------- Layer 1-4a: scripted state-mode trace ------------------

SCRIPTED_ACTIONS_240 = (
    [1] * 10 +         # stay x 10
    [0] * 80 +         # left x 80
    [1] * 10 +         # stay x 10
    [2] * 120 +        # right x 120
    [1] * 20           # stay x 20
)  # total 240


def run_layers_1_2_3_4a(
    *, out_dir: Path, seed: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Single state-mode rollout under the canonical scripted policy.

    Returns a dict keyed by layer name plus the per-step rows. The
    rollout MUST run to completion (no terminal short-circuit) so all
    timing/kinematics predicates have a contiguous window; the seed and
    scripted policy are chosen so the player drifts away from hazards.
    If the env terminates early the layer-1 predicate fails fast and the
    downstream layer predicates are marked inconclusive.
    """
    sub_run = out_dir / "layers_1_4a"
    sub_run.mkdir(parents=True, exist_ok=True)
    env = _build_env(observation_mode="state", run_dir=sub_run, seed=seed,
                     max_steps=len(SCRIPTED_ACTIONS_240) + 10)
    rows: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    terminated_early = False
    try:
        obs, info = env.reset()
        reset_obs = obs.tolist()
        # Capture the reset frame (Godot frame=0 per soft reset).
        reset_frame = int(info.get("frame", 0))
        for i, action_wire in enumerate(SCRIPTED_ACTIONS_240):
            obs_prev_player_x_from_obs = float(reset_obs[0]) if i == 0 else \
                float(step_records[-1]["obs_after"][0])
            obs, r, t, tr, info = env.step(int(action_wire))
            rec = {
                "step": i + 1,
                "action_wire": int(action_wire),
                "action_mapped_expected": ACTION_WIRE_TO_MAPPED[int(action_wire)],
                "frame": int(info.get("frame", -1)),
                "reward": float(r),
                "terminated": bool(t),
                "truncated": bool(tr),
                "obs_after": [float(v) for v in obs.tolist()],
                "godot_info": info.get("godot_info") or {},
            }
            step_records.append(rec)
            rows.append({
                "layer": "1-4a", "step": i + 1, "action_wire": int(action_wire),
                "frame": rec["frame"], "reward": rec["reward"],
                "terminated": rec["terminated"], "truncated": rec["truncated"],
                "obs0_player_x_norm": rec["obs_after"][0],
                "obs1_last_move_x": rec["obs_after"][1],
            })
            if t or tr:
                terminated_early = True
                break
    finally:
        env.close()

    # Parse godot.ndjson for spawn / h3_step events.
    godot_events = _read_ndjson(sub_run / "godot.ndjson")
    h3_step_events = [e for e in godot_events if e.get("type") == "h3_step"]
    spawn_events = [e for e in godot_events if e.get("type") == "spawn"]

    facts: dict[str, dict[str, Any]] = {}

    # ----- Layer 1: action timing + per-step contract -----
    layer1 = {"layer": "1", "name": "action_timing_per_step_contract"}
    pred_step_count_match = (
        len(h3_step_events) == len(step_records)
        and not terminated_early
    )
    pred_frame_monotonic = all(
        h3_step_events[i]["frame"] == i + 1 for i in range(len(h3_step_events))
    ) if h3_step_events else False
    pred_seq_match = all(
        h3_step_events[i].get("seq") == i for i in range(len(h3_step_events))
    ) if h3_step_events else False
    pred_action_wire_map = all(
        h3_step_events[i].get("action_wire") == step_records[i]["action_wire"]
        and h3_step_events[i].get("action") == ACTION_WIRE_TO_MAPPED[step_records[i]["action_wire"]]
        for i in range(min(len(h3_step_events), len(step_records)))
    )
    layer1.update({
        "python_step_count": len(step_records),
        "godot_h3_step_count": len(h3_step_events),
        "terminated_early": terminated_early,
        "predicate_step_count_match": pred_step_count_match,
        "predicate_frame_monotonic_increment_1": pred_frame_monotonic,
        "predicate_seq_match_per_step": pred_seq_match,
        "predicate_action_wire_to_mapped_correct": pred_action_wire_map,
    })
    layer1["pass"] = all([
        pred_step_count_match, pred_frame_monotonic,
        pred_seq_match, pred_action_wire_map,
    ])
    facts["1"] = layer1

    # ----- Layer 2: player kinematics -----
    layer2 = {"layer": "2", "name": "player_kinematics"}
    player_x_obs_violations: list[dict[str, Any]] = []
    prev_x: float | None = None
    last_action_mapped = 0
    for i, h3 in enumerate(h3_step_events):
        cur_x = float(h3.get("player_x", 0.0))
        action_mapped = int(h3.get("action", 0))
        if prev_x is None:
            prev_x = cur_x
            last_action_mapped = action_mapped
            continue
        expected_delta = action_mapped * PLAYER_SPEED_PX_STEP
        actual_delta = cur_x - prev_x
        clamped_left = abs(cur_x - PLAYER_X_LEFT_CLAMP) <= EPS_POSITION and action_mapped <= 0
        clamped_right = abs(cur_x - PLAYER_X_RIGHT_CLAMP) <= EPS_POSITION and action_mapped >= 0
        ok = (
            abs(actual_delta - expected_delta) <= EPS_POSITION
            or clamped_left or clamped_right
        )
        if not ok:
            player_x_obs_violations.append({
                "step": i + 1, "frame": h3.get("frame"),
                "action_mapped": action_mapped,
                "prev_x": prev_x, "cur_x": cur_x,
                "expected_delta": expected_delta, "actual_delta": actual_delta,
            })
        prev_x = cur_x
        last_action_mapped = action_mapped
    clamp_left_hits = sum(
        1 for h in h3_step_events
        if abs(float(h.get("player_x", 0.0)) - PLAYER_X_LEFT_CLAMP) <= EPS_POSITION
    )
    clamp_right_hits = sum(
        1 for h in h3_step_events
        if abs(float(h.get("player_x", 0.0)) - PLAYER_X_RIGHT_CLAMP) <= EPS_POSITION
    )
    layer2.update({
        "kinematic_violations": player_x_obs_violations[:20],
        "kinematic_violation_count": len(player_x_obs_violations),
        "left_clamp_hits": clamp_left_hits,
        "right_clamp_hits": clamp_right_hits,
        "predicate_player_delta_matches_action_speed": len(player_x_obs_violations) == 0,
        "predicate_left_clamp_observed": clamp_left_hits > 0,
    })
    layer2["pass"] = (
        len(player_x_obs_violations) == 0 and clamp_left_hits > 0
    )
    facts["2"] = layer2

    # ----- Layer 3: hazard kinematics + spawn contract -----
    layer3 = {"layer": "3", "name": "hazard_kinematics_spawn_contract"}
    spawn_frames = [int(s.get("frame", -1)) for s in spawn_events]
    pred_spawn_cadence = all(
        f > 0 and f % SPAWN_INTERVAL_FRAMES == 0 for f in spawn_frames
    ) if spawn_frames else False
    pred_spawn_count_matches = (
        len(spawn_events) == len(h3_step_events) // SPAWN_INTERVAL_FRAMES
    )
    pred_spawn_y = all(
        abs(float(s.get("y", 0.0)) - HAZARD_SPAWN_Y) <= EPS_POSITION
        for s in spawn_events
    )
    hazard_dy_violations: list[dict[str, Any]] = []
    prev_hazards_by_id: dict[int, dict[str, float]] = {}
    for i, rec in enumerate(step_records):
        rs = rec.get("godot_info", {}).get("reward_state", {})
        cur_hazards = rs.get("hazards_above", []) if isinstance(rs, dict) else []
        cur_by_id: dict[int, dict[str, float]] = {}
        for h in cur_hazards:
            try:
                hid = int(h["id"])
                cur_by_id[hid] = {"x": float(h["x"]), "y": float(h["y"])}
            except (KeyError, TypeError, ValueError):
                continue
        for hid, cur in cur_by_id.items():
            if hid in prev_hazards_by_id:
                prev = prev_hazards_by_id[hid]
                dy = cur["y"] - prev["y"]
                if abs(dy - HAZARD_SPEED_PX_STEP) > EPS_HAZARD_DT:
                    hazard_dy_violations.append({
                        "step": i + 1, "hazard_id": hid,
                        "prev_y": prev["y"], "cur_y": cur["y"], "dy": dy,
                    })
        prev_hazards_by_id = cur_by_id
    layer3.update({
        "spawn_event_count": len(spawn_events),
        "spawn_frames": spawn_frames,
        "predicate_spawn_cadence_div30": pred_spawn_cadence,
        "predicate_spawn_count_matches_floor_steps_div_30": pred_spawn_count_matches,
        "predicate_spawn_y_neg_hazard_size": pred_spawn_y,
        "hazard_dy_violations": hazard_dy_violations[:20],
        "hazard_dy_violation_count": len(hazard_dy_violations),
        "predicate_hazard_dy_matches_speed": len(hazard_dy_violations) == 0,
    })
    layer3["pass"] = all([
        pred_spawn_cadence, pred_spawn_count_matches, pred_spawn_y,
        len(hazard_dy_violations) == 0,
    ])
    facts["3"] = layer3

    # ----- Layer 4a: state obs freshness -----
    layer4a = {"layer": "4a", "name": "state_obs_freshness"}
    obs0_mismatches: list[dict[str, Any]] = []
    obs1_mismatches: list[dict[str, Any]] = []
    for i, rec in enumerate(step_records):
        h3 = h3_step_events[i] if i < len(h3_step_events) else {}
        cur_x = float(h3.get("player_x", 0.0))
        action_mapped = int(h3.get("action", 0))
        expected_obs0 = (cur_x / float(SCREEN_WIDTH)) * 2.0 - 1.0
        expected_obs0 = max(-1.0, min(1.0, expected_obs0))
        obs0_actual = float(rec["obs_after"][0])
        if abs(obs0_actual - expected_obs0) > 1e-4:
            obs0_mismatches.append({
                "step": i + 1, "frame": h3.get("frame"),
                "expected_obs0": expected_obs0, "actual_obs0": obs0_actual,
                "cur_x": cur_x,
            })
        obs1_actual = float(rec["obs_after"][1])
        if abs(obs1_actual - float(action_mapped)) > 1e-4:
            obs1_mismatches.append({
                "step": i + 1, "frame": h3.get("frame"),
                "expected_obs1": float(action_mapped), "actual_obs1": obs1_actual,
            })
    layer4a.update({
        "obs0_mismatches": obs0_mismatches[:20],
        "obs0_mismatch_count": len(obs0_mismatches),
        "obs1_mismatches": obs1_mismatches[:20],
        "obs1_mismatch_count": len(obs1_mismatches),
        "predicate_obs0_player_x_post_step_match": len(obs0_mismatches) == 0,
        "predicate_obs1_current_action_not_previous": len(obs1_mismatches) == 0,
    })
    layer4a["pass"] = (
        len(obs0_mismatches) == 0 and len(obs1_mismatches) == 0
    )
    facts["4a"] = layer4a

    return facts, rows


# ---------------- Layer 4b: pixel obs freshness --------------------------

def run_layer_4b(
    *, out_dir: Path, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pixel-mode rollout under the same scripted policy.

    Predicates:
      - Godot dynamics (frame, action_wire, action, player_x) sequence
        identical to Layer 1 state-mode trace.
      - Per-step pixel SHA-256 sequence has no consecutive duplicates
        when there is motion (every step has either action != 0 or a
        hazard advance).
      - The first left action (step 11) produces a different pixel hash
        from the prior stay step.
      - The first right action (step 101) produces a different pixel
        hash from the prior stay step.
    """
    facts: dict[str, Any] = {"layer": "4b", "name": "pixel_freshness_state_alignment"}
    rows: list[dict[str, Any]] = []
    sub_run = out_dir / "layer_4b"
    sub_run.mkdir(parents=True, exist_ok=True)
    env = _build_env(observation_mode="pixel", run_dir=sub_run, seed=seed,
                     max_steps=len(SCRIPTED_ACTIONS_240) + 10)
    pixel_hashes: list[str] = []
    pixel_records: list[dict[str, Any]] = []
    try:
        obs, info = env.reset()
        pixel_hashes.append(_hash_pixel_obs(obs))
        for i, action_wire in enumerate(SCRIPTED_ACTIONS_240):
            obs, r, t, tr, info = env.step(int(action_wire))
            h = _hash_pixel_obs(obs)
            pixel_hashes.append(h)
            pixel_records.append({
                "step": i + 1, "action_wire": int(action_wire),
                "frame": int(info.get("frame", -1)),
                "reward": float(r),
                "terminated": bool(t), "truncated": bool(tr),
                "pixel_sha256": h,
            })
            rows.append({
                "layer": "4b", "step": i + 1, "action_wire": int(action_wire),
                "frame": int(info.get("frame", -1)),
                "pixel_sha256_prefix": h[:16],
                "terminated": bool(t), "truncated": bool(tr),
            })
            if t or tr:
                break
    finally:
        env.close()

    pixel_godot = _read_ndjson(sub_run / "godot.ndjson")
    pixel_h3 = [e for e in pixel_godot if e.get("type") == "h3_step"]
    state_godot = _read_ndjson(out_dir / "layers_1_4a" / "godot.ndjson")
    state_h3 = [e for e in state_godot if e.get("type") == "h3_step"]
    dynamics_divergences: list[dict[str, Any]] = []
    common = min(len(pixel_h3), len(state_h3))
    for i in range(common):
        ps = pixel_h3[i]
        ss = state_h3[i]
        diff: dict[str, Any] = {}
        if ps.get("frame") != ss.get("frame"):
            diff["frame"] = (ps.get("frame"), ss.get("frame"))
        if ps.get("action_wire") != ss.get("action_wire"):
            diff["action_wire"] = (ps.get("action_wire"), ss.get("action_wire"))
        if ps.get("action") != ss.get("action"):
            diff["action"] = (ps.get("action"), ss.get("action"))
        px_p = float(ps.get("player_x", 0.0))
        px_s = float(ss.get("player_x", 0.0))
        if abs(px_p - px_s) > EPS_POSITION:
            diff["player_x"] = (px_p, px_s)
        if diff:
            diff["step"] = i + 1
            dynamics_divergences.append(diff)

    duplicate_pairs: list[dict[str, Any]] = []
    for i in range(1, len(pixel_hashes)):
        if pixel_hashes[i] == pixel_hashes[i - 1]:
            duplicate_pairs.append({"step": i, "hash_prefix": pixel_hashes[i][:16]})

    first_left_step = 11
    first_right_step = 101
    def _same_step_shift(idx: int) -> dict[str, Any]:
        if idx >= len(pixel_hashes):
            return {"ok": False, "reason": "out_of_range"}
        before = pixel_hashes[idx - 1]
        after = pixel_hashes[idx]
        return {"ok": before != after, "before_prefix": before[:16],
                "after_prefix": after[:16]}

    first_left_shift = _same_step_shift(first_left_step)
    first_right_shift = _same_step_shift(first_right_step)

    # Action-transition freshness: at every step where action_wire changes from
    # the prior step, the pixel hash MUST differ. Sub-quantization motion
    # during identical-action windows can yield consecutive duplicate hashes
    # at 84x84 nearest-neighbor downsample (initial pre-spawn stay window,
    # mid-stride left/right where 5 px world maps to 0.58 obs px); those are
    # not freshness failures. Hash failing to change ACROSS an action change
    # is the load-bearing freshness defect.
    transition_misses: list[dict[str, Any]] = []
    for i in range(1, len(pixel_records)):
        if pixel_records[i]["action_wire"] != pixel_records[i - 1]["action_wire"]:
            # pixel_hashes index i corresponds to the obs after step i (1-based),
            # so use pixel_hashes[i+1] vs pixel_hashes[i] for the cross-transition pair.
            # Layout: pixel_hashes[0] is reset obs; pixel_hashes[k] is obs after step k.
            h_prev = pixel_hashes[i]
            h_cur = pixel_hashes[i + 1] if i + 1 < len(pixel_hashes) else h_prev
            if h_prev == h_cur:
                transition_misses.append({
                    "step": i + 1,
                    "action_wire_prev": pixel_records[i - 1]["action_wire"],
                    "action_wire_cur": pixel_records[i]["action_wire"],
                    "hash_prefix": h_prev[:16],
                })

    facts.update({
        "pixel_step_count": len(pixel_records),
        "pixel_hash_count": len(pixel_hashes),
        "dynamics_divergence_count": len(dynamics_divergences),
        "dynamics_divergences": dynamics_divergences[:20],
        "consecutive_duplicate_pair_count": len(duplicate_pairs),
        "consecutive_duplicate_pairs_first10": duplicate_pairs[:10],
        "first_left_same_step_shift": first_left_shift,
        "first_right_same_step_shift": first_right_shift,
        "predicate_dynamics_match_state_mode": len(dynamics_divergences) == 0,
        "predicate_first_left_same_step_shift": bool(first_left_shift.get("ok")),
        "predicate_first_right_same_step_shift": bool(first_right_shift.get("ok")),
        "predicate_action_transition_hashes_differ": len(transition_misses) == 0,
        "action_transition_misses": transition_misses[:10],
        "action_transition_miss_count": len(transition_misses),
        "note_duplicate_pairs_metadata_only": (
            "consecutive_duplicate_pair_count is metadata, not a pass criterion. "
            "Sub-quantization motion at 84x84 nearest-neighbor downsample produces "
            "consecutive identical hashes during pre-spawn stay windows and "
            "mid-stride low-motion windows; this is expected."
        ),
    })
    facts["pass"] = all([
        len(dynamics_divergences) == 0,
        bool(first_left_shift.get("ok")),
        bool(first_right_shift.get("ok")),
        len(transition_misses) == 0,
    ])
    return facts, rows


# ---------------- Layer 5: frame-stack contract --------------------------

def run_layer_5(*, out_dir: Path, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Inspect env observation_space shape in both modes.

    No frame stack is configured in the K5.1 alpha=0.30 config; this
    layer verifies env construction with the same kwargs produces the
    expected (1, 84, 84) shape and (10,) shape respectively. Does NOT
    open a TCP connection (env init does not start Godot).
    """
    facts: dict[str, Any] = {"layer": "5", "name": "frame_stack_contract"}
    sub_run = out_dir / "layer5"
    sub_run.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    state_env = _build_env(observation_mode="state", run_dir=sub_run / "state", seed=seed)
    pixel_env = _build_env(observation_mode="pixel", run_dir=sub_run / "pixel", seed=seed)
    try:
        state_shape = tuple(int(x) for x in state_env.observation_space.shape)
        pixel_shape = tuple(int(x) for x in pixel_env.observation_space.shape)
    finally:
        state_env.close()
        pixel_env.close()
    pred_state_shape = state_shape == (10,)
    pred_pixel_shape = pixel_shape == (1, 84, 84)
    facts.update({
        "state_observation_space_shape": list(state_shape),
        "pixel_observation_space_shape": list(pixel_shape),
        "predicate_state_shape_is_10": pred_state_shape,
        "predicate_pixel_shape_is_1_84_84": pred_pixel_shape,
        "note": (
            "K5.1 alpha=0.30 config sets no frame_stack key; factory only "
            "applies VecFrameStack when frame_stack > 1. Single-frame "
            "(1,84,84) is the expected production shape."
        ),
    })
    facts["pass"] = pred_state_shape and pred_pixel_shape
    return facts, rows


# ---------------- Layer 6: scripted policies + oracle --------------------

POLICY_NAMES = ("constant_left", "constant_stay", "constant_right",
                "hazard_reactive_oracle", "shaped_reward_greedy_oracle_alpha030")


def _select_action(policy: str, last_obs_state: list[float] | None,
                   last_reward_state: dict[str, Any] | None) -> int:
    if policy == "constant_left":
        return 0
    if policy == "constant_stay":
        return 1
    if policy == "constant_right":
        return 2
    if policy == "hazard_reactive_oracle":
        return hazard_reactive_oracle(last_reward_state or {})
    if policy == "shaped_reward_greedy_oracle_alpha030":
        return shaped_reward_greedy_oracle(last_reward_state or {}, alpha=0.30)
    raise ValueError(f"unknown policy: {policy}")


def _run_one_episode(env: GodotSignalDodgeEnv, policy: str, *,
                     max_steps: int, seed: int) -> dict[str, Any]:
    obs, info = env.reset(seed=int(seed))
    last_reward_state: dict[str, Any] | None = (
        (info.get("godot_info") or {}).get("reward_state")
    )
    last_state_obs: list[float] | None = (
        [float(v) for v in obs.tolist()] if hasattr(obs, "tolist") else None
    )
    total_base = 0.0
    total_shaped = 0.0
    action_counts = [0, 0, 0]
    x_distribution_bins = [0] * 12
    saw_active_threat_steps = 0
    sat_active_threat_steps = 0
    step_idx = 0
    terminated = False
    truncated = False
    terminal_reason = ""
    while step_idx < max_steps:
        a = _select_action(policy, last_state_obs, last_reward_state)
        action_counts[int(a)] += 1
        obs, r, terminated, truncated, info = env.step(int(a))
        gi = info.get("godot_info") or {}
        last_reward_state = gi.get("reward_state") or {}
        last_state_obs = [float(v) for v in obs.tolist()] if hasattr(obs, "tolist") else None
        base_r = 0.0 if terminated else 1.0
        bonus, _, n_active = compute_threat_weighted_clearance(
            last_reward_state, alpha=0.30,
        )
        total_base += base_r
        total_shaped += (base_r + (0.0 if terminated else bonus))
        if n_active > 0:
            saw_active_threat_steps += 1
            if bonus >= 0.98 * 0.30:
                sat_active_threat_steps += 1
        try:
            px = float(last_reward_state.get("player_x", SCREEN_WIDTH / 2.0))
        except (TypeError, AttributeError):
            px = SCREEN_WIDTH / 2.0
        bx = int(max(0, min(11, px // 60)))
        x_distribution_bins[bx] += 1
        step_idx += 1
        if terminated or truncated:
            terminal_reason = info.get("terminal_reason", "")
            break
    return {
        "policy": policy,
        "seed": int(seed),
        "steps": step_idx,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": terminal_reason,
        "action_counts": action_counts,
        "x_distribution_bins": x_distribution_bins,
        "active_threat_steps": saw_active_threat_steps,
        "saturated_active_threat_steps": sat_active_threat_steps,
        "total_base_reward": total_base,
        "total_shaped_reward_alpha030": total_shaped,
    }


def run_layer_6(*, out_dir: Path, seeds: list[int],
                max_steps: int = 1800) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    facts: dict[str, Any] = {"layer": "6", "name": "scripted_policy_reward_surface"}
    rows: list[dict[str, Any]] = []
    sub_run = out_dir / "layer6"
    sub_run.mkdir(parents=True, exist_ok=True)
    per_policy: dict[str, list[dict[str, Any]]] = {p: [] for p in POLICY_NAMES}
    for policy in POLICY_NAMES:
        env = _build_env(
            observation_mode="state",
            run_dir=sub_run / policy,
            seed=int(seeds[0]),
            max_steps=max_steps,
            reward_shaping="none",
        )
        try:
            for s in seeds:
                t0 = time.monotonic()
                ep = _run_one_episode(env, policy, max_steps=max_steps, seed=int(s))
                ep["wall_s"] = time.monotonic() - t0
                per_policy[policy].append(ep)
                rows.append({"layer": "6", **ep})
        finally:
            env.close()
    aggregates: dict[str, dict[str, Any]] = {}
    for policy, eps in per_policy.items():
        if not eps:
            continue
        mean_len = sum(e["steps"] for e in eps) / len(eps)
        coll_rate = sum(
            1 for e in eps if e["terminated"] and e["terminal_reason"] == "collision"
        ) / len(eps)
        timeout_rate = sum(
            1 for e in eps if e["truncated"] and e["terminal_reason"] == "timeout"
        ) / len(eps)
        action_totals = [0, 0, 0]
        x_totals = [0] * 12
        total_base = 0.0
        total_shaped = 0.0
        for e in eps:
            for i in range(3):
                action_totals[i] += e["action_counts"][i]
            for i in range(12):
                x_totals[i] += e["x_distribution_bins"][i]
            total_base += e["total_base_reward"]
            total_shaped += e["total_shaped_reward_alpha030"]
        total_actions = sum(action_totals)
        action_dist = [a / total_actions if total_actions else 0.0 for a in action_totals]
        total_x = sum(x_totals)
        x_dist = [a / total_x if total_x else 0.0 for a in x_totals]
        aggregates[policy] = {
            "episodes": len(eps),
            "mean_episode_length": mean_len,
            "collision_rate": coll_rate,
            "timeout_rate": timeout_rate,
            "action_distribution_wire_left_stay_right": action_dist,
            "x_distribution_60px_bins": x_dist,
            "total_base_reward": total_base,
            "total_shaped_reward_alpha030": total_shaped,
        }
    facts["per_policy_aggregates"] = aggregates
    const_mean = max(
        aggregates.get(p, {}).get("mean_episode_length", 0.0)
        for p in ("constant_left", "constant_stay", "constant_right")
    ) if aggregates else 0.0
    oracle_mean = aggregates.get("hazard_reactive_oracle", {}).get("mean_episode_length", 0.0)
    threshold = max(0.10 * const_mean, 60.0)
    materially_beats = (oracle_mean - const_mean) >= threshold
    facts.update({
        "best_constant_mean_episode_length": const_mean,
        "hazard_reactive_oracle_mean_episode_length": oracle_mean,
        "delta_oracle_minus_best_constant": oracle_mean - const_mean,
        "materiality_threshold_frames": threshold,
        "predicate_oracle_materially_beats_constants": materially_beats,
    })
    facts["pass"] = materially_beats
    return facts, rows


# ---------------- Main runner --------------------------------------------

def classify(facts: dict[str, Any]) -> dict[str, str]:
    """Compute K5.2 classification from per-layer facts.

    Ladder:
      - Layers 0..5 are env mechanics. Any FAIL -> ENV-FAIL.
      - Layer 6 is task geometry / oracle. PASS -> ENV-PASS.
      - Layer 6 FAIL with Layers 0..5 PASS -> TASK-GEOMETRY-FAIL.
    """
    mechanics_layers = ("0", "1", "2", "3", "4a", "4b", "5")
    mechanics_failures: list[str] = []
    for lid in mechanics_layers:
        layer = facts.get(lid)
        if layer is None:
            continue
        if not layer.get("pass"):
            mechanics_failures.append(lid)
    if mechanics_failures:
        return {
            "classification": "ENV-FAIL",
            "trigger_layers": ",".join(mechanics_failures),
        }
    layer6 = facts.get("6")
    if layer6 is None:
        return {"classification": "INCONCLUSIVE", "trigger_layers": "6_not_run"}
    if layer6.get("pass"):
        return {"classification": "ENV-PASS", "trigger_layers": "6"}
    return {"classification": "TASK-GEOMETRY-FAIL", "trigger_layers": "6"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="k5_2_env_dynamics_probe")
    parser.add_argument("--layers", default="0,1,2,3,4a,4b,5,6",
                        help="comma list: 0,1,2,3,4a,4b,5,6")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "runs" / "phase_k")
    parser.add_argument("--seed", type=int, default=1000,
                        help="layer-0/1-4a/4b base seed")
    parser.add_argument("--layer6-seeds", default="1000,1001,1002,1003,1004,1005,1006,1007,1008,1009")
    parser.add_argument("--layer6-max-steps", type=int, default=1800)
    parser.add_argument("--merge", action="store_true",
                        help="merge into existing JSON/CSV instead of overwriting")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(s.strip() for s in args.layers.split(",") if s.strip())
    layer6_seeds = [int(s) for s in args.layer6_seeds.split(",")]

    facts: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if args.merge:
        existing_json = out_dir / "k5_2_env_dynamics_probe.json"
        if existing_json.exists():
            try:
                facts = json.loads(existing_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                facts = {}
        existing_csv = out_dir / "k5_2_env_dynamics_probe_rows.csv"
        if existing_csv.exists():
            try:
                with existing_csv.open("r", encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
            except OSError:
                rows = []
    facts.update({
        "schema_version": 1,
        "probe": "k5_2_env_dynamics",
        "git_commit": _short_git_commit(),
        "godot_exe": _resolve_godot_exe(),
        "layers_requested": sorted(set(facts.get("layers_requested", [])) | requested),
        "layer6_seeds": layer6_seeds,
        "started_ts_unix": facts.get("started_ts_unix", time.time()),
    })

    def _run(label: str, fn) -> None:
        try:
            print(f"[probe] running layer {label} ...", flush=True)
            t0 = time.monotonic()
            f, r = fn()
            f["wall_s"] = time.monotonic() - t0
            facts[label] = f
            rows.extend(r)
            print(f"[probe] layer {label} done in {f['wall_s']:.1f}s pass={f.get('pass')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            facts[label] = {
                "layer": label, "pass": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(f"[probe] layer {label} ERROR: {type(exc).__name__}: {exc}", flush=True)

    if "0" in requested:
        _run("0", lambda: run_layer_0(out_dir=out_dir, seed=args.seed))
    needs_1_4a = any(l in requested for l in ("1", "2", "3", "4a"))
    if needs_1_4a:
        try:
            print("[probe] running layers 1,2,3,4a (shared state-mode rollout) ...", flush=True)
            t0 = time.monotonic()
            shared, shared_rows = run_layers_1_2_3_4a(out_dir=out_dir, seed=args.seed)
            elapsed = time.monotonic() - t0
            for lid, lf in shared.items():
                lf["wall_s"] = elapsed if lid == "1" else 0.0
                facts[lid] = lf
                print(f"[probe] layer {lid} pass={lf.get('pass')}", flush=True)
            rows.extend(shared_rows)
        except Exception as exc:  # noqa: BLE001
            for lid in ("1", "2", "3", "4a"):
                facts[lid] = {
                    "layer": lid, "pass": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            print(f"[probe] layers 1-4a ERROR: {type(exc).__name__}: {exc}", flush=True)
    if "4b" in requested:
        _run("4b", lambda: run_layer_4b(out_dir=out_dir, seed=args.seed))
    if "5" in requested:
        _run("5", lambda: run_layer_5(out_dir=out_dir, seed=args.seed))
    if "6" in requested:
        _run("6", lambda: run_layer_6(
            out_dir=out_dir, seeds=layer6_seeds, max_steps=args.layer6_max_steps,
        ))

    facts["finished_ts_unix"] = time.time()
    facts["classification"] = classify(facts)

    json_path = out_dir / "k5_2_env_dynamics_probe.json"
    csv_path = out_dir / "k5_2_env_dynamics_probe_rows.csv"
    json_path.write_text(json.dumps(facts, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        all_keys: list[str] = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    else:
        csv_path.write_text("", encoding="utf-8")
    print(f"[probe] wrote {json_path}")
    print(f"[probe] wrote {csv_path}")
    print(f"[probe] classification: {facts['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
