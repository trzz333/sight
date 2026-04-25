"""Evaluator.

Primary join key is command seq (python.decision.seq <-> godot.controller_cmd_applied.seq).
Frame is the canonical simulation coordinate after the join; timestamps are for latency
diagnostics only.

Reconstruction:
- Hazard y is reconstructed from spawn events plus the locked Signal Dodge constants:
      y(frame) = spawn_y + HAZARD_SPEED * (frame - spawn_frame) / PHYSICS_HZ
- Nearest-hazard gap uses AABB edge distance between the player rect and each hazard rect.
- No frame-by-frame hazard_tick logging required until reconstruction proves unreliable.
"""

from .reconcile import evaluate, load_ndjson, reconcile

__all__ = ["evaluate", "load_ndjson", "reconcile"]
