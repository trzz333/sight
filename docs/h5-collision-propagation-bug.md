# H5 collision-propagation bug

Evidence captured during the Step 2A-lite 2-seed negative-control live
smoke (commit `a3d3b82`). The smoke artifacts under
`runs/rl/signal_dodge_ppo_h4_pixel/h5_negative_controls_smoke/` show a
Godot/Python protocol drift: Godot detects collisions and emits death
NDJSON events, but the per-step TCP reply never carries
`terminated=True`, so the Python rollout loop treats every episode as a
1800-step timeout. This document is the diagnostic record so the bug-fix
slice can be scoped against concrete numbers.

This bug, not profile ease, is the cause of the 100% non-saturation
result reported in the Step 2A-lite session. Until it is fixed, ANY
H5 non-saturation evidence on Signal Dodge is invalid because the env
cannot end an episode on collision. Hardening the profile would have
hidden the defect; one of the strongest cases against Step 2B was
exactly this scenario, surfaced in the diagnostic step.

## Evidence: Godot detects collisions and deaths

`godot.ndjson` event counts for the 2-seed × 1800-step rollout:

| policy        | spawn | collision | death | episode_start | h3_step |
| ------------- | ----- | --------- | ----- | ------------- | ------- |
| stay_only     |   121 |        18 |    18 |             4 |    3600 |
| seeded_random |   120 |        14 |    14 |             4 |    3600 |
| untrained_cnn |   120 |        18 |    18 |             4 |    3600 |

Spawn rate: 120 spawns / 1800 frames at `SPAWN_INTERVAL_FRAMES=30`
matches the configured profile. Hazards spawn fine. Collisions happen
fine on the Godot side. The profile is NOT zero-hazard.

## Evidence: Python never sees terminated=True

`python.ndjson` for the stay_only run (3610 lines):

```
{type: env_start, godot_pid: 15208}
{type: reset,    episode_id: ep-000001, seed: 1000, frame: 0, terminated: false, truncated: false}
{type: obs_metadata, ...}
{type: step,     episode_id: ep-000001, frame: 1, reward: 1.0, terminated: false, truncated: false, terminal_reason: ""}
{type: step,     episode_id: ep-000001, frame: 2, reward: 1.0, terminated: false, truncated: false, terminal_reason: ""}
... (3600 step events, all terminated=false, truncated=false, terminal_reason="")
{type: close,    ...}
```

Across all 3600 step events for all three policies, `terminated` is
`false`, `truncated` is `false`, and `terminal_reason` is the empty
string. The Python env never receives a death signal from the TCP wire.

Note that the `episode_id` in `python.ndjson` stays `ep-000001` for the
entire first seed's rollout. The env's internal episode counter never
advances on a collision because the protocol replies always say
terminated=false. The Godot side does advance (4 episode_starts) because
its own gameplay logic detects the collisions, but the controller's
step replies do not surface them.

## Root cause: GDScript step-flag race

`games/signal-dodge/scripts/main.gd` design intent (line 432 comment):
"`_on_player_died` fires synchronously inside the `move_action()` call
below if a collision occurs."

Reality: hazards move on Godot physics ticks. Player movement happens
inside `move_action`, called from `_h3_perform_step`. A hazard physics
tick can run BETWEEN two Python step requests and trigger
`_on_player_died` without `move_action` being on the stack.

Critical lines in `main.gd`:

- Line 226: `_on_player_died` sets `_h3_step_terminated = true` (correct).
- Line 417: at the start of every `_h3_perform_step` call,
  `_h3_step_terminated = false` (the bug surface).
- Line 432-435: `terminated: bool = _h3_step_terminated` is read AFTER
  `move_action` runs. If the collision fired in between steps (i.e.
  before the start-of-step reset), the flag was set true, then wiped
  false by line 417 before line 435 ever read it.

Sequence that produces the observed symptom:

1. Python sends step N to Godot via TCP.
2. Godot runs physics ticks; on some tick, a hazard intersects the
   player. `_on_player_died` fires. `_h3_step_terminated = true`.
   Death and collision NDJSON events log to `godot.ndjson`.
3. Eventually Godot enters `_h3_perform_step` for step N. Line 417:
   `_h3_step_terminated = false`. The between-step collision signal
   is gone.
4. Line 432 applies the player action. No new collision (player is
   already dead but the GDScript scene state is still alive).
5. Line 435 reads `_h3_step_terminated = false`. Reply sent with
   `terminated: false`.
6. Python loop continues stepping. Repeat for 1800 frames.
7. Godot's own gameplay loop separately detects more collisions on
   subsequent physics ticks. Each one logs a fresh
   collision+death event pair, none of which propagate to Python.

The Godot-side `episode_start: 4` (one per Python reset, including the
initial connect-side reset) confirms that gameplay episodes are NOT
ending on collision in the controller path. The collision-detection
exists; the propagation to the next step's reply is missing.

## Suggested fix paths

Not fixing in this slice; capturing options for the bug-fix scope.

1. **Sticky terminated flag.** Replace the start-of-step reset on
   line 417 with a guard that only clears `_h3_step_terminated` on
   `episode_start`/reset, not on every `_h3_perform_step`. Then any
   between-step collision is consumed by the very next step reply.
   Smallest change.

2. **Pending-collision queue.** Add a separate
   `_h3_pending_collision: bool` that survives the start-of-step
   reset. `_on_player_died` sets it true. The start of
   `_h3_perform_step` ORs it into `_h3_step_terminated` after the
   reset, then clears the pending flag. Slightly more explicit; same
   semantics.

3. **Step-gated physics.** Pause hazard physics between Python step
   requests so collisions can only fire inside `move_action`. The
   biggest architectural change; risks reproducibility drift vs the
   H3 same-seed posture. Probably not worth it.

(1) is the cleanest. The H3 same-seed reproducibility test should be
re-run after the fix to confirm no regression in the H4 capture-path
determinism (per the H4 closure record).

## Test additions for the fix slice

- A targeted unit test against a fake transport in which the death
  event fires on a physics tick between two step requests, asserting
  the next step reply carries `terminated=True` and
  `terminal_reason="collision"`.
- An integration check on the new live smoke that
  `mean_episode_length < 0.80 * max_steps` for at least one of the
  three negative controls on the current H4 pixel profile, since the
  Godot side already reports collisions at the current spawn rate.
  This is what the 2A-lite smoke SHOULD have shown if the propagation
  were correct.
- A regression test that re-reads the failing smoke artifacts as
  fixtures and asserts they would still be classified as `passed=False`
  by the harness under the canonical thresholds (the saturation
  decision remains correct given the input; only the input was wrong).

## What the original H5 Step 2B prompt called for

The Step 2B prompt asked for additive H5 difficulty parameterization
(spawn_interval_frames, hazard_speed, hazards_per_spawn) plus an H5
hard pixel YAML. None of that landed; the diagnostic killed the slice
upstream. Once the propagation bug is fixed and the H4 profile's
non-saturation result is re-measured truthfully, the decision on
whether to harden the profile is a separate, downstream question.
