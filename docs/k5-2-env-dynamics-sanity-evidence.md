# K5.2 env-dynamics sanity evidence

## Posture

K5.1 (alpha=0.30 high-force clearance reward, 10k timesteps,
deterministic-argmax eval) terminated on a single-action attractor at
`stay`, `player_x=360`, `6060/6060 reached eval steps`. K3.5c baseline
(no shaping, same budget) terminated on `left`, `player_x=16`,
`8457/8457 reached eval steps`. Two distinct reward regimes, same
weight-invariant collapse signature. Per the K5.2 charter, before any
further training or hyperparameter changes, the env mechanics and task
geometry must be ruled out as the cause.

K5.2 is an evidence-only probe. No training. No reward edits. No env
edits. The probe layers the production `GodotSignalDodgeEnv` /
`GodotH3Transport` path with scripted-policy instrumentation and reads
the resulting NDJSON streams plus the live wire payloads to evaluate a
ladder of predicates.

Tool: `tools/k5_2_env_dynamics_probe.py` (commit at the tip of this
slice). Artifacts: `runs/phase_k/k5_2_env_dynamics_probe.json` (full
predicate facts) and `runs/phase_k/k5_2_env_dynamics_probe_rows.csv`
(per-step rows). `runs/` is gitignored; artifacts are reproducible from
the tool.

Run conditions: Windows 11, StrongerJr; Godot 4.6.2-stable_win64;
`SIGHT_GODOT_EXE` set inline; state-mode layers headless, pixel-mode
layer windowed; seed 1000 for layers 0/1/2/3/4a/4b/5; seeds
1000..1009 for layer 6; `max_steps=1800` for layer 0 and layer 6;
`max_steps=250` (240 scripted + 10 slack) for layers 1/2/3/4a/4b.

## Classification

**ENV-PASS** (trigger layer 6).

All layer-0 through layer-5 mechanics predicates passed. The layer-6
hazard-reactive 1-step geometry oracle survives a mean of 1762.8
frames per episode across 10 seeds vs. the best constant policy
(`constant_left`) at 845.7 frames; the delta of 917.1 frames exceeds
the materiality threshold of 84.6 frames (max of 10% of best-constant
mean or 60 frames). Signal Dodge under the K5.1 alpha=0.30 budget,
spawn rate, hazard speed, and player speed is therefore a solvable
task with a 1-step lookahead policy that has access to raw geometry,
under the same env transport the trained policies use.

The K3.5c / K5.1 weight-invariant single-action collapse is not
explained by env mechanics (collision propagation, action timing,
player or hazard kinematics, observation freshness, frame-stack
contract) and not explained by task geometry (the oracle clears the
materiality bar by over an order of magnitude relative to the
threshold). Cause is downstream of the env, in the learning pipeline
(PPO + CnnPolicy + deterministic-argmax eval + 10k-step budget).

## Predicate ladder

Each layer reports one or more named predicates plus a layer-level
`pass`. Mechanics layers (0-5) gate ENV-FAIL; layer 6 gates
TASK-GEOMETRY-FAIL vs ENV-PASS.

### Layer 0 - collision-propagation preflight (state mode)

PASS. Forced collision at `player_x=360` under repeated `stay` action.
Terminal step at frame `terminal_step` carries `terminated=true`,
`terminal_reason="collision"`, and a populated `collision_info` dict
(`hazard_x`, `hazard_y`, `player_x`, `player_y`, `survival_time`,
`frame`). Step-after-terminal raises (predicate C records the
exception kind for audit). Soft reset clears the sticky terminal flag
and the first five post-reset steps return `terminated=false,
truncated=false`.

This dynamically confirms the H5 collision-propagation fix
(`docs/h5-collision-propagation-bug.md`, "Fix result" section). The
sticky `_h3_step_terminated` flag set in `_on_player_died` is now
propagated through the very next `_h3_perform_step` reply rather than
being wiped by an unconditional start-of-step reset.

Predicates: `predicate_A_terminal_is_collision`,
`predicate_B_collision_info_present`,
`predicate_C_step_after_terminal_rejects`,
`predicate_D_reset_clears`. All true.

### Layer 1 - action timing + per-step contract (state mode)

PASS. A 240-step canonical scripted policy
(`stay*10, left*80, stay*10, right*120, stay*20`) is driven through
the env. Per-step Godot NDJSON `h3_step` events are joined to Python
`step()` calls.

Predicates:
- `predicate_step_count_match`: exactly 240 `h3_step` events for 240
  Python steps, no early termination.
- `predicate_frame_monotonic_increment_1`: `h3_step[i].frame == i+1`
  for `i in [0, 240)`. No idle physics ticks advance frame counter.
- `predicate_seq_match_per_step`: `h3_step[i].seq == i` for all i.
  TCP sequence numbering aligns with Python step order.
- `predicate_action_wire_to_mapped_correct`: wire action 0->mapped -1,
  1->0, 2->+1, with no off-by-one between Python step input and Godot
  applied mapped action.

All true.

### Layer 2 - player kinematics (state mode)

PASS. Reuses the layer-1 trace. For each consecutive pair of
`h3_step` events, the player_x delta matches the applied mapped
action times the per-step speed (5 px/step at speed=300/60) within
1e-3 absolute, or the player is clamped at one of the wall positions
(16 or 704).

The scripted policy `left*80` from x=360 reaches x=16 after exactly
(360-16)/5 = 68.8 -> 69 steps, then clamps for the remaining 11 steps
before the stay window. The `right*120` window then traverses from
x=16 across the screen.

Predicates:
- `predicate_player_delta_matches_action_speed`: zero kinematic
  violations across 240 steps.
- `predicate_left_clamp_observed`: 12 steps observed at
  abs(player_x - 16) <= 1e-3. (Right clamp not observed in this trace
  because the right window does not last long enough to reach 704
  before the trace ends.)

### Layer 3 - hazard kinematics + spawn contract (state mode)

PASS. Reuses the layer-1 trace plus the Godot `spawn` events from the
same NDJSON.

Predicates:
- `predicate_spawn_cadence_div30`: 8 spawn events at frames
  {30, 60, 90, 120, 150, 180, 210, 240}. All frame numbers divisible
  by 30.
- `predicate_spawn_count_matches_floor_steps_div_30`: 8 spawns ==
  floor(240/30).
- `predicate_spawn_y_neg_hazard_size`: all spawn `y` values equal
  -24.0 = `-HAZARD_SIZE`.
- `predicate_hazard_dy_matches_speed`: for every hazard tracked
  across consecutive steps in `reward_state.hazards_above`, the
  per-step y delta equals 200/60 = 3.333... px within 5e-3. Zero
  violations.

This confirms Godot's hazard physics tick produces exactly one 1/60s
advance per consumed `_h3_perform_step`, with no double-step,
sub-step, or skipped-step drift across the 240-step window.

### Layer 4a - state obs freshness (state mode)

PASS. Per-step state-mode obs is the length-10 vector defined in
`docs/sight-h3-plan.md` section 2. The probe asserts the post-step
observation reflects the post-step world state.

Predicates:
- `predicate_obs0_player_x_post_step_match`: for every step,
  `obs[0]` equals `clamp((player_x/720)*2 - 1, -1, 1)` within 1e-4.
  Zero mismatches over 240 steps.
- `predicate_obs1_current_action_not_previous`: for every step,
  `obs[1]` equals the CURRENT applied mapped action (not the
  previous step's action). Zero mismatches over 240 steps. This rules
  out an off-by-one in the `_h3_last_move_x` update relative to obs
  construction.

### Layer 4b - pixel obs freshness + state-pixel alignment (pixel mode)

PASS. Same 240-step scripted policy in pixel mode, same seed 1000,
windowed Godot launch. The probe asserts pixel obs reflects post-step
world state and that the cross-mode dynamics are bit-equivalent.

Predicates:
- `predicate_dynamics_match_state_mode`: 0 divergences across 240
  steps when comparing pixel-mode `h3_step` events against the
  state-mode `h3_step` events on `frame`, `action_wire`, `action`,
  `player_x` (with 1e-3 tolerance on player_x). Same-seed
  determinism holds across observation modes.
- `predicate_first_left_same_step_shift`: the SHA-256 hash of the
  pixel obs at the first-left step (step 11) differs from the prior
  stay step (step 10). Confirms obs captures motion on the same
  physics tick as the action.
- `predicate_first_right_same_step_shift`: identical assertion for
  the first-right step (step 101).
- `predicate_action_transition_hashes_differ`: at every step where
  `action_wire[i] != action_wire[i-1]`, the pixel hash differs.
  Zero misses. This is the load-bearing freshness predicate.

The original "no consecutive duplicate hashes" predicate was reframed
during this slice: 4 of 5 raw duplicate pairs occurred in the
pre-spawn stay window (frames 1-5) where the world is genuinely
visually static apart from a 1-digit step-counter glyph in the corner
that is sub-quantization at 84x84 grayscale L8 with nearest-neighbor
downsample; the 5th was a mid-stride low-motion left step where the
5 px world-space player motion maps to 0.58 obs-space px and
nearest-neighbor sometimes rounds two consecutive steps to the same
grid cell. These are properties of the downsample, not freshness
failures. The action-transition predicate is the correct instrument
and it is clean.

### Layer 5 - frame-stack contract (env inspection only)

PASS. Construct `GodotSignalDodgeEnv` in state and pixel modes with
the K5.1 alpha=0.30 config kwargs. Observation space shapes are
`(10,)` and `(1, 84, 84)` respectively. The K5.1 config sets no
`frame_stack` key; the factory only wraps with `VecFrameStack` when
`frame_stack > 1`. Single-frame `(1, 84, 84)` is the production
shape under which the K3.5c and K5.1 trained policies were learned
and evaluated.

### Layer 6 - scripted-policy reward surface + hazard-reactive oracle (state mode)

PASS. Five scripted policies, 10 seeds each (1000..1009), max 1800
steps per episode. Env runs with `reward_shaping=none` and the
shaped-alpha=0.30 total is computed Python-side from the per-step
`reward_state` wire payload via `compute_threat_weighted_clearance`.
All numbers are aggregated across the 10-seed batch.

Policies:

| policy                           | mean_ep_len | collision_rate | timeout_rate | action distribution L/S/R | total_base | total_shaped_alpha030 |
| -------------------------------- | -----------:| --------------:| ------------:| ------------------------- | ----------:| ---------------------:|
| constant_left                    |      845.7  |          0.9   |        0.1   | 1.000 / 0.000 / 0.000     |    8448.0  |             10351.95  |
| constant_right                   |      689.7  |          0.9   |        0.1   | 0.000 / 0.000 / 1.000     |    6888.0  |              8410.15  |
| constant_stay                    |      606.0  |          1.0   |        0.0   | 0.000 / 1.000 / 0.000     |    6050.0  |              7166.56  |
| hazard_reactive_oracle           |     1762.8  |          0.1   |        0.9   | 0.232 / 0.521 / 0.248     |   17627.0  |             22048.17  |
| shaped_reward_greedy_oracle_a030 |     1462.8  |          0.6   |        0.4   | 0.086 / 0.818 / 0.096     |   14622.0  |             18227.36  |

Player-x distribution (12 60-px bins; bin 0 is x in [0,60), bin 11 is
x in [660,720)):

- `constant_left`: 93% in bin 0 (clamped at x=16). Confirms drift
  to wall.
- `constant_right`: 91% in bin 11 (clamped at x=704). Symmetric.
- `constant_stay`: 100% in bin 6 (x in [360,420), the initial
  player_x with no horizontal motion).
- `hazard_reactive_oracle`: spread across all 12 bins, with edges
  bin 0 (19.1%) and bin 11 (24.3%) over-represented relative to the
  middle bins (3.7%-13.3%). The oracle uses the walls as defensive
  positions when both lateral directions point into a hazard
  corridor and stay would leave the player on a collision line.
- `shaped_reward_greedy_oracle_alpha030`: 82% stay, with edges
  spilled to bin 0 (15.4%) and bin 11 (24.6%). The shaped reward
  surface itself biases toward center-stay even when threats are
  present, because the threat-weighted-clearance bonus is averaged
  across all hazards above the player and a small lateral motion
  rarely changes the mean clearance enough to clear the
  no-motion-cost null hypothesis.

Materiality: `best_constant_mean_episode_length=845.7`,
`hazard_reactive_oracle_mean_episode_length=1762.8`,
`delta=917.1`, `threshold=max(0.10*845.7, 60)=84.57`. Oracle exceeds
the threshold by 832.5 frames. `predicate_oracle_materially_beats_constants=true`.

Two corollary observations carried in the per-policy aggregates,
relevant to the K3.5c / K5.1 collapse analysis even though they are
not part of the K5.2 pass criterion:

1. The greedy oracle on the shaped-reward surface
   (`shaped_reward_greedy_oracle_alpha030`) emits `stay` 81.8% of
   the time and reaches mean_ep_len 1462.8 with collision_rate 0.6.
   This is decisively better than the constant-stay policy (606.0,
   collision_rate 1.0) and decisively worse than the
   geometry-first hazard-reactive oracle (1762.8, collision_rate
   0.1). The shaped-reward surface IS biased toward stay-at-center.
   This is consistent with the K5.1 trained-policy collapse to
   stay-at-360, but the K3.5c collapse to constant-left under
   unshaped reward shows the constant-action pathology is not
   caused by the shaping; the shaping is a separate, smaller
   contributor.

2. The base-reward total under the hazard-reactive oracle
   (17627.0) is approximately 2.1x the best constant
   (8448.0 = constant_left). This is a strong signal that, given
   the geometry, a learning algorithm with the right inductive bias
   should be able to discover the survival policy from base reward
   alone within ~10k timesteps. PPO with CnnPolicy on (1, 84, 84)
   single-frame pixel input is the candidate that should be
   examined.

## What this rules in vs. rules out

Rules out, given the predicate evidence:

- Env collision-propagation defect (Layer 0).
- Off-by-one or stale action plumbing between Python step and Godot
  physics (Layer 1).
- Player or hazard kinematics drift, sub-step physics, frame-skip
  (Layers 1, 2, 3).
- Stale state-obs or pixel-obs payloads, observation/action timing
  inversion, between-mode determinism divergence (Layers 4a, 4b).
- Frame-stack misconfiguration relative to the K5.1 production
  config (Layer 5).
- Task-geometry difficulty: the task is solvable with a 1-step
  lookahead policy from raw geometry under the same budget
  (Layer 6).

Rules in (not exhaustive):

- The K3.5c / K5.1 weight-invariant deterministic-argmax fixed
  point is downstream of the env, in the learning pipeline.
- The shaped-reward surface (alpha=0.30, lookahead_band=270,
  safe_lateral_distance=180) is biased toward center-stay (see
  Layer 6 shaped-reward-greedy oracle). This is necessary context
  for any future reward-shape revision but does not by itself
  explain the K3.5c unshaped collapse to constant-left.
- The most parsimonious K5.1 hypothesis space (per Grok's
  unbidden suggestion in the RED-classification packet response)
  is deterministic-argmax convergence to a fixed point in the
  CnnPolicy logit space, independent of the converged weight set.
  Falsifying experiment: re-evaluate the same K5.1 checkpoint
  under stochastic action sampling instead of argmax and report
  whether per-step action distribution exhibits any spread.
  `tools/h5_stochastic_eval.py` already exists in the repo from a
  prior phase and is a candidate harness for that test.

## Out of scope of K5.2 (preserved for the next round)

- Any training or hyperparameter changes.
- Reward-shape revision.
- Stochastic-eval falsification of the deterministic-argmax fixed
  point hypothesis (it is the recommended next experiment, not
  part of this evidence pack).
- Architecture changes (CnnPolicy width, frame_stack > 1,
  per-action-type head split).
- Charter capability-target revision.

## Reproduction

```
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"
cd /d C:\Projects\Sight

REM Mechanics layers (single Python invocation, ~30s, fits MCP timeout).
python tools\k5_2_env_dynamics_probe.py --layers 0,1,2,3,4a,5 --out-dir runs\phase_k --seed 1000

REM Pixel layer (windowed Godot launch).
python tools\k5_2_env_dynamics_probe.py --layers 4b --out-dir runs\phase_k --seed 1000 --merge

REM Layer 6 (50 episodes, ~11 minutes wall, exceeds MCP 4-min timeout).
REM Use the .bat + sentinel pattern under C:\Users\maste\AppData\Local\Temp\.
python tools\k5_2_env_dynamics_probe.py --layers 6 --out-dir runs\phase_k --seed 1000 --merge
```

Outputs:
- `runs/phase_k/k5_2_env_dynamics_probe.json`
- `runs/phase_k/k5_2_env_dynamics_probe_rows.csv`
- `runs/phase_k/layer0/`, `runs/phase_k/layers_1_4a/`,
  `runs/phase_k/layer_4b/`, `runs/phase_k/layer5/`,
  `runs/phase_k/layer6/<policy>/` per-layer NDJSON evidence
  directories.
