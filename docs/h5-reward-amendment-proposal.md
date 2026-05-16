# H5 Reward Amendment Proposal - Threat-Weighted Clearance Reward

Docs-only amendment proposal. No code, no training. Authored after Jeff
approved the scope amendment of H5 plan section 7 to permit exactly one
bounded reward-shaping variant for H5 continuation.

## 1. Approval reference

Approval recorded on 2026-05-16:

> Approved: amend H5 section 7 to permit exactly one bounded
> reward-shaping variant for H5 continuation, threat-weighted clearance
> reward. Base `+1/step` survival reward remains preserved; shaping
> coefficient starts at `alpha = 0.05`. No target-environment change,
> no observation-channel change, no commercial-game scope change, no
> platform automation, no hyperparameter sweep before the first
> shaped-reward diagnostic, and no training in the docs-only amendment
> slice.

This proposal documents the technical design under that approval. It
does not authorize code. Code lands in a subsequent slice after this
doc is reviewed.

## 2. Rationale

The H5 behavior audit (`docs/h5-behavior-audit-evidence.md`) closed the
diagnostic question of why two distinct trained networks produced
indistinguishable in-game behavior. Both the `MlpPolicy` on state
observations and the `CnnPolicy/NatureCNN` on pixel observations
converge on the same policy: drive left, wedge against `x = 16`,
survive until a hazard happens to spawn near the left edge.

The audit identifies the proximate cause as the reward function, not
the perception channel, encoder, observation contract, entropy
coefficient, or training budget. Under `+1/step` survival reward, the
left wall is a gradient fixed point: `action = -1` is locally stable
once the agent clamps at `x = 16`, and any exploratory deviation from
that wall is high-variance and on average reward-negative because it
exposes the agent to right-side hazards. Wall-hugging is the gradient
descent's correct answer to the survival objective; it is not a
training pathology.

This proposal addresses the reward gradient. It adds a small per-step
bonus that distinguishes "near a hazard" from "far from any hazard",
so that the per-step reward at `x = 16` while a hazard is approaching
is strictly less than the per-step reward after moving away. The base
survival reward is preserved so survival remains the dominant signal;
the shaping term modulates only the marginal value of lateral position
when active threats exist.

## 3. Formula

Notation:

- `player_x`, `player_y`: current player position (player_y is fixed
  at `SCREEN_HEIGHT - SIZE = 508`).
- For each hazard `i` currently above or at the player
  (`hazard_y_i <= player_y`):
  - `vertical_distance_i = player_y - hazard_y_i` (non-negative)
  - `lateral_distance_i = abs(hazard_x_i - player_x)`
- `lookahead_band`: vertical range (px) over which an above-player
  hazard contributes to the shaping bonus. Initial value `270`
  (`SCREEN_HEIGHT / 2`).
- `safe_lateral_distance`: lateral distance (px) at which a hazard's
  contribution to clearance saturates at 1.0. Initial value `180`
  (`SCREEN_WIDTH / 4`).

- `alpha`: shaping coefficient. Initial value `0.05`.

Per-step computation:

```text
For each hazard i with hazard_y_i <= player_y:
    vertical_weight_i =
        clamp(1.0 - vertical_distance_i / lookahead_band, 0.0, 1.0)
    lateral_clearance_i =
        clamp(lateral_distance_i / safe_lateral_distance, 0.0, 1.0)

W = sum(vertical_weight_i)

if W > 0:
    clearance_bonus =
        alpha * (sum(vertical_weight_i * lateral_clearance_i) / W)
else:
    clearance_bonus = 0.0

non_terminal_step_reward = 1.0 + clearance_bonus
collision_terminal_reward = 0.0
truncation_terminal_reward = 0.0
```

Properties of the formulation:

- The shaping term is bounded in `[0.0, alpha]`. With `alpha = 0.05`,
  the per-step reward lies in `[1.0, 1.05]` non-terminal and stays at
  `0.0` on collision. Over a full 1800-step episode the maximum
  cumulative shaping contribution is `90.0`, against a base of `1800.0`.
  Survival therefore remains the dominant gradient signal.
- The shaping is `0.0` when no hazards are above the player (between
  spawns or in the brief windows after all live hazards have passed),
  so the base survival reward is unchanged in those frames.
- The weighting by `vertical_weight_i` is intentional: it makes
  imminent threats dominate the bonus, so the gradient encourages
  motion away from the highest-priority hazard rather than averaging
  uniformly across all visible hazards.

- Hazards below the player are excluded; they have already passed and
  cannot collide with the player. This matches the existing
  threat-priority filter in `_h3_sort_hazards_by_threat()`
  (`games/signal-dodge/scripts/main.gd`).
- `lateral_clearance_i` uses absolute lateral distance, not signed
  distance. The shaping rewards being far from hazards on either
  side; it does not encode a preferred direction.

## 4. Initial constants

| Constant                 | Initial value | Derivation                                                  |
| ------------------------ | ------------- | ----------------------------------------------------------- |
| `alpha`                  | `0.05`        | Per Jeff-approved scope amendment.                          |
| `lookahead_band`         | `270` px      | `SCREEN_HEIGHT / 2`. Hazards in upper half count as active. |
| `safe_lateral_distance`  | `180` px      | `SCREEN_WIDTH / 4`. ~6.4x the 28 px collision threshold.    |

Game geometry from `games/signal-dodge/scripts/main.gd` and `player.gd`:

- `SCREEN_WIDTH = 720`, `SCREEN_HEIGHT = 540`.
- Player half = 16 px; hazard half = 12 px.
- Player y is fixed at `SCREEN_HEIGHT - SIZE = 508` (`player.gd`).
- Player clamps in `x` to `[16, 704]`.
- Hazards spawn at `y = -24` and fall until `y > 564`.
- Collision alignment threshold (lateral) = `player_half + hazard_half
  = 28` px (`agent.gd`).

Initial values are derived from screen geometry, not tuned. The
implementation slice may revise them after a smoke run if the smoke
run produces a degenerate gradient (for example, if `lookahead_band`
is so wide that the bonus is non-zero almost every frame, or if
`safe_lateral_distance` is so small that the bonus saturates as soon
as the agent moves a few pixels off-center). Any revision must be
recorded in `config_effective.yaml` and reflected in this doc.

## 5. Implementation constraints

Binding for the H5 continuation slice that implements this reward:

1. **No wall-specific penalty.** No explicit penalty term that
   references `x = 16` or `x = 704`. The signal that wall-hugging is
   bad must come entirely from low clearance during an active threat.
2. **No future-collision oracle.** The shaping term is computed from
   the current frame's hazard positions only. No simulation of
   future hazard trajectories, no time-to-collision estimate fed
   back as reward.
3. **No target-environment change.** Same Signal Dodge profile, same
   hazard density, same player kinematics, same screen dimensions
   as the post-H4 baseline.
4. **No observation-channel change.** Pixel mode remains pixel mode;
   state mode remains state mode. Both can be run under the new
   reward separately. The reward is computed inside the env wrapper,
   not in the observation.
5. **No commercial-game scope change.** Charter ethics armor holds.
6. **No platform automation.** No new environments added in this
   slice.
7. **No hyperparameter sweep before the first shaped-reward
   diagnostic.** The first slice runs the existing entropy recipe
   (`ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`,
   per `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`) only,
   so the reward change is the single independent variable against
   Phase E.
8. **Reward components must be logged separately.** Each step must
   record `base_reward` (the `+1.0` or `0.0`) and `clearance_bonus`
   (the shaping term) as distinct fields in `python.ndjson`, so the
   evidence pass can separate base survival from shaping
   contribution without re-deriving.

9. **Reward computation lives Python-side.** The shaping bonus is
   computed in the env wrapper from the `state` dict the Godot side
   already exposes (player position, hazard list with id/x/y). The
   GDScript reward source remains the existing sparse-survival
   signal; the Python env layer adds the bonus before returning
   `(obs, reward, terminated, truncated, info)`. This keeps Godot
   side unchanged and makes the shaping ablate-able by config.
10. **Config-gated.** A new YAML key `reward_shaping` selects the
    shape; the default value remains `none` (preserving the H4 and
    Phase-A-through-E reward). Existing configs and existing runs
    must replay byte-identically with `reward_shaping: none`. The
    new shaped-reward config will set `reward_shaping:
    threat_weighted_clearance` and surface `alpha`,
    `lookahead_band`, `safe_lateral_distance` as tunable.
11. **No charter amendment.** This is an H5 plan amendment only.
    Charter mission, scope, non-goals, and ethics armor are
    unchanged.

## 6. Confirmation criteria

The reward-shape hypothesis is confirmed if the first shaped-reward
H5 diagnostic (3 seeds, 10k timesteps each, entropy recipe held
fixed, only `reward_shaping` changed from `none` to
`threat_weighted_clearance`) shows ALL of the following:

1. The trained `CnnPolicy` no longer converges to the left-wall
   fixed point:
   - `action = -1` does not exceed 60% of frames in representative
     episodes (down from 100% in Phase E seed 2).
   - representative episodes spend less than 50% of frames at
     `x = 16` (down from > 90% in Phase E seed 2).
   - `wall_hugging_into_collision` is no longer the dominant
     classified failure mode in the behavior audit replay on the
     new artifact.

2. H5 plan section 6 GREEN bar becomes plausible or clears on the
   shaped-reward artifact:
   - trained policy beats the best negative control by at least
     25% mean reward OR 25% mean episode length on the configured
     profile (where "reward" for the trained policy is interpreted
     as the sum of base survival reward plus clearance bonus, and
     for negative controls as base survival reward alone; the
     evidence summary must report both totals and the base-only
     comparator).
   - trained policy reduces collision rate by at least 20 percentage
     points vs the best negative control.
3. Same-seed behavior changes for the expected reason:
   - on the cross-policy same-seed pairs that previously produced
     identical lengths (1000, 1004, 1008), trajectories now diverge,
     with the shaped-reward trained model showing measurable lateral
     motion away from active hazards in `godot.ndjson`.
   - per-step `clearance_bonus` summaries show non-zero accumulation
     during hazard approach windows.

If criterion 1 holds but criterion 2 fails, the result is partial:
the reward has broken the local optimum but training budget,
hyperparameters, or further reward iteration may be needed for the
GREEN bar. That outcome justifies further work inside the amended
scope but does not retroactively justify additional reward variants
beyond the one approved here without a new amendment.

## 7. Falsification criteria

The reward-shape hypothesis is falsified, or at least downgraded, if
the first 3-seed 10k shaped-reward run produces all of the following
behavior-audit signatures:

- collision rate remains at or above 0.90 (i.e., within ~3 percentage
  points of the Phase E / state-comparator 0.933 baseline),
- representative episodes still wedge at `x = 16` for the majority of
  frames,
- `action = -1` still dominates policy output at > 80%,
- `wall_hugging_into_collision` remains the dominant classified
  failure mode.

If falsified, the next experiment lever is no longer reward shaping
inside this formulation. The remaining levers, in order of expected
information gain:

1. Game-state-dynamics sanity check (velocity observation, action
   timing, frame-stack contract, observation freshness). This is a
   Grok-trigger-worthy RL/Godot sanity check per charter phase-gate
   pattern.
2. A different bounded reward formulation (would require a new
   amendment, not an extension of this one).
3. Action space or step kinematics change (would require an H5
   scope revision, not an amendment).

A falsified result must not be followed by hyperparameter tuning of
the entropy recipe within the failed reward; that would conflate
"reward is wrong" with "training is wrong".

## 8. No-training-before-code-approval statement

This proposal is docs-only. No reward code is implemented in this
slice. No training runs are executed. The next operational step is
GPT-led design review of this proposal, followed by a Jeff approval
of the implementation slice (which lands the env-wrapper reward
computation, the config schema change, and the additional
`python.ndjson` logging fields). Training runs against the shaped
reward begin only after that implementation slice is committed and
tests are green.

## 9. What this proposal does not change

- Charter (`docs/sight-charter.md`) mission, scope, non-goals, ethics
  armor, role split, and phase-gate pattern remain unchanged.
- H5 plan sections 1 through 6, 8, 9, and 10 remain unchanged. Only
  section 7's reward-shaping exclusion is amended, narrowly.
- All H5 hard preconditions (transport literal pinning, per-reset
  `obs_metadata` persistence, default-tier pytest green, charter
  invariants) remain in force.
- The H5 GREEN bar (section 6) is unchanged in structure. Only the
  reward column for the trained policy gains an additional
  shaping-aware reporting requirement noted above.
- The H5 non-saturation rule (section 5) is unchanged.
- The four-policy baseline suite (section 3) is unchanged. The shaped
  reward applies to the trained `CnnPolicy` only; the three negative
  controls (stay-only, seeded random, untrained `CnnPolicy`) continue
  to be evaluated under their existing reward expectation (base
  `+1/step`), so their scores remain comparable across phases.

## 10. Open items routed to GPT for design review

These are not blockers to committing this proposal; they are the
specific items GPT should weigh before the implementation slice is
authorized.

- Whether the negative controls should also be re-evaluated under
  shaped reward for an apples-to-apples reward-axis comparison, in
  addition to the base-reward comparison above. The argument for:
  cleaner GREEN-bar math. The argument against: changing the
  negative controls retroactively re-bases every prior eval.

- Whether the initial `lookahead_band = 270` and
  `safe_lateral_distance = 180` should be smoke-tested with a single
  short un-trained run (random or stay-only) before training, to
  confirm the per-step bonus is not pathological at the chosen
  constants. A pre-training smoke is cheap and falls inside the
  no-training-before-code-approval rule once the code lands.
- Whether the implementation slice should also emit a per-step
  `active_hazard_count_above_player` field to make future audits of
  the shaping behavior easier without a re-derivation pass.
- Whether to add a regression test that fixes `reward_shaping: none`
  byte-equality of total reward against a stored Phase E artifact,
  to guarantee the default code path is byte-identical to the
  pre-amendment behavior.

End of proposal.
