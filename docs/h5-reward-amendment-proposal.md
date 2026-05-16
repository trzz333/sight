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

# Wire base reward is preserved exactly. The Python env wrapper reads
# base_reward = float(resp["reward"]) from the Godot step response and
# only modulates it when shaping is enabled. Godot's existing rule
# (games/signal-dodge/scripts/main.gd) is "reward = 0.0 if terminated
# else 1.0", so collision yields 0.0 and timeout/truncation yields 1.0;
# neither value is hard-coded on the Python side.

if shaping enabled and not terminated:
    reward = base_reward + clearance_bonus
else:
    reward = base_reward
```

Properties of the formulation:

- The shaping term is bounded in `[0.0, alpha]`. With `alpha = 0.05`,
  the per-step reward lies in `[1.0, 1.05]` for non-terminal steps,
  stays at `1.0 + clearance_bonus` on the truncation step (matching
  Godot's `+1.0` final non-collision step), and stays at `0.0` on the
  collision terminal step. Over a full 1800-step episode the maximum
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
8. **Reward components must be logged separately when shaping is
   enabled.** When `reward_shaping` is set to
   `threat_weighted_clearance`, each step records `base_reward`
   (the value Godot returned, `+1.0` or `0.0`), `clearance_bonus`
   (the shaping term in `[0, alpha]`), the final `reward`,
   `active_hazard_count_above_player`, and `threat_weight_sum`
   as distinct fields in `python.ndjson`, so the evidence pass can
   separate base survival from shaping contribution without
   re-deriving. When `reward_shaping` is `none`, these shaped-mode
   fields are not emitted; the default log shape is byte-identical
   to the pre-amendment behavior so the regression test in item 12
   can prove byte-equality against a freshly recomputed default-path
   reference rather than a stored Phase E artifact.

9. **Reward computation lives Python-side.** The shaping bonus is
   computed in the env wrapper from a per-step `reward_state` dict
   that the Godot side adds to step `info`: player position
   (`player_x`, `player_y`) and the list of hazards above the player
   (`{id, x, y}`). The GDScript reward source remains the existing
   sparse-survival signal; the Python env layer adds the bonus
   before returning `(obs, reward, terminated, truncated, info)`.
   This keeps the Godot reward path and physics unchanged and makes
   the shaping ablate-able by config. Adding `reward_state` under
   step `info` is a forward-compatible wire extension per
   `src/sight_agent/protocol.py`: `info` is required, but unknown
   sub-keys are tolerated by existing consumers.
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
12. **GREEN math uses base reward.** H5 plan section 6 GREEN bar
    comparisons (trained vs negative controls, 25% mean-reward
    threshold) use the Godot base survival reward, episode length,
    and collision rate. The shaped total (`base + clearance`) is
    reported separately as a training-diagnostic quantity, not as
    a free comparator advantage; including it in the GREEN math
    against base-only controls would overstate learning by the
    shaping mass alone.
13. **Regression test for default path.** A deterministic
    fake-transport/golden unit test must prove that
    `reward_shaping: none` returns the exact Godot-supplied reward,
    preserves termination and truncation handling, and produces a
    `python.ndjson` event schema byte-identical to the pre-amendment
    default path. The test does not depend on a committed Phase E
    artifact; `runs/` is gitignored and CI cannot rely on a local
    artifact.

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
     profile. GREEN math uses the **Godot base survival reward**
     for both the trained policy and the negative controls
     (apples-to-apples). The shaped total (base + clearance) is
     reported alongside as a training-diagnostic; including it in
     the GREEN bar against base-only controls would overstate the
     advantage by the shaping mass alone.
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

These were the items GPT weighed before authorizing the implementation
slice. Resolutions are recorded inline so the implementation contract
is traceable from this document alone.

- **Negative controls re-evaluated under shaped reward.** Resolved:
  **no**. Keeping `stay_only`, `seeded_random`, and `untrained_cnn`
  under base sparse-survival reward preserves comparability across
  phases. Re-running controls under shaped reward is permitted as a
  later analysis pass but is not a blocker for the first shaped
  diagnostic. The H5 plan baseline suite (section 3) is unchanged.
- **Pre-training smoke at the initial constants.** Resolved: **yes**,
  blocking before the first shaped training run. After the
  implementation slice commits and tests pass, run a no-training
  smoke with `lookahead_band = 270`, `safe_lateral_distance = 180`,
  `alpha = 0.05` using a stay-only or seeded-random / untrained
  policy. The smoke must verify `clearance_bonus` stays in
  `[0.0, 0.05]`, is not always zero, is not saturated near `0.05`
  almost every frame, appears during active-hazard windows, and that
  `reward_shaping: none` still behaves identically. The smoke uses
  no training and therefore falls inside the no-training-before-
  code-approval rule.
- **`active_hazard_count_above_player` log field.** Resolved:
  **yes**, plus `threat_weight_sum`. Both are cheap audit fields
  that remove future re-derivation; both are emitted only when
  `reward_shaping` is `threat_weighted_clearance`. See implementation
  constraint 8.
- **Regression test for `reward_shaping: none` byte-equality.**
  Resolved: **yes, but not against a stored Phase E artifact.**
  `runs/` is gitignored and CI cannot depend on a local artifact.
  Use a deterministic fake-transport/golden unit test that proves
  default-path reward, termination, truncation, and `python.ndjson`
  schema are unchanged. See implementation constraint 13.

End of proposal.
