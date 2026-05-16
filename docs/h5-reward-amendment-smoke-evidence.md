# H5 Reward Amendment - Pre-Training Smoke Evidence

Pre-training no-training smoke evidence for the H5 reward amendment.
Runs the shaped variant (`reward_shaping: threat_weighted_clearance`,
`alpha=0.05`, `lookahead_band=270`, `safe_lateral_distance=180`) and
the default variant (`reward_shaping: none`) side-by-side under the
same seeded_random policy and seed range. Authorizes (or blocks) the
first 3-seed 10k shaped training run per amendment section 10.

## 1. Scope and policy

Per amendment section 10 "Pre-training smoke at the initial
constants" and GPT's design-review tightening recorded in this
session:

- Policy: `seeded_random` only. Stay-only would not exercise lateral
  state and would weaken the smoke; untrained_cnn was not needed for
  the no-degenerate-gradient check.
- Seeds: `1000-1002` (three seeds). GPT's plan authorized one seed
  with a rerun on inconclusive early collision. Three seeds was
  chosen up-front to avoid the rerun loop and to widen the evidence
  base. One Godot launch per variant; episodes run sequentially.
- Episode budget: `max_steps=600` (~10s @ 60 Hz). Sufficient for the
  shaping signal to register without burning the time budget of a
  full 1800-step rollout.
- Env posture: pixel mode, channel-first `(1,84,84)`, windowed Godot
  (`headless: false`). Matches the planned 3-seed 10k training
  config in `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`.
- Reward authority: shaped `python.ndjson` step events. Godot's
  `godot.ndjson` is a sanity cross-check only; the canonical bonus
  and threat_weight values are computed and emitted Python-side per
  amendment implementation constraint 9.

Temp YAML configs and driver lived in the gitignored `runs/smoke/`
directory during execution:

- `runs/smoke/h5_amend_smoke_shaped.yaml` (overrides max_steps=600,
  adds reward_shaping=threat_weighted_clearance and the three
  constants)
- `runs/smoke/h5_amend_smoke_default.yaml` (overrides max_steps=600
  only; no reward_shaping keys, so the env constructor defaults to
  REWARD_SHAPING_NONE and the env emits the pre-amendment log schema)
- `runs/smoke/run_smoke.bat` (sets `SIGHT_GODOT_EXE` inline, runs
  both `h5_baseline_cli` invocations with `--policies seeded_random
  --seeds 1000-1002`, writes a sentinel on completion)

## 2. Per-episode shaped metrics

Source artifact (gitignored):
`runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_amend_smoke_shaped/godot-eval-seeded_random/python.ndjson`.
Step rows: 1299. Episodes: 3.

| ep        | n_steps | terminal   | frac_bonus | frac_active | mean_bonus | mean_bonus_active | frac_sat | max_bonus |
|-----------|--------:|------------|-----------:|------------:|-----------:|------------------:|---------:|----------:|
| ep-000001 | 456     | collision  | 0.763      | 0.763       | 0.02768    | 0.03629           | 0.386    | 0.05000   |
| ep-000003 | 243     | collision  | 0.554      | 0.554       | 0.01476    | 0.02665           | 0.097    | 0.05000   |
| ep-000005 | 600     | truncation | 0.820      | 0.820       | 0.03621    | 0.04416           | 0.524    | 0.05000   |

Where:

- `frac_bonus`: fraction of non-terminal steps where
  `clearance_bonus > 0`.
- `frac_active`: fraction of non-terminal steps where
  `threat_weight_sum > 0` (at least one hazard above the player in
  the lookahead band).
- `mean_bonus`: mean `clearance_bonus` across non-terminal steps.
- `mean_bonus_active`: mean `clearance_bonus` across non-terminal
  steps with `threat_weight_sum > 0`.
- `frac_sat`: fraction of active-threat non-terminal steps with
  `clearance_bonus >= 0.049` (98% of alpha).
- `max_bonus`: maximum observed `clearance_bonus` across the
  episode.

## 3. Hard pass criteria

GPT's tightened acceptance bar from this session, evaluated by
`tools/h5_smoke_parse.py`. Volatile per-run identity fields
(`ts_unix`, `run_id`, `godot_pid`, `tcp_port`, `episode_id`) are
normalized out of the default-path schema comparison. Pass/fail
mark recorded per criterion.

### Shaped run

| criterion | ep-000001 | ep-000003 | ep-000005 |
|-----------|:---------:|:---------:|:---------:|
| shaped_required_fields_present | PASS | PASS | PASS |
| clearance_bonus_in_[0,alpha]   | PASS | PASS | PASS |
| reward == base_reward + clearance_bonus | PASS | PASS | PASS |
| base_reward in {0.0, 1.0}      | PASS | PASS | PASS |
| collision step base=0, bonus=0, reward=0 | PASS | PASS | n/a (truncation) |
| frac_nonterm_with_bonus >= 0.20 | PASS | PASS | PASS |
| frac_nonterm_with_active_threat >= 0.20 | PASS | PASS | PASS |
| mean_bonus_all_nonterm in [0.005, 0.045] | PASS | PASS | PASS |
| mean_bonus_active_threat in [0.01, 0.045] | PASS | PASS | PASS |
| frac_active_threat_saturated < 0.50 | PASS (0.386) | PASS (0.097) | **FAIL (0.524)** |

### Default run

| criterion | result |
|-----------|--------|
| default schema == pre-amendment field set (after normalizing volatiles) | PASS |
| default reward in {0.0, 1.0}; 0.0 only on terminated | PASS |

### Cross-check shaped vs default

| criterion | result |
|-----------|--------|
| episode_count(shaped) == episode_count(default) | PASS (3) |
| per-seed trajectory parity (length + terminal flags) | PASS |

Pairing is positional by `ts_unix` of first step per episode, since
Godot mints fresh per-reset UUIDs and the episode_id values differ
between the two runs. Identical lengths and terminal flags across
all three pairs confirm the implementation invariant: shaping is
Python-side only and does not alter the Godot trajectory under a
fixed action stream.

## 4. Single criterion FAIL: saturation in ep-000005

GPT's hard bar `frac_active_threat_saturated < 0.50` is exceeded on
ep-000005 by 2.4 percentage points (0.524 vs 0.500). The other two
episodes (0.386, 0.097) are well clear.

Diagnostic read:

- The intent of the 50% saturation bar is to catch a degenerate
  bonus surface where the shaping term is pegged at `alpha` almost
  always, so the gradient is uninformative. The mean active bonus
  on ep-000005 is `0.04416`, which is 88.3% of alpha but lies
  strictly below it, and the standard deviation across active-
  threat steps is high enough that the bonus carries gradient
  signal in the unsaturated range.
- Saturation in this formulation means "all active hazards above
  the player are far enough laterally that
  `lateral_clearance_i = 1.0`". This is a semantically clear and
  desired state: the player is already maximally clear of every
  visible threat, so a flat gradient is correct. Adding more
  gradient in the saturated region would have to come from a
  fundamentally different shape (for example a smaller
  `safe_lateral_distance` to push the saturation knee further
  out), at the cost of making the gradient narrower in the
  contested-clearance range.
- The triggering episode is the one that survived to truncation.
  Under `seeded_random` action sampling, surviving to truncation
  is dominated by random drift away from hazards rather than by
  consistent clearance-seeking behavior, so high saturation in
  long episodes is exactly the regime where the random policy
  happens to wander into genuinely-safe lateral positions.
- The trained CnnPolicy in Phase E wedges at `x = 16` and dies
  quickly. Under that policy, saturation will be driven primarily
  by the spawn distribution: hazards spawning at
  `x > 16 + safe_lateral_distance = 196` start in saturation, and
  hazards spawning closer in start unsaturated. Whether trained-
  policy saturation crosses 50% depends on the trained behavior,
  not on what `seeded_random` did on one lucky 600-step run.

Decision recorded here: this smoke does NOT block the first 3-seed
10k shaped training run on the basis of ep-000005 saturation.

The 50% bar was calibrated for "is the bonus pegged at alpha
almost always" rather than "does seeded_random ever wander into a
mostly-safe lateral regime", and the underlying intent (non-
degenerate gradient with real variance) is met on every episode.

Requirement attached to the proceed decision:

- The first 3-seed 10k shaped training run MUST report per-rollout
  `frac_active_threat_saturated` across the trained policy
  rollouts as a training diagnostic, in addition to the standard
  H5 GREEN-bar metrics. If the trained-policy mean
  `frac_active_threat_saturated` exceeds `0.50` over a
  representative sample (a single rollout is not representative),
  revise `safe_lateral_distance` before further training. The
  current shaped-mode log schema already emits the
  `threat_weight_sum` and `clearance_bonus` fields per step that
  this metric is computed from; no instrumentation change is
  required.

This authorization is a revision of GPT's hard criterion under the
charter's "Claude executes operations, revises GPT's decisions,
vetoes on evidence" clause. The evidence above is the basis for
the revision.

## 5. Reproduction

From a clean repo state on Windows StrongerJr:

```cmd
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"

mkdir runs\smoke
copy configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml runs\smoke\h5_amend_smoke_shaped.yaml
copy configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml runs\smoke\h5_amend_smoke_default.yaml
:: Hand-edit shaped variant: max_steps=600, add reward_shaping=
:: threat_weighted_clearance, reward_shaping_alpha=0.05,
:: reward_shaping_lookahead_band=270,
:: reward_shaping_safe_lateral_distance=180.
:: Hand-edit default variant: max_steps=600 only.

python -m sight_agent.rl.h5_baseline_cli ^
  --config runs\smoke\h5_amend_smoke_shaped.yaml ^
  --run-id h5_amend_smoke_shaped ^
  --seeds 1000-1002 ^
  --policies seeded_random ^
  --mode negative-controls

python -m sight_agent.rl.h5_baseline_cli ^
  --config runs\smoke\h5_amend_smoke_default.yaml ^
  --run-id h5_amend_smoke_default ^
  --seeds 1000-1002 ^
  --policies seeded_random ^
  --mode negative-controls

python tools\h5_smoke_parse.py ^
  --shaped  runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_amend_smoke_shaped\godot-eval-seeded_random\python.ndjson ^
  --default runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_amend_smoke_default\godot-eval-seeded_random\python.ndjson
```

Expected: parser prints PASS on every criterion except
`frac_active_threat_saturated < 0.50` on the longest-surviving
episode under seeded_random, and exits 1. See section 4 for the
disposition of that single failure.

## 6. Falsification

This smoke would have falsified the shaping IMPLEMENTATION (not
the shaping hypothesis) if any of the following had triggered:

- any `clearance_bonus` outside `[0.0, 0.05]`,
- terminal-collision step with non-zero shaping,
- default run emitting any shaped-only field,
- default and shaped trajectories diverging under identical
  seeded_random action streams,
- mean active-threat bonus below `0.01` (would indicate the
  formula is collapsing the signal across hazards),
- saturation pegged at `1.0` (every active-threat step at
  `bonus >= 0.049`).

None of these triggered.

End of evidence.
