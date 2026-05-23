# K5.5 State-Observation PPO Control - Evidence

## Verdict

`STATE-CONTROL-FAIL-ACTIVE-BAD`

0 of 3 train seeds clear the 930.27 material survival bar. Pooled
non-stay action fraction 0.411 (>= 0.20). Pooled collision rate 0.967
(>= 0.80). Mechanical run clean (`mechanical_ok: true`).

Interpretation: handing the policy the production 10-dim hazard
geometry directly, with a memoryless MlpPolicy under the K5.1-matched
budget and shaping, does NOT rescue the agent. The pixel representation
is not the primary blocker. The blocker is upstream of perception:
PPO objective, budget, credit assignment, or reward geometry.

Caveat on the "ACTIVE" label: the bucket name overstates the policies.
All three seeds converged to degenerate single-action policies, not to
moving-but-misaligned policies. seed0 is constant-left, seed1 and seed2
are constant-stay. The 0.411 pooled non-stay fraction is entirely
seed0's constant left; pooled right fraction is 0.0. No seed produces
hazard-conditioned or bidirectional motion. K5.6 scoping must not
assume an active policy that needs steering corrected.

## Run identity

- Config: `configs/rl/signal_dodge_ppo_h5_state_shaped_alpha030.yaml`
- Train run dirs (under `runs/rl/signal_dodge_ppo_h5_state_shaped_alpha030/`):
  - `k5_5_state_alpha030_seed0_10k`
  - `k5_5_state_alpha030_seed1_10k`
  - `k5_5_state_alpha030_seed2_10k`
- Model fingerprints (SHA-256, param_count):
  - seed0: `6cff8486e70b...` 9988 params
  - seed1: `bf4153320019...` 9988 params
  - seed2: `15e8ab2b5d7f...` 9988 params
  - Three distinct trained networks (distinct SHA-256).
- Train seeds: 0, 1, 2. Eval seeds: 1000-1009. Max steps: 1800.
- Reward scale verification: all three summary.json record
  `reward_scale_divisor: 30.0`, `reward_scale_applied: true`,
  `status: ok`, `total_timesteps: 10000`. config_effective confirms
  `observation_mode: state`, `policy: MlpPolicy`,
  `reward_shaping_alpha: 0.30`.
- Observation schema: production 10-dim state vector, unchanged. No
  new env schema. No raw player_y, no explicit arrival_steps, no
  hazard velocity, no frame_stack.

## Per-train-seed eval table

| seed | mean len | median | min | max | collision | timeout | reward mean | L / S / R action frac | player_x min/mean/max |
|---|---|---|---|---|---|---|---|---|---|
| seed0 | 845.7 | 588.0 | 243 | 1800 | 0.90 | 0.10 | 1035.20 | 1.000 / 0.000 / 0.000 | 16.0 / 136.3 / 355.0 |
| seed1 | 606.0 | 303.0 | 183 | 1263 | 1.00 | 0.00 | 716.66 | 0.000 / 1.000 / 0.000 | 360.0 / 360.0 / 360.0 |
| seed2 | 606.0 | 303.0 | 183 | 1263 | 1.00 | 0.00 | 716.66 | 0.000 / 1.000 / 0.000 | 360.0 / 360.0 / 360.0 |

Non-stay fraction: seed0 1.000, seed1 0.000, seed2 0.000.

Per-eval-seed episode lengths:

- seed0 (constant-left): 1383, 483, 1293, 603, 1443, 363, 573, 273,
  1800, 243. One timeout (eval seed 1008, survived to 1800).
- seed1 (constant-stay): 333, 273, 843, 963, 1203, 1263, 543, 183,
  183, 273.
- seed2 (constant-stay): 333, 273, 843, 963, 1203, 1263, 543, 183,
  183, 273.

seed1 and seed2 produce bit-identical per-eval-seed lengths despite
distinct model weights. See "Bit-identical anomaly resolution" below.

## Pooled table

- n episodes: 30 (3 models x 10 eval seeds)
- episode length: mean 685.9, median 543.0, min 183.0, max 1800.0
- collision rate: 0.967 (29 of 30 episodes ended in collision)
- timeout rate: 0.033 (1 of 30)
- action fractions: left 0.411, stay 0.589, right 0.000
- non-stay action fraction: 0.411
- success_seed_count (mean len >= 930.27): 0 of 3

Classification inputs: success_seed_count 0; pooled_collision_rate
0.967; pooled_stay 0.589; pooled_nonstay 0.411; pooled_left 0.411;
pooled_right 0.000. First matching bucket in packet order:
`STATE-CONTROL-FAIL-ACTIVE-BAD` (success_seed_count == 0,
pooled_nonstay >= 0.20, pooled_collision >= 0.80; FAIL-STAY rejected
because pooled_stay 0.589 < 0.90).

## Comparison anchors

- K5.1 pixel alpha030 deterministic: mean 606.0, collision 1.00.
  K5.5 seed1 and seed2 reproduce exactly 606.0 / collision 1.00 with
  constant-stay; the K5.1 stay-pinned failure recurs under state obs.
- K5.2 best constant: 845.7. K5.5 seed0 (constant-left) scores
  exactly 845.7; it learned the best constant action and nothing more.
- Material survival bar: 930.27. No seed clears it.
- K5.2 / K5.4 hazard oracle: 1762.8. K5.5 best seed mean 845.7,
  a 917-step gap below the oracle.

## Bit-identical anomaly resolution

The handoff active-anomaly (distinct trained weights producing
identical per-seed eval outcomes, hypothesized as either an eval
pipeline defect or a deterministic-argmax convergence) is resolved by
K5.5 in favor of the second hypothesis. seed1 and seed2 are distinct
networks (distinct SHA-256) and produce bit-identical per-eval-seed
lengths. seed0 is a distinct argmax behavior and produces a distinct
length set. Distinct weights yield identical eval if and only if they
converge to the same deterministic-argmax policy. The eval pipeline is
not defective; it is using the loaded model. Bit-identical eval is a
symptom of degenerate convergence to identical constant policies, not
a measurement bug.

## Routing

`STATE-CONTROL-FAIL-ACTIVE-BAD` routes K5.6 to a PPO objective,
reward-geometry, and credit-assignment audit. Not frame_stack. Not a
CNN feature-extractor change. State observation has exonerated
single-frame pixels as the sole blocker: perfect low-dimensional
hazard geometry produces the same degenerate-constant-policy failure.

Open scoping question for GPT: whether the next lever is reward
geometry (the threat_weighted_clearance shaping may be satisfiable by
a constant action), the 10k budget, or the PPO credit-assignment path.
The constant-left convergence of seed0 scoring exactly the K5.2 best
constant suggests the objective is being optimized correctly toward a
degenerate optimum, which points at reward geometry over budget.
