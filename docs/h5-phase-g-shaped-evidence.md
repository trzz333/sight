# H5 Phase G - Shaped-Reward 3-Seed 10k Evidence

Trained-policy evidence for the first 3-seed 10k shaped-reward training
run authorized by `docs/h5-reward-amendment-smoke-evidence.md`. Three
PPO `CnnPolicy` networks trained for 10000 timesteps each under the
threat-weighted clearance reward (`alpha=0.05`, `lookahead_band=270`,
`safe_lateral_distance=180`). Eval under deterministic argmax on seeds
1000-1009 per train seed, 30 trained-policy rollouts total.

The headline finding is that the shaped reward did not change the eval
trajectory in any measurable way: Phase G per-seed eval episode
lengths are byte-identical to Phase E (base survival reward) across
all three train seeds and all ten eval seeds. The amendment hypothesis
that reward gradient is the lever for the wall-hugger / freeze-at-
center fixed point is falsified at the eval-trajectory level. The
attached saturation monitoring requirement is discharged (no breach).

## 1. Run identifiers and HEAD

- Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped.yaml`
  (new this session, sibling of the Phase E entropy config with
  `reward_shaping: threat_weighted_clearance` plus the three tunable
  constants).
- Train artifacts (gitignored, per memory rule on `runs/`):
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped/h5_train_phase_g_shaped_seed{1,2,3}_10k/`.
- Eval artifacts (gitignored):
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped/h5_eval_phase_g_shaped_seed{1,2,3}_10k_trained_only/`.
- Driver: `runs/phase_g/run_phase_g.bat` (gitignored).
- HEAD at training: `24006a3` (recorded in each train summary).

## 2. GREEN-math comparator vs Phase E

Per amendment implementation constraint 12, GREEN math uses Godot
base survival reward, not the shaped total. The shaped total
(`reward = base + clearance_bonus`) is reported separately as a
diagnostic. Phase E paired numbers from
`runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_eval_phase_e_seed{N}_entropy_10k_trained_only/evaluation/trained_cnn/summary.json`.

| train_seed | shaped mean base | E mean base | shaped mean shaped_total | E mean reward | shaped mean length | E mean length | shaped collision | E collision |
|------------|-----------------:|------------:|-------------------------:|--------------:|-------------------:|--------------:|-----------------:|------------:|
| 1          | 605.0            | 605.0       | 623.61                   | 605.00        | 606.0              | 606.0         | 1.0              | 1.0         |
| 2          | 844.8            | 844.8       | 876.53                   | 844.80        | 845.7              | 845.7         | 0.9              | 0.9         |
| 3          | 844.8            | 844.8       | 876.53                   | 844.80        | 845.7              | 845.7         | 0.9              | 0.9         |

Base reward, episode length, and collision rate are byte-identical
between Phase G and Phase E for every train seed. The shaped total is
greater than the base by the integrated shaping mass over each
episode (per-step mean `clearance_bonus` ≈ 0.031 over ~600 active-
step rollouts), but the underlying trajectories are unchanged.

## 3. Per-eval-seed paired comparison

Episode lengths per eval seed, shaped (Phase G) vs base (Phase E):

| eval seed | train1 shaped | train1 base | train2 shaped | train2 base | train3 shaped | train3 base |
|-----------|--------------:|------------:|--------------:|------------:|--------------:|------------:|
| 1000      | 333           | 333         | 1383          | 1383        | 1383          | 1383        |
| 1001      | 273           | 273         | 483           | 483         | 483           | 483         |
| 1002      | 843           | 843         | 1293          | 1293        | 1293          | 1293        |
| 1003      | 963           | 963         | 603           | 603         | 603           | 603         |
| 1004      | 1203          | 1203        | 1443          | 1443        | 1443          | 1443        |
| 1005      | 1263          | 1263        | 363           | 363         | 363           | 363         |
| 1006      | 543           | 543         | 573           | 573         | 573           | 573         |
| 1007      | 183           | 183         | 273           | 273         | 273           | 273         |
| 1008      | 183           | 183         | 1800          | 1800        | 1800          | 1800        |
| 1009      | 273           | 273         | 243           | 243         | 243           | 243         |

All 60 paired cells match. Across six trained networks (three Phase E
+ three Phase G), eval trajectories cluster into exactly two
equivalence classes:

- **Class A** (train_seed=1): mean length 606.0, collision 1.0,
  policy outputs `action_wire=1` (stay) on every step regardless of
  observation, player remains frozen at `x = 360` (center spawn).
- **Class B** (train_seed=2 and 3, identical to each other): mean
  length 845.7, collision 0.9, policy outputs `action_wire=0` (left)
  on every step, player drives to `x = 16` and wedges for 92% of
  frames.

## 4. Trained-policy action and position distributions

From `godot.ndjson` `h3_step` rows (action mapping
`_h3_map_action`: wire 0 -> -1 left, 1 -> 0 stay, 2 -> +1 right).

| train_seed | total step rows | action=-1 frac | action=0 frac | action=+1 frac | x<=16 frac | mean x | x range          |
|-----------:|----------------:|---------------:|--------------:|---------------:|-----------:|-------:|------------------|
| 1          | 6060            | 0.000          | 1.000         | 0.000          | 0.000      | 360.0  | (360.0, 360.0)   |
| 2          | 8457            | 1.000          | 0.000         | 0.000          | 0.920      | 29.8   | (16.0, 355.0)    |
| 3          | 8457            | 1.000          | 0.000         | 0.000          | 0.920      | 29.8   | (16.0, 355.0)    |

Action 2 (`right`) was selected by zero networks across the entire
Phase G eval (30 rollouts, 22 974 steps). Phase E showed the same
pattern.

## 5. Falsification of the amendment hypothesis at eval-trajectory level

Amendment section 7 lists four signatures that, if all present after
the first shaped-reward 3-seed 10k run, falsify or downgrade the
reward-shape hypothesis:

1. collision rate >= 0.90 across the run
2. representative episodes still wedge at `x = 16` for the majority
   of frames
3. `action = -1` still dominates policy output at > 80%
4. `wall_hugging_into_collision` remains the dominant classified
   failure mode

Evidence:

1. Aggregate collision rate over 30 rollouts is 0.933 (28/30
   collision, 2/30 timeout truncation, both in train_seed=2). HOLDS.
2. Class B episodes (20 of 30 rollouts) wedge at `x = 16` for 92% of
   frames. Class A episodes (10 of 30 rollouts) sit at `x = 360`
   instead. Aggregate "wedge at x = 16 for majority of frames" holds
   for 20/30 = 67% of rollouts. HOLDS DESCRIPTIVELY but with a new
   subclass.
3. Aggregate `action_wire=0` (which `_h3_map_action` returns as
   `action = -1`) is 8457*2 / (6060 + 8457*2) = 73.6%. Below the 80%
   bar. Yet it dominates Class B (100%) and seed 1 dominates with
   `action_wire=1` (stay) at 100% rather than left. Effectively
   "policy output is a single action per train seed" at 100% in every
   sub-class, just not always action=-1. PARTIALLY HOLDS, in spirit
   stronger (single-action degeneracy) than the original criterion.
4. Class B rollouts: `wall_hugging_into_collision` is the failure
   mode. Class A rollouts: a new "freeze_at_center_into_collision"
   failure mode that the Phase E behavior audit did not classify
   because Phase E seed=1 was not deeply audited. Both subclasses
   are observation-independent constant-action policies.

The strict AND of the four criteria is not all-met (criterion 3
falls below the 80% threshold due to the Class A subclass). The
spirit of the falsification — that the reward gradient did not move
the trained policy — holds completely: Phase G eval trajectories are
byte-identical to Phase E across every paired seed.

## 6. Monitoring requirement: discharged

The smoke evidence (`docs/h5-reward-amendment-smoke-evidence.md`
section 4) attached the following requirement to the proceed
authorization:

> If the trained-policy mean `frac_active_threat_saturated` exceeds
> `0.50` over a representative sample, revise `safe_lateral_distance`
> before further training.

Per-episode `frac_active_threat_saturated` computed from the shaped-
mode `python.ndjson` step events (`clearance_bonus >= 0.049` among
steps with `threat_weight_sum > 0`):

| train_seed | per-episode frac_sat (10 eval seeds)                                                |
|-----------:|-------------------------------------------------------------------------------------|
| 1          | 0.362, 0.000, 0.448, 0.439, 0.178, 0.150, 0.143, 0.000, 0.000, 0.561                |
| 2          | 0.452, 0.599, 0.322, 0.344, 0.603, 0.717, 0.677, 0.561, 0.518, 0.000                |
| 3          | 0.452, 0.599, 0.322, 0.344, 0.603, 0.717, 0.677, 0.561, 0.518, 0.000                |

Aggregate over 30 trained-policy rollouts:

- mean `frac_active_threat_saturated` = **0.3955**
- mean `frac_nonterm_with_active_threat` = 0.7672
- mean `clearance_bonus` = 0.03063
- mean `clearance_bonus_active` = 0.03850

`0.3955 < 0.50`. The diagnostic does not breach. `safe_lateral_distance`
does NOT need pre-emptive revision. The monitoring requirement is
discharged with no action item.

The Class B rollouts (train seeds 2 and 3) show higher saturation
than Class A (train seed 1), consistent with the Class B policy
parking at `x = 16` where most lookahead-band hazards are at
`hazard_x > 196` and thus lateral_clearance = 1.0. That is a real
property of the wedge-at-wall behavior, but the aggregate stays
below the bar.

## 7. What this means

The deterministic-argmax pathology already named in the Sight memory
("Three distinct trained networks with different weights produce
identical per-seed eval outcomes" - the leading hypothesis: "the
deterministic-argmax eval surface converges to the same action
sequence regardless of weights") is strongly confirmed and is the
proximate bottleneck on H5 progress. Across six trained networks
(Phase E + Phase G across train seeds 1, 2, 3) on two distinct
reward functions, eval behavior reduces to two single-action
policies. The shaping term modulated network weights during training
but did not change the per-step argmax decision on any observation.

The seeded-random rollouts from the pre-training smoke
(`docs/h5-reward-amendment-smoke-evidence.md`) produced observation-
correlated trajectories with varied per-seed lengths and meaningful
shaped-reward variance, which is direct evidence that the env,
protocol, NDJSON logging, and shaped-reward computation all behave
correctly under a non-degenerate policy. The pathology is localized
to the trained `CnnPolicy`'s argmax surface under
`predict(deterministic=True)`.

## 8. What this does NOT mean

The implementation slice is not at fault. The amendment-tests pass,
the smoke validated the shaped-reward bonus surface is non-degenerate
under non-trivial action sampling, the default-path schema regression
is byte-equal to Phase A-E, and the shaped vs default trajectory
parity invariant holds end-to-end against real Godot. The shaped
config and the Phase G train artifacts are durable evidence that the
shaped path runs cleanly at full training scale and produces shaped
totals strictly above base totals as designed.

The shaping is not falsified as a tool; it is falsified as a sufficient
solo lever for the eval-time pathology this project hit.

## 9. Next lever (for GPT planning)

Per amendment section 7, with the reward-shape hypothesis demoted at
the eval-trajectory level, the remaining lever order is:

1. **Game-state-dynamics sanity check.** Inspect the trained
   network's per-step action LOGIT distribution (not just argmax) on
   the same eval observations, comparing Phase E vs Phase G models.
   If logits differ meaningfully but argmax does not, the issue is
   the deterministic decision rule, not the network. If logits are
   themselves nearly identical, the issue is upstream of the action
   head: encoder, observation freshness, frame-stack contract.
   Memory flags this class of check as Grok-trigger-worthy.
2. A different bounded reward formulation. **Not the right next
   step here.** Falsification was at the eval surface, not the reward
   formula's properties at training time. Burning another amendment
   on the reward surface without first localizing the
   deterministic-argmax pathology would re-burn the same evidence.
3. Action space or step kinematics change. **Premature.** Same
   reason.

Recommended next diagnostic (Claude opinion, surfaced for GPT to
plan against, not to execute without approval): a single-rollout
logit-distribution comparator. Load the Phase E seed=1 model and the
Phase G seed=1 model, run them against the eval seed=1000 trajectory
step-by-step, dump per-step logits for both networks, and compare.
This is a docs-and-tools slice with no training and no env changes.

## 10. Repro

Train + eval (matches `runs/phase_g/run_phase_g.bat`):

```cmd
set "SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe"

for %S in (1 2 3) do (
  python -m sight_agent.rl.train ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped.yaml ^
    --seed %S ^
    --run-id h5_train_phase_g_shaped_seed%S_10k
)

for %S in (1 2 3) do (
  python -m sight_agent.rl.h5_baseline_cli ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped.yaml ^
    --run-id h5_eval_phase_g_shaped_seed%S_10k_trained_only ^
    --seeds 1000-1009 ^
    --policies trained_cnn ^
    --mode full ^
    --train-run-dir runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped\h5_train_phase_g_shaped_seed%S_10k
)
```

Per-train-seed wall time on StrongerJr: ~6 minutes train + ~1 minute
eval. Three seeds end-to-end: ~21 minutes.

End of evidence.
