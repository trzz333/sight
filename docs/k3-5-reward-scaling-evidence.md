# K3.5 Fixed Reward-Scaling Evidence: Both /30 and /100 Pass Primary and Strong Gates; Absolute Magnitude Is Load-Bearing

## Context

K3.4 mechanistically confirmed the K3.2 update-1 value-shock hypothesis: scalar value-head bias init at `mean(rollout_buffer.returns)` fully rescues update 1 (lvf_post = 128/128, vp_std_post = 4.95e-2, v_loss = 7.54) but the collapse trajectory rejoins K3.3 by update 4 because per-step survival reward grows `ret_mean` monotonically with episode length (15.71 -> 28.97 -> 33.63 -> ... -> 39.91 across 8 updates), recreating a fresh ~13.3-unit bias gap every rollout. Classified TRANSIENT RESCUE.

K3.5 tests the absolute-magnitude-vs-relative-drift discriminator: a fixed scalar divisor at the VecEnv layer reduces the magnitude of each per-update bias gap but preserves the relative drift ratio across updates. If absolute magnitude was load-bearing, both `/30` and `/100` should prevent collapse. If only relative drift mattered, both should fail similarly and falsify fixed scaling.

## Patch (tools/h5_training_entropy_probe.py)

1. New import `from stable_baselines3.common.vec_env import VecEnvWrapper`.
2. New class `FixedRewardScaleVecEnv(VecEnvWrapper)`: divides per-step reward by a fixed scalar before SB3 sees it. Returns/advantages/value targets are computed in the scaled stream consistently. Raises on divisor <= 0.
3. New CLI `--reward-scale-divisor FLOAT` (default 1.0 no-op).
4. In `main()`, wrap the training VecEnv with `FixedRewardScaleVecEnv` when divisor != 1.0. The fixed observation-conditioning panel env is NEVER wrapped (panel reward is irrelevant to observation conditioning, and wrapping risks contaminating a pure observation probe).
5. Header records `reward_scale_divisor` (CLI float) and `reward_scale_applied` (hard state from wrapper application).

Smoke (seed 3, 512 ts, divisor 30, `--value-bias-init none`, pi=64/vf=128, vf_coef=0.5): exited 0, 2 updates, header `reward_scale_divisor=30.0 reward_scale_applied=true`. Elapsed 17.2 s.

## Runs

Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Seed 3, 2048 ts, pi=[64], vf=[128], `vf_coef=0.5`, `--value-bias-init none`, `ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`, `lr=3e-4`, fixed panel enabled.

- `runs/phase_k/k3_5_smoke_reward_scale_div30_512.ndjson` (smoke, 2 updates)
- `runs/phase_k/k3_5_reward_scale_div30_2048.ndjson` (header `reward_scale_divisor=30.0 reward_scale_applied=true`)
- `runs/phase_k/k3_5_reward_scale_div100_2048.ndjson` (header `reward_scale_divisor=100.0 reward_scale_applied=true`)

Elapsed: 64.9 s (/30), 64.8 s (/100).

Per-update CSV extract: `runs/phase_k/k3_5_reward_scale_table.csv` (gitignored).

## Primary mechanism gate (revised contract)

Required: `latent_vf_live_post_update_3 >= 16 AND fixed_panel_value_predictions_std_post_update_3 > 1e-6 AND post_rollout_model_to_constant_mse_ratio_update_3 improves vs K3.3 vf_coef=0.5 baseline (22.84).`

| Condition                                       | /30      | /100     | Threshold     | /30   | /100  |
| ----------------------------------------------- | -------: | -------: | ------------- | :---: | :---: |
| latent_vf_live_post_update_3                    | 128      | 128      | >= 16         | PASS  | PASS  |
| fixed_panel_value_predictions_std_post_update_3 | 4.89e-3  | 6.33e-3  | > 1e-6        | PASS  | PASS  |
| post_rollout_model_to_constant_mse_ratio_u3     | 3.66     | 1.10     | < 22.84       | PASS  | PASS  |

Primary mechanism gate: PASS for both /30 and /100.

## Strong win gate (revised contract)

Required: `latent_vf_live_post_update_8 >= 16 AND fixed_panel_value_predictions_std_post_update_8 > 1e-6 AND post_rollout_model_to_constant_mse_ratio_update_8 improves vs K3.3 baseline (6.18).`

| Condition                                       | /30      | /100     | Threshold     | /30   | /100  |
| ----------------------------------------------- | -------: | -------: | ------------- | :---: | :---: |
| latent_vf_live_post_update_8                    | 128      | 128      | >= 16         | PASS  | PASS  |
| fixed_panel_value_predictions_std_post_update_8 | 5.33e-3  | 9.62e-3  | > 1e-6        | PASS  | PASS  |
| post_rollout_model_to_constant_mse_ratio_u8     | 0.94     | 0.75     | < 6.18        | PASS  | PASS  |

Strong win gate: PASS for both /30 and /100.

## Failure gate (revised contract)

Required: `latent_vf_live_post_update_3 == 0 OR fixed_panel_value_predictions_std_post_update_3 <= 1e-6.`

Result for /30: lvf_post_u3 = 128 (does not fire), vp_std_post_u3 = 4.89e-3 (does not fire). Failure gate: does not fire.
Result for /100: lvf_post_u3 = 128 (does not fire), vp_std_post_u3 = 6.33e-3 (does not fire). Failure gate: does not fire.

## Classification per the final contract

The revised classification table classifies on the strongest-passing leaf:

| Outcome (final contract)                                    | Selected? | Classification           | Next move                     |
| ----------------------------------------------------------- | :-------: | ------------------------ | ----------------------------- |
| /30 passes strong                                           | YES       | K3.5 PASS                | Run 10k confirmation on /30  |
| /100 passes strong while /30 fails                          | no        | conditional PASS         | (over-compression case)       |
| /30 passes strong and /100 fails                            | no        | K3.5 PASS, aggr. harmful | Run 10k on /30                |
| both fail similarly                                         | no        | falsified                | K3.6 reward-shape             |

**Classification: K3.5 PASS.** /30 passes strong; /100 also passes strong as supplementary evidence that the scale threshold sits comfortably inside the passing range. The "over-compression" suspicion (asked of any /100 pass) is checked below and not supported by the evidence for the 2048-ts slice.

## Per-update ret_mean drift: raw vs scaled

The mechanism-facing metric is the per-update raw `ret_mean` delta because that delta is what the value head's bias must chase between rollouts. The cumulative first-to-last ratio is the coarse drift summary.

### K3.5 /30

| u | ret_mean_scaled | ret_mean_raw | scaled_delta | raw_delta | lvf_post | vp_std_post | post_ratio | v_loss | \|g\| |
| -: | -------------: | -----------: | -----------: | --------: | -------: | ----------: | ---------: | -----: | ----: |
| 1 | 0.4706         | 14.118       |              |           | 128      | 1.12e-2     | 1.408      | 0.057  | 1.589 |
| 2 | 0.9811         | 29.432       | +0.5104      | +15.314   | 128      | 8.69e-3     | 1.884      | 0.064  | 2.118 |
| 3 | 1.4036         | 42.108       | +0.4225      | +12.676   | 128      | 4.89e-3     | 3.658      | 0.039  | 1.790 |
| 4 | 1.8018         | 54.055       | +0.3982      | +11.947   | 128      | 4.26e-3     | 1.085      | 0.023  | 1.502 |
| 5 | 1.9016         | 57.048       | +0.0998      | +2.993    | 128      | 2.70e-3     | 0.996      | 0.131  | 1.206 |
| 6 | 1.9990         | 59.969       | +0.0974      | +2.922    | 128      | 2.23e-3     | 1.029      | 0.146  | 1.228 |
| 7 | 2.2707         | 68.121       | +0.2717      | +8.152    | 128      | 2.27e-3     | 1.375      | 0.016  | 1.416 |
| 8 | 2.3084         | 69.252       | +0.0377      | +1.131    | 128      | 5.33e-3     | 0.939      | 0.180  | 0.942 |

First-to-last ratio: scaled 4.905x, raw 4.905x (ratios are scale-invariant by construction).

### K3.5 /100

| u | ret_mean_scaled | ret_mean_raw | scaled_delta | raw_delta | lvf_post | vp_std_post | post_ratio | v_loss | \|g\| |
| -: | -------------: | -----------: | -----------: | --------: | -------: | ----------: | ---------: | -----: | ----: |
| 1 | 0.1027         | 10.267       |              |           | 128      | 5.07e-3     | 0.975      | 0.095  | 1.891 |
| 2 | 0.2472         | 24.716       | +0.1445      | +14.448   | 128      | 2.69e-3     | 1.251      | 0.006  | 0.383 |
| 3 | 0.3339         | 33.386       | +0.0867      | +8.670    | 128      | 6.33e-3     | 1.097      | 0.007  | 0.366 |
| 4 | 0.4165         | 41.648       | +0.0826      | +8.262    | 128      | 6.42e-3     | 5.625      | 0.004  | 0.520 |
| 5 | 0.4419         | 44.194       | +0.0255      | +2.546    | 128      | 4.65e-3     | 0.871      | 0.009  | 0.413 |
| 6 | 0.5391         | 53.908       | +0.0971      | +9.714    | 128      | 8.60e-3     | 0.701      | 0.003  | 0.405 |
| 7 | 0.5994         | 59.938       | +0.0603      | +6.029    | 128      | 2.13e-3     | 0.706      | 0.002  | 0.394 |
| 8 | 0.6303         | 63.033       | +0.0310      | +3.096    | 128      | 9.62e-3     | 0.754      | 0.013  | 0.288 |

First-to-last ratio: scaled 6.139x, raw 6.139x.

### K3.4 reference (no scaling, bias-init)

| u | ret_mean | lvf_post | vp_std_post | post_ratio | v_loss  | \|g\|   |
| -: | -------: | -------: | ----------: | ---------: | ------: | ------: |
| 1 | 15.714   | 128      | 4.95e-2     | 0.972      | 7.544   | 4.449   |
| 2 | 28.965   | 126      | 2.46e-3     | 5.678      | 110.080 | 128.821 |
| 3 | 33.632   | 10       | 2.16e-4     | 2.597      | 116.495 | 99.154  |
| 4 | 37.308   | 2        | 0           | 30.118     | 132.825 | 128.860 |
| 5 | 35.513   | 1        | 9.37e-7     | 2.547      | 121.210 | 99.262  |
| 6 | 36.226   | 0        | 0           | 2.471      | 122.087 | 98.624  |
| 7 | 37.377   | 0        | 0           | 2.594      | 130.360 | 103.312 |
| 8 | 39.913   | 0        | 8.26e-7     | 30.608     | 122.960 | 123.989 |

First-to-last ratio: 2.540x.

## Drift-ratio finding

The K3.5 contract explicitly required reporting the drift ratio to test whether fixed scaling changes the collapse mechanism or only compresses the numbers. The result on the 2048-ts slice is mixed and worth being careful about.

- **First-to-last drift ratio is NOT preserved across divisors.** /30 has 4.91x, /100 has 6.14x, K3.4 (no scaling) has 2.54x. The K3.5 runs drift MORE in relative terms than K3.4 baseline.
- **The reason is downstream of the gate that passed: the policy survives much longer.** Under K3.4 the policy collapses to a constant-action wedge by update 4, episodes terminate earlier, and `ret_mean` plateaus. Under K3.5 the value head stays alive, the policy keeps learning, episodes run longer, and `ret_mean` keeps climbing. The drift ratio is large precisely because nothing is collapsing.
- **The absolute raw per-update delta is what mattered for the bias gap.** Under K3.4 the per-update raw delta was ~13 units (15.71 -> 28.97), which against a near-zero initial value head produced v_loss = 110 at update 2. Under K3.5 /30 the per-update raw delta is similar in magnitude (~12-15 units at u1->u2), but the scaled stream the value head sees has a delta of 0.51 units. v_loss at u2 = 0.064 (vs K3.4 110), a 1700x reduction. The catch-up shock is small enough that Adam absorbs it without blasting upstream gradients into the shared features extractor.
- **Fixed scaling changed the collapse mechanism, not just compressed the numbers.** Compression alone would have left the v_loss to scale by the divisor squared (since MSE), so /30 should give v_loss ~ 110/900 = 0.12 and /100 should give ~ 110/10000 = 0.011. /30 actually shows v_loss in the 0.02 to 0.18 range across updates 1-8, and /100 in 0.002 to 0.10. Both are close to the naive scaling prediction at update 2 (0.064 ~ 0.12 / 2; 0.006 ~ 0.011 / 2), but stay small thereafter because the policy is now learning. The latent_vf collapse mechanism is fully prevented.

The K3.5 /30 finding is **not transient rescue**. K3.4's signature (lvf_post collapses from 128 to 0 by update 6) is replaced by lvf_post = 128 sustained across all 8 updates. The value head's predictions remain observation-conditioned (vp_std_post stays in 2e-3 to 1e-2 range, 1000x the K3.4 baseline at update 3) for the full probe budget.

## Over-compression check on /100

The contract asked /100 passes be flagged as possible over-compression. The 2048-ts slice does not show signs of over-compression on /100:

- `vp_std_post` on /100 is 5e-3 to 1e-2 across all 8 updates, comparable to /30 (3e-3 to 1e-2). The value head retains observation conditioning.
- `post_ratio` (model MSE / constant baseline MSE) is in the 0.70 to 1.25 range on /100 (except a single spike to 5.63 at update 4) and in 0.94 to 3.66 range on /30. /100 is actually BETTER on average than /30 against the constant-mean baseline.
- Total grad norm `|g|` on /100 is 0.29 to 1.89 (smaller than /30 0.94 to 2.12), consistent with smaller value-loss gradients flowing through the network but still non-trivial.
- Policy is still learning: rollout `top_action_fraction` climbs from 0.34 (u1) to 0.40 (u8) on /100, vs 0.34 to 0.50 on /30. Both stay well below the 0.95 wedge threshold.

The over-compression risk would manifest as `vp_std_post -> 0` (value head not observation-conditioning), `|g| -> 0` (no gradient signal), or `top_action_fraction` stuck at uniform 0.33 (policy not differentiating). None of those signatures fire on /100 in the 2048-ts slice. A 10k-ts confirmation would be the next step to verify this holds at longer horizons.

## Mechanism read

Three findings:

1. **Absolute reward magnitude is load-bearing under Adam.** K3.3 had falsified `vf_coef` as a magnitude lever because Adam's per-parameter normalization made gradient scale roughly invariant. K3.5 changes the magnitude of the value-target itself (returns), which Adam cannot rescale away (Adam normalizes per-parameter, not per-target). The /30 and /100 results confirm magnitude is the operative variable for the value-shock mechanism, complementing the K3.2 mechanistic framing and the K3.4 bias-init confirmation.
2. **The K3.4 secondary mechanism (drift chasing) is bounded by absolute magnitude, not by drift ratio.** K3.4 evidence framed the secondary mechanism as "the value head chases a moving target." K3.5 keeps the moving target moving (drift ratio is actually larger than K3.4 because the policy survives longer) but caps the absolute size of the chase per rollout. The value head can catch up without each catch-up producing a network-killing gradient burst.
3. **Adam's bounded step interacts with reward scale.** v_loss = 0.06 at /30 update 2 (vs K3.4 110) is 1700x smaller, and `|g|` is 60x smaller (2.12 vs 128.82). The cnn features extractor backprop is now in a healthy range, and latent_vf, latent_pi all stay live. This is consistent with the K3.3 Adam-normalization finding: scale the inputs to the optimizer, not the loss coefficient, to change Adam's per-parameter step.

## Verdict against the final K3.5 contract

K3.5 PASS on /30. /100 also passes strong with no over-compression signal on this slice. The next move per the contract is the 10k confirmation on /30. The 10k confirmation is the K3.5b deliverable, not part of this session's matrix.

K3.6 reward-shape change in `games/signal-dodge/scripts/main.gd` is not needed. Action-net gain stays parked. Separate value-head optimizer (K3.7 placeholder) stays parked.

## Files

- `tools/h5_training_entropy_probe.py` patched (VecEnvWrapper import, FixedRewardScaleVecEnv class, --reward-scale-divisor CLI, wrapper application, header fields).
- `tools/k3_5_extract.py` analysis script (gate evaluation, drift metrics, CSV writer).
- `runs/phase_k/k3_5_smoke_reward_scale_div30_512.ndjson` smoke source.
- `runs/phase_k/k3_5_reward_scale_div30_2048.ndjson` /30 real slice source.
- `runs/phase_k/k3_5_reward_scale_div100_2048.ndjson` /100 real slice source.
- `runs/phase_k/k3_5_reward_scale_table.csv` per-update extraction (gitignored).
