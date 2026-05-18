# K3.4 Value-Head Bias Init Evidence: First-Rollout-Mean Bias Init Prevents the Update-1 Value Shock, Then Collapse Resumes by Update 4

## Context

K3.3 falsified `vf_coef` as a value-shock intervention: under Adam normalization, scaling the value-loss gradient by 10x changed parameter-update magnitude by less than 0.01. The K3.3 evidence narrowed the next-lever space to interventions that change WHAT the value head is asked to predict, not how strongly the gradient is weighted: return rescaling, reward rescaling, value-head bias init, or a separate value-head optimizer.

K3.4 implements the smallest of those: scalar value-head bias init at `mean(rollout_buffer.returns)` of the first rollout. Tests directly whether eliminating the initial scalar-value-target gap that K3.2 implicated (`val_mean = -0.066` vs `ret_mean = 15.71` at update 1) prevents the update-1 latent_vf collapse.

## Patch (tools/h5_training_entropy_probe.py)

1. New CLI `--value-bias-init {none,first-rollout-mean}`, default `none`.
2. `InstrumentedPPO.__init__` accepts `value_bias_init_mode` kwarg.
3. In `train()`, after `rollout_buffer` is bound and `rb_obs_flat` built, BEFORE `pre_action_net` / `pre_policy_state` / `pre_fixed_panel_state` snapshots and BEFORE the first optimizer step: if mode is `first-rollout-mean` and `_probe_update_idx == 0`, mutate `policy.value_net.bias` to `mean(rollout_buffer.returns)`. Only the scalar head bias is mutated; `value_net.weight` and `mlp_extractor.value_net` are untouched. Mutation fires exactly once.
4. Immediately after the mutation, re-run `policy.predict_values(rb_obs_flat)` and compute `compute_value_fit_stats` to capture `pre_rollout_after_value_bias_init` (the live policy fit before the first optimizer step).
5. Per-update record carries an additive `value_fit.pre_rollout_after_value_bias_init` (non-null only on update 1 with mutation, null otherwise per GPT revision 2 schema) and a peer `value_bias_init` block (mutation diagnostic, non-null only on update 1 when applied).
6. Header records `value_bias_init_mode` (CLI string) and `value_bias_init_applied` (hard state from the InstrumentedPPO `_value_bias_init_record`).

Smoke (seed 3, 512 ts, `--value-bias-init first-rollout-mean`, pi=64/vf=128): exited 0, 2 updates, `old_bias=0.0 new_bias=15.713981`, header `value_bias_init_applied=True`, additive fields populate cleanly. Existing K3.x fields preserved.

## Run

Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Seed 3, 2048 ts, pi=[64], vf=[128], fixed panel enabled, default `vf_coef=0.5`, `ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`, `lr=3e-4`.

- `runs/phase_k/k3_4_value_bias_first_rollout_mean_2048.ndjson` (header `value_bias_init_mode=first-rollout-mean`, `value_bias_init_applied=true`, `vf_coef=0.5`)
- `runs/phase_k/k3_4_smoke_value_bias_first_rollout_mean_512.ndjson` (smoke, 2 updates)
- `runs/phase_k/k3_4_value_bias_first_rollout_mean_table.csv` per-update extraction (gitignored).

Elapsed: 64.3 s for the 2048-ts slice.

## Value-bias-init diagnostic block

```
mode                  : first-rollout-mean
applied               : true
applied_before_pre_snapshot: true
update_idx            : 1
old_bias              : 0.000000
new_bias              : 15.713981
rollout_returns_mean  : 15.713981
rollout_returns_std   :  2.719093
```

The mutation fires at the point GPT specified: between `collect_rollouts` (which computed returns against the pre-mutation value head) and the first optimizer step (which would otherwise face the full `ret_mean - val_mean ~= 15.78` shock).

`pre_rollout_after_value_bias_init` confirms the effect on the live policy before any optimizer step:

```
value_pred_mean       : 15.648
value_pred_std        :  0.0197
model_mse             :  7.382
constant_baseline_mse :  7.393
model_to_constant_mse_ratio: 0.998
explained_variance    :  0.0022
```

The bias-initialized policy roughly matches the constant-mean baseline. Observation conditioning (value_pred_std 0.0197) is preserved at the same magnitude as the K3.3 pre-update state.

## Primary mechanism gate (revised contract)

Required: `latent_vf_live_post_update_3 >= 16 AND fixed_panel_value_predictions_std_post_update_3 > 1e-6 AND post_rollout_model_to_constant_mse_ratio_update_3 improves vs K3.3 vf_coef=0.5 baseline.`

| Condition | K3.4 | Threshold | Result |
| --- | --- | --- | --- |
| latent_vf_live_post_update_3 | 10 | >= 16 | FAIL |
| fixed_panel_value_predictions_std_post_update_3 | 2.16e-4 | > 1e-6 | PASS |
| post_rollout_model_to_constant_mse_ratio_update_3 | 2.60 | < 22.84 (K3.3 baseline) | PASS |

Primary mechanism gate: FAIL (lvf live dims at update 3 = 10, below the 16-dim threshold).

## Strong win gate (revised contract)

Required: `latent_vf_live_post_update_8 >= 16 AND fixed_panel_value_predictions_std_post_update_8 > 1e-6 AND post_rollout_model_to_constant_mse_ratio_update_8 improves vs K3.3 baseline.`

| Condition | K3.4 | Threshold | Result |
| --- | --- | --- | --- |
| latent_vf_live_post_update_8 | 0 | >= 16 | FAIL |
| fixed_panel_value_predictions_std_post_update_8 | 8.26e-7 | > 1e-6 | FAIL |
| post_rollout_model_to_constant_mse_ratio_update_8 | 30.61 | < 6.18 (K3.3 baseline) | FAIL |

Strong win gate: FAIL.

## Failure gate (revised contract)

Required: `latent_vf_live_post_update_3 == 0 OR fixed_panel_value_predictions_std_post_update_3 <= 1e-6.`

| Condition | K3.4 | Threshold | Result |
| --- | --- | --- | --- |
| latent_vf_live_post_update_3 | 10 | == 0 | does not fire |
| fixed_panel_value_predictions_std_post_update_3 | 2.16e-4 | <= 1e-6 | does not fire |

Failure gate: does not fire.

## Verdict against revised K3.4 contract

Primary gate FAIL on lvf threshold. Strong gate FAIL across all three subgates. Failure gate does not fire. Per the revised decision rule, the literal "primary passes AND strong fails -> transient rescue" branch does not apply because primary does not pass, but the SPIRIT of the rule matches this case exactly: the intervention rescues the early collapse, then the collapse trajectory resumes within a single rollout cycle.

Classification: **transient rescue**. Do not run 10k confirmation. Escalate to K3.5 Python-side reward scaling wrapper per the revised decision rule.

## What the bias-init did and did not do

Per-update fixed-panel state (post-update):

```
u | lvf_post | vp_std_post  | post_ratio | ret_mean | val_mean (rollout) | post_val_mean
1 |   128    | 4.95e-02     |  0.972     |  15.71   |  -0.066            |  15.754
2 |   126    | 2.46e-03     |  5.678     |  28.97   |  15.671            |  24.013
3 |    10    | 2.16e-04     |  2.597     |  33.63   |  24.015            |  25.571
4 |     2    | 4.46e-07     | 30.118     |  37.31   |  25.572            |  26.401
5 |     1    | 1.06e-06     |  2.547     |  35.51   |  26.401            |  27.189
6 |     0    | 8.50e-07     |  2.471     |  36.23   |  27.189            |  27.939
7 |     0    | 8.83e-07     |  2.594     |  37.38   |  27.939            |  28.664
8 |     0    | 8.26e-07     | 30.608     |  39.91   |  28.664            |  29.371
```

Same column slice for the K3.3 `vf_coef=0.5` baseline at the same seed and net_arch:

```
u | lvf_post | vp_std_post  | post_ratio | ret_mean | val_mean (rollout) | post_val_mean
1 |    70    | 6.11e-04     |  6.805     |  15.71   |  -0.066            |   9.162
2 |     2    | 1.05e-04     |  0.0002    |  23.49   |   9.162            |   9.960
3 |     0    | 0.00e+00     | 22.838     |  24.06   |   9.960            |  10.640
8 |     0    | 0.00e+00     |  6.180     |  25.49   |  13.263            |  13.904
```

What the bias-init achieves:

- **Update 1 rescue is total.** lvf_post = 128/128, vp_std_post = 4.95e-2 (81x the K3.3 baseline 6.11e-4), `value_loss_mean = 7.54` (vs K3.3 baseline 111.21), `total_grad_norm_mean = 4.45` (vs K3.3 baseline 135.32). The update-1 value shock that K3.2 implicated as the latent_vf killer is eliminated.
- **Update 2 rescue is substantial.** lvf_post = 126/128 (K3.3 baseline already collapsed to 2). The MLP value branch is still observation-conditioning.
- **Update 3 partial.** lvf_post drops to 10/128. K3.3 baseline is at 0.
- **Update 4 onward.** Collapse trajectory resumes and reaches the same dead-MLP state as K3.3 by update 6.

What the bias-init does NOT achieve: it does not prevent the update-2 chase. Returns climb from 15.71 at update 1 to 28.97 at update 2 (episodes get longer because the surviving policy plays longer), so the value head must close a new 13.3-unit gap. v_loss at u2 = 110, comparable in magnitude to K3.3 baseline u1 v_loss = 111. The bias init delays the value shock by exactly one rollout but does not prevent it once returns drift upward.

## Mechanism read

Three things matter:

1. **The K3.2 update-1 value-shock mechanism is now directly confirmed.** When the bias-init eliminates the scalar value-target gap, latent_vf survives update 1 with all 128 dims intact and observation-conditioning rises 81x on the fixed panel. The K3.2 framing of the collapse was correct in mechanism.
2. **The value head chases a moving target.** Per-step survival reward gives `return = sum(gamma^t * 1) = (1 - gamma^T) / (1 - gamma)`, which grows with episode length T. As the policy survives longer, T grows, and `returns_mean` rises monotonically (15.71 -> 28.97 -> 33.63 -> ... -> 39.91 across 8 updates). The value head's bias cannot keep up: `post_val_mean` lags `ret_mean` by 10 to 13 units throughout updates 2-8. Each catch-up is a fresh value shock.
3. **The optimizer cannot rescue this from the value-loss side alone under Adam.** K3.3 already showed `vf_coef` cannot reduce the per-update parameter change. Bias init removes one shock; the next shock arrives one rollout later, with comparable magnitude (v_loss 110 at u2 vs 111 at u1 in K3.3), and the trajectory rejoins K3.3.

This pattern is consistent across grad norms: `gn_scalar_vh` at u2 (41.32 in K3.4 vs 11.15 in K3.3 baseline) is LARGER under bias-init because the bias must close a 13.3-unit gap rather than the K3.3 baseline's already-partially-closed gap. `gn_mlp_vf_w` at u2 = 31.25 in K3.4 vs 11.15 in K3.3 baseline. The bias-init front-loads the value-side gradient that K3.3 spread across updates 1 and 2.

## Other observations

- `cnn_features` live dims: 251 -> 142 -> 131 -> 129 -> 128 -> 128 -> 128 -> 128. Steady descent during the value-MLP collapse phase, then pins at 128. K3.3 baseline pinned at 129 from update 2 onward. The bias-init shifts the cnn collapse curve earlier (because the value-MLP gradient survives longer to backprop into the shared features extractor) but the asymptote is the same.
- `latent_pi` live dims: 64 -> 64 -> 48 -> 42 -> 40 -> 40 -> 40 -> 40. K3.3 baseline asymptotes to 40 by update 4. Same value.
- Rollout `values.std` (the stored values from `collect_rollouts`): u1 = 0.0197, u2 = 0.0492. Update-2 values were predicted by the post-update-1 policy whose `value_net.bias = ~9.16` was set by the optimizer's first step. Note `value_pred_std` after bias init alone = 0.0197 (identical to K3.3 pre-update). Bias init did not destroy observation conditioning; the first optimizer step DID inflate it 5x to 0.0995 on rollout obs and to 0.0494 on the fixed panel.
- Entropy stays > 1.092 across all 8 updates (K-C classification: no collapse). K3.3 baseline entropy also stayed above 1.09 (the classifier is matched by per-step survival policy preserving symmetric action distribution in deterministic mode; this is consistent across K3.x and is not what K3.4 was testing).
- `constant_action_attractor` is True for every update (matches K3.3). The bias-init does not affect the deterministic argmax wedge on the fixed panel.

## Recommended K3.5

Per the revised decision rule and the mechanism read, the next intervention is **Python-side reward scaling wrapper**:

1. Wrap the env with a `gym.Wrapper` that divides per-step reward by a fixed scale (e.g., 100) or by a running estimate so returns live in O(1) range regardless of episode length.
2. Hold seed, n_steps, batch_size, n_epochs, lr, gamma, gae_lambda, clip_range, ent_coef, max_grad_norm, vf_coef, and net_arch constant.
3. Run a 2048-ts seed-3 slice with the wrapper and observe whether lvf_post stays > 16 at update 3 and update 8.

This is preferred over Godot reward rescaling (option c in the K3.3 evidence) for two reasons:

- It keeps the game spec untouched until the scale hypothesis is validated.
- It keeps returns, GAE, value targets, and advantages in the same scale (Python wrapper scales the reward stream before SB3 sees it; SB3 computes returns from the scaled stream consistently).

If reward scaling also fails to keep lvf alive at update 8, the deeper pathology is reward shape rather than reward magnitude, and K3.6 escalates to terminal-only or episode-normalized rewards in `games/signal-dodge/scripts/main.gd`. A separate value-head optimizer is reserved for K3.7 if the target-scale hypothesis is fully exhausted.

## Files

- `tools/h5_training_entropy_probe.py` patched (CLI `--value-bias-init`, InstrumentedPPO kwarg, bias-init in `train()`, `value_fit.pre_rollout_after_value_bias_init`, `value_bias_init` record block, header fields).
- `runs/phase_k/k3_4_smoke_value_bias_first_rollout_mean_512.ndjson` smoke source.
- `runs/phase_k/k3_4_value_bias_first_rollout_mean_2048.ndjson` real slice source.
- `runs/phase_k/k3_4_value_bias_first_rollout_mean_table.csv` per-update extraction (gitignored).
