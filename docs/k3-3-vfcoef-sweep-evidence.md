# K3.3 vf_coef Sweep Evidence: vf_coef Cannot Prevent latent_vf Collapse Under Adam

## Context

K3.2 Stage 1 evidence identified a value-loss-magnitude shock at update 1 that collapses `mlp_extractor.value_net` (latent_vf) from 128 to 0 live dims by update 3 on the fixed observation-conditioning panel. The shock was hypothesized to be driven by `vf_coef * value_loss` gradient magnitude. K3.3 tests this directly: hold seed, n_steps, batch_size, n_epochs, lr, gamma, gae_lambda, clip_range, ent_coef, max_grad_norm, and net_arch constant; vary only `vf_coef` across {0.5, 0.1, 0.05}.

K3.3 also adds new per-update diagnostics to the probe so the result can be read directly from per-submodule gradient norms and pre/post value-fit on rollout obs.

## Patch (tools/h5_training_entropy_probe.py)

1. New CLI `--vf-coef` overrides `hyperparams["vf_coef"]` before model construction. Header now records the override.
2. `compute_value_fit_stats(values, returns, returns_std_for_ratio)` returns `value_pred_{mean,std,min,max}`, `model_mse`, `constant_baseline_mse`, `model_to_constant_mse_ratio`, `explained_variance`.
3. New per-update `value_fit` block with `constant_baseline_mse`, `returns_std`, `pre_rollout`, `post_rollout`. Post-rollout is re-evaluated via `policy.predict_values(rb_obs_flat)` after the optimizer step.
4. `compute_module_grad_norm_weights_only(module)` returns L2 grad norm over weight tensors only (rank >= 2), excluding biases.
5. `grad_norms_preclip` now exposes per-submodule keys: `features_extractor_{mean,max}`, `mlp_extractor_{mean,max}`, `action_net_{mean,max}`, `value_net_{mean,max}` (legacy: scalar value head), plus the new `mlp_policy_net_{mean,max}`, `mlp_value_net_{mean,max}`, `mlp_value_net_weights_{mean,max}`, `scalar_value_head_{mean,max}`. The legacy keys remain present for backward compat with K3.x downstream readers.

Smoke (seed 3, 512 ts, `--vf-coef 0.1`, pi=64/vf=128) exited 0; new fields populate; header `vf_coef=0.1` confirmed.

## Runs

Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Seed 3, 2048 ts each, pi=[64], vf=[128], fixed panel enabled.

- `runs/phase_k/k3_3_baseline_vfcoef_0_5_2048.ndjson` (header `vf_coef=0.5`)
- `runs/phase_k/k3_3_vfcoef_0_1_2048.ndjson` (header `vf_coef=0.1`)
- `runs/phase_k/k3_3_vfcoef_0_05_2048.ndjson` (header `vf_coef=0.05`)

Per-update CSV extract: `runs/phase_k/k3_3_vfcoef_sweep_table.csv`.

## Primary mechanism gate (per GPT contract)

Required: `latent_vf_live_post_update_3 > 0 AND fixed_panel_value_predictions_std_post_update_3 > 1e-6 AND grad_norm_mlp_value_net_mean_update_1 materially lower than baseline.`

Result:

```
vf_coef | gn_mlp_vf_mean u1 | lvf_post u1  u2  u3 | vp_std_post u1     u2       u3
0.5     | 41.18              | 70  2  0           | 0.000611 0.000105 0.000000
0.1     |  8.24              | 71  2  0           | 0.000604 0.000160 0.000000
0.05    |  4.12              | 71  2  0           | 0.000592 0.000204 0.000000
```

- `gn_mlp_value_net_mean` at update 1 scales exactly with vf_coef (41.18 / 8.24 / 4.12 = 5x / 1.6x / 1x against vf=0.05 baseline, perfectly tracking the 10x / 2x / 1x ratio in vf_coef).
- `latent_vf_live_post_update_3 == 0` for all three vf_coef values.
- `fixed_panel_value_predictions_std_post_update_3 < 1e-6` for all three.

Primary mechanism gate: FAIL on all three runs.

## Stronger win gate

Required: `latent_vf_live_post_update_3 >= 16 AND latent_vf_live_final >= 16 AND post_rollout_model_to_constant_mse_ratio improves versus vf_coef=0.5.`

Result: latent_vf_live is 0 for both update 3 and update 8 across all three runs. Stronger win gate: FAIL.

## Failure gate

Required: `vf_coef=0.1 and 0.05 both still hit latent_vf_live_post_update_3 == 0.`

Result: TRUE for both. Failure gate fires.

## The Adam-normalization finding

The K3.3 sweep shows something the K3.2 hypothesis did not predict: trajectories are nearly bit-identical across a 10x vf_coef range.

Max delta across (vf=0.5, vf=0.1, vf=0.05), all 8 updates:

```
ret_mean       u1=0.000  u2=0.001  u3=0.008  u4=0.027  u8=0.118
val_mean       u1=0.000  u2=0.001  u3=0.010  u4=0.001  u8=0.000
post_val_mean  u1=0.001  u2=0.010  u3=0.001  u4=0.000  u8=0.000
val_std        u1=0.000  u2=0.000  u3=0.000  u4=0.000  u8=0.000
```

The grad norms scale linearly with vf_coef (`gn_mlp_value_net_mean` at update 1: 41.18 vs 8.24 vs 4.12, exactly 5:1.6:1) but the resulting parameter updates and post-update value predictions are within 0.01 of each other.

This is the signature of Adam optimization, which SB3 uses for PPO by default. Adam's running second moment v_hat is approximately proportional to the squared gradient, so the effective step `m_hat / sqrt(v_hat)` is approximately scale-invariant in the gradient. Reducing vf_coef by 10x reduces the value-loss gradient by 10x, but Adam compensates by reducing v_hat proportionally, leaving the parameter update direction and magnitude approximately unchanged. The value head's bias still jumps from -0.066 to 9.16 in one update across all three vf_coef values.

This means `vf_coef` is effectively a no-op as a value-loss-magnitude intervention under Adam. The K3.2 "value-loss shock" framing was correct in mechanism (the value head's bias overshoot in update 1 destroys latent_vf), but `vf_coef` is the wrong knob to address it. The correct levers act on what Adam cannot rescale away:

1. **Return target rescaling**: divide returns by a running std before computing value loss. SB3 has `PPO(normalize_advantage=True)` for advantages but no built-in return normalization. Custom return-normalization wrapper or PPO subclass.
2. **Reward rescaling**: divide environment rewards by a fixed or running scale so returns naturally live in a smaller range (`return = sum(gamma^t * reward)` is currently 15 to 44 here, value head init is near zero, so the bias gap is enormous at init).
3. **Value-head output bias init**: set the scalar value head's bias to an estimate of `mean(returns)` at construction so the gap to close in update 1 is small.
4. **Separate value-head optimizer with lower learning rate**: bypass Adam's auto-rescale by giving the value head a small lr that Adam cannot inflate.
5. **Reward redesign**: terminal-only or episode-normalized rewards instead of per-step survival, which would change `returns_mean` from "scales with episode length" to "bounded."

## Other observations

- `cnn_features` live dims and `latent_pi` collapse trajectory are also bit-identical across the three runs, confirming the value-side gradient drives the upstream collapse through the shared `features_extractor` and `mlp_extractor`. K3.1 noted `cnn_features` stay stable (~129 live dims) which holds here.
- `gn_mlp_policy_net_mean` at update 1 is ~ 0.03 in the K3.3 smoke (and proportional in the 2048 runs), two orders of magnitude below the value-side. The shared `features_extractor` receives the full value-loss gradient flowing back from `mlp_extractor.value_net`, so the policy side cannot escape the collapse simply by being decoupled.
- Returns grow from 15.71 at update 1 to 25.49 at update 8 in the baseline (vs 15.71 to 25.38 in vf=0.1 and 15.71 to 25.38 in vf=0.05). Same growth, driven by the same constant-action attractor that survives longer episodes.

## Verdict against GPT's K3.3 contract

Failure gate fires. Per the contract: "next lever is not action-net gain. It is return/value scaling: return normalization, reward rescaling, value-head output-scale init, or reward redesign."

The Adam finding sharpens which of those levers will actually work. Return normalization or reward rescaling change what the value head is asked to predict, and Adam cannot rescale that away. Value-head bias init reduces the update-1 overshoot directly. A separate value-head optimizer would be the most surgical intervention but adds machinery.

## Recommended K3.4

Three cheap interventions ranked by surgical-ness:

1. **Value-head bias init at `mean(returns_first_rollout)`**. Smallest patch. Run a single seed-3 2048-ts slice and observe whether update-1 lvf_post stays > 16.
2. **Return normalization wrapper** that divides returns by a running std before passing to PPO's value loss. Run a single seed-3 2048-ts slice.
3. **Reward rescaling**: change per-step survival reward from 1.0 to 0.01 in `games/signal-dodge/scripts/main.gd` (or via an env wrapper). Run a single seed-3 2048-ts slice. Note: this changes the game's reward shape and merits its own commit and evidence trail; it is a charter-relevant change, not a hyperparameter.

Action-net gain stays parked.

## Files

- `tools/h5_training_entropy_probe.py` patched (CLI, value_fit, per-submodule grad norms, weights-only grad)
- `runs/phase_k/k3_3_baseline_vfcoef_0_5_2048.ndjson` source
- `runs/phase_k/k3_3_vfcoef_0_1_2048.ndjson` source
- `runs/phase_k/k3_3_vfcoef_0_05_2048.ndjson` source
- `runs/phase_k/k3_3_vfcoef_sweep_table.csv` per-update extraction (gitignored, local-only)
