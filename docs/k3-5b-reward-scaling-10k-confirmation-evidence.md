# K3.5b 10k Confirmation Evidence: Reward-Scaling /30 Holds Across 40 Updates; Deterministic-Argmax Panel Wedge Is Mechanistically Separable

## Context

K3.5 (2048-ts seed-3 slice) classified as PASS: both `/30` and `/100` divisors passed the primary and strong gates, with `latent_vf_live_post = 128/128` across all 8 updates and `vp_std_post` in the 2e-3 to 1e-2 range (vs K3.4 baseline collapsing to 0 by update 4). Per the final K3.5 contract decision rule, the next move was the 10k confirmation on `/30`. K3.5b runs that confirmation.

## Run

Config: `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Seed 3, **10000 ts**, pi=[64], vf=[128], `vf_coef=0.5`, `--value-bias-init none`, `--reward-scale-divisor 30`, `ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`, `lr=3e-4`, fixed panel enabled.

- `runs/phase_k/k3_5_confirm_reward_scale_div30_10000.ndjson` (header `reward_scale_divisor=30.0 reward_scale_applied=true`, `value_bias_init_applied=false`)
- Per-update CSV extract: `runs/phase_k/k3_5_confirm_reward_scale_div30_10000_table.csv` (gitignored).
- Analysis script: `tools/k3_5b_extract.py`.

Elapsed: **313.5 s**, 40 PPO updates (ceil(10000/256) = 40). Wall-time tracks the 2048-ts slice projection (64.9 s × 5 = 324.5 s; actual 313.5 s).

Launched via bat-sentinel pattern (`%TEMP%\run_k3_5_confirm.bat`, `PYTHONUNBUFFERED=1`, sentinel `%TEMP%\k3_5_confirm_done.sentinel`) because 5+ min wall time exceeds the MCP shell timeout ceiling. Sentinel returned exit code 0.

## Sustained-life gates across 40 updates

The K3.5b gate is structural: the K3.5 primary and strong gates must hold across the full probe, not just at updates 3 and 8. The fail condition is `latent_vf_live` collapsing below 16 dims OR `vp_std_post` falling to 0 OR a wedge forming (rollout `top_action_fraction >= 0.95`) at any point in the 40 updates.

| Metric                                          | Value         | Threshold      | Result |
| ----------------------------------------------- | ------------: | -------------- | :----: |
| latent_vf_live_post min across 40 updates       | 128 at u1     | >= 16          | PASS   |
| fixed_panel_value_predictions_std_post min      | 2.23e-3 at u6 | > 1e-6         | PASS   |
| rollout top_action_fraction max across 40       | 0.629 at u6   | < 0.95 (wedge) | PASS   |
| post_rollout_model_to_constant_mse_ratio max    | 25.143 at u35 | (note below)   | mixed  |

`post_ratio` max of 25.143 at update 35 is a single-update spike. The K3.5 strong-gate threshold (6.18) was the K3.3 baseline at update 8 only; across 40 updates the post_ratio can fluctuate. The aggregate is healthy: 32 of 40 updates have `post_ratio < 1.0` (better than the constant baseline), and only update 35 exceeds the K3.5 strong threshold. The spike is not accompanied by a collapse of latent_vf (still 128 dims) or vp_std (1.69e-1, well above 1e-6). Classified as transient and not gate-failing.

K3.5 reward scaling holds across the full 10k probe budget on the contract's structural gates.

## Reward drift across the full probe

Raw `ret_mean` first-to-last ratio: 14.118 (u1) -> 75.639 (u40) = **5.36x**. K3.5 2048-ts /30 was 4.91x; the longer probe extends the drift modestly without breaking the value head's ability to track it.

Per-update raw `ret_mean` trajectory (rounded): 14.1, 29.4, 42.1, 54.1, 57.0, 60.0, 68.1, 69.3, 70.3, 72.6, 80.8, 76.3, 77.1, 78.4, 74.5, 68.0, 63.7, 74.6, 78.6, 74.4, 63.5, 65.1, 70.4, 72.3, 83.5, 74.3, 81.0, 87.8, 87.1, 91.3, 85.9, 84.5, 93.4, 88.4, 87.3, 86.5, 76.9, 74.6, 74.2, 75.6.

The drift saturates around u30 (max 93.4 at u33) and then mean-reverts to ~75. The episodes are not getting monotonically longer; the policy reaches a survival ceiling near 90-step episodes and oscillates around it. The K3.4 secondary-mechanism concern (value head chases an ever-growing target) is bounded under K3.5 scaling: the value head closes the gap each rollout because the per-update raw delta is small enough (typically 1-15 units, scaled to 0.03-0.5) for Adam to absorb.

## The deterministic-argmax panel wedge is NOT resolved by K3.5

The fixed observation-conditioning panel registers `panel_top_argmax_fraction = 1.000` and `panel_num_det_actions = 1` on EVERY update from update 1 onward. Under deterministic argmax, the policy outputs the same action across all 32 panel observations from the very first forward pass.

This is consistent with prior K-series findings: K3.4 evidence noted "constant_action_attractor is True for every update (matches K3.3). The bias-init does not affect the deterministic argmax wedge on the fixed panel." K3.5 also does not affect it.

The mechanism worth being explicit about: the wedge is present at **update 1**, before any optimizer step on a reward-scaled rollout. That means the wedge is **not formed by training collapse under any K-series intervention**. It is an initialization artifact of the action head (or a deterministic-argmax property of the freshly-initialized CnnPolicy that survives all K3.x interventions).

At the same time, **rollout-time stochastic behavior is healthy**:

- Rollout `top_action_fraction` max is 0.629 at u6, well below the 0.95 wedge threshold.
- The top action shifts between `left` and `right` across rollouts (8 left-dominant, 22 right-dominant, 0 stay-dominant out of 40 updates).
- Sampling distribution (`mean_probs`) keeps all three actions in play.

The pattern is: the policy distribution is non-degenerate under stochastic sampling, but the argmax under deterministic eval is uniform across inputs.

This is mechanistically separable from the latent_vf collapse that K3.5 addressed. K3.5 fixed the value-shock pathway. The panel wedge is a different problem: it lives in the action-head decision surface and is present from initialization.

## What K3.5 closes and what it does not

**K3.5 closes (PASS):**
- The K3.2 update-1 value-shock collapse mechanism. Confirmed via K3.4 bias-init (transient rescue) and K3.5 reward scaling (sustained rescue across 40 updates).
- The Adam-normalization-defeats-vf_coef finding from K3.3. K3.5 confirms the correct lever for Adam is to change what the value head is asked to predict (reward magnitude), not the loss coefficient.
- The K3.4 secondary-mechanism "value head chases a moving target" framing. Bounded by absolute magnitude under K3.5; the value head catches up each rollout without producing network-killing gradient bursts.
- Latent representation health: `latent_vf` 128/128, `latent_pi` 64/64, `cnn_features` mostly stable. `|g|` in 0.76 to 2.20 range across 40 updates (vs K3.3 baseline 13-99 range).

**K3.5 does NOT close (and is not asked to):**
- The deterministic-argmax panel wedge. Present at update 1 in the reward-scaled run. Independent of training collapse.
- The original Phase K eval anomaly (bit-identical per-seed eval results across distinct trained models). The deterministic-argmax wedge is the upstream candidate explanation for this anomaly: if the policy's argmax surface is independent of input from initialization onward, deterministic eval can produce the same action sequence regardless of how much the policy network's weights have been trained.

## Verdict against the final K3.5 contract

K3.5b 10k confirmation: **PASS**. K3.5 closes as a successful intervention for the value-shock pathway. Reward scaling at divisor 30 is the active recipe.

K3.6 reward-shape change in `games/signal-dodge/scripts/main.gd` is NOT needed. Action-net gain stays parked. Separate value-head optimizer (K3.7) stays parked.

## Recommended next-phase scoping (for GPT)

Two independent paths emerge from K3.5b. They are scope-distinct and Jeff/GPT should decide direction:

1. **Pursue the deterministic-argmax panel wedge as the active K-series target.** The wedge is present at update 1 of a freshly-initialized CnnPolicy, mechanistically separate from value collapse, and the most plausible explanation for the original Phase K bit-identical eval anomaly. Candidate K4 mechanisms: action-head initialization, CnnPolicy default weight scaling, log-softmax numerical behavior at near-uniform initialization producing tie-breaking on a single dimension, or features-extractor output uniformity producing identical logits to all panel obs. Diagnostic slice would inspect the raw action-net logits on the panel at update 0 (before any training step).
2. **Step back to the original Phase K eval anomaly directly.** Run a K3.5-recipe (`--reward-scale-divisor 30`, no bias init, otherwise default H5 entropy config) full-budget training (10k or 25k), checkpoint at multiple stages, and run the standard H5 eval pipeline against those checkpoints. If eval results now differ across checkpoints, K3.5 has resolved the anomaly downstream of the value head. If eval results remain bit-identical, the deterministic-argmax wedge is confirmed as the operative mechanism and K4 becomes the focused intervention.

Path 2 is the cheaper falsification and answers a direct question about whether K3.5's value-head rescue translates to eval-time behavior change. Path 1 is the deeper mechanism investigation. Both can be sequenced.

## Files

- `tools/k3_5b_extract.py` analysis script.
- `runs/phase_k/k3_5_confirm_reward_scale_div30_10000.ndjson` 10k confirmation source.
- `runs/phase_k/k3_5_confirm_reward_scale_div30_10000_table.csv` per-update CSV (gitignored).
- `%TEMP%\run_k3_5_confirm.bat` bat-sentinel launcher (local-only, not in repo).
- `%TEMP%\k3_5_confirm_log.txt` stdout log (local-only, not in repo).
