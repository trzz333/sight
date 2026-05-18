# K3.2 Stage 1 Evidence: Returns Distribution and Value-Optimization Failure

## Context

K3.1 localized the variance-collapse locus to `mlp_extractor.value_net`. By update 3, `latent_vf` had 0 of 128 dims live and stayed dead through update 40, while `cnn_features` remained stable around 129 of 512 live dims. The K3.1 evidence did not answer whether the value-MLP collapse was correct (returns near-constant under current Signal Dodge reward shaping, so a constant predictor is L2-optimal) or pathological (returns have real variance and the value head fails to fit it).

K3.2 Stage 1 answers that question by extracting per-update rollout returns, advantages, values, value-error, MSE, and EV from the existing K3.1 NDJSON. No code patch, no new run.

## Source

`runs/phase_k/k3_1_seed3_pi64_vf128_feature_chain_10k.ndjson` (41 rows: 1 header + 40 updates).
Per-update CSV extract: `runs/phase_k/k3_2_stage1_returns_table.csv`.

Config: seed 3, total_timesteps 10000, n_steps 256, batch_size 64, n_epochs 4, lr 3e-4, gamma 0.99, gae_lambda 0.95, clip_range 0.2, vf_coef 0.5, max_grad_norm 0.5, ent_coef 0.01, policy_kwargs net_arch pi=[64] vf=[128].

## Decision table

```
u  | ts    | ret_m   ret_s  ret_min ret_max | adv_m   adv_s  pos% | val_m    val_s    | const_mse  model_mse  ratio  | EV    | v_loss  | gn_mlp gn_act gn_vscalar | lvf_pre lvf_post vp_pre vp_post
 1 |   256 | 15.714  2.719  0.935  16.766  | 15.780  2.716 100.0 | -0.0658  0.0197   |     7.394   256.379  34.68  | 0.002 | 111.21  | 41.18  0.60   69.14      | 128     70       0.0076 0.0006
 2 |   512 | 23.486  2.467 10.069  24.429  | 14.324  2.467 100.0 |  9.1619  0.0008   |     6.088   211.268  34.70  | 0.000 | 200.00  | 11.15  0.67  157.67      |  70      2       0.0006 0.0001
 3 |   768 | 24.061  2.872  0.000  25.093  | 14.100  2.872  99.6 |  9.9604  0.0002   |     8.247   207.065  25.11  | 0.000 | 198.16  |  0.87  0.66  156.48      |   2      0       0.0001 0.0000
 4 |  1024 | 23.108  4.742  0.000  25.659  | 12.468  4.742  96.5 | 10.6405  0.0000   |    22.490   177.939   7.91  | 0.000 | 170.21  |  0.01  0.47  138.04      |   0      0       0.0000 0.0000
 5 |  1280 | 23.577  4.790  0.000  26.214  | 12.268  4.790  96.1 | 11.3087  0.0000   |    22.941   173.451   7.56  | 0.000 | 166.01  |  0.01  0.56  135.83      |   0      0       0.0000 0.0000
40 | 10240 | 43.841  1.811 33.995  44.533  | 10.514  1.811 100.0 | 33.3279  0.0000   |     3.279   113.813  34.71  | 0.000 | 107.79  |  0.00  0.85  116.10      |   0      0       0.0000 0.0000
```

## Classification (all 40 updates)

- `returns_class`: `real_returns_variance` for all 40 updates. `ret_std` ranges from 1.81 (update 40) to 5.30 (update 17), `ret_range` from 10.5 to 26.2. Returns are never near-constant or low-effective.
- `model_class`: `model_not_beating_constant` for all 40 updates. `model_to_constant_mse_ratio` ranges from 7.56 (update 5) to 53.4 (update 18). `explained_variance` is 0.002 at update 1 and 0.000 from update 2 onward.
- `advantage_class`: `advantage_nondegenerate` for all 40 updates. `adv_std` ranges from 1.81 to 5.30. The K-B `adv_std < 0.05` detector does not fire.


## Mechanism

The Stage 1 data is sharper than the GPT contract Branch B ("ambiguous between unobservable returns and failed value optimization") allows for. The collapse is not ambiguous, it is a value-side optimization shock that destroys the value-MLP capacity in the first three updates.

1. **At init (pre-update 1)**: the network is functional. `latent_vf` has 128 of 128 live dims, fixed-panel `value_predictions` span [-0.074, -0.050] across 32 panel items with std 0.0076, and rollout `values` have std 0.0197. The value head is conditioning on observations.

2. **Update 1**: returns mean is 15.71, values mean is -0.066. Value loss is 111. With `vf_coef=0.5` and `lr=3e-4`, the scalar value head receives a gradient norm of 69.1, and the shared MLP extractor receives gradient norm 41.2. The bias of the value head moves up sharply: post-update 1, fixed-panel `value_predictions_std` drops to 0.0006 (a 12x compression). Post-update 1 rollout `latent_vf` live dims drop from 128 to 70.

3. **Update 2**: returns mean is 23.49, values mean is now 9.16 (bias caught up partially). But `val_std` is 0.0008 across rollout (collapsed). `latent_vf` live dims drop from 70 to 2. The scalar value head gradient is now 157.7, more than 2x update 1, because the bias is chasing a moving target.

4. **Update 3**: `latent_vf` live dims drop from 2 to 0. From this point the value-MLP branch is dead, and only the scalar value head's bias can update.

5. **Updates 4 to 40**: with `latent_vf` at 0 live dims, the value head's input is effectively zero everywhere, so the bias is the only learnable scalar. The bias chases `returns_mean` (which grows from 23 at update 4 to 43.84 at update 40 as episodes lengthen) but never catches up. At update 40 the bias is 33.33 vs returns_mean 43.84, a persistent 10.5 lag.

6. **Consequence for policy**: persistent under-prediction means advantages are persistently positive, 96 to 100 percent positive fraction, mean 10 to 15 across all updates. Every action looks "good" in policy-gradient terms. The policy converges to a single argmax (`stay`) and the constant-action-attractor flag is True for every update.

## Disambiguation against alternative explanations

- **Not near-constant returns.** `ret_std` is 1.81 to 5.30 across all 40 updates, `ret_range` 10.5 to 26.2. The value MLP is not L2-correctly collapsing onto a constant target.
- **Not advantage degeneracy in the K-B sense.** `adv_std` is 1.81 to 5.30. The K-B `adv_std < 0.05` detector does not fire on any update.
- **Not insufficient observation signal at init.** `cnn_features` live dims are stable around 129 across all 40 updates. `latent_vf` had 128 live dims at init and produced a non-trivial fixed-panel `value_predictions_std` of 0.0076. Observations were predictive enough for the value head to start learning before collapse.
- **Not a pure policy-side bottleneck.** `latent_pi` also collapses by update 3, but the K3 evidence already showed pi widening alone cannot rescue. Stage 1 here points to the value-side gradient as the upstream driver.

## Naming caveat

`grad_norms_preclip.value_net_mean` in the NDJSON is the final scalar value head (`policy.value_net`), not the `mlp_extractor.value_net` submodule. `grad_norms_preclip.mlp_extractor_mean` is mixed pi and vf branches. Stage 1 thus indicates value-side gradient pressure but cannot isolate the `mlp_extractor.value_net` submodule gradient. The Stage 2 patch contract specifies adding `grad_norm_mlp_value_net_params_mean/max` and `grad_norm_mlp_value_net_weights_mean/max` to close this.

## Verdict against GPT's decision table

Formal classification: Branch B (real returns variance + model not beating constant + latent_vf collapses by update 3). Stage 2 patch is justified by the contract.

Sharper read: the value-MLP collapse is not "ambiguous between unobservable returns and failed value optimization." It is a value-loss-magnitude shock at update 1 that destroys `latent_vf` faster than it can learn, after which only the scalar bias can update against a moving `returns_mean`. The next intervention space per GPT's contract is the value-side knobs (`vf_coef`, value LR split, value normalization, value-head init), not the action-net-gain experiment and not pi widening.

## Recommended Stage 2 scope

Per GPT's K3.2 contract, patch `tools/h5_training_entropy_probe.py` and run seed 3, 2048 ts, pi=[64], vf=[128] with the listed digest fields. The CNN-to-returns CV R^2 probe remains useful as a sanity check that observations *can* predict returns (the init `value_predictions_std=0.0076` is suggestive but not conclusive), but the bigger lever based on Stage 1 mechanism is to test reduced `vf_coef` (for example 0.1 or 0.05) or a value-head init that starts predictions in the right magnitude range, on a parallel slice.

Stage 2 contract has not been adjusted. Next prompt should decide whether to add a `vf_coef` sweep slice alongside the planned 2048 ts probe run, or keep the probe pure-diagnostic per contract and run `vf_coef` interventions as K3.3.

## Files

- `runs/phase_k/k3_1_seed3_pi64_vf128_feature_chain_10k.ndjson` source
- `runs/phase_k/k3_2_stage1_returns_table.csv` extracted per-update table
