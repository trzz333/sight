# H5 Phase I activation comparator evidence

Status: VERIFIED end-to-end on StrongerJr. Two invocations of the activation comparator complete on the stay tape (333 steps, 309 unique obs) and the left tape (1383 steps, 1381 unique obs), each loading all four pixel-CNN policies and probing the full SB3 actor path.

## 1. TL;DR

Phase H concluded that the trained pixel-CNN policies are observation-insensitive at the policy-output layer because every model selects the same action on every step regardless of which trajectory it sees. Phase I refines that finding inside the actor: **features DO vary across observations, raw action logits DO vary across observations, but the learned raw-logit margin is many times larger than the per-step variation, so the argmax never crosses an action boundary**. The policy reads the screen; the learned ranking just never flips.

The two equivalence classes localize to two different points on the GPT pre-Phase-I branch table:

- **Class A (train_seed=1; Phase E seed1 and Phase G seed1)**: Branch C — actor layers vary modestly, but the action head produces near-constant raw logits for the selected action. Selected-action raw-logit std is 0.011 to 0.020 against a raw-logit margin of 6.5 to 11.1. Margin-to-std ratio 540 to 630. The action head is operating as a near-constant classifier on this trajectory class.
- **Class B (train_seed=2; Phase E seed2 and Phase G seed2)**: Branch D — actor layers vary substantially, raw logits vary substantially (selected-action std 0.05 to 0.56 against margins 0.83 to 6.31), but argmax remains `left` because no excursion crosses the margin. The smallest headroom appears in Phase E seed2 on the left tape: mean raw-logit margin 0.830, minimum 0.738, selected-action std 0.165. Ratio 4.5. The closest any model gets to flipping argmax.

Branch E (tensor extraction mismatch) is falsified at near-machine precision: `softmax(action_net(mlp_extractor(extract_features(obs))[0]))` matches `policy.get_distribution(obs).distribution.probs` within 1.2e-7 across all four models on both tapes. The actor path extraction is sound.

Phase G's reward shaping had a measurable and consistent effect inside the network: it widened raw-logit margins by 1.7x to 5x while keeping argmax choices fixed. Class A E1 → G1 margin 6.55 → 11.09. Class B E2 → G2 margin on left tape 0.83 → 4.27. The shaping made each network more confident on the same already-selected action, never moving any network across basins.

Branch C and Branch D both license one and only one next slice from the pre-Phase-I scope note: **stochastic-action eval ablation**, with the qualification that it is operationally meaningful only for Class B. Class A is saturated past the point where stochastic sampling would change anything (probability entropy 0.0002 for G1, 0.011 for E1). Class B's left action probability mean is 0.62 (E2) or 0.98 (G2) per Phase H; stochastic sampling would draw `left` 62% of the time for E2 with materially different actions on the other 38%, revealing whether the underlying entropy gradient corresponds to any policy improvement once the deterministic-argmax mask is removed.

The Class A bias-only argmax matches the projection argmax matches the observed argmax: `stay` consistently. The Class B equivalent: `left` consistently. **Biases are operationally negligible**: they are order 1e-3 to 4e-3 while raw-logit projection means are order 6 to 11. The argmax is driven by `W @ latent_pi` magnitudes, not bias gaps. The strict "constant classifier dominated by bias" sub-signature within Branch C does not apply; the constant-classifier behavior arises from the action_net producing near-constant projection outputs despite varying features.

## 2. Why Phase H forces this slice

Phase H established four claims with HIGH confidence: (a) all four model.zip artifacts are distinct at the SHA-256 layer and at the policy state-dict parameter-hash layer; (b) observation hashes are 92.8% to 99.86% unique across the two tapes used here, so observation freshness is intact; (c) cross-basin same-argmax fraction is 0.0000 on both tapes, with G1 picking `stay` on 100% of left-tape steps and G2 picking `left` on 100% of stay-tape steps; (d) within-basin probability-space agreement is essentially perfect for Class A (prob_l1 mean 0.003) and substantial-but-not-perfect for Class B (prob_l1 mean 0.72).

What Phase H could not decide on its own: whether the constant-action policies arise from
- (i) an encoder that has stopped producing observation-correlated features, or
- (ii) a policy MLP that varies but a final action head that ignores its inputs, or
- (iii) all actor layers varying with one action permanently ranked first because the trained margin is too wide for any excursion to cross.

The Phase I comparator was scoped to extract activations at every actor layer (`features_extractor` output, `mlp_extractor.policy_net` output, raw `action_net` logits before softmax, `value_net` output), measure per-layer variance, compute action_net bias gaps and W@latent projection statistics, and verify SB3 consistency. The required deliverable was a verdict on which of the three hypotheses above holds, with a margin-aware ratio rather than a fishing expedition on PCA spectra.

## 3. Tool and methodology

Tool: `tools/h5_activation_compare.py`. Docs-and-tools-only slice. No training. No env source changes. No config changes. No reward variant. Reuses the fingerprinting, tape parser, env factory wiring, and observation hashing from `tools/h5_logit_compare.py`.

Methodology:

1. Load all four target models. Fingerprint each by file SHA-256, archive member list, parameter count, and a blake2b-128 policy state-dict digest. Abort if any two SHA-256 values match.
2. Build one Godot Signal Dodge env in `eval` mode from `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Seed at the requested eval seed and reset.
3. Drive the env once with the fixed behavior tape, recording every observation (uint8, channel-first (1, 84, 84)) into an in-memory list. Close the env. The env is deterministic at fixed eval seed and fixed tape, so a single pass produces the same observation sequence every model would see.
4. For every recorded observation and every loaded model: call `model.policy.obs_to_tensor(obs)`, then directly call `model.policy.extract_features(obs_tensor)`, `model.policy.mlp_extractor(features)` to obtain `(latent_pi, latent_vf)`, `model.policy.action_net(latent_pi)` for raw logits, `model.policy.value_net(latent_vf)` for the value scalar, `torch.softmax(raw_logits)` for manual probabilities, and `model.policy.get_distribution(obs_tensor).distribution.probs` for the SB3 distribution probabilities. Record `max |manual - sb3|` per model as the path-validity check.
5. Compute per-layer summaries (norm mean and std, per-dim std with fraction-above-threshold buckets at 1e-6, 1e-5, 1e-4, adjacent-step L2, first-vs-last L2, covariance trace, effective rank via singular-value participation ratio, top-1/3/10 PCA explained variance fractions) for `features`, `latent_pi`, and `latent_vf`.
6. Compute raw-logit summaries: per-action mean and std across steps, raw top1-top2 margin mean/median/min/p95/max, argmax fractions, probability entropy mean/min/max, and the consistency error.
7. Inspect the action_net Linear layer: weight row norms per action, biases per action, all pairwise bias gaps, bias-only argmax, `W @ latent_pi` per-action statistics, and the **decisive ratio**: standard deviation of W@latent on the top-bias action divided by the absolute bias gap between top-bias and second-bias actions. If this ratio is small the head is bias-dominated; if large the head's projection is the dominant signal.
8. Write a single JSON summary per tape with the header (fingerprints, eval seed, tape, n_steps, elapsed, ran_at_utc), the observation hashes summary, and the per-model results.

Two invocations: `--behavior-tape stay --eval-seed 1000 --max-steps 1800` and `--behavior-tape left --eval-seed 1000 --max-steps 1800`. Both load all four models at once. Total Phase I driver wall time on StrongerJr was about 1 minute 25 seconds for two sequential windowed-Godot pixel-mode runs followed by activation collection. Per-tape elapsed: 3.3 seconds stay, 13.2 seconds left.

## 4. Model fingerprints carried forward

All four model fingerprints from Phase H are preserved bit-identical here. The Phase I comparator re-loads from the same train run directories and re-fingerprints; SHA-256, parameter count, and policy state-dict blake2b-128 hash match Phase H section 3.

| Label | SHA-256 (16 hex prefix) | param hash (blake2b-128) | param count | size bytes |
|---|---|---|---|---|
| phase_e_seed1 | `2126014dae0cedbf` | `b59518de034867e6a8ff6db638827561` | 5036004 | 20237577 |
| phase_e_seed2 | `5e5fbba2d7580bfb` | `9dee74fb5b87e63e1b2f2816f8c03baa` | 5036004 | 20237577 |
| phase_g_seed1 | `51e02ecdb6234b64` | `638672a424d62cb360a5f356ef2de677` | 5036004 | 20237577 |
| phase_g_seed2 | `705f56596a69a415` | `a587646124d06fc87690965988f38ba3` | 5036004 | 20237577 |

Parameter count uniform at 5,036,004 across all four artifacts: NatureCNN (1 input channel, three convs with channel widths 32, 64, 64, then a flatten and a 512-unit Linear) plus an `action_net` Linear(512, 3) and a `value_net` Linear(512, 1). Default SB3 `ActorCriticCnnPolicy` `net_arch=None` resolves to empty MLP layers, so `mlp_extractor.policy_net` and `mlp_extractor.value_net` are identity modules. This is confirmed in section 6 below: `latent_pi`, `latent_vf`, and `features` are bit-identical across every step and every model.

## 5. Observation freshness, rechecked

Phase I re-collects observation hashes during its env pass. Counts match Phase H section 4 within the same eval-seed-and-tape pair, confirming the trajectories are reproducible across sessions:

| Tape | Steps | Unique obs hashes | All distinct |
|---|---:|---:|---:|
| stay | 333 | 309 | false |
| left | 1383 | 1381 | false |

The stay-tape obs sequence reproduces Phase H's 309/333. The left-tape obs sequence reproduces Phase H's 1381/1383. The 24 duplicate-hash steps on the stay tape and the 2 duplicate-hash steps on the left tape are reproducible and consistent with brief near-stationary stretches before terminal events.

## 6. Actor-path extraction validity check

The SB3 consistency check (`max |softmax(raw_logits) - dist.probs|`) returns near-zero across all eight model-tape pairs:

| Model | Stay tape cons_err | Left tape cons_err |
|---|---|---|
| phase_e_seed1 | 6.98e-10 | 8.15e-10 |
| phase_e_seed2 | 5.96e-8 | 1.19e-7 |
| phase_g_seed1 | 1.64e-11 | 1.82e-11 |
| phase_g_seed2 | 1.16e-9 | 1.19e-7 |

All errors are at or below 1.2e-7. Branch E (tensor extraction mismatch) is falsified. The directly-extracted actor path matches the SB3 distribution path to within numerical noise of float32 softmax.

Additionally, `latent_pi.shape == latent_vf.shape == features.shape == (1, 512)` for every model and every step. Component-wise comparison shows `latent_pi`, `latent_vf`, and `features` are bit-identical: every per-layer summary statistic for `latent_pi` matches the corresponding `features` statistic to all reported decimal places (norm mean, norm std, per-dim std, adjacent-step L2, effective rank). This is the predicted consequence of default `CnnPolicy` `net_arch=None`: `mlp_extractor.policy_net = mlp_extractor.value_net = nn.Sequential()` (identity). The actor latent IS the encoder output.

Practical consequence for the branch table: GPT's Branch B (encoder sees, actor latent collapses) is empty by construction in this configuration. Branch C and Branch D are the only live verdict options once Branches A and E are falsified.

## 7. Stay tape results (333 steps, 309 unique obs)

Feature-layer activity:

| Model | features norm mean | features norm std | per-dim std mean | per-dim std p95 | adj-step L2 mean | first-vs-last L2 | eff rank | top1 PCA | top3 PCA | frac dim std > 1e-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase_e_seed1 | 143.249 | 0.254 | 0.0073 | 0.0282 | 0.077 | 0.632 | 1.78 | 0.747 | 0.801 | 0.318 |
| phase_e_seed2 | 122.261 | 2.472 | 0.0627 | 0.2314 | 0.658 | 1.713 | 1.04 | 0.983 | 0.986 | 0.338 |
| phase_g_seed1 | 161.524 | 0.268 | 0.0078 | 0.0306 | 0.087 | 0.588 | 1.87 | 0.729 | 0.781 | 0.316 |
| phase_g_seed2 | 129.022 | 2.942 | 0.0751 | 0.2615 | 0.664 | 1.568 | 1.03 | 0.985 | 0.988 | 0.332 |

Class A (E1, G1): features norm shifts by ~0.18% across the trajectory (norm std 0.25 on mean 143-162). Adjacent-step L2 about 0.08 against per-step norm of about 150 (0.05% per-step change). About one-third of the 512 feature dims show std above 1e-4. Effective rank 1.78-1.87 with top-3 PCA capturing 78-80%: features lie roughly on a 2-d manifold in the 512-dim space.

Class B (E2, G2): features norm shifts by ~2% across the trajectory (norm std 2.5-2.9 on mean 122-129). Adjacent-step L2 about 0.66, an order of magnitude larger than Class A's 0.08. Effective rank 1.03-1.04 with top-1 PCA at 0.98: features lie essentially on a 1-d manifold but with substantial movement along that single direction. Same proportion (~33%) of feature dims active.

Raw-logit layer:

| Model | argmax | raw margin mean | raw margin min | raw margin max | per-action std (selected) | entropy mean |
|---|---|---:|---:|---:|---:|---:|
| phase_e_seed1 | stay 100% | 6.5452 | 6.5221 | 6.5627 | stay 0.0120 | 0.0111 |
| phase_e_seed2 | left 100% | 1.4099 | 1.3651 | 1.4734 | left 0.0481 | 0.6650 |
| phase_g_seed1 | stay 100% | 11.0860 | 11.0502 | 11.1145 | stay 0.0176 | 0.0002 |
| phase_g_seed2 | left 100% | 6.3064 | 6.1066 | 6.6425 | left 0.1549 | 0.0141 |

Action_net audit:

| Model | top bias | bias (top) | bias gap (top - second) | proj std (top action) | ratio = proj std / bias gap | weight row norm (top) |
|---|---|---:|---:|---:|---:|---:|
| phase_e_seed1 | stay | 0.002885 | 0.003308 | 0.0120 | 3.62 | 0.0494 |
| phase_e_seed2 | left | 0.000674 | 0.000261 | 0.0481 | 183.94 | 0.0247 |
| phase_g_seed1 | stay | 0.004012 | 0.002980 | 0.0176 | 5.91 | 0.0680 |
| phase_g_seed2 | left | 0.004017 | 0.004222 | 0.1549 | 36.70 | 0.0551 |

Class A on stay tape: ratio 3.6-5.9. The W@latent projection moves substantially more than the bias gap, so the head is projection-driven, not bias-driven. But the projection std is small in absolute terms (0.012-0.018) against margins of 6.5-11.1 — margin headroom is ~540x to 630x the projection std. The action head emits effectively the same raw logits at every step.

Class B on stay tape: ratio 38-184. The projection vastly dominates the bias gap, so the head is reading the encoder. Projection std 0.05-0.15 against margins 1.4-6.3 — margin headroom 28x (E2) to 41x (G2) the projection std. Real per-step variation, never crosses argmax.

## 8. Left tape results (1383 steps, 1381 unique obs)

Feature-layer activity:

| Model | features norm mean | features norm std | per-dim std mean | per-dim std p95 | adj-step L2 mean | first-vs-last L2 | eff rank | top1 PCA | frac dim std > 1e-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| phase_e_seed1 | 141.401 | 0.239 | 0.0076 | 0.0277 | 0.105 | 2.089 | 2.56 | 0.622 | 0.318 |
| phase_e_seed2 | 77.454 | 8.517 | 0.2138 | 0.8008 | 0.845 | 44.260 | 1.01 | 0.993 | 0.332 |
| phase_g_seed1 | 159.718 | 0.300 | 0.0091 | 0.0345 | 0.116 | 1.968 | 2.20 | 0.673 | 0.318 |
| phase_g_seed2 | 88.950 | 10.539 | 0.2684 | 0.9598 | 0.859 | 34.675 | 1.01 | 0.995 | 0.332 |

Class A on the left tape sees more obs diversity than on the stay tape (effective rank rises to 2.20-2.56 from 1.78-1.87; first-vs-last L2 rises to 1.97-2.09 from 0.59-0.63). The encoder responds to that diversity with proportionally larger variation (adj-step L2 0.105-0.116 vs 0.077-0.087). But the per-dimension std mean stays small (0.008-0.009).

Class B on the left tape shows **dramatic** feature variation: norm std 8.5 (E2) and 10.5 (G2) on a norm mean of 77-89 — about 11-12% of the mean. First-vs-last L2 is 44.3 (E2) and 34.7 (G2), so the final-step features are about 30-50% of one norm-length away from the initial-step features. Effective rank 1.01-1.02 with top-1 PCA at 0.99: a single high-variance direction is doing nearly all the work.

Raw-logit layer:

| Model | argmax | raw margin mean | raw margin min | raw margin max | per-action std (selected) | entropy mean |
|---|---|---:|---:|---:|---:|---:|
| phase_e_seed1 | stay 100% | 6.4589 | 6.4380 | 6.5618 | stay 0.0112 | 0.0120 |
| phase_e_seed2 | left 100% | 0.8301 | 0.7375 | 1.4617 | left 0.1648 | 0.8880 |
| phase_g_seed1 | stay 100% | 10.9601 | 10.9280 | 11.1084 | stay 0.0196 | 0.0002 |
| phase_g_seed2 | left 100% | 4.2650 | 3.7233 | 6.8061 | left 0.5584 | 0.0903 |

Action_net audit:

| Model | top bias | bias gap (top - second) | proj std (top action) | ratio = proj std / bias gap |
|---|---|---:|---:|---:|
| phase_e_seed1 | stay | 0.003308 | 0.0112 | 3.40 |
| phase_e_seed2 | left | 0.000261 | 0.1648 | 630.72 |
| phase_g_seed1 | stay | 0.002980 | 0.0196 | 6.59 |
| phase_g_seed2 | left | 0.004222 | 0.5584 | 132.26 |

**The smallest margin headroom across all eight model-tape configurations is phase_e_seed2 on the left tape**: margin min 0.738, margin mean 0.830, selected-action projection std 0.165. Ratio mean-margin / selected-std is 5.0. Ratio min-margin / selected-std is 4.5. This is the cell where stochastic-action sampling would be most likely to surface a non-`left` action: probability entropy mean 0.888 (out of 1.099 for discrete-3 uniform), and Phase H reported the per-action probability means as left 0.621, right 0.271, stay 0.108.

Class A models on the left tape behave essentially as they did on the stay tape: raw margin 6.5-11, selected-action std 0.011-0.020, ratio ~3.4-6.6 between projection-std and bias-gap (so projection-driven), but margin-to-std ratio ~560-580 (saturated argmax).

## 9. Cross-tape within-model comparison

For each model, comparing its activity on the stay tape vs the left tape:

| Model | features norm std (stay → left) | features adj L2 (stay → left) | features first-vs-last L2 (stay → left) | raw margin (stay → left) | selected std (stay → left) |
|---|---|---|---|---|---|
| phase_e_seed1 | 0.254 → 0.239 | 0.077 → 0.105 | 0.632 → 2.089 | 6.5452 → 6.4589 | 0.0120 → 0.0112 |
| phase_e_seed2 | 2.472 → 8.517 | 0.658 → 0.845 | 1.713 → 44.260 | 1.4099 → 0.8301 | 0.0481 → 0.1648 |
| phase_g_seed1 | 0.268 → 0.300 | 0.087 → 0.116 | 0.588 → 1.968 | 11.0860 → 10.9601 | 0.0176 → 0.0196 |
| phase_g_seed2 | 2.942 → 10.539 | 0.664 → 0.859 | 1.568 → 34.675 | 6.3064 → 4.2650 | 0.1549 → 0.5584 |

Three observations.

First, Class A networks (E1, G1) are nearly invariant across the two tapes: features-norm-std barely moves, adj-step L2 increases by about 30% (from 0.08 to 0.10), and raw-logit margin changes by less than 2%. The action head is producing essentially the same raw logits on both trajectories. This is the strongest possible expression of Branch C for the seed=1 basin.

Second, Class B networks (E2, G2) show large changes across tapes: features norm std triples to quadruples (2.5 → 8.5 for E2; 2.9 → 10.5 for G2), first-vs-last L2 rises from ~1.6 to 34-44 (20x to 30x), and the raw-logit margin SHRINKS by 40% for E2 (1.41 → 0.83) and by 32% for G2 (6.31 → 4.27). The selected-action projection std also triples to quadruples. The seed=2 basin's policy IS reading the observation and responding to it; the response just stays below the margin threshold to flip argmax.

Third, Phase G reward shaping (E1 → G1, E2 → G2) is observable inside the network without changing argmax behavior. On both tapes Phase G widens raw margins by 1.7x (Class A) to 4.5-5.1x (Class B). The shaped reward biased the optimizer toward sharper distributions on the same already-selected action; it did not move any model across basins. This is consistent with Phase G's eval-trajectory falsification: same trajectories, same argmax, sharper internal distributions.

## 10. Action-net bias and weight inspection

Action_net is a single Linear layer mapping 512-d latent_pi to 3-d raw logits. Weights and biases are completely model-specific (no shared parameters across the four), but the small absolute scale of the biases means the projection `W @ latent_pi` is the dominant contributor to the raw logits.

Biases per model (rounded to 6 decimal places):

| Model | bias[left] | bias[stay] | bias[right] | bias-only argmax |
|---|---:|---:|---:|---|
| phase_e_seed1 | -0.000866 | 0.002885 | -0.000422 | stay |
| phase_e_seed2 | 0.000674 | 0.000231 | 0.000413 | left |
| phase_g_seed1 | 0.001032 | 0.004012 | -0.003373 | stay |
| phase_g_seed2 | 0.004017 | -0.000206 | -0.002058 | left |

Weight row norms per model:

| Model | norm(W[left]) | norm(W[stay]) | norm(W[right]) |
|---|---:|---:|---:|
| phase_e_seed1 | 0.0338 | 0.0494 | 0.0208 |
| phase_e_seed2 | 0.0247 | 0.0273 | 0.0178 |
| phase_g_seed1 | 0.0185 | 0.0680 | 0.0317 |
| phase_g_seed2 | 0.0551 | 0.0197 | 0.0278 |

Three things to note.

First, **biases agree with argmax in every case**: every model's bias-only argmax matches the observed argmax. But the magnitudes are tiny — 1e-3 to 4e-3 — while projection means are 6 to 11. Removing the biases entirely would not change argmax for any of the four networks at any observed step on either tape. The biases are along for the ride; the projections do the work.

Second, **the weight row norm for the selected action is the largest of the three** in every case. E1 and G1 have the largest `W[stay]` row (0.049 and 0.068); E2 and G2 have the largest `W[left]` row (0.025 vs 0.027 vs 0.018 for E2; 0.055 vs 0.020 vs 0.028 for G2). The action head learned to maximize the row norm of the selected action's weight. Combined with a feature vector that has consistently large norm (77-162) and an effective rank near 1 in Class B, the projection mean is dominated by the alignment of `W[selected]` with the principal feature direction.

Third, **Phase G's shaping increased the selected-action weight-row norm**: E1's W[stay] 0.049 → G1's W[stay] 0.068 (37% increase). E2's W[left] 0.025 → G2's W[left] 0.055 (123% increase). The shaping is finding extra reward signal that the optimizer uses to grow the selected-action weight row, which combined with already-large feature norm produces the wider raw-logit margins observed in section 7 and section 8.

## 11. Branch-table verdict

Mapping the Phase I scope-note branch table to the measured evidence:

- **Branch A: encoder collapse.** Falsified. Features show non-trivial per-dim std (mean 0.007-0.27 on 512 dims), per-trajectory adjacent-step L2 of 0.08-0.86, and first-vs-last L2 of 0.59 to 44.26. About a third of feature dimensions have std above 1e-4 in every model and tape. The encoder is producing observation-correlated output.
- **Branch B: encoder sees, actor latent collapses.** Empty by construction. Default `CnnPolicy` `net_arch=None` resolves to `nn.Sequential()` for both `mlp_extractor.policy_net` and `mlp_extractor.value_net`, so `latent_pi == latent_vf == features` for every model and step. Section 6 verified this bit-identically. The actor latent IS the encoder output; there is no separate MLP latent for it to collapse on.
- **Branch C: encoder and actor latent vary, action head ignores them.** **Verdict for Class A (phase_e_seed1, phase_g_seed1)**. Features vary modestly (eff rank 1.78-2.56, adj-step L2 0.08-0.12). Raw logits for the selected action vary tiny (std 0.011-0.020). Bias-only argmax matches actual argmax, but biases are 1e-3 to 4e-3 against projection means of 6.5-11.1, so the constant-classifier behavior arises from the action_net producing near-constant W@latent outputs (selected-action projection std ~1% of the projection mean), not from a bias-dominated regime. Margin-to-std ratio 540-630.
- **Branch D: all actor layers vary, but argmax stays constant.** **Verdict for Class B (phase_e_seed2, phase_g_seed2)**. Features vary substantially (eff rank 1.01-1.04 with large norm std 2.5-10.5, adj-step L2 0.66-0.86, first-vs-last L2 1.6-44.3). Raw logits vary substantially (selected-action std 0.05-0.56). Probabilities vary (entropy 0.09-0.89). But argmax remains `left` because no step's projection excursion exceeds the raw margin. Smallest headroom: E2 on left tape, mean-margin/selected-std = 5.0, min-margin/selected-std = 4.5.
- **Branch E: tensor extraction mismatch.** Falsified at near-machine precision. Max `|manual softmax - SB3 distribution probs|` is 1.19e-7 across all eight model-tape pairs. The actor-path extraction is operationally identical to SB3's distribution path.

The Phase H "policies are observation-insensitive" verdict is REFINED, not falsified:

- True at the argmax surface: every trained network's argmax is constant per train seed, regardless of trajectory. Phase H established this.
- False at the feature surface: every trained network's `features_extractor` output responds measurably to observation differences. Phase I established this.
- False at the raw-logit surface for Class B: per-action raw-logit std reaches 0.56 on a 4.3 mean margin (G2 left tape). The policy distribution IS changing across steps.
- True at the raw-logit surface for Class A: per-action raw-logit std is bounded by 0.02 on margins of 6.5-11.1. The action head emits effectively the same raw logits at every step despite varying features. This is the strict "action head ignores them" signature.

Two different verdicts for two different basins. The unified reason: the trained raw-logit margins in both basins are larger than the per-step projection variation that the encoder produces. Class A's margins (6.5-11.1) are 100x larger than its selected-action projection std (0.011-0.020). Class B's margins (0.8-6.3) are 4.5x to 41x larger than its selected-action projection std (0.05-0.56). Both basins are below the threshold at which observation excursions could flip argmax under deterministic eval.

## 12. Next slice recommendation

GPT's Branch D entry explicitly says: "Next slice: stochastic-action eval becomes legitimate again, especially for Class B." Phase I's measurements make that recommendation precise.

**Operational next slice (docs-and-tools-only): Phase J stochastic-action eval ablation on the four trained networks, weighted toward Class B.**

Scope:

1. Run the existing trained-policy eval path under `model.predict(obs, deterministic=False)` (categorical sampling from `dist.distribution`) on the same 10-seed `runs/phase_h`/`runs/phase_g` eval set (seeds 1000-1009). The H5 baseline CLI already accepts the deterministic flag at the YAML level; switch to `deterministic: false` for the Phase J eval config or thread a CLI override.
2. Run a per-seed paired comparison: same model, same env seed, deterministic eval vs stochastic eval, per-step action sequence diff, terminal type, episode length, collision/truncation.
3. Report per-model: action-distribution shift, fraction of steps where stochastic sample differs from deterministic argmax, episode length under stochastic eval, terminal reason histogram.
4. Specifically target Phase E seed2 (highest entropy 0.888, smallest min margin 0.738) and Phase G seed2 (entropy 0.090, min margin 3.72). Phase G seed1 (entropy 0.0002) is the negative control: stochastic eval there should be operationally indistinguishable from deterministic.

Predicted outcomes (MEDIUM confidence, source LABEL: PHASE_I_MEASUREMENT_PROJECTION):

- **Phase G seed1 stochastic eval**: per-step sample differs from argmax in < 0.02% of steps (entropy 0.0002). Episode length and terminal reason unchanged. Falsification target only.
- **Phase E seed1 stochastic eval**: per-step sample differs from argmax in roughly 1% of steps (entropy 0.011 implies ~0.5% non-argmax probability mass). Small effect on episode length and terminal reason.
- **Phase G seed2 stochastic eval**: per-step sample differs from argmax in roughly 1-3% of steps (entropy 0.090). Modest effect on trajectories; left still dominates.
- **Phase E seed2 stochastic eval**: per-step sample differs from argmax in roughly 30-40% of steps (entropy 0.888, left 0.62, right 0.27, stay 0.11). Substantial trajectory change. **This is the cell most likely to produce a different terminal outcome and a different per-step trajectory under stochastic eval.**

What stochastic eval would PROVE if it changes terminal outcomes for E2:
- The deterministic-argmax eval surface was indeed masking trained-policy behavior in Class B.
- The Class B network has learned a non-trivial action distribution and is "playing the game" probabilistically even though its argmax never moves.
- Reward shaping (E2 → G2) collapsed that distribution to near-deterministic without changing argmax, removing the stochastic-eval-recoverable signal.

What stochastic eval would PROVE if it does NOT change terminal outcomes for E2:
- Even with 38% non-`left` per-step actions, the env-trajectory class is robust to action sampling, suggesting the env dynamics are dominated by a small number of decision points (likely the hazard-onset frames) and the trained-policy entropy is uninformatively distributed across irrelevant frames.
- The H5 plateau is structural in the env-policy coupling, not recoverable by removing eval determinism.

Either outcome localizes the H5 bottleneck further and decides between (a) eval-only fix path (use stochastic eval, possibly with temperature tuning) and (b) train-only fix path (force wider exploration or different objective shaping during training).

**Secondary Phase I observation that may merit a separate slice eventually but is NOT the cheapest next step:** the seed=1 basin's raw-logit margins are 10x larger than the seed=2 basin's in only 10k training steps with the same hyperparameters and the same random-policy initialization distribution. Class A E1 margin 6.5 vs Class B E2 margin 0.83. Either the train_seed=1 random initialization put the action_net `W[stay]` weight row close to alignment with the principal feature direction such that PPO's policy gradient amplified the margin much faster, or the train_seed=1 early-rollout collisions produced a sharper TD signal that drove the optimizer harder. Diagnosing this would require a training-time gradient-flow probe across a few seeds and is Grok-trigger territory per charter (RL internals + ambiguity).

## 12.1 Grok sanity check status

GPT's Phase I scope note included a Grok prompt asking whether the SB3 actor-path extraction points (`extract_features`, `mlp_extractor`, `action_net`, `value_net`, `get_distribution`) are valid for PPO `CnnPolicy` and whether any failure modes could make the branch logic misleading. That prompt has NOT been relayed to Grok yet. The reasons to relay it before Phase J:

- The Phase I extraction yielded `manual_softmax == SB3 distribution probs` to within 1.2e-7 in every cell; the path is operationally correct.
- The `mlp_extractor` returning identity-equivalent latent_pi and latent_vf is the documented SB3 default when `net_arch=None`; no Grok input needed there.
- A Phase J stochastic-action eval ablation does NOT depend on activation-extraction correctness; it uses `model.predict(obs, deterministic=False)` which is the standard SB3 inference API.

A Grok sanity check is still valuable as a pre-execution safety net for any FUTURE slice that probes deeper into the action_net training dynamics (e.g. the seed=1 vs seed=2 margin asymmetry described above). Recommended Grok prompt for future relay, verbatim from GPT:

> We are instrumenting SB3 PPO `CnnPolicy` trained on channel-first pixel observations. We plan to compare `extract_features(obs_tensor)`, `mlp_extractor(features)`, `action_net(latent_pi)`, raw softmax probabilities, and SB3 `get_distribution` probabilities across fixed observation tapes. Goal is to localize observation-insensitive constant-action policies to encoder collapse vs actor-latent collapse vs final action-head bias. Are these extraction points valid for SB3 PPO CnnPolicy, and what failure modes could make this branch logic misleading?

Jeff can relay this before scoping the seed-asymmetry training-time probe, not before Phase J stochastic eval.

## 13. Reproduction recipe

Environment requirements identical to Phase H:

- Windows 11, repo at `C:\Projects\Sight`, Python at `C:\Users\maste\AppData\Local\Python\bin\python.exe`
- `stable-baselines3`, `torch`, `gymnasium`, `numpy` from the same env used by Phase E and Phase G training
- Godot 4.6.2 at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`
- The four target train run directories under `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed{1,2}_entropy_10k/` and `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped/h5_train_phase_g_shaped_seed{1,2}_10k/`, each containing `model.zip`. Source commits as recorded in Phase H section 10.

Single-tape invocation (cmd.exe):

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
"C:\Users\maste\AppData\Local\Python\bin\python.exe" tools\h5_activation_compare.py ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
    --models phase_e_seed1=runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed1_entropy_10k,phase_e_seed2=runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed2_entropy_10k,phase_g_seed1=runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped\h5_train_phase_g_shaped_seed1_10k,phase_g_seed2=runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped\h5_train_phase_g_shaped_seed2_10k ^
    --eval-seed 1000 ^
    --max-steps 1800 ^
    --behavior-tape stay ^
    --out-dir runs\phase_i
```

Switch `--behavior-tape stay` to `--behavior-tape left` for the Class B trajectory. Both invocations write a single summary JSON file each plus a Godot-eval sidecar directory.

Two-run driver. The driver used for this evidence is at `%TEMP%\phase_i_driver.bat` (not committed; ephemeral). It calls the comparator twice in sequence with the four-model load list and the two tapes, writes `runs/phase_i/driver.log` and `runs/phase_i/driver.done` sentinels, redirects all stdout and stderr to the log file with `>> %LOG% 2>&1`. Total wall time on StrongerJr was about 1 minute 25 seconds for both runs.

Output artifacts (all gitignored under `runs/`; durable evidence in this doc):

- `runs/phase_i/activation_compare_stay.summary.json`: per-tape header (fingerprints, eval seed, tape, n_steps, elapsed, models), obs hash summary, per-model summaries.
- `runs/phase_i/activation_compare_left.summary.json`: same shape for the left tape.
- `runs/phase_i/activation_stay_godot/`, `runs/phase_i/activation_left_godot/`: Godot env eval NDJSON sidecars from `make_env`.
- `runs/phase_i/driver.log`, `runs/phase_i/driver.done`: driver progress and completion sentinel.

Re-running on the same set of train run directories must produce bit-identical fingerprint values for every model (the four SHA-256 prefixes and the four parameter hashes from Phase H section 3 carried forward to Phase I section 4). Re-running with the same eval seed must produce a bit-identical observation hash sequence per tape; downstream summary JSON values differ only in non-deterministic wall-time fields. Activation summary statistics (norms, eff rank, raw-logit margin, projection std, ratio) are deterministic.

This slice deliberately does not write any artifact to the gitignored `runs/` tree beyond the per-tape summary JSON and Godot eval sidecar. Durable evidence lives in this document plus the committed `tools/h5_activation_compare.py`.
