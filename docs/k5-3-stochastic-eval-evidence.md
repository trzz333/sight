# K5.3 - Stochastic Eval Evidence Pack

**Phase:** K (post-ENV-PASS diagnostic slice)
**Trigger:** GPT K5.3 execution packet, Grok GREEN on K5.2 ENV-PASS.
**Tool:** `tools/h5_stochastic_eval.py` (K5.3-patched).
**Config:** `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml`
**Checkpoint:** `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k/model.zip`
- sha256 `e5eac667157f43d88ace9f11f72bc26c329c5abeb1d5401c7bfd2fe6ff40abc9`
- size 20,237,581 bytes; param_count 5,036,004
- state_dict blake2b16 `159b5f9be9ae4aec0dcd601b427a7102`
**Eval seeds:** 1000-1009; **replicates:** 5/seed; **max_steps:** 1800; **deterministic:** False
**policy_sample_seed rule:** `eval_seed + 10_000_000 + replicate_idx`
**Wall time:** 759.9 s; **ran_at_utc:** 2026-05-20T04:56:58Z

---

## Verdict

**Primary bucket: SOFT-BAD-POLICY.**
**near_tie_bias overlay: false.**

The learned CnnPolicy distribution is soft, not collapsed. Stochastic sampling moves off the argmax on 18.9% of steps with a mean entropy of 0.607 nats and a top1-top2 margin centered at 0.704. Yet sampled mean episode length (543.0) is below both the K5.1 deterministic baseline (606.0) and the K5.2 layer-6 best-constant survival bar (845.7), and is well below the material-survival bar (930.27 = 845.7 + 84.57). The pathology is not "argmax kills a usable distribution" and is not "distribution collapsed to a delta." It is "the learned probabilities are misranked or poorly conditioned": there is real probability mass off stay, but sampling from it is worse than the already-bad argmax.

---

## Comparison vs K5.1 deterministic baseline

| eval_seed | det length | stoch length_mean | length_min | length_max | terminal_differs |
|----------:|-----------:|------------------:|-----------:|-----------:|-----------------:|
| 1000 |  333 |  393.0 |  333 |  603 | 0 |
| 1001 |  273 |  381.0 |  183 |  723 | 0 |
| 1002 |  843 |  669.2 |  423 |  964 | 0 |
| 1003 |  963 | 1011.0 |  783 | 1353 | 0 |
| 1004 | 1203 |  753.0 |  273 | 1293 | 0 |
| 1005 | 1263 |  543.2 |  214 |  903 | 0 |
| 1006 |  543 |  369.8 |  243 |  543 | 0 |
| 1007 |  183 |  465.0 |  183 |  963 | 0 |
| 1008 |  183 |  503.8 |  183 | 1067 | 0 |
| 1009 |  273 |  341.0 |  273 |  399 | 0 |
| **mean** | **606.0** | **543.0** | -- | -- | **0/10** |

All 50 stochastic episodes end in collision (collision_rate=1.00, timeout_rate=0.00). All seed-level terminal flags match the deterministic baseline (collision/timeout). On 6 of 10 seeds the stochastic mean is lower than the deterministic baseline; on 4 seeds it is higher; the unweighted stochastic mean (543.0) trails the deterministic mean (606.0) by ~10%.

---

## K5.3 step-weighted statistics

n_steps evaluated: **27,150**

### Step-weighted action fractions

| action | sampled fraction | argmax fraction |
|:-------|-----------------:|----------------:|
| left  | 0.0830 | 0.0000 |
| stay  | 0.8112 | **1.0000** |
| right | 0.1059 | 0.0000 |

`sampled_differs_from_argmax_step_fraction = 0.1888`

### Entropy (nats) over the 3-action categorical

| stat | value |
|:-----|------:|
| mean | 0.6069 |
| std  | 0.0432 |
| min  | 0.5507 |
| p05  | 0.5588 |
| p50  | 0.6022 |
| p95  | 0.6972 |
| max  | 0.7808 |

### top1 - top2 probability margin

| stat | value |
|:-----|------:|
| mean | 0.7040 |
| std  | 0.0321 |
| min  | 0.5654 |
| p05  | 0.6368 |
| p50  | 0.7082 |
| p95  | 0.7378 |
| max  | 0.7432 |
| fraction < 0.05 | **0.0000** |

### Per-action probability stats

| action | mean | std | min | p05 | p50 | p95 | max |
|:-------|-----:|----:|----:|----:|----:|----:|----:|
| left  | 0.0775 | 0.0098 | 0.0657 | 0.0673 | 0.0762 | 0.0980 | 0.1207 |
| stay  | 0.8132 | 0.0209 | 0.7224 | 0.7694 | 0.8160 | 0.8353 | 0.8388 |
| right | 0.1093 | 0.0111 | 0.0956 | 0.0975 | 0.1078 | 0.1326 | 0.1570 |

`argmax_action_step_counts`: left=0, stay=27150, right=0. `argmax_concentrated_on_single_action = true`, `argmax_dominant_action = stay`.

---

## Classification logic applied

K5.3 packet thresholds (Grok GREEN proposal, GPT amendment that NEAR-TIE-BIAS is an overlay, not a bucket):

Primary bucket, ordered:
1. `ARGMAX-ARTIFACT` requires `diff_step_fraction > 0.05` AND `sampled_mean_episode_length > 930.27`.
   - Observed: diff_step_fraction = 0.189 (pass), sampled_mean_episode_length = 543.0 (**fail** the 930.27 bar).
2. `POLICY-DIST-COLLAPSE` requires `diff_step_fraction <= 0.05` OR `entropy_nats.mean < 0.1`.
   - Observed: 0.189 (above 0.05), 0.607 (above 0.1). **Both clauses false.**
3. `SOFT-BAD-POLICY` requires `diff_step_fraction > 0.05` AND `sampled_mean_episode_length <= 930.27`.
   - Observed: 0.189 > 0.05 (pass), 543.0 <= 930.27 (pass). **Match.**

Overlay:
- `near_tie_bias` requires `top1_top2_margin_lt_0p05_fraction > 0.50` AND argmax constant.
  - Observed: 0.000 (fail). Overlay = false.

---

## Interpretation

The K5.1 alpha=0.30 checkpoint has a non-degenerate softmax distribution (stay ~0.81, right ~0.11, left ~0.08) but the action ranking is wrong relative to the env's solvability geometry. K5.2 layer 6 showed a hazard-reactive 1-step oracle reaches 1762.8 mean frames on the same 10 seeds, so a much better policy is available in the function class. K3.5c on the unshaped surface collapsed to constant-left; this shaped alpha=0.30 checkpoint has shifted toward constant-stay under argmax but the soft tail favors right over left (0.106 vs 0.080), which is a third action-distribution shape over the same env-task pair. Three trained networks, three different soft distributions, three argmax fixed points: argmax is reading off the structure that PPO learned, not flattening it.

Sampling-makes-survival-worse implies the learned probabilities encode dodge intent that is on average more wrong than the always-stay behavior the argmax produces. That is consistent with a policy that has been pushed by shaping toward staying close to the column center but has not learned hazard-conditioned action selection: the residual non-stay probability mass is not aligned with hazard kinematics.

This is not an evaluation-protocol problem. It is a learning-pipeline problem inside PPO + CnnPolicy + single-frame (1,84,84) + 10k budget on the shaped surface.

---

## Routing for K5.4

GPT K5.3 packet decision tree, branch matched: "sampled differs from argmax but survival does not improve - misranked or poorly conditioned action probabilities. Next move: logit/obs probe on hazard-relative states, then architecture or frame_stack."

Recommended K5.4 scope: logit/obs probe on hazard-relative states using a small fixed set of synthetic or replay-derived observations where the hazard-relative-x is known. Quantify whether the CnnPolicy's logit ordering correlates with the optimal-action geometry that the K5.2 layer-6 oracle exploits. Candidate harness: `tools/h5_logit_compare.py` already has margin and entropy aggregation; the slice that does not exist yet is generating a controlled obs set where the ground-truth optimal action is known by construction.

Lower-priority candidates that K5.3 evidence makes premature:
- frame_stack=4 retrain. Single-frame observability is a candidate root cause but only if K5.4 shows logits do not correlate with the relevant kinematic feature even on hand-crafted obs that contain it.
- CnnPolicy width sweep. No evidence yet that capacity is binding rather than the optimization target.
- Longer-budget retrain. Risks training a deeper version of the same misranked surface.
- Reward-shape revision. K5.2 oracle on the shaped surface already reached 1462.8 frames; the shaping is consistent with a usable policy in the function class. Revisit only if K5.4 demonstrates that PPO cannot align logits with hazard kinematics under this shape.

---

## Reproduction

```
cd /d C:\Projects\Sight

REM Run directly via Python (may exceed default MCP timeout; see .bat alternative below)
"C:\Users\maste\AppData\Local\Python\bin\python.exe" tools\h5_stochastic_eval.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml ^
  --models k5_1_alpha030_seed0_10k=runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030\k5_1_alpha030_seed0_10k ^
  --seeds 1000-1009 ^
  --replicates 5 ^
  --max-steps 1800 ^
  --out-dir runs\phase_k\k5_3_stochastic_eval ^
  --label-suffix k5_1_alpha030

REM .bat + sentinel pattern used for K5.3 actual run:
REM   C:\Users\maste\AppData\Local\Temp\k5_3_run_stochastic_eval.bat
REM Sentinel: C:\Users\maste\AppData\Local\Temp\k5_3_stochastic_eval.done
REM Log:      C:\Users\maste\AppData\Local\Temp\k5_3_stochastic_eval.log
```

Outputs:
- `runs/phase_k/k5_3_stochastic_eval/stochastic_eval_k5_1_alpha030.summary.json`
- `runs/phase_k/k5_3_stochastic_eval/stochastic_eval_k5_1_alpha030.ndjson`

---

## Notes

- K5.1 deterministic baseline values for seeds 1000-1009 hardcoded in `tools/h5_stochastic_eval.py` under `DETERMINISTIC_BASELINES["k5_1_alpha030_seed0_10k"]` per GPT K5.3 packet. All 10 deterministic episodes collide; lengths 333/273/843/963/1203/1263/543/183/183/273 (mean 606.0).
- Classifier smoke-tested before launch on five synthetic distributions including the GPT amendment case where NEAR-TIE-BIAS overlay coexists with ARGMAX-ARTIFACT primary.
- The 27,150-step pool aggregates evenly across the 50 episodes; per-step weighting was applied for all threshold comparisons.
- Stay-fraction inflation (sampled 81.1% vs argmax 100.0%) shows the soft tail off argmax is small in absolute mass but real and non-zero on every evaluated step.
- The 0.704 mean top1-top2 margin combined with 0.607 nats mean entropy describes a distribution that is concentrated but not delta-like: a clean softmax sitting roughly two-thirds of the way from uniform (~1.099 nats for 3-action uniform) to delta.
