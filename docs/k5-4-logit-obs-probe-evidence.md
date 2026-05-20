# K5.4 - Replay-Derived Logit/Obs Oracle-Alignment Probe

**Phase:** K (post-K5.3 diagnostic slice)
**Trigger:** GPT K5.4 execution packet, Grok GREEN on K5.3 SOFT-BAD-POLICY.
**Tool:** `tools/k5_4_logit_obs_probe.py`
**Config:** `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml`
**Checkpoint:** `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k/model.zip`
- sha256 `e5eac667157f43d88ace9f11f72bc26c329c5abeb1d5401c7bfd2fe6ff40abc9`
- size 20,237,581 bytes; param_count 5,036,004
- state_dict blake2b16 `159b5f9be9ae4aec0dcd601b427a7102`
**Seeds:** 1000-1009; **collectors:** stay, left, right, oracle, seeded_random, sweep; **max_steps:** 600
**Episodes:** 60; **aligned samples:** 26,079; **wall time:** 726.8 s; **ran_at_utc:** 2026-05-20T15:31:10Z

---

## Verdict

**Primary bucket: STAY-BIASED-MISRANKING.**
**Expansion pass needed: no.** Coverage exceeded all minimums (26,079 >= 3000 total; left 4074, stay 16678, right 5327, each >= 200).

The K5.1 alpha=0.30 CnnPolicy emits argmax `stay` on **100.000% of all 26,079 production-pixel observations**, including every observation where the K5.2 hazard-reactive oracle calls for `left` or `right`. The confusion matrix collapses to three cells, all in the `argmax_stay` column. Per-oracle-label top-1 accuracy is `stay` 1.000, `left` 0.000, `right` 0.000. There is no observation, in any geometry bucket or arrival window, on which argmax leaves `stay`.

This is not an evaluation-protocol problem and not a soft-noise problem. The learned policy's top action is structurally pinned to `stay` and is blind to hazard geometry at the argmax level. K5.3 already showed the soft tail off `stay` has real mass (entropy 0.61 nats); K5.4 shows that mass is not conditioned on hazard position - the argmax never moves and the residual probability does not rank the oracle action.

---

## Run identity

| field | value |
|:------|:------|
| tool | `tools/k5_4_logit_obs_probe.py` |
| model checkpoint sha256 | `e5eac667...40abc9` |
| state_dict blake2b16 | `159b5f9be9ae4aec0dcd601b427a7102` |
| param_count | 5,036,004 |
| n_episodes | 60 |
| n_aligned_samples | 26,079 |
| collectors | stay, left, right, oracle, seeded_random, sweep |
| seeds | 1000-1009 |
| max_steps | 600 |

`reward_state` extraction note: the K5.4 tool reads `info["godot_info"]["reward_state"]`. The Godot wire info is forwarded nested under `info["godot_info"]` by `src/sight_agent/rl/godot_env.py` `_build_info`. The first K5.4 launch read a flat `info["reward_state"]` path, recorded 0 rows on every episode, and was killed and corrected before the run reported here. The flat path was confirmed dead by a standalone Godot smoke test; the nested path returned `reward_state` on 10/10 steps with keys `hazards_above`, `player_x`, `player_y`.

---

## Coverage table

| dimension | bucket | samples |
|:----------|:-------|--------:|
| collector | stay | 4188 |
| collector | left | 4935 |
| collector | right | 4545 |
| collector | oracle | 6000 |
| collector | seeded_random | 3115 |
| collector | sweep | 3296 |
| oracle label | left | 4074 |
| oracle label | stay | 16678 |
| oracle label | right | 5327 |
| player_x bucket | left_wall | 5799 |
| player_x bucket | center | 9678 |
| player_x bucket | right_wall | 4977 |
| player_x bucket | other | 5625 |
| arrival bucket | none | 7740 |
| arrival bucket | le15 | 8055 |
| arrival bucket | le30 | 8484 |
| arrival bucket | le60 | 1800 |
| nearest_dx bucket | none | 7740 |
| nearest_dx bucket | left | 8655 |
| nearest_dx bucket | centerline | 1197 |
| nearest_dx bucket | right | 8487 |
| imminent_threat | false | 7740 |
| imminent_threat | true | 18339 |

Coverage gap: `arrival_bucket=gt60` has 0 samples. The geometry definition only labels a hazard imminent at arrival_steps <= 60, so any hazard beyond that window is not counted and `gt60` is unreachable by construction. This is a definitional artifact, not a sampling shortfall, and does not weaken the verdict.

---

## Oracle label distribution

| oracle label | count | fraction |
|:-------------|------:|---------:|
| left  | 4074  | 0.156 |
| stay  | 16678 | 0.640 |
| right | 5327  | 0.204 |

The 0.640 stay fraction is the reason overall top-1 accuracy reads 0.6395 despite the policy never selecting a non-stay action. Overall accuracy here equals the oracle stay-label fraction exactly: a constant-stay predictor scores the stay-label share and nothing else. Overall accuracy must not be read as competence; per-label accuracy is the load-bearing metric.

---

## Confusion matrix (oracle label x argmax)

| oracle \ argmax | left | stay | right |
|:----------------|-----:|-----:|------:|
| left  | 0 | 4074  | 0 |
| stay  | 0 | 16678 | 0 |
| right | 0 | 5327  | 0 |

Every off-diagonal mass is in the `argmax_stay` column. The policy has exactly one argmax behavior across the entire production-pixel observation set.

---

## Oracle top-1 accuracy

**Overall:** 0.6395 (= oracle stay-label fraction; see above).

**By oracle label:**

| oracle label | accuracy | mean oracle rank | mean p_oracle |
|:-------------|---------:|-----------------:|--------------:|
| left  | 0.000 | 3.00 | 0.0905 |
| stay  | 1.000 | 1.00 | 0.7691 |
| right | 0.000 | 2.00 | 0.1268 |

When the oracle calls `left`, the policy ranks `left` dead last (rank 3.00) on every such observation. When the oracle calls `right`, the policy ranks `right` second (rank 2.00). The policy's fixed probability ordering is `stay > right > left` on essentially every observation, regardless of which direction the hazard geometry requires.

**By player_x bucket:**

| bucket | n | accuracy | argmax stay frac |
|:-------|--:|---------:|-----------------:|
| left_wall  | 5799 | 0.821 | 1.000 |
| center     | 9678 | 0.585 | 1.000 |
| right_wall | 4977 | 0.823 | 1.000 |
| other      | 5625 | 0.383 | 1.000 |

**By arrival bucket:**

| bucket | n | accuracy | argmax stay frac |
|:-------|--:|---------:|-----------------:|
| none | 7740 | 1.000 | 1.000 |
| le15 | 8055 | 0.533 | 1.000 |
| le30 | 8484 | 0.490 | 1.000 |
| le60 | 1800 | 0.271 | 1.000 |

**By nearest_dx bucket:**

| bucket | n | accuracy | argmax stay frac |
|:-------|--:|---------:|-----------------:|
| none       | 7740 | 1.000 | 1.000 |
| left       | 8655 | 0.485 | 1.000 |
| centerline | 1197 | 0.086 | 1.000 |
| right      | 8487 | 0.546 | 1.000 |

Every bucket on every geometry dimension shows `argmax stay fraction = 1.000`. Per-bucket accuracy is entirely a function of how often `stay` happens to be the oracle action in that bucket. The `none` arrival bucket scores 1.000 because with no imminent hazard the oracle returns `stay` and the policy also stays. The `le60` bucket scores worst (0.271) because the closest imminent hazards most often demand motion and the policy never provides it. Accuracy degrades monotonically as the hazard gets closer (none 1.000, le15 0.533, le30 0.490, le60 0.271), which is the signature of a policy that ignores hazard proximity.

---

## p_oracle and margin statistics

| metric | mean | std | min | p05 | p50 | p95 | max |
|:-------|-----:|----:|----:|----:|----:|----:|----:|
| p_oracle overall | 0.5319 | 0.3176 | 0.0663 | 0.0768 | 0.7397 | 0.8162 | 0.8364 |
| p_oracle_minus_best_wrong overall | 0.1649 | 0.6304 | -0.7711 | -0.7381 | 0.5914 | 0.7085 | 0.7396 |
| entropy_nats overall | 0.6847 | 0.0719 | 0.5538 | 0.5765 | 0.7296 | 0.7545 | 0.7814 |
| top1_top2_margin overall | 0.6438 | 0.0561 | 0.5648 | 0.5887 | 0.6100 | 0.7260 | 0.7412 |

The bimodal `p_oracle` distribution (std 0.318, p05 0.077, p95 0.816) is the stay-vs-non-stay split: when the oracle action is `stay` the policy assigns it ~0.77; when the oracle action is `left` or `right` the policy assigns it ~0.09-0.13. `p_oracle_minus_best_wrong` has mean +0.165 but p05 -0.738, again the same split: strongly positive on stay-labeled observations, strongly negative on motion-labeled ones. The distribution is soft (entropy 0.685 nats, well above the 0.10 collapse floor) and the margin (0.644 mean) is consistent with K5.3's 0.704 - a concentrated but non-delta softmax. The probability mass exists; it is simply not conditioned on hazard direction.

---

## Representative failure buckets

- `arrival_bucket=le60`, n=1800, accuracy 0.271: closest imminent hazards, where motion is most often required, are where the policy fails hardest.
- `nearest_dx_bucket=centerline`, n=1197, accuracy 0.086: when a hazard is on the player's centerline (abs dx <= 40), the oracle almost always calls for a dodge; the policy stays and is wrong 91.4% of the time.
- `oracle_label=left`, n=4074, accuracy 0.000, mean rank 3.00: the policy never ranks `left` anywhere but last. The single largest blind spot.
- `player_x_bucket=other`, n=5625, accuracy 0.383: mid-field positions away from walls, where dodging both directions is live, are scored low because the policy provides no dodging.

---

## Interpretation

K5.4 confirms and sharpens K5.3. K5.3 classified the checkpoint SOFT-BAD-POLICY: soft distribution, real off-argmax mass, but sampling does not improve survival. K5.4 explains why. The argmax is pinned to `stay` on 100% of production-pixel observations, and the soft probability ordering is a fixed `stay > right > left` that does not respond to hazard geometry. The residual non-stay mass that K5.3 saw under stochastic sampling is real but undirected: sampling from it produces motion uncorrelated with where the hazard is, which is why K5.3 sampled survival (543.0) was worse than deterministic stay (606.0).

The K5.2 layer-6 oracle reaches 1762.8 mean frames on these seeds, and the K5.4 oracle collector survived all 10 seeds to the 600-step cap, so a hazard-conditioned policy is reachable in the env and the function class. The K5.1 checkpoint has not learned the hazard-position-to-action mapping at all. The pathology is in representation or training, not eval protocol, not env, not reward scale alone.

This routes to the GPT K5.4 packet branch for STAY-BIASED-MISRANKING: "K5.1 learned a real soft distribution but the top action is stuck at stay even when geometry calls for motion. This supports representation/objective misranking."

---

## Routing for K5.5

GPT K5.4 packet routing for STAY-BIASED-MISRANKING: "routes to frame_stack=4 or state-observation PPO control. Prefer state-observation control first if cheap, because it isolates pixel representation."

Recommended K5.5 scope: a state-observation PPO control run. Train PPO on the same Signal Dodge task and the same 10k-step budget, alpha=0.30 shaping, but with a low-dimensional state-vector observation (player_x, player_y, nearest-hazard relative geometry) instead of single-frame pixels. This isolates one variable: whether the stay-pinned argmax is caused by the single-frame pixel representation failing to encode hazard geometry, or by the PPO objective / budget failing to learn the mapping even when the geometry is handed to it directly.

- If the state-observation control learns a hazard-conditioned policy at 10k steps, the pixel representation is implicated and K5.6 is frame_stack=4 or a CNN feature-extractor change.
- If the state-observation control also collapses to stay-pinned argmax, the representation is exonerated and the problem is the PPO objective, budget, or exploration; K5.6 is a longer-budget or PPO-hyperparameter slice.

This is the cheap variable-isolating move and should precede frame_stack=4, which is more expensive and only worth running once the representation is actually implicated. Capacity sweeps, longer-budget retrains, and reward-shape revision remain premature.

Note for K5.5 scoping: this is a new training run, which is a scope/direction change. GPT should scope it and Grok should phase-gate it before Claude executes.

---

## Reproduction

```
cd /d C:\Projects\Sight

REM .bat + sentinel pattern used for the K5.4 run:
REM   C:\Users\maste\AppData\Local\Temp\k5_4_run_logit_obs_probe.bat
REM   sentinel C:\Users\maste\AppData\Local\Temp\k5_4_logit_obs_probe.done
REM   log      C:\Users\maste\AppData\Local\Temp\k5_4_logit_obs_probe.log

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\k5_4_logit_obs_probe.py ^
  --config configs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml ^
  --model-label k5_1_alpha030_seed0_10k ^
  --model-dir runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030\k5_1_alpha030_seed0_10k ^
  --seeds 1000-1009 ^
  --collector-policies stay,left,right,oracle,seeded_random,sweep ^
  --max-steps 600 ^
  --out-dir runs\phase_k\k5_4_logit_obs_probe
```

Outputs:
- `runs/phase_k/k5_4_logit_obs_probe/k5_4_logit_obs_probe.ndjson`
- `runs/phase_k/k5_4_logit_obs_probe/k5_4_logit_obs_probe.summary.json`

---

## Notes

- The K5.4 tool reuses `h5_logit_compare.load_and_fingerprint`, `h5_logit_compare.get_action_logits`, `h5_logit_compare.obs_hash`, and `k5_2_env_dynamics_probe.hazard_reactive_oracle` / `HAZARD_SPEED_PX_STEP` per the GPT K5.4 packet. It does not modify `h5_logit_compare.py`.
- All 26,079 samples are real Godot viewport pixels through the production H3 pixel path. No synthetic observations, no Godot source or debug-protocol edits, consistent with the GPT amendment that K5.4 must be replay-derived.
- The `info["godot_info"]["reward_state"]` nesting is the correct path. The K5.3 evidence doc described its `player_x` extraction as opportunistic; in fact K5.3 was reading the dead flat path and `player_x` was always None in K5.3 output. The K5.3 verdict (SOFT-BAD-POLICY) does not depend on `player_x` and stands; the K5.3 evidence doc's player_x line is technically true but should be read as "not captured" rather than "opportunistically captured".
- Overall oracle top-1 accuracy (0.6395) is not a competence signal. It equals the oracle stay-label fraction because the policy is a constant-stay predictor. Future readers should use the per-label accuracy block, not the overall figure.
- The monotone accuracy decay across arrival buckets (none 1.000, le15 0.533, le30 0.490, le60 0.271) is direct evidence that the policy does not respond to hazard proximity; a hazard-aware policy would show the opposite or flat profile.
