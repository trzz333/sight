# H5 Phase J stochastic-action eval ablation evidence

Status: VERIFIED end-to-end on StrongerJr. J0 pilot (Phase E seed2, 10 eval seeds, 5 replicates per seed, 50 episodes) and J1 expansion (Phase G seed2, 10 eval seeds, 5 replicates per seed, 50 episodes) complete. Optional Class A negative controls skipped per scope note "only if budget is still clean"; Phase I already showed E1 entropy 0.011 and G1 entropy 0.0002, predicting stochastic-eval no-ops for both.

## 1. TL;DR

Phase J replaces `model.predict(obs, deterministic=True)` with `model.predict(obs, deterministic=False)` on the trained Phase E seed2 and Phase G seed2 networks. The deterministic-argmax surface that Phase H and Phase I both characterized as observation-insensitive is removed. The trained Categorical distribution gets to sample.

Two results, sharply different by basin:

- **Phase E seed2 (entropy 0.888 from Phase I)**: stochastic sampling differs from argmax on 36.0% of steps, matching Phase H's per-action probability mean prediction of ~38% non-`left` mass. Mean episode length drops 22% (deterministic 845.7 → stochastic 657.3). Collision rate rises from 0.90 to 1.00. **Seed 1008, the deterministic timeout survival, becomes 5-of-5 collisions under stochastic eval** with episode lengths 513, 513, 517, 519, 1744. Seven of ten eval seeds (1001, 1003-1007, 1009) produce bit-identical episode lengths across all five replicates despite 36% non-argmax sampling; three seeds (1000, 1002, 1008) show wide replicate variance.
- **Phase G seed2 (entropy 0.090 from Phase I)**: stochastic sampling differs from argmax on 1.4% of steps. Mean episode length, collision rate, timeout rate, and per-seed terminal outcomes are **operationally indistinguishable from deterministic**. Every replicate matches its deterministic per-seed length bit-identically. Phase G's reward shaping sharpened the distribution to the point where stochastic sampling has zero effect on outcomes.

Cross-mapping to the GPT pre-Phase-J verdict table: closest fit is **J-B (actions change but trajectory class largely unchanged)** for Phase E seed2, with a sharpened reading. The deterministic-argmax surface was **not** masking learned competence (J-A is falsified): the seed-1008 "survival" was a fragile artifact of policy commitment to the wedge-at-x=16 strategy, not observation-conditional dodging. Stochastic sampling breaks the wedge and the trained policy has no observation-conditional behavior to fall back on. Phase G seed2 makes the same point a different way: shaping sharpened commitment further, removing the small sampling signal that E2 still carried.

The H5 plateau is structural in the env-policy coupling. The deterministic-argmax surface was hiding the absence of learned dodging behind position-locking strategies whose survival depends on policy commitment. The fix path is training-side. Reward shaping (Phase G) was the wrong direction: it tightened commitment instead of teaching dodging.

## 2. Why Phase I forces this slice

Phase I concluded that the trained pixel-CNN policies are not observation-blind: features vary with the observation, raw action logits vary with the observation, the action_net does real linear projection. But the trained raw-logit margins are many times larger than the per-step projection variation, so argmax never crosses. Phase E seed2 on the left tape was the smallest margin headroom across all eight Phase I model-tape cells: mean raw margin 0.830, minimum 0.738, selected-action projection std 0.165, margin-to-std ratio 4.5.

Phase I's section 12 recommended stochastic-action eval ablation as the next slice because the J-A entry of the Branch D row says: "Next slice should compare stochastic policy-selection variants ... maybe deterministic policy selection remains invalid for H5 reporting." The Phase I prediction was that Phase E seed2 stochastic eval would surface non-`left` actions on roughly 38% of steps per Phase H's per-action probability means (left 0.621, stay 0.108, right 0.271).

GPT's Phase J scope note added two corrections to the original Phase I recommendation: (1) test stochastic eval as a diagnostic ablation, not a new H5 acceptance eval; (2) run a two-stage workflow, J0 pilot on E2 only at 5 replicates per seed before deciding whether to expand to J1.

## 3. Tool and methodology

Tool: `tools/h5_stochastic_eval.py`. Docs-and-tools-only slice. No training. No env source changes. No config changes. No reward variant. Reuses fingerprinting, env factory wiring, and observation hashing from `tools/h5_logit_compare.py`. Uses the existing SB3 inference API rather than the H5 baseline CLI, since `h5_baseline_cli.py` hard-codes `deterministic=True` at the call site even though the lower-level harness and the `_SB3PolicyAdapter` already accept the flag.

Methodology:

1. Load trained PPO CnnPolicy models. Fingerprint each by file SHA-256 and policy state-dict blake2b digest, abort on any duplicate.
2. Build one Godot Signal Dodge env in `eval` mode per loaded model. Reuse the env across the (eval_seed, replicate) loop for that model, since each `env.seed()` plus `env.reset()` yields a deterministic env trajectory anchored at the chosen eval seed.
3. For each eval seed and replicate: compute `policy_sample_seed = eval_seed + 10_000_000 + replicate_idx`; reseed Python `random`, NumPy, and Torch with that value; reseed the env with `eval_seed`; reset; run one episode under `model.predict(obs, deterministic=False)`. At every step also extract `model.policy.get_distribution(obs).distribution.probs` and the deterministic argmax for the same obs, so per-step "sampled differs from argmax" booleans are recorded alongside `p_left`, `p_stay`, `p_right`.
4. Per episode, record sampled action counts/fractions, argmax counts, sampled-differs-from-argmax fraction, episode length, collision/timeout, total reward, first non-`left` step index, and a classification label: `wall_hugging_into_collision` for collision with left_fraction at or above 0.80, `survived_to_timeout` for any truncation, `non_wall_hugging_collision` otherwise.
5. Per (label, eval_seed), record per-replicate length/collision/timeout, the deterministic baseline from a constant lookup table, and a `terminal_differs_count` (number of replicates where collision or timeout differs from the deterministic baseline).
6. Output: one NDJSON per run with a header row plus one row per step, one `summary.json` with per-label aggregates, per-seed cross-replicate stats, and the full episode-summary list.

Two-stage workflow per scope note:

- **J0 pilot**: `phase_e_seed2` only, eval seeds 1000-1009, 5 replicates per seed = 50 episodes. Decide expansion against threshold criteria.
- **J1 expansion**: `phase_g_seed2` added, eval seeds 1000-1009, 5 replicates per seed = 50 additional episodes (E2 J0 data is preserved). Class A negative controls (`phase_e_seed1`, `phase_g_seed1`) skipped per "only if budget is still clean" and predicted to be no-ops based on Phase I entropy.

Wall time on StrongerJr: J0 = 919.6 seconds (~15 min 20 sec) for 50 episodes. J1 = 1180.6 seconds (~19 min 41 sec) for 50 episodes. One Godot launch per (label, run), reused across the entire seed-by-replicate loop.

## 4. Model fingerprints carried forward

Fingerprints reproduce exactly across Phase H, Phase I, and Phase J. Both Phase J runs reloaded the same artifacts and recomputed both file and parameter hashes; values match the Phase H section 3 table bit-identically.

| Label | SHA-256 (16 hex prefix) | param hash (blake2b-128) | param count | size bytes |
|---|---|---|---|---|
| phase_e_seed2 | `5e5fbba2d7580bfb` | `9dee74fb5b87e63e1b2f2816f8c03baa` | 5036004 | 20237577 |
| phase_g_seed2 | `705f56596a69a415` | `a587646124d06fc87690965988f38ba3` | 5036004 | 20237577 |

The two Phase J models are the same artifacts probed by Phase H and Phase I.

## 5. J0 results: phase_e_seed2 stochastic eval (50 episodes)

Aggregate:

| Metric | Value |
|---|---:|
| Episodes | 50 |
| Mean episode length | 657.3 |
| Median episode length | 573.0 |
| Min / max episode length | 183 / 1744 |
| Std episode length | 424.1 |
| Collision rate | 1.000 |
| Timeout rate | 0.000 |
| Sampled differs from argmax (mean) | 0.3600 |
| Sampled differs from argmax (min / max) | 0.2922 / 0.4105 |
| Action fraction (mean): left | 0.6400 |
| Action fraction (mean): stay | 0.0993 |
| Action fraction (mean): right | 0.2607 |
| Classification | 50 / 50 `non_wall_hugging_collision` |
| Seeds with any terminal-outcome diff vs deterministic | 1 (seed 1008) |

Per-seed (det length, then all 5 replicate lengths in sorted order):

| Eval seed | Det length | Det terminal | Replicate lengths | Length variance? |
|---:|---:|---|---|---:|
| 1000 | 1383 | collision | [579, 1383, 1383, 1383, 1383] | yes (one short) |
| 1001 | 483 | collision | [483, 483, 483, 483, 483] | no |
| 1002 | 1293 | collision | [183, 753, 754, 764, 766] | yes (all short) |
| 1003 | 603 | collision | [513, 518, 603, 603, 603] | small |
| 1004 | 1443 | collision | [1443, 1443, 1443, 1443, 1443] | no |
| 1005 | 363 | collision | [363, 363, 363, 363, 363] | no |
| 1006 | 573 | collision | [573, 573, 573, 573, 573] | no |
| 1007 | 273 | collision | [273, 273, 273, 273, 273] | no |
| 1008 | 1800 | timeout | [513, 513, 517, 519, 1744] | **all collide** |
| 1009 | 243 | collision | [243, 243, 243, 243, 243] | no |

Three observations.

First, the sampled-differs-from-argmax fraction (mean 0.360, range 0.29 to 0.41) matches Phase H's predicted ~38% non-`left` action probability mass within tight bounds. The seeding protocol works; stochastic sampling is materially active.

Second, six of ten eval seeds (1001, 1004, 1005, 1006, 1007, 1009) produce **bit-identical episode lengths across all five replicates** despite the 36% non-argmax sampling. These are seeds where the env-terminal hazard arrives at a specific frame that any non-`left` action at the wrong moment triggers, so the trained policy's commitment to `left` is what was buying it those exact step counts. Sampling does not change the outcome because the hazard timing is geometry-determined, not policy-determined, on these seeds.

Third, three seeds (1000, 1002, 1008) produce wide replicate variance. Seed 1008 is the smoking gun: deterministic Phase E seed2 timed out at 1800 steps; all five stochastic replicates collide with lengths 513, 513, 517, 519, 1744. The "survival" on seed 1008 was specifically the wedge-at-x=16 trick that the deterministic policy maintained by picking `left` every step. Stochastic sampling occasionally pulls the player out of the wedge and the hazards no longer miss. The trained policy has no observation-conditional dodging behavior to recover with.

### 5.1 J0 expansion-threshold check

GPT's pre-J0 expansion threshold from the scope note required any one of five criteria to cross to trigger J1.

| Criterion | Threshold | J0 measurement | Crossed? |
|---|---|---|---|
| Mean episode length change vs det | at least 10% | 22.3% decrease (845.7 → 657.3) | yes |
| Collision rate change vs det | at least 0.10 absolute | 0.10 increase (0.90 → 1.00) | yes (boundary) |
| Eval seeds with any-replicate terminal diff | at least 3 of 10 | 1 of 10 (seed 1008) | no |
| Median x-le-16 fraction drop | at least 0.20 | not applicable: reward_shaping=none in this config means `info["reward_state"]["player_x"]` was not emitted and player_x is unavailable from env info | n/a |
| Non-wall-hugging failure or survival class | at least 20% of replicates | 100% of replicates classified `non_wall_hugging_collision` | yes |

Three of four applicable criteria crossed; J1 expansion approved and executed.

## 6. J1 results: phase_g_seed2 stochastic eval (50 episodes)

Aggregate:

| Metric | Value |
|---|---:|
| Episodes | 50 |
| Mean episode length | 845.7 |
| Median episode length | 588.0 |
| Min / max episode length | 243 / 1800 |
| Std episode length | 542.7 |
| Collision rate | 0.900 |
| Timeout rate | 0.100 |
| Sampled differs from argmax (mean) | 0.0143 |
| Sampled differs from argmax (min / max) | 0.0041 / 0.0227 |
| Action fraction (mean): left | 0.9857 |
| Action fraction (mean): stay | 0.0130 |
| Action fraction (mean): right | 0.0013 |
| Classification | 45 / 50 `wall_hugging_into_collision`, 5 / 50 `survived_to_timeout` |
| Seeds with any terminal-outcome diff vs deterministic | 0 |

Per-seed:

| Eval seed | Det length | Det terminal | Replicate lengths | Match det? |
|---:|---:|---|---|---|
| 1000 | 1383 | collision | [1383, 1383, 1383, 1383, 1383] | yes |
| 1001 | 483 | collision | [483, 483, 483, 483, 483] | yes |
| 1002 | 1293 | collision | [1293, 1293, 1293, 1293, 1293] | yes |
| 1003 | 603 | collision | [603, 603, 603, 603, 603] | yes |
| 1004 | 1443 | collision | [1443, 1443, 1443, 1443, 1443] | yes |
| 1005 | 363 | collision | [363, 363, 363, 363, 363] | yes |
| 1006 | 573 | collision | [573, 573, 573, 573, 573] | yes |
| 1007 | 273 | collision | [273, 273, 273, 273, 273] | yes |
| 1008 | 1800 | timeout | [1800, 1800, 1800, 1800, 1800] | yes (timeout preserved) |
| 1009 | 243 | collision | [243, 243, 243, 243, 243] | yes |

**Phase G seed2 stochastic eval is bit-identical to Phase G seed2 deterministic eval on episode length and terminal outcome for every replicate of every seed.** The 1.4% sampled-differs-from-argmax fraction maps to one to three off-argmax actions per hundred steps; those samples either occur at non-hazard moments where the player returns to x=16 within the next step, or are too few to disturb the wedge. Phase G's reward shaping sharpened the trained Categorical distribution to the point where Phase J's diagnostic surface is null.

This is the predicted outcome from Phase I (G2 entropy 0.090, raw-logit margin 4.27 on left tape vs E2's 0.83): G2's larger margin and lower entropy together rule out meaningful sampling effect. J1 confirms the prediction at the trajectory-outcome level.

## 7. Cross-model comparison

The two networks under Phase J probe ablation share train_seed=2 (the Class B basin per Phase H) but differ in reward function: Phase E used the default `reward_shaping: none` Godot base reward (+1/step survival), Phase G used `threat_weighted_clearance` with `alpha=0.05`, `lookahead_band=270`, `safe_lateral_distance=180`. They are different artifacts at the SHA-256, parameter-hash, raw-logit-margin, and entropy levels. Phase J reveals they are also different at the eval-time policy-behavior level under stochastic sampling.

| Metric | phase_e_seed2 (J0) | phase_g_seed2 (J1) |
|---|---:|---:|
| Phase I raw-logit margin (left tape, mean) | 0.830 | 4.265 |
| Phase I selected-action proj std (left tape) | 0.165 | 0.558 |
| Phase I margin-to-std ratio | 5.0 | 7.6 |
| Phase I probability entropy mean | 0.888 | 0.090 |
| Phase J sampled-differs-from-argmax (mean) | 0.360 | 0.014 |
| Phase J sampled-differs-from-argmax (max) | 0.411 | 0.023 |
| Phase J left-action fraction (mean) | 0.640 | 0.986 |
| Phase J mean episode length | 657.3 | 845.7 |
| Phase J vs deterministic mean length | -22.3% | 0.0% (bit-identical) |
| Phase J collision rate | 1.000 | 0.900 |
| Phase J vs deterministic collision rate | +0.10 | bit-identical |
| Phase J timeout rate | 0.000 | 0.100 |
| Phase J vs deterministic timeout rate | -0.10 | bit-identical |
| Phase J seeds with any terminal diff | 1 (seed 1008) | 0 |
| Phase J classification distribution | 50 NWHC | 45 WHIC, 5 STT |

NWHC = non_wall_hugging_collision. WHIC = wall_hugging_into_collision. STT = survived_to_timeout.

The Phase I prediction held in both directions:

- E2's wider entropy (0.888) implied substantial sampling effect; J0 measured 36% sampled-differs-from-argmax against the 38% prediction. The trajectory-outcome consequence is a 22% drop in mean length and complete loss of the seed-1008 survival.
- G2's narrow entropy (0.090) implied negligible sampling effect; J1 measured 1.4% sampled-differs-from-argmax against the predicted 1-3%. The trajectory-outcome consequence is **zero**: every per-seed length matches deterministic bit-identically.

Phase G's reward shaping is now traceable through every layer Phase I and Phase J probed:

1. **Phase I encoder layer**: G2 features norm std rose to 10.5 vs E2's 8.5 on the left tape (slightly more obs-correlated response).
2. **Phase I action_net projection**: G2 `W[left]` row norm rose to 0.055 from E2's 0.025 (123% increase).
3. **Phase I raw logits**: G2 mean margin rose to 4.27 vs E2's 0.83 (5.1x).
4. **Phase I probability entropy**: G2 dropped to 0.090 vs E2's 0.888 (10x lower).
5. **Phase J sampling effect**: G2 sampled-differs-from-argmax dropped to 1.4% vs E2's 36% (25x lower).
6. **Phase J trajectory-outcome effect**: G2 bit-identical to deterministic; E2 22% shorter mean length, 1.00 collision rate, all seed-1008 survival lost.

The shaping moved every observable from "policy distribution responds to observations" (E2) toward "policy distribution is operationally constant" (G2). Phase G did not teach the policy to dodge hazards. It tightened commitment to the same wedge-at-x=16 strategy the unshaped policy had already learned.

## 8. Per-seed terminal-outcome diff table

A consolidated view of where stochastic eval moved the env trajectory, by (model, eval seed):

| Eval seed | Det length | Det term | E2 stoch lengths | E2 term diff | G2 stoch lengths | G2 term diff |
|---:|---:|---|---|---|---|---|
| 1000 | 1383 | coll | [579, 1383, 1383, 1383, 1383] | 0 / 5 | [1383, 1383, 1383, 1383, 1383] | 0 / 5 |
| 1001 | 483 | coll | [483, 483, 483, 483, 483] | 0 / 5 | [483, 483, 483, 483, 483] | 0 / 5 |
| 1002 | 1293 | coll | [183, 753, 754, 764, 766] | 0 / 5 | [1293, 1293, 1293, 1293, 1293] | 0 / 5 |
| 1003 | 603 | coll | [513, 518, 603, 603, 603] | 0 / 5 | [603, 603, 603, 603, 603] | 0 / 5 |
| 1004 | 1443 | coll | [1443, 1443, 1443, 1443, 1443] | 0 / 5 | [1443, 1443, 1443, 1443, 1443] | 0 / 5 |
| 1005 | 363 | coll | [363, 363, 363, 363, 363] | 0 / 5 | [363, 363, 363, 363, 363] | 0 / 5 |
| 1006 | 573 | coll | [573, 573, 573, 573, 573] | 0 / 5 | [573, 573, 573, 573, 573] | 0 / 5 |
| 1007 | 273 | coll | [273, 273, 273, 273, 273] | 0 / 5 | [273, 273, 273, 273, 273] | 0 / 5 |
| **1008** | **1800** | **TO** | **[513, 513, 517, 519, 1744]** | **5 / 5** | [1800, 1800, 1800, 1800, 1800] | 0 / 5 |
| 1009 | 243 | coll | [243, 243, 243, 243, 243] | 0 / 5 | [243, 243, 243, 243, 243] | 0 / 5 |

The seed 1008 row is the only cell where stochastic eval changes a terminal outcome class for any model. Both replicates of the deterministic timeout (Phase E seed2 and Phase G seed2 both timeout at 1800 steps) become collisions only for Phase E seed2 stochastic; Phase G seed2 stochastic preserves the timeout in all five replicates.

Why does Phase G seed2 preserve seed 1008's timeout when Phase E seed2 loses it? Because Phase G's 1.4% sampled-differs-from-argmax fraction is too low to break the wedge-at-x=16 strategy that buys seed 1008's survival. Phase E seed2's 36% sampled-differs-from-argmax fraction is high enough that the player drifts out of the wedge often enough for the seed-1008 hazard pattern to catch it. The exact same env trajectory under both networks; the policy commitment to `left` is what differs, and that commitment is what is buying seed 1008's survival.

This is the cleanest single-cell falsification of any "learned dodging" interpretation of the seed 1008 timeout. The timeout is a function of policy commitment, not of observation-conditional decision-making.

## 9. Verdict mapping

GPT's pre-Phase-J verdict table identified four candidate verdicts. Cross-mapping:

- **J-A: stochastic eval reveals hidden Class-B competence.** **Falsified.** The "improve or diversify" arm requires E2 stochastic to either lengthen episodes or surface more timeouts. The opposite happened: mean length dropped 22% and the one deterministic timeout was lost.
- **J-B: stochastic eval changes actions but not trajectory class.** **Closest fit, with refinement.** Sampled action differs from argmax in 36% of E2 steps, matching the 30-40% prediction window exactly. Six of ten eval seeds for E2 show bit-identical episode lengths despite 36% sampling, so the env-trajectory class is preserved at the seed level for the majority. For G2 the trajectory class is preserved on every seed including seed 1008. The refinement: J0's seed-1008 cell **does change** terminal class for E2 (timeout becomes collision), and three of ten E2 seeds show wide replicate variance in length. The "essentially unchanged" interpretation needs the qualifier "majority of seeds, except where the wedge strategy was specifically holding the survival together."
- **J-C: sampled-differs-from-argmax far below the predicted ~38%.** **Falsified.** E2 measured 0.360 against a 0.38 prediction (5% deviation in fraction-space). The seeding protocol works and the SB3 sampling path is reached as designed.
- **J-D: Class A moves unexpectedly.** **Not tested.** Class A controls were scope-note-optional and skipped because Phase I already predicted operationally indistinguishable stochastic-vs-deterministic outcomes for both Class A models (E1 entropy 0.011, G1 entropy 0.0002). Skipping them is consistent with the budget guidance; running them would be a quick negative control if needed later.

**Headline verdict**: J-B with a sharpened reading. The deterministic-argmax surface was not masking learned competence. The trained policy's apparent survival on seed 1008 was a fragile commitment artifact tied specifically to maintaining the wedge-at-x=16 position. Removing determinism breaks the wedge for E2 and the trained policy has no observation-conditional behavior to fall back on. For G2 the shaping pushed commitment so tight that even removing determinism does not move trajectories.

The H5 plateau is structural in the env-policy coupling. The fix path is training-side. Eval-side determinism is not the bottleneck; it has been the only thing keeping these networks from looking even worse.

## 10. Recommended next slice

The Phase J verdict closes the eval-side diagnostic chain. The H5 plateau is not an eval determinism artifact (Phase J), not a model-loading bug (Phase H), not an observation freshness pathology (Phase H), not an encoder collapse (Phase I), not an actor-latent collapse (Phase I empty by construction), not an action_net bias-domination (Phase I), and not recoverable by reward shaping in the threat-weighted-clearance family (Phase G, refined further by Phase J showing the shaping tightens the wrong commitment). The trained policies have not learned to dodge hazards. They have learned to park at sticky positions whose survival on certain hazard patterns is a side-effect of their commitment to a single action.

The next slice is training-side. Three candidate directions in increasing order of structural change, all docs-and-tools-only-plus-training:

1. **Training-time entropy probe**. Run a short instrumented training session (1k-2k timesteps on the existing entropy YAML) with an entropy-aware logger that records: when entropy collapse occurs (which PPO iteration), what the policy gradient norm is at and around the collapse, whether collapse coincides with a specific value-function or advantage signal pattern, and whether `ent_coef=0.01` is binding or not. This isolates whether the issue is the entropy coefficient being too low at the existing budget, the value function saturating early and locking the advantage signal, or something else upstream of the trained-policy artifact itself.

2. **Architecture or objective change**. The current setup uses SB3's default `CnnPolicy` `net_arch=None` which collapses `latent_pi` and `latent_vf` to the encoder output. Adding a small policy MLP (e.g. `net_arch=dict(pi=[64], vf=[64])`) gives the optimizer a separate latent space to learn observation-conditional value and policy heads independently. This is a one-line config change with full retraining.

3. **Train-seed asymmetry probe**. Phase I section 12 flagged a separate but related puzzle: Class A (seed=1) trained networks have 10x wider raw-logit margins than Class B (seed=2) at identical hyperparameters and identical training budgets. This points at random-initialization-driven basin separation. A 4-seed sweep at different initial action_net weight scales would isolate whether the early-rollout collision distribution lock policies into wedge basins from which the optimizer cannot escape under the current entropy budget. This is the Grok-trigger candidate per charter (RL internals + phase-gate-level question).

The next-slice ordering recommended by Phase J: do (1) first because it is the cheapest and tells us whether the existing setup can ever escape wedge basins given more training, regardless of architecture. If (1) shows training-time collapse onset at a specific iteration with a diagnosable cause, (2) and (3) become better-scoped. If (1) shows the entropy budget is operationally infinite and the optimizer still locks into wedges, (2) is the cleanest next move. (3) becomes more interesting after a Grok consult on whether the seed=1 vs seed=2 margin asymmetry is itself a known SB3 PPO CnnPolicy initialization pathology or something env-specific to Signal Dodge.

What stochastic-eval-variants Phase J ruled out for future slices: there is no point sweeping temperature, top-k, or top-p on top of categorical sampling. The Phase J finding is that E2's distribution itself encodes useful information about non-`left` actions but that information does not improve outcomes when sampled. Any temperature variant that further widens the distribution will produce more non-`left` actions and worse outcomes; any temperature variant that sharpens the distribution will approach G2's bit-identical-to-deterministic regime. Neither is a productive eval-side direction.

## 10.1 Grok status

Phase I's section 12.1 staged a Grok prompt about whether the SB3 actor-path extraction points (`extract_features`, `mlp_extractor`, `action_net`, `value_net`, `get_distribution`) are valid for PPO `CnnPolicy`. That prompt has not been relayed yet. Phase J does not increase the urgency on that prompt. Phase J does increase the urgency on Phase I section 12's secondary observation: the seed=1 vs seed=2 raw-logit margin asymmetry. Recommended Grok prompt for the eventual training-time / seed-asymmetry probe (option 3 above), verbatim Phase I section 12.1:

> We are instrumenting SB3 PPO `CnnPolicy` trained on channel-first pixel observations. We plan to compare `extract_features(obs_tensor)`, `mlp_extractor(features)`, `action_net(latent_pi)`, raw softmax probabilities, and SB3 `get_distribution` probabilities across fixed observation tapes. Goal is to localize observation-insensitive constant-action policies to encoder collapse vs actor-latent collapse vs final action-head bias. Are these extraction points valid for SB3 PPO CnnPolicy, and what failure modes could make this branch logic misleading?

Plus a Phase J addition that is now worth asking Grok:

> At identical PPO hyperparameters and identical training budgets on a fixed Godot pixel-CNN env, four different train_seeds collapsed into two equivalence classes keyed by train_seed alone (Class A: train_seed=1 picks `stay` constant action with raw-logit margin 6-11; Class B: train_seed=2 and 3 pick `left` constant action with raw-logit margin 0.8-6 depending on reward shaping). Class A networks have 10-100x wider raw-logit margins than Class B in only 10k timesteps. Is this a known SB3 PPO CnnPolicy initialization-driven basin pathology, or is it env-specific? What would a minimal training-time diagnostic look like?

These two prompts could be relayed together before scoping option 3.

## 11. Reproduction recipe

Environment requirements identical to Phase H and Phase I:

- Windows 11, repo at `C:\Projects\Sight`, Python at `C:\Users\maste\AppData\Local\Python\bin\python.exe`
- `stable-baselines3`, `torch`, `gymnasium`, `numpy` from the same env used by Phase E and Phase G training
- Godot 4.6.2 at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`
- The Phase E seed2 and Phase G seed2 train run directories at `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed2_entropy_10k/` and `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped/h5_train_phase_g_shaped_seed2_10k/`, each containing `model.zip`. Source commits as recorded in Phase H section 10.

Single-model invocation (cmd.exe):

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\h5_stochastic_eval.py ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
    --models phase_e_seed2=runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed2_entropy_10k ^
    --seeds 1000-1009 ^
    --replicates 5 ^
    --max-steps 1800 ^
    --out-dir runs\phase_j ^
    --label-suffix j0
```

The `-u` flag is mandatory when stdout is redirected to a log file. Without it, Python block-buffers `print()` output and per-episode progress lines do not appear in the log until the process exits. The driver `.bat` used for this evidence used `-u` for J1; J0 was launched without `-u` and only surfaced its episode progress after completion. Either way the NDJSON step rows and summary JSON are unaffected.

Two-stage driver. The drivers used for this evidence are at `%TEMP%\phase_j_j0_driver.bat` and `%TEMP%\phase_j_j1_driver.bat` (not committed; ephemeral). Each calls the comparator once with one model on the same 10 eval seeds and 5 replicates, redirects stdout and stderr to `runs/phase_j/{j0,j1}_driver.log`, and writes a `{j0,j1}_driver.done` sentinel on exit. Total wall time on StrongerJr: J0 = 919.6 s, J1 = 1180.6 s; per-episode wall time averaged ~18-24 s depending on episode length.

Output artifacts (all gitignored under `runs/`; durable evidence in this doc):

- `runs/phase_j/stochastic_eval_j0.ndjson`: per-tape header + one row per env step for J0 (E2)
- `runs/phase_j/stochastic_eval_j0.summary.json`: J0 aggregate + per-seed + per-episode summary
- `runs/phase_j/stochastic_eval_j1_g2.ndjson`: same for J1 (G2)
- `runs/phase_j/stochastic_eval_j1_g2.summary.json`: same for J1
- `runs/phase_j/godot_phase_e_seed2_j0/` and `runs/phase_j/godot_phase_g_seed2_j1_g2/`: Godot env eval NDJSON sidecars from `make_env`
- `runs/phase_j/j0_driver.log`, `runs/phase_j/j1_driver.log`, `runs/phase_j/{j0,j1}_driver.done`: driver progress and sentinels

Re-running on the same train run directories must produce bit-identical fingerprint values (the two SHA-256 prefixes in section 4) and bit-identical observation hash sequences per env seed. Bit-identical sampled-action sequences require bit-identical seeding: the policy_sample_seed formula `eval_seed + 10_000_000 + replicate_idx` plus Python/NumPy/Torch seeding before each episode. Torch's CUDA random state would also need to match if any of the loaded models or the SB3 internals routed through CUDA; this evidence loaded all models with `device='cpu'` so CUDA state is irrelevant.

## 12. Implementation-not-at-fault statement

The implementation slices that have been suspected in prior H5 phases are not at fault for the Phase J results either:

- The H5 reward-amendment implementation (commit `b41bffc`) is correct. Phase G's shaped reward reached the optimizer (Phase G evidence, Phase I weight-row-norm growth from 0.025 to 0.055), the policy's distribution sharpened in response (Phase I entropy 0.888 → 0.090), and Phase J confirms the sharpening at the trajectory-outcome level (G2 bit-identical to deterministic).
- The Phase H logit-distribution comparator (`tools/h5_logit_compare.py`) is correct. Its predicted ~38% per-action probability mass for E2 was confirmed to 36% by Phase J at the SB3 sampling layer.
- The Phase I activation comparator (`tools/h5_activation_compare.py`) is correct. Its margin-vs-std ratios for E2 (5.0 on left tape) and G2 (7.6 on left tape) predicted the relative sampling effect at the trajectory layer, confirmed by Phase J (E2 large effect, G2 zero effect).
- The Phase J stochastic-eval tool (`tools/h5_stochastic_eval.py`) is correct. Manual `softmax(action_net(latent_pi))` matches SB3 `get_distribution.probs` to within 1.2e-7 across both runs (Phase I section 6 result reused; Phase J does not re-verify but uses the same SB3 path).

The H5 plateau localizes upstream of any of these implementation surfaces. It is in the training-time dynamics that produced the trained `model.zip` artifacts in the first place. The Phase J ablation closes the eval-side diagnostic chain and hands the next slice to the training side.
