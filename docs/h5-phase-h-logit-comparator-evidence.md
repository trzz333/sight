# H5 Phase H logit-distribution comparator evidence

Status: VERIFIED end-to-end on StrongerJr, four runs complete, two equivalence classes covered, optional cross-basin runs included.

## 1. TL;DR

The H5 amendment hypothesis was falsified in Phase G at the eval-trajectory level. Phase H tests a deeper layer: are the trained CNN policies even using the observation? Answer, with HIGH confidence and direct measurement: **no**. Each trained network has collapsed to a train-seed-keyed constant action that is invariant to the observation.

- Phase E seed1 vs Phase G seed1 on a stay tape at eval seed 1000 produce **identical action distributions in probability space** (prob_l1 mean 0.0029, sym_kl mean 0.0057) despite distinct file SHAs and distinct parameter hashes. Both saturate on `stay` with mean top1-top2 margin 0.997 (E1) and 1.0000 (G1).
- Phase E seed2 vs Phase G seed2 on a left tape at eval seed 1000 produce the **same argmax (`left`) on every step** but with materially different probability distributions: E2 mean top1-top2 margin 0.350 (entropy 0.888), G2 mean margin 0.968 (entropy 0.090), prob_l1 mean 0.724, sym_kl mean 0.884. Reward shaping sharpened confidence in Class B without shifting which action was chosen.
- Phase G seed1 vs Phase G seed2 on either tape produce same_argmax fraction **0.0000**: G1 picks `stay` on 100% of steps regardless of which trajectory it sees; G2 picks `left` on 100% of steps regardless of which trajectory it sees. prob_l1 mean approaches the theoretical maximum of 2.0 (1.996 on stay tape, 1.969 on left tape); sym_kl mean 7.5 to 8.7.
- Observation freshness is intact across every run: 309 of 333 unique observation hashes on the stay tape (max consecutive repeat 5, all near the late pre-collision frames), 1381 of 1383 unique on the left tape (max consecutive repeat 2).

The cross-basin runs are the smoking gun. G1 receiving the left-tape observation sequence and G2 receiving the stay-tape observation sequence both still pick their basin's constant action with near-saturated probability. The networks ignore the observation at the policy-output layer. The proximate pathology is upstream of the action head: feature extraction, policy-latent MLP, or both are producing near-observation-invariant outputs per trained network.

The branch the eval-trajectory falsification cannot decide alone is now decidable. Deterministic argmax is not the bottleneck. Reward shaping is not the bottleneck. Eval seed pairing across cells is not the bottleneck. The bottleneck is observation insensitivity in the trained policy network itself.

## 2. Tool, methodology, and scope

Tool: `tools/h5_logit_compare.py`. Docs-and-tools-only slice. No training. No env source changes. No config changes. Loads existing committed Phase E and Phase G `model.zip` artifacts from `runs/rl/.../h5_train_phase_*` directories on disk.

Methodology:

1. Load two or more SB3 PPO CnnPolicy checkpoints. Fingerprint every model at load time by file SHA-256, archive member list, parameter count, and a blake2b-128 hash over the concatenated policy state-dict tensor bytes in sorted key order. Abort with a "model-loading bug suspected" RuntimeError if any two models share an identical file SHA-256 or an identical parameter hash.
2. Build a single Godot Signal Dodge VecEnv from `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml` in `eval` mode. Seed the env at the requested eval seed and reset.
3. At each env step: hash the observation (blake2b-12-hex), record observation shape, dtype, min, max, mean. Query every loaded model on the same observation tensor: extract the SB3 `CategoricalDistribution.distribution` log-probs and probabilities, the entropy, the argmax, and the top1-top2 margin. Compute pairwise centered-logit L2, prob L1, and symmetric KL between every model pair on that same observation. Then apply a **fixed behavior tape action** (not any model's action) and step the env. Stop on terminal or after `max_steps`.
4. Write one NDJSON header row carrying all fingerprints, the tape, the eval seed, and the elapsed wall time, then one row per step. Write a sibling `.summary.json` with per-model and pairwise aggregates plus observation-hash aggregates.

The fixed-tape design is the key control: trajectory does not depend on any model's predictions. Every model is asked for its action distribution on the same observation sequence. This is what GPT named "do not let each model control the environment while comparing logits" in the Phase H scope note.

The categorical logits returned by SB3's `policy.get_distribution(obs_tensor).distribution.logits` are PyTorch's stored `log_softmax(raw_logits)`, not the raw action-head pre-softmax scalars. All pairwise metrics in this evidence (centered logit L2, prob L1, symmetric KL) are well-defined on normalized log-probs; raw action-head logit extraction would couple to SB3-internal MLP layout and is deferred unless a later slice requires it.

Scope of the four runs:

| Run | Models | Tape | Eval seed | Steps |
|---|---|---|---|---|
| Class A (within-basin, reward variant comparison) | Phase E seed1 vs Phase G seed1 | stay | 1000 | 333 (collision) |
| Class B (within-basin, reward variant comparison) | Phase E seed2 vs Phase G seed2 | left | 1000 | 1383 (truncation) |
| Cross-basin secondary | Phase G seed1 vs Phase G seed2 | stay | 1000 | 333 |
| Cross-basin secondary | Phase G seed1 vs Phase G seed2 | left | 1000 | 1383 |

Class A and Class B follow the Phase G/E equivalence-class structure recorded in the handoff and in `docs/h5-phase-g-shaped-evidence.md`. Class A = train_seed 1, natural action `stay`, ends in collision around step 600 in trained eval; on the stay tape at eval seed 1000 the env terminates at step 333. Class B = train_seed 2 or 3, natural action `left`, longer episodes; on the left tape at eval seed 1000 the env truncates at step 1383.

Per-run wall time was 12 to 41 seconds. Total driver wall time was about 2 minutes 17 seconds across all four runs including four windowed Godot launches.

## 3. Model-loading verification

All four loaded models have distinct file SHA-256 digests and distinct policy state-dict parameter hashes. File size, archive structure, and parameter count are uniform across all four artifacts (this is the SB3 PPO CnnPolicy NatureCNN backbone with `Discrete(3)` action head; 5,036,004 parameters; 20,237,577 bytes per `model.zip`).

| Label | SHA-256 (16 hex prefix) | param hash (blake2b-128) | param count | size bytes |
|---|---|---|---|---|
| phase_e_seed1 | `2126014dae0cedbf` | `b59518de034867e6a8ff6db638827561` | 5036004 | 20237577 |
| phase_e_seed2 | `5e5fbba2d7580bfb` | `9dee74fb5b87e63e1b2f2816f8c03baa` | 5036004 | 20237577 |
| phase_g_seed1 | `51e02ecdb6234b64` | `638672a424d62cb360a5f356ef2de677` | 5036004 | 20237577 |
| phase_g_seed2 | `705f56596a69a415` | `a587646124d06fc87690965988f38ba3` | 5036004 | 20237577 |

The file-layer model-loading hypothesis named as the highest-priority pretest in the GPT branch table is **falsified**: distinct artifacts are reaching the loader, distinct state-dicts are reaching the policy network, and no two models in the comparator share a fingerprint at either layer.

## 4. Observation freshness

Observation hashing per step rules out a stale-observation or frame-stack-gross-failure pathology before any policy-internal claim. The Godot env delivers fresh observations across both tapes.

| Run | Steps | Unique obs hashes | All-distinct | Max consecutive repeat |
|---|---|---|---|---|
| Class A (stay tape) | 333 | 309 | false | 5 |
| Class B (left tape) | 1383 | 1381 | false | 2 |
| Cross-basin (stay) | 333 | 309 | false | 5 |
| Cross-basin (left) | 1383 | 1381 | false | 2 |

On the stay tape (player frozen at x=360) 92.8% of frames produce a unique observation hash. On the left tape (player wedged at x=16 after a few steps) 99.86% of frames are unique. The 5-step max repeat run on the stay tape is consistent with brief stationary stretches near the collision frame at step 333; it is not a frame-stack or observation-cache failure. Observation freshness is **not** the proximate pathology.

## 5. Class A: Phase E seed1 vs Phase G seed1 on stay tape (eval seed 1000)

Trajectory: 333 steps. Tape action `stay` (action_wire=1) every step. Env terminates at step 333 (collision). Both models pick `stay` on 100% of steps; same-argmax fraction 1.0.

Per-model action distribution:

| Model | argmax | entropy mean | margin mean | margin min | mean prob(left) | mean prob(stay) | mean prob(right) | same-as-tape |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phase_e_seed1 | `stay` 100% | 0.0111 | 0.9971 | 0.9970 | 2.69e-5 | 0.9985 | 0.00143 | 1.0000 |
| phase_g_seed1 | `stay` 100% | 0.0002 | 1.0000 | 1.0000 | 1.53e-5 | 0.99998 | 5.25e-7 | 1.0000 |

Pairwise distributional distances:

| Metric | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| centered_logit_l2 | 6.2458 | 6.2428 | 6.2611 | 6.2615 |
| prob_l1 | 0.0029 | 0.0029 | 0.0029 | 0.0030 |
| sym_kl | 0.0057 | 0.0057 | 0.0058 | 0.0058 |

Interpretation. **Both models put essentially all probability mass on `stay`** with E1 at 99.85% and G1 at 99.998%. The probability-space distance is negligible: prob_l1 mean 0.0029 (theoretical max 2.0) and sym_kl mean 0.0057 (theoretical min 0.0). The 6.25 mean centered-logit L2 looks large in isolation but reflects only the magnitude of the log-prob gap on the off-actions: G1 is more confident on `stay` so its `left` and `right` log-probs are more negative. In the metric that determines downstream behavior, probability space, the two networks are operationally identical in Class A.

This is the strongest possible expression of **reward shaping had no effect on the policy function in Class A**. The shaped reward signal sharpened G1's confidence on the same already-selected action; it did not select a different action, and would not even under stochastic-action eval, because both networks place over 99.8% probability on the same action at every step on this trajectory.

The Class A within-basin run alone cannot decide whether the policy is reading the observation. It is consistent with two hypotheses:
- (a) Both networks correctly read this observation and correctly agree that `stay` is the value-maximizing action under both reward functions.
- (b) Both networks ignore the observation and have collapsed to a constant `stay` action in the train_seed=1 basin.

The cross-basin run in section 7 discriminates between (a) and (b) decisively.

## 6. Class B: Phase E seed2 vs Phase G seed2 on left tape (eval seed 1000)

Trajectory: 1383 steps. Tape action `left` (action_wire=0) every step. Env truncates at step 1383 (out of 1800-step budget). Both models pick `left` on 100% of steps; same-argmax fraction 1.0.

Per-model action distribution:

| Model | argmax | entropy mean | margin mean | margin min | mean prob(left) | mean prob(stay) | mean prob(right) | same-as-tape |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| phase_e_seed2 | `left` 100% | 0.8880 | 0.3500 | 0.3103 | 0.6210 | 0.1079 | 0.2711 | 1.0000 |
| phase_g_seed2 | `left` 100% | 0.0903 | 0.9676 | 0.9499 | 0.9830 | 0.0153 | 0.00168 | 1.0000 |

Pairwise distributional distances:

| Metric | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| centered_logit_l2 | 4.0778 | 3.8018 | 5.0565 | 6.3409 |
| prob_l1 | 0.7239 | 0.7383 | 0.7505 | 0.7596 |
| sym_kl | 0.8844 | 0.8454 | 0.9804 | 1.0460 |

Interpretation. Class B is the only run in this evidence pass where reward shaping had a measurable effect on the policy distribution. Phase E seed2 was hovering on `left` with only 62% probability mass on `left` and 38% spread across `stay` (11%) and `right` (27%). Mean entropy 0.888 is close to but not at the discrete-3 uniform ceiling of 1.0986. The argmax is `left` only by margin, with mean top1-top2 margin 0.350 and minimum 0.310. Phase G seed2 sharpened `left` to 98.3% probability with mean margin 0.968.

Under **stochastic-action eval**, this would matter. A categorical sample from E2's distribution would draw `left` about 62% of the time, `right` about 27%, `stay` about 11%, producing materially different per-step actions than the argmax surface. Under the same stochastic-action eval, G2 would draw `left` 98% of the time and behave essentially like its deterministic argmax. Stochastic eval would therefore be a meaningful diagnostic **for Class B alone**.

But stochastic eval is not the cheapest decisive next step, because the cross-basin runs in section 7 show that even Class B's argmax selection is observation-insensitive at the policy-output layer. The probability distribution differs in magnitude; the action choice itself is locked in.

GPT's pre-Phase-H branch table assumed that margin sizes near saturation argued against deterministic argmax being the bottleneck. Class B partially recovers that branch: margins are far from saturated (0.35 mean for E2), so stochastic eval would reveal distributional differences that argmax masks. The cross-basin runs supersede this conclusion. Stochastic eval would expose the entropy gradient between Phase E and Phase G but would not address the observation-insensitivity revealed in section 7.

## 7. Cross-basin: Phase G seed1 vs Phase G seed2 on both tapes (smoking gun)

These two runs are the decisive evidence. G1 and G2 share the exact same hyperparameter recipe, the exact same reward function, the exact same training timesteps, and differ only in train_seed. They are presented with the **same observation sequence** at each step.

Same-argmax fraction across both tapes: **0.0000**.

### 7.1 Cross-basin on stay tape (333 steps)

Per-model action distribution on Class A's natural trajectory (the trajectory G1 itself produces under trained eval):

| Model | argmax | entropy mean | margin mean | mean prob(left) | mean prob(stay) | mean prob(right) | same-as-tape |
|---|---|---:|---:|---:|---:|---:|---:|
| phase_g_seed1 | `stay` 100% | 0.0002 | 1.0000 | 1.53e-5 | 0.99998 | 5.25e-7 | 1.0000 |
| phase_g_seed2 | `left` 100% | 0.0141 | 0.9963 | 0.9981 | 0.00184 | 6.54e-5 | 0.0000 |

Pairwise distributional distances:

| Metric | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| centered_logit_l2 | 12.4561 | 12.4139 | 12.6356 | 12.6678 |
| prob_l1 | 1.9963 | 1.9961 | 1.9972 | 1.9974 |
| sym_kl | 8.6799 | 8.6437 | 8.8350 | 8.8637 |

### 7.2 Cross-basin on left tape (1383 steps)

Per-model action distribution on Class B's natural trajectory:

| Model | argmax | entropy mean | margin mean | mean prob(left) | mean prob(stay) | mean prob(right) | same-as-tape |
|---|---|---:|---:|---:|---:|---:|---:|
| phase_g_seed1 | `stay` 100% | 0.0002 | 1.0000 | 1.74e-5 | 0.99998 | 6.18e-7 | 0.0000 |
| phase_g_seed2 | `left` 100% | 0.0903 | 0.9676 | 0.9830 | 0.0153 | 0.00168 | 1.0000 |

Pairwise distributional distances:

| Metric | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| centered_logit_l2 | 11.3535 | 11.2190 | 11.8225 | 12.7472 |
| prob_l1 | 1.9693 | 1.9632 | 1.9898 | 1.9978 |
| sym_kl | 7.4946 | 7.3241 | 8.0869 | 8.9350 |

### 7.3 What this proves

G1 receiving the stay-tape obs sequence and G1 receiving the left-tape obs sequence both produce **the exact same per-step action distribution** to four decimal places: argmax `stay` at 99.998% mean probability with mean margin 1.0000 and entropy 0.0002. The 1383-step left-tape trajectory shows a markedly different scene (the player has wedged at x=16, hazard geometry on screen is entirely different from the stay-tape's player-at-center-x=360 view) yet G1's action distribution at every step is indistinguishable from its action distribution on the stay tape.

The mirror holds for G2. G2 receiving the left-tape obs sequence and G2 receiving the stay-tape obs sequence both produce argmax `left` at 98+% probability with mean margin 0.97. The two trajectories share no obs hashes in common (the obs hash universes for the two tapes are entirely disjoint by construction; player x-position differs by hundreds of pixels for the bulk of frames) and yet G2 picks the same constant action throughout.

**The trained policies do not read the observation at the policy-output layer.** Each network has collapsed to a basin-keyed constant action. The basin is fully determined by the training seed; reward shaping (Phase E vs Phase G) does not move a network across basins; the action choice within each basin is independent of the observation actually presented to the network.

This is exactly what is required to produce the Phase G falsification result documented in `docs/h5-phase-g-shaped-evidence.md`: byte-identical eval trajectories across all 30 paired (train_seed, eval_seed) cells for Phase E and Phase G, regardless of reward function. The deterministic-argmax pathology suspected after Phase D and characterized after Phase G has a deeper cause: there is no observation dependence in the trained policies for argmax to mask.

## 8. Branch-table mapping per the Phase H scope note

GPT's pre-Phase-H branch table, evaluated against the evidence above:

- **"Params or model fingerprints are identical when they should not be: model-loading/path bug. Stop and fix tooling before interpreting policy behavior."** → Falsified. All four models have distinct SHA-256 and distinct parameter hashes (section 3). Model loading is sound.
- **"Obs hash is constant or nearly constant while the env state should be changing: observation freshness or frame-stack contract is the lead suspect."** → Falsified. 309/333 and 1381/1383 unique obs hashes, max consecutive repeat 2 to 5 (section 4). Observation freshness is intact.
- **"Params differ, obs changes, logits/probs differ, argmax same, margins small: deterministic argmax eval is the likely bottleneck. Next slice: stochastic-action eval ablation on existing models."** → Partially applies to Class B (E2 vs G2, prob_l1=0.72, sym_kl=0.88, mean margin 0.35 for E2). Does not apply to Class A. The cross-basin runs supersede it.
- **"Params differ, obs changes, logits/probs differ, argmax same, margins huge: not merely eval determinism. The trained distributions have collapsed to the same confident action. Stochastic eval is lower value; inspect training/objective/action-prior next."** → Applies to Class A (E1 vs G1, mean margins 0.997 and 1.0000, prob_l1=0.003). The cross-basin runs reveal the deeper cause.
- **"Params differ, obs changes, logits/probs nearly identical: move upstream. Next slice: activation comparator across `features_extractor`, policy latent, and action head. This is where Grok becomes worth pulling."** → Applies to Class A in probability space (prob_l1 mean 0.0029, sym_kl mean 0.0057), and the cross-basin runs (section 7) show that the deeper problem is observation insensitivity per-network, not merely cross-network log-prob similarity.

The cross-basin runs add a category the original branch table did not name: **within-network observation invariance**. Each trained network produces near-saturated argmax on a constant action regardless of the observation it sees. This is upstream of "logits differ but argmax does not": it is "logits do not respond to the observation".

## 9. Recommended next slice

The cheapest decisive next slice is an **activation comparator inside the policy network**, not stochastic-action eval. Stochastic-action eval would partially expose the Class B entropy gradient (E2 at 0.62 left vs G2 at 0.98 left) but cannot address Class A (both networks above 99.8% on `stay`) or the cross-basin invariance.

The activation comparator slice (docs-and-tools-only, no training, no env change):

1. Capture the activation tensor at the output of `model.policy.features_extractor` for each of the four models on a fixed step-by-step observation sequence (the same stay-tape and left-tape sequences used here).
2. Compute per-model: mean activation magnitude across steps, per-step variance of each feature dimension, principal-component spectrum, and the L2 distance between paired-step feature vectors across the trajectory.
3. Within-model branch: if `features_extractor` output is nearly constant across observations (very low across-step variance), the encoder has collapsed and the policy MLP is doing nothing because there is no signal to do anything with. Next slice after that: training-time gradient flow probe and possibly weight-initialization or learning-rate audit on the encoder.
4. Within-model branch: if `features_extractor` output varies meaningfully across observations but `mlp_extractor.policy_net` output is near-constant, the bottleneck is the policy MLP, not the encoder. Next slice: action-net weight inspection and policy-MLP variance probe.
5. Within-model branch: if both encoder and policy MLP outputs vary meaningfully but the final action logits saturate to the same argmax, the action-net has a heavily biased final linear layer. This is the least likely outcome given Class A's near-identical probability distributions across two distinctly-trained networks; if it appears, treat as a training-objective pathology.

This is the slice the charter names as Grok-trigger eligible: RL internals on the boundary between feature extraction, policy MLP, and the action head, with a phase-gate-level finding about whether the H5 baseline can ever learn observation-conditioned behavior under the current configuration. Pulling Grok here matches the charter's "weak-domain questions (RL internals)" and "phase-gate sanity checks" triggers.

What this slice would change as a verdict: if encoder outputs vary substantially across steps and policy MLP outputs collapse, that points at policy-MLP saturation, not encoder collapse. If encoder outputs themselves are near-constant, the H5 pixel-CNN baseline is mis-configured (likely the NatureCNN-with-only-10k-timesteps regime is insufficient to escape an initialization-dominated equilibrium, or the observation normalization is wrong, or the frame-stack contract is silently degraded). Either way, the next experiment is no longer "another reward variant" or "another hyperparameter sweep"; it is a structural inspection.

Out of scope for the next slice unless the activation comparator implicates it: stochastic-action eval (only meaningful for Class B distributional differences), more training timesteps (does not address observation insensitivity), more reward variants (falsified at the trajectory level in Phase G), entropy coefficient sweep (Phase D vs E showed it changed entropy magnitude but not the trajectory equivalence classes).

## 10. Reproduction recipe

Environment requirements:

- Windows 11, repo at `C:\Projects\Sight`, Python at `C:\Users\maste\AppData\Local\Python\bin\python.exe`
- `stable-baselines3`, `torch`, `gymnasium`, `numpy` already installed and on the Python path (same env used by Phase E and Phase G training)
- Godot 4.6.2 at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`
- The four target train run directories under `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed{1,2}_entropy_10k/` and `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped/h5_train_phase_g_shaped_seed{1,2}_10k/`, each containing `model.zip`. Phase E artifacts come from commit `dc0a8a3` (Phase E 3-seed evidence); Phase G artifacts come from commit `716ed73` (Phase G shaped 3-seed evidence).

Single-run invocation (cmd.exe):

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe
"C:\Users\maste\AppData\Local\Python\bin\python.exe" tools\h5_logit_compare.py ^
    --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml ^
    --models phase_e_seed1=runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed1_entropy_10k,phase_g_seed1=runs\rl\signal_dodge_ppo_h5_pixel_entropy_shaped\h5_train_phase_g_shaped_seed1_10k ^
    --eval-seed 1000 ^
    --max-steps 1800 ^
    --behavior-tape stay ^
    --out runs\phase_h\logit_compare_classA_stay.ndjson
```

Behavior tape accepts `stay`, `left`, `right`, or a comma-separated sequence of action ints. The env at eval seed 1000 deterministically terminates around step 333 on the stay tape and truncates at step 1383 on the left tape; setting `--max-steps 1800` is a generous upper bound, not an episode length expectation.

Four-run driver. The full driver used for this evidence is the same `.bat` pattern as Phase G's `run_phase_g.bat`, with one Python invocation per run and `>> driver.log 2>&1` redirection. Output sentinels:

- `runs/phase_h/driver.log`: per-run stdout and stderr concatenated
- `runs/phase_h/driver.done`: written after the final run exits
- `runs/phase_h/logit_compare_<case>.ndjson`: per-run NDJSON with one header row plus one row per step
- `runs/phase_h/logit_compare_<case>.summary.json`: per-run aggregates
- `runs/phase_h/logit_compare_<case>_godot/`: Godot eval NDJSON sidecar (gitignored)

Re-running on the same set of train run directories must produce bit-identical fingerprint values for every model (the four SHA-256 prefixes and the four parameter hashes in section 3). Re-running with the same eval seed must produce a bit-identical observation-hash sequence per tape; downstream NDJSON rows differ only in non-deterministic wall-time fields (`elapsed_seconds`, `ran_at_utc`).

This slice deliberately does not write to the gitignored `runs/` tree from the comparator process beyond the per-run NDJSON outputs. The durable evidence is this document plus the committed `tools/h5_logit_compare.py`.

## 11. Implementation-not-at-fault statement

The H5 amendment implementation slice and the Phase G shaped-reward implementation slice are not at fault for the H5 plateau. The amendment tests and the shaped reward surface were correct: the smoke evidence in `docs/h5-reward-amendment-smoke-evidence.md` proved the bonus surface was non-degenerate under non-trivial action sampling, and the Phase G evidence in `docs/h5-phase-g-shaped-evidence.md` proved the shaped reward did reach the training optimizer. The bottleneck is not in reward computation, not in the env-protocol layer, not in the NDJSON logging, and not in eval seed pairing. The bottleneck is upstream of the policy-output layer in the trained network itself.

Tooling notes added by Phase H to the operational memory:

- SB3 `policy.get_distribution(obs_tensor).distribution.logits` returns log_softmax, not raw action-head logits. Pairwise comparison metrics in this comparator are well-defined on the normalized form. Raw action-head logit extraction would require touching `policy.mlp_extractor.policy_net` and `policy.action_net` directly and is deferred.
- The pixel-mode windowed-Godot wall time on StrongerJr is about 12 seconds for a 333-step Class A run, about 41 seconds for a 1383-step Class B run. Four sequential runs over a sentinel-and-poll driver complete in about 2 minutes 17 seconds total. This is well under the 4-minute DC tool-call ceiling per run; the sentinel-and-poll pattern is used here only because the driver fires four runs in sequence.
- The DC `start /B` background-launch quirk reported in prior sessions reproduces: `start "" /B cmd /c <bat>` from inside an `interact_with_process`-spawned shell can crash the shell with "Input redirection is not supported". The reliable detach pattern is `start "title" /MIN cmd /c <bat>` (separate minimized window). The .bat then runs to completion regardless of whether the spawning shell stays alive, and the sentinel file is the source of truth for completion.
