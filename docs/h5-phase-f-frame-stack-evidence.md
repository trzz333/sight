# H5 Phase F: Frame-Stack Observation Contract Evidence

Status: Phase F complete, negative result. Phase G NOT triggered.

## Summary

Phase F tested the hypothesis that single-frame pixel observations under the H4 contract `(1, 84, 84)` were the blocker preventing the CNN policy from learning velocity-aware avoidance on Signal Dodge. The Phase F change wrapped train and eval VecEnvs with `VecFrameStack(n_stack=4, channels_order="first")`, giving the policy a `(4, 84, 84)` observation. PPO recipe and total timesteps were held constant from Phase D/E.

Result: under the (4, 84, 84) contract at 10,000 timesteps with `ent_coef=0.01`, the trained CNN policy did not beat the Phase E aggregate, did not clear the H5 25% reward/length gap vs the hardest frame-stack negative control (stay_only), and did not reach the revised Phase G collision trigger of <= 0.80. Two of three training runs showed policy degeneration: seed 1 exhibited entropy collapse from iteration 33 onward in training logs, and seed 3 trained_cnn eval metrics are byte-identical to the stay_only baseline (same per-seed episode lengths across all 10 eval seeds).

## Protocol

Diagnostic-not-selection, per Phase E rule. Three independent train seeds {1, 2, 3} were trained at 10,000 timesteps each on the same frame-stack YAML, then evaluated externally over the locked eval seed set 1000-1009 with `trained_cnn` only. Aggregation is pooled mean across train seeds. No best-of-N seed picking.

Negative controls were rerun under the (4, 84, 84) observation contract before any trained sweep, because the `untrained_cnn` baseline is shape-sensitive (its input layer and random-policy logits change under a different observation shape).

Frame-stack contract validation: `run_start.env_smoke.obs_shape` is `[4, 84, 84]` in every Phase F train run (verified seed 3 events.ndjson).

## Configuration

- Config: `configs/rl/signal_dodge_ppo_h5_pixel_frame_stack.yaml`
- Env: `godot:signal-dodge-v0`, `max_steps=1800`, `frame_stack=4`, `pixel_channels=1`, `headless=false`
- PPO recipe: `n_steps=256`, `batch_size=64`, `n_epochs=4`, `ent_coef=0.01`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2` (held constant from Phase D/E)
- Total timesteps: 10,000 per train seed
- Eval seeds: 1000-1009 (10 seeds, locked from Phase E)
- Git commit at experiment time: `42d5533`

## Run IDs and artifact paths

Negative controls (under frame_stack=4):
- `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\h5_eval_phase_f_frame_stack_negative_controls_10seed\`

Trained sweep:
- `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\h5_train_phase_f_frame_stack_seed1_10k\`
- `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\h5_train_phase_f_frame_stack_seed2_10k\`
- `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\h5_train_phase_f_frame_stack_seed3_10k\`
- `runs\rl\signal_dodge_ppo_h5_pixel_frame_stack\h5_eval_phase_f_frame_stack_seed<N>_10k_trained_only\` for N in {1,2,3}

Seed 3 train required one retry; the initial run hit `GodotTransportError: recv timed out after 5.0s` at step 0, a transient Godot startup race. The retry started from a cleared run directory and completed normally with the same seed and config, producing `status=ok` and the same `config_hash=f3d0246a8f2e1b1e87237a135737d0dc11e694bcaf4a3975853ae1c2eb38a2b6` as before. Seed 3 results are reproducible from the second attempt's saved `model.zip`.

## Negative controls under (4, 84, 84)

CLI stdout: `passed=True saturated_negative_controls=[]`.

| Policy | collision_rate | mean_length | mean_reward | length_ratio | saturated |
|---|---|---|---|---|---|
| stay_only | 1.0 | 606.0 | 605.0 | 0.337 | false |
| seeded_random | 1.0 | 414.3 | 413.3 | 0.230 | false |
| untrained_cnn | 1.0 | 373.3 | 372.3 | 0.207 | false |

Non-saturation gate cleared: max length_ratio is 0.337 (< 0.80 threshold), max timeout_rate is 0.0 (< 0.50 threshold).

Best (hardest-to-beat) frame-stack negative for reward/length is `stay_only` at 605/606. Best (lowest-collision) is tied at 1.0 across all three.

## Trained sweep results

| Train seed | mean_reward | mean_length | collision_rate | timeout_rate |
|---|---|---|---|---|
| 1 | 844.8 | 845.7 | 0.9 | 0.1 |
| 2 | 688.8 | 689.7 | 0.9 | 0.1 |
| 3 | 605.0 | 606.0 | 1.0 | 0.0 |
| **pooled mean** | **712.87** | **713.80** | **0.933** | **0.067** |

Phase E aggregate reference: `mean_reward=764.87`, `mean_length=765.80`, `collision_rate=0.933`.

Pooled Phase F is worse than Phase E on reward and length by ~52 each, and identical on collision rate.

## Phase G trigger evaluation

Revised trigger per GPT correction (frame-stack-aware):
- `collision_rate <= best_frame_stack_negative_collision - 0.20 = 1.0 - 0.20 = 0.80`
- 25% reward/length gap vs best frame-stack negative (read as hardest-to-beat = stay_only at 605/606): requires >= 756.25 reward, >= 757.50 length
- Not worse than Phase E aggregate: requires >= 764.87 reward, >= 765.80 length

| Criterion | Threshold | Phase F pooled | Result |
|---|---|---|---|
| collision_rate <= 0.80 | <= 0.80 | 0.933 | FAIL |
| 25% reward gap vs stay_only | >= 756.25 | 712.87 | FAIL |
| 25% length gap vs stay_only | >= 757.50 | 713.80 | FAIL |
| Not worse than Phase E reward | >= 764.87 | 712.87 | FAIL |
| Not worse than Phase E length | >= 765.80 | 713.80 | FAIL |

Even under the loosest reading of "best frame-stack negative" (= lowest = untrained_cnn at 372/373), the 25% gap target (466.25 / 466.62) is cleared, but the Phase E aggregate comparison still fails on both reward and length. Under any defensible reading of the revised Phase G trigger, Phase F does NOT trigger Phase G.

## Failure mode: policy degeneration

Two of three train seeds show degeneration to a near-deterministic stay policy.

Seed 1 (training logs, runs/rl/.../h5_train_phase_f_frame_stack_seed1_10k):
- Iterations 1-31: normal PPO training. `entropy_loss` ranges from -1.08 to -0.45, `approx_kl` from 1e-3 to 0.6, `clip_fraction` 0.04 to 0.94.
- Iteration 32 (timestep 8192): `clip_fraction=0.938`, `approx_kl=2.09`, `entropy_loss=-0.095`. Sharp drop.
- Iterations 33-40 (timesteps 8448-10240): `approx_kl=0.0`, `clip_fraction=0`, `entropy_loss` in `[-0.0005, -0.0012]`, `policy_gradient_loss` ~1e-7. Policy frozen.
- Final 20% of training was deterministic.

Seed 3 (eval result): trained_cnn per-seed `(seed -> episode_length)`:
- 1000 -> 333, 1001 -> 273, 1002 -> 843, 1003 -> 963, 1004 -> 1203, 1005 -> 1263, 1006 -> 543, 1007 -> 183, 1008 -> 183, 1009 -> 273
- These are byte-identical to the stay_only baseline per-seed episode lengths under the same frame-stack negative-controls run. The trained policy under `deterministic=true` eval is selecting the stay action at every step.

Seed 2 shows a partial pattern: one timeout (seed 1001, length 1800), nine collisions, mean length 689.7. Less complete collapse than seed 3 but the policy was unable to consistently improve over stay-only behavior.

## Diagnosis

Under the Phase D entropy recipe (`ent_coef=0.01`), the policy collapses to a low-entropy degenerate solution within 10,000 timesteps under the (4, 84, 84) observation contract. The collapse is consistent with the policy learning that the stay action gives positive per-step survival reward and rapidly committing to it. Stacked frames did not, at this budget and recipe, surface a velocity-aware avoidance signal strong enough to compete with the trivial stay solution.

This is not a failure of the frame-stack contract (the contract propagates correctly). It is a failure of the recipe-budget pair to find a non-degenerate policy under that contract. The Phase F hypothesis (single-frame perception is the blocker) is not falsified by this result, but it is also not supported. The dominant blocker at 10,000 timesteps is policy collapse, not perception.

## Provenance

All metrics in this document are read directly from JSON artifacts in `runs/rl/signal_dodge_ppo_h5_pixel_frame_stack/` produced by `h5_baseline_cli` at git commit `42d5533`. No hand-calculated or remembered numbers.

- Negative-control summaries: `evaluation/{stay_only,seeded_random,untrained_cnn}/summary.json`
- Train summaries: `h5_train_phase_f_frame_stack_seed{1,2,3}_10k/summary.json` (all `status=ok`)
- Eval summaries: `h5_eval_phase_f_frame_stack_seed{1,2,3}_10k_trained_only/evaluation/trained_cnn/summary.json`
- Train log for seed 1 entropy-collapse observation: `runs/rl/h5_phasef_sweep.stdout.log`

## Open questions for GPT

1. "Best frame-stack negative" definition. Hardest-to-beat (stay_only at 605/606) vs lowest (untrained_cnn at 372/373) was ambiguous in the Phase G trigger spec. Phase F fails under either reading on the Phase E comparison, so the ambiguity did not change this decision, but the convention should be resolved before Phase G or its successor.
2. Next Phase F or successor experiment. Options on the table: raise `ent_coef` (e.g., 0.05) to delay collapse, raise `total_timesteps` past collapse onset to see if the policy recovers, change the per-step reward shaping (currently survival = +1/step which incentivizes stay), or move to a different perception change (e.g., grayscale -> color, or larger pixel grid). Not a Claude decision.
