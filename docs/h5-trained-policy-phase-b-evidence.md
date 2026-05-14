# H5 Trained-Policy Slice (Phase B, 2048 timesteps)

First trained-CnnPolicy evidence slice on the existing H4 pixel Signal
Dodge profile after the collision-propagation bug fix. Smoke-grade
evidence, NOT closure-grade.

## Runs

- Train: `runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_b_2048/`
  - 2048 timesteps, 32 PPO updates, 460s wall
  - n_steps=64, batch_size=32, n_epochs=1 (existing H4 YAML defaults)
  - `model.zip` produced at commit `fb5b32e`
- Eval: `runs/rl/signal_dodge_ppo_h4_pixel/h5_eval_phase_b_10seed/`
  - 10 seeds (1000-1009), mode=full, all 4 policies
  - Per-policy `evaluation/<policy>/summary.json` and run-level
    `index.json`

## Per-policy aggregate (10 seeds)

| policy        | mean_reward | mean_length | collision_rate | timeout_rate | length_ratio |
| ------------- | ----------- | ----------- | -------------- | ------------ | ------------ |
| stay_only     |       605.0 |       606.0 |            1.0 |          0.0 |        0.337 |
| seeded_random |       413.3 |       414.3 |            1.0 |          0.0 |        0.230 |
| untrained_cnn |       605.0 |       606.0 |            1.0 |          0.0 |        0.337 |
| trained_cnn   |       688.8 |       689.7 |            0.9 |          0.1 |        0.383 |

Best negative control: stay_only / untrained_cnn (tied, identical
per-seed values).

## Saturation gate

`saturation_decision.passed = true`. All three negative controls
length_ratio well below the canonical 0.80 threshold and
timeout_rate=0.0 well below 0.50. Profile is NOT saturated.

## H5 section 6 GREEN bar

| criterion                           | value   | threshold | result   |
| ----------------------------------- | ------- | --------- | -------- |
| mean_reward gap vs best neg control |  +13.9% |    25%    | FAIL     |
| mean_length gap vs best neg control |  +13.8% |    25%    | FAIL     |
| collision_rate reduction (pp)       |    10pp |   20pp    | FAIL     |
| same-seed pixel determinism         | not run | PASS req  | DEFERRED |

Verdict: NOT closure-grade. Trained policy is measurably above the
negative-control floor but falls short of GREEN on all three
quantitative bars.

## Untrained_cnn collapses to stay_only

Under deterministic eval the freshly-initialized SB3 PPO CnnPolicy
argmaxes to action 1 (stay) on every observation. Across all 10 seeds
`untrained_cnn` per-seed reward / length / collision flags are byte-
equal to `stay_only`. The H4 closure record and `docs/sight-h5-plan.md`
section 1 anticipated this: a freshly initialized CnnPolicy at step-0
hazard density is not learning evidence, and the deterministic argmax
collapses to a constant action. Practically, the H5 negative-control
suite delivers 2 independent baselines on this profile (stay_only and
seeded_random), not 3.

## Diagnosis: blocker is insufficient training

Of the four candidate blockers from the H5 execution prompt:

- (a) insufficient training: PRIMARY. 32 PPO updates against a random-
  initialized CnnPolicy on 84x84 grayscale pixel input is far below
  typical training budgets for pixel observation spaces. The training
  trace shows entropy_loss collapsing from -1.10 to -0.07 by iter 26,
  with periodic value_loss spikes (1.07e+03, 1.14e+03) at iters 13,
  20, 30. The policy departed random-init and started exploring but
  did not converge.
- (b) profile headroom: NOT THE BLOCKER. Best negative control
  length_ratio = 0.337 leaves >60% of the 1800-step max_steps budget
  for a trained policy to exploit. Hardening the profile would shrink
  that headroom, not enlarge it.
- (c) reward/eval metric issue: NO EVIDENCE. Per-step +1 survival
  reward produces sensible per-seed numbers (e.g. trained_cnn seed
  1001 reaches a 1800-step timeout).
- (d) runtime/ops issue: NO. Both train and eval completed cleanly.

## Implication for next slice

The natural next move is a larger training budget on the same H4 pixel
profile, not profile hardening. Concretely a 10K-50K timestep training
run with the existing hyperparameters, followed by the same 10+ seed
full-mode eval. Step 2B (profile hardening) remains unjustified at
section-5 saturation thresholds.

## Code changes and artifacts

- No code changes were required for this slice. Existing
  `sight_agent.rl.train` and `sight_agent.rl.h5_baseline_cli` already
  cover the train -> save -> load -> 4-policy eval path end-to-end.
- Run artifacts under `runs/` are gitignored per H4 convention.
- This doc is the only on-disk durable summary of the slice.
