# H5 Trained-Policy Slice (Phase C, 10000 timesteps)

Second trained-CnnPolicy evidence slice on the existing H4 pixel Signal
Dodge profile. ~5x the Phase B training budget. NOT closure-grade and
NOT a budget-scaling result.

## Commands

Training:

```
python -m sight_agent.rl.train \
  --config configs/rl/signal_dodge_ppo_h4_pixel.yaml \
  --total-timesteps 10000 \
  --run-id h5_train_phase_c_10k
```

Trained-only eval:

```
python -m sight_agent.rl.h5_baseline_cli \
  --config configs/rl/signal_dodge_ppo_h4_pixel.yaml \
  --run-id h5_eval_phase_c_10k_trained_only \
  --seeds 1000-1009 \
  --mode full \
  --policies trained_cnn \
  --train-run-dir runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_c_10k
```

Full 4-policy eval was NOT run, per Phase C decision rule. See
"Decision" below.

## Runs

- Train: `runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_c_10k/`
  - 10000 timesteps requested, 10048 executed (PPO rollout aligned
    to n_steps=64)
  - 156 PPO updates
  - 1431 s wall (SB3 `time_elapsed`, ~24 min)
  - `model.zip` 20.2 MB
  - YAML defaults: n_steps=64, batch_size=32, n_epochs=1,
    ent_coef=0.0, learning_rate=3e-4
- Eval: `runs/rl/signal_dodge_ppo_h4_pixel/h5_eval_phase_c_10k_trained_only/`
  - 10 seeds (1000-1009), `--mode full --policies trained_cnn`
  - ~191 s aggregate per-seed `elapsed_seconds`, ~3 min wall

## Phase C trained_cnn aggregate (10 seeds 1000-1009)

| policy        | mean_reward | mean_length | collision_rate | timeout_rate | length_ratio |
| ------------- | ----------- | ----------- | -------------- | ------------ | ------------ |
| trained_cnn   |       688.8 |       689.7 |            0.9 |          0.1 |        0.383 |

Saturation gate: `length_ratio=0.383` well below the 0.80 ceiling,
`timeout_rate=0.1` well below 0.50. `saturated=false`. Profile is
NOT saturated, same as Phase B.

## Comparison to Phase B (2048 ts) and Phase B negative-control baseline

Phase B and Phase C trained_cnn aggregates are identical to the
significant figures recorded:

| metric           | Phase B (2048 ts) | Phase C (10000 ts) | delta |
| ---------------- | ----------------- | ------------------ | ----- |
| mean_reward      |             688.8 |              688.8 |   0.0 |
| mean_episode_len |             689.7 |              689.7 |   0.0 |
| collision_rate   |               0.9 |                0.9 |   0.0 |
| timeout_rate     |               0.1 |                0.1 |   0.0 |
| length_ratio     |             0.383 |              0.383 | 0.000 |

Per-seed values are bit-identical across the two runs. Seed 1001
timeouts at 1800 in both runs. Seed 1006 collides at length 183 in
both runs. Seeds 1000, 1002-1005, 1007-1009 produce identical
collision lengths in both runs.

Phase B published negative-control aggregate (best = stay_only and
untrained_cnn tied):

| metric           | best neg control | Phase C trained_cnn | gap   |
| ---------------- | ---------------- | ------------------- | ----- |
| mean_reward      |            605.0 |               688.8 | +13.9% |
| mean_episode_len |            606.0 |               689.7 | +13.8% |
| collision_rate   |              1.0 |                 0.9 | -10 pp |

Negative-control re-run was deliberately skipped this round. The H4
pixel profile is deterministic at the seed level for `stay_only`,
`untrained_cnn`, and `seeded_random`; the published Phase B
aggregates remain authoritative. The H5 plan's "full eval only if
trained_cnn clears the bars" criterion explicitly governs this.

## H5 section 6 GREEN bar

| criterion                           | value   | threshold | result   |
| ----------------------------------- | ------- | --------- | -------- |
| mean_reward gap vs best neg control |  +13.9% |   +25%    | FAIL     |
| mean_length gap vs best neg control |  +13.8% |   +25%    | FAIL     |
| collision_rate reduction (pp)       |    10pp |   20pp    | FAIL     |
| saturation gate                     |   pass  |   pass    | PASS     |
| same-seed pixel determinism         | not run | PASS req  | DEFERRED |

Verdict: NOT closure-grade. Identical magnitude of failure to Phase
B. None of the three quantitative bars cleared.

## Decision: skip full 4-policy eval

Per the Phase C execution prompt: "If trained_cnn is clearly below
the bars, do not spend wall-time on full 4-policy eval. Document
the result and recommend the next budget."

Phase C trained_cnn is bit-identical to Phase B trained_cnn. The
negative controls are seed-deterministic and were measured in Phase
B; re-measuring them would add no information. Full 4-policy eval
skipped to preserve wall time. Determinism re-check also skipped
since the H5 plan only requires it after a passing full eval.

## Diagnosis: NOT insufficient training. ent_coef=0.0 collapse.

The Phase B closure-grade diagnosis recorded "insufficient training"
as the primary blocker. Phase C falsifies that hypothesis. Of the
four candidate blockers:

- (a) insufficient training: FALSIFIED. 5x more timesteps and 156
  PPO updates (vs 32 in Phase B) produced bit-identical eval results.
  Additional training of this exact recipe does not move the policy.
- (b) hyperparams: PRIMARY. The committed
  `configs/rl/signal_dodge_ppo_h4_pixel.yaml` is explicitly tagged
  "Smoke-cheap PPO values" sized for H4's "constructs and runs while
  writing artifacts" acceptance, not for learning. Specifically
  `ent_coef=0.0` (SB3 default) drives premature entropy collapse:
    - Phase B entropy_loss reached -0.07 by iter 26 of 32.
    - Phase C entropy_loss reached -0.07 by iter 36 of 156, then
      drifted to -0.003 by iter ~100 and stayed there.
    - From iter ~80 onward Phase C shows `approx_kl=0.0`,
      `clip_fraction=0`, `policy_gradient_loss` in the 1e-7 to 1e-8
      range. With near-zero entropy and near-zero advantage signal,
      the policy gradient has nothing to optimize. The argmax is
      frozen.
  Same recipe with non-zero `ent_coef` (e.g. 1e-2), or larger
  `n_steps` / `n_epochs`, is the credible next experiment.
- (c) profile headroom: NOT THE BLOCKER. Length ratio 0.383 leaves
  62% of the 1800-step budget on the table. Same conclusion as
  Phase B.
- (d) eval/reward issue: NO EVIDENCE.
- (e) runtime/ops: NO. Train and eval both completed cleanly.

## Implication for next slice

Increasing the training budget further is not the right next move.
Phase C demonstrates that this hyperparameter recipe converges to a
fixed deterministic policy well before 10K timesteps. A 50K or 100K
run with the same YAML would burn wall time without changing
behavior.

The credible next slice is a hyperparameter pass at the SAME or
slightly LARGER timestep budget. Concrete candidates for GPT to
choose among:

- `ent_coef` 1e-2 or 1e-3 (most direct fix for the collapse mode).
- `n_steps` 256 with `batch_size` 64, `n_epochs` 4-10 (closer to
  Atari-style PPO baselines; larger rollouts to estimate advantage
  before policy updates).
- both of the above combined.

Whichever combination, the smoke-cheap defaults should be replaced
with a learning-grade recipe before another budget bump.

This recommendation is GPT's to confirm; this doc records the
empirical falsification of the "more timesteps" hypothesis only.

## Code changes and artifacts

- No code changes were required for this slice. Existing
  `sight_agent.rl.train` and `sight_agent.rl.h5_baseline_cli` cover
  the train -> save -> trained-only eval path end-to-end.
- Run artifacts under `runs/` are gitignored per H4 convention.
- This doc is the only on-disk durable summary of the slice.
