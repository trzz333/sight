# H5 Trained-Policy Slice (Phase D, entropy recipe, 10000 timesteps)

Third trained-CnnPolicy evidence slice. First with a learning-grade
recipe (`ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`)
on a dedicated H5 config. NOT closure-grade. Falsifies the
entropy-collapse hypothesis from Phase C.

## Crash recovery note

Claude Desktop crashed twice during this session, both crashes after
the train and trained-only eval commands had completed cleanly but
before any evidence doc or handoff edits landed. State was recovered
by inspecting `git status`, the live process table, and on-disk run
artifacts. No live `python`/`Godot` processes were running at recovery
time. No re-training was performed. Run dirs and run ids match the
originally intended targets; no `_recovery1` variant was needed.

## Commands

Training (executed pre-crash):

```
python -m sight_agent.rl.train \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --total-timesteps 10000 \
  --run-id h5_train_phase_d_entropy_10k
```

Trained-only eval (executed pre-crash):

```
python -m sight_agent.rl.h5_baseline_cli \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --run-id h5_eval_phase_d_entropy_10k_trained_only \
  --seeds 1000-1009 \
  --mode full \
  --policies trained_cnn \
  --train-run-dir runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_d_entropy_10k
```

Full 4-policy eval was NOT run, per the same Phase C decision rule
(see "Decision" below).

## Config and hyperparameter diff vs the H4 smoke-cheap config

New file:
`configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`
(config_hash `013a8f0d12248900d4bbba84b0f29c9fc890981e43f94d7897deb60164898c81`).

| field           | H4 smoke (Phase B/C) | Phase D entropy |
| --------------- | -------------------- | --------------- |
| `n_steps`       |                   64 |             256 |
| `batch_size`    |                   32 |              64 |
| `n_epochs`      |                    1 |               4 |
| `ent_coef`      |    not set (SB3 0.0) |            0.01 |
| `learning_rate` |               0.0003 |          0.0003 |
| `eval_freq`     |                   64 |            2048 |

All env, observation, device, max_steps, checkpoint, and Godot
settings unchanged from the H4 pixel config.

## Runs

- Train: `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_d_entropy_10k/`
  - 10000 timesteps requested, 10240 executed (PPO rollout aligned
    to `n_steps=256`)
  - 40 PPO iterations; `train/n_updates = 156` at run end
    (`n_epochs * (iters - 1) = 4 * 39`)
  - 338.91 s wall (run_end `elapsed_seconds`, ~5m39s); SB3
    `time/time_elapsed = 340`, `time/fps = 30`
  - `summary.json status = ok`
  - `model.zip` present, `events.ndjson` 47 lines (run_start, 40
    train_metrics, 5 eval, run_end)
  - `git_commit = 509192b`
- Eval: `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_eval_phase_d_entropy_10k_trained_only/`
  - 10 seeds (1000-1009), `--mode full --policies trained_cnn`
  - Per-seed `elapsed_seconds` aggregate ~191 s (~3 min wall),
    matching Phase C's per-seed timing within wall-clock noise
  - `evaluation/trained_cnn/summary.json` and
    `evaluation/index.json` both present

## Training entropy / approx_kl / clip_fraction / pg trajectory

Sampled `train_metrics` events (iter, total_timesteps, entropy_loss,
approx_kl, clip_fraction, policy_gradient_loss, value_loss,
explained_variance):

```
iter   ts      ent      kl       clip   pg          vl       ev
   2   512   -1.0909  0.00325   0.168  -0.002642   85.96    0.001
  10  2560   -1.0380  0.01191   0.518   0.021392   17.65    0.006
  13  3328   -0.7581  0.10780   0.338   0.019623  201.35    0.010
  17  4352   -0.2050  0.06902   0.098   0.014374  181.86    0.010
  23  5888   -0.1493  0.04959   0.030   0.003332  160.88    0.028
  24  6144   -0.0374  0.00000   0.000  -0.000010  275.71    0.036
  28  7168   -0.5510  0.04581   0.294   0.011339  136.84    0.127
  34  8704   -0.6656  0.00567   0.102   0.003469  162.23    0.409
  36  9216   -0.0879  0.14336   0.075   0.016872  102.27    0.365
  40 10240   -0.0168  0.00000   0.000  -0.000063  114.81    0.455
```

Observations:

- `H(uniform-over-3)` is `ln(3) ~= 1.0986`. The recipe maintained
  near-maximum entropy through iter ~13 (`entropy_loss` from -1.09
  down to -0.76) and substantial entropy through iter ~23
  (-0.15 range).
- `approx_kl` was substantial (0.01-0.12) through iter 13 and
  intermittently substantial through iter 36. `clip_fraction` was
  substantial (0.1-0.5) through iter 13 and again at iter 28.
- The policy DID re-collapse to near-deterministic by the end of
  training (iter 38-40: `entropy_loss` -0.03 to -0.02,
  `approx_kl ~ 0`, `clip_fraction ~ 0`).
- Value function learned: `explained_variance` rose monotonically
  in trend from 0.001 (iter 2) to 0.455 (iter 40).

## In-training deterministic eval was locked from step 2048

The five SB3 in-training eval points (each `n_eval_episodes=1`,
`deterministic=true`) produced bit-identical `mean_reward`:

```
step   2048: mean_reward 212.0, std 0.0
step   4096: mean_reward 212.0, std 0.0
step   6144: mean_reward 212.0, std 0.0
step   8192: mean_reward 212.0, std 0.0
step  10240: mean_reward 212.0, std 0.0
```

The deterministic eval trajectory at the run seed locked in by
step 2048 (~iter 8, well inside the high-entropy phase) and never
moved across the remaining 32 rollouts and ~8000 timesteps.

## Phase D trained_cnn aggregate (10 seeds 1000-1009)

| policy        | mean_reward | mean_length | collision_rate | timeout_rate | length_ratio |
| ------------- | ----------- | ----------- | -------------- | ------------ | ------------ |
| trained_cnn   |       688.8 |       689.7 |            0.9 |          0.1 |        0.383 |

Saturation gate: `length_ratio=0.383` below 0.80 ceiling,
`timeout_rate=0.1` below 0.50. `saturated=false`.

## Per-seed comparison vs Phase C

Episode lengths per seed across Phase C (smoke recipe, 10k ts) and
Phase D (entropy recipe, 10k ts):

| seed | C ep_len | D ep_len | C term      | D term      |
| ---- | -------- | -------- | ----------- | ----------- |
| 1000 |      903 |      903 | collision   | collision   |
| 1001 |     1800 |     1800 | timeout     | timeout     |
| 1002 |      273 |      273 | collision   | collision   |
| 1003 |      363 |      363 | collision   | collision   |
| 1004 |      753 |      753 | collision   | collision   |
| 1005 |     1203 |     1203 | collision   | collision   |
| 1006 |      183 |      183 | collision   | collision   |
| 1007 |      693 |      693 | collision   | collision   |
| 1008 |      423 |      423 | collision   | collision   |
| 1009 |      303 |      303 | collision   | collision   |

Bit-identical. Only `elapsed_seconds` differs (wall-clock noise).
Phase D and Phase C deterministic eval trajectories are byte-equal
across all 10 evaluation seeds despite materially different training
trajectories.

## Phase B / Phase C / Phase D aggregate equivalence

| metric           | Phase B (2048 ts) | Phase C (10000 ts) | Phase D (entropy 10000 ts) |
| ---------------- | ----------------- | ------------------ | -------------------------- |
| mean_reward      |             688.8 |              688.8 |                      688.8 |
| mean_episode_len |             689.7 |              689.7 |                      689.7 |
| collision_rate   |               0.9 |                0.9 |                        0.9 |
| timeout_rate     |               0.1 |                0.1 |                        0.1 |
| length_ratio     |             0.383 |              0.383 |                      0.383 |

Three different training configurations produced the same
deterministic eval policy at the eval seed set.

## Comparison to Phase B / C published best negative-control baseline

Phase B negative-control aggregate (best = stay_only and
untrained_cnn tied):

| metric           | best neg control | Phase D trained_cnn | gap   |
| ---------------- | ---------------- | ------------------- | ----- |
| mean_reward      |            605.0 |               688.8 | +13.9% |
| mean_episode_len |            606.0 |               689.7 | +13.8% |
| collision_rate   |              1.0 |                 0.9 | -10 pp |

Same gap magnitudes as Phase B and Phase C. The negative controls
are seed-deterministic on the H4 pixel profile; the published Phase
B aggregates remain authoritative. Per the H5 plan's "full eval only
if trained_cnn clears the bars" rule, no negative-control re-run is
warranted.

## H5 section 6 GREEN bar

| criterion                              | value   | threshold | result   |
| -------------------------------------- | ------- | --------- | -------- |
| mean_reward gap vs best neg control    |  +13.9% |   +25%    | FAIL     |
| mean_length gap vs best neg control    |  +13.8% |   +25%    | FAIL     |
| collision_rate reduction (pp)          |    10pp |   20pp    | FAIL     |
| saturation gate                        |   pass  |   pass    | PASS     |
| same-seed pixel determinism            | not run | PASS req  | DEFERRED |

Verdict: NOT closure-grade. Identical failure magnitude to Phase B
and Phase C.

## Decision: skip full 4-policy eval

Per the Phase D execution prompt decision rule, full eval runs only
when `trained_cnn` clears at least one of reward/length AND clears
collision, or when gaps are >= 20% on reward/length or
`collision_rate <= 0.8`. Phase D meets none of these conditions
(13.9% reward gap, 13.8% length gap, 0.9 collision rate). Full
4-policy eval skipped.


## Entropy-collapse verdict: FALSIFIED

The Phase C evidence doc named `ent_coef=0.0` driving premature
entropy collapse as the primary blocker and recommended raising
`ent_coef` and enlarging `n_steps`/`n_epochs`. Phase D implements
that exact recommendation. The result:

- `ent_coef=0.01` successfully delivered an entropy budget. Entropy
  stayed near uniform (`entropy_loss > -0.7`) through iter ~13 and
  remained substantial (`entropy_loss > -0.15`) through iter ~23.
- `approx_kl` and `clip_fraction` confirm the policy was actively
  updating across those iterations.
- The in-training deterministic eval at step 2048 (~iter 8, deep
  inside the high-entropy phase) was bit-identical to the
  in-training eval at step 10240. The eval-relevant argmax locked
  in DURING the high-entropy training phase.
- All 10 external eval seeds (1000-1009) produced trajectories
  byte-equal to Phase C's smoke-recipe trajectories.

Adding entropy to the loss did not change the argmax-selected
actions at the eval states. The entropy-collapse hypothesis is
falsified as the primary blocker. Whatever is preventing learning
operates orthogonally to action-distribution entropy at this
profile, this architecture, and this training budget.

## Implication for the next slice

The "smoke-cheap hyperparameters are the blocker" framing (Phase B
closure-grade diagnosis, Phase C secondary diagnosis) is empirically
dead. Three different hyperparameter recipes produce bit-identical
external eval trajectories. The deterministic eval policy is
invariant under:

- timestep budget changes (2048 vs 10000 vs 10000)
- `ent_coef` changes (0.0 vs 0.01)
- `n_steps` changes (64 vs 256)
- `batch_size` changes (32 vs 64)
- `n_epochs` changes (1 vs 4)
- `eval_freq` changes (64 vs 2048)

What was NOT changed across the three slices:

- random seed (`seed=0` in every config)
- policy architecture (SB3 NatureCNN under `CnnPolicy`)
- observation pipeline (`(1, 84, 84)` grayscale, channel-first)
- reward function (sparse `+1` per non-terminal step)
- environment (Signal Dodge H4 profile)

Candidate next experiments (GPT to choose; ranked by how directly
they break the determinism above):

1. **Seed sweep.** Train the entropy recipe at seeds 1, 2, 3 and
   evaluate each. If the eval-policy invariance is a seed=0 fixed
   point, a different seed will produce different argmax behavior.
   If multiple seeds also converge to the same eval policy, the
   invariance is structural and architectural changes are required.
2. **Frame stacking.** Single-frame `(1, 84, 84)` may not encode
   velocity information. SB3's `VecFrameStack(n=4)` is the standard
   fix. This changes the observation contract and is more invasive
   than a config knob.
3. **State-observation comparator.** Train the same PPO on
   `observation_mode=state` and compare. If state-mode learns and
   pixel-mode does not, the failure is in the perception path, not
   PPO. The H3 state-mode pipeline already exists.
4. **Reward shaping.** Dense reward (proximity-to-hazard penalty,
   center-screen bonus, or action-cost). This is a charter-amendment
   decision per the H5 plan section 7 non-goals, not a knob.

Increasing the timestep budget further within the current recipe
is not on this list. Three different recipes have failed at this
profile.

This recommendation is GPT's call to refine. This doc records the
empirical falsification of the entropy-collapse hypothesis only.

## Code changes and artifacts

- No code changes were required for this slice. The new H5 entropy
  config is a YAML-only addition.
- Run artifacts under `runs/` are gitignored per H4 convention.
- This doc and `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`
  are the on-disk durable summary of the slice.
