# H5 Trained-Policy Slice (Phase E, entropy recipe, three-seed sweep)

Diagnostic seed-sensitivity test. Not a seed-selection experiment.
H5 learning evidence may not be claimed from best-of-N train-seed
selection. Any best individual train seed is reported descriptively
only.

## Diagnostic-not-selection preamble

Phase D established that on `seed=0`, three different recipes
(Phase B smoke, Phase C 10k smoke, Phase D 10k entropy) produced
byte-equal external eval trajectories. Phase E tests whether the
eval-policy invariance is a seed=0 fixed point or structural across
train seeds. Phase E does NOT create a candidate acceptance seed.
The bars for H5 acceptance per `docs/sight-h5-plan.md` section 6
require pre-registered replication or a multi-train-seed protocol;
this run is neither.

## Commands

Trains:

```
python -m sight_agent.rl.train \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --seed 1 --total-timesteps 10000 \
  --run-id h5_train_phase_e_seed1_entropy_10k

python -m sight_agent.rl.train \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --seed 2 --total-timesteps 10000 \
  --run-id h5_train_phase_e_seed2_entropy_10k

python -m sight_agent.rl.train \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --seed 3 --total-timesteps 10000 \
  --run-id h5_train_phase_e_seed3_entropy_10k
```

Trained-only evals (same shape per seed):

```
python -m sight_agent.rl.h5_baseline_cli \
  --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml \
  --run-id h5_eval_phase_e_seed<N>_entropy_10k_trained_only \
  --seeds 1000-1009 --mode full --policies trained_cnn \
  --train-run-dir runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed<N>_entropy_10k
```

`<N>` in `{1, 2, 3}`. Negative controls not rerun this slice; the
Phase B/C/D negative-control aggregate (best = `stay_only` =
`untrained_cnn` tied at 605.0 reward, 606.0 length, 1.0 collision)
remains authoritative.

## Training dynamics per train seed

All three trains ran the same Phase D entropy recipe
(`ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`,
`learning_rate=3e-4`, `eval_freq=2048`), 10240 timesteps executed
(10000 requested, PPO rollout aligned), 40 PPO iterations,
`train/n_updates=156`.

Entropy trajectory (sampled `entropy_loss` from `train_metrics`
events, `H(uniform-over-3) = ln(3) ~= 1.0986`):

| iter | seed 0 (Phase D) | seed 1     | seed 2     | seed 3     |
| ---- | ---------------- | ---------- | ---------- | ---------- |
|    2 |          -1.0909 |     ~-1.09 |     ~-1.09 |     ~-1.09 |
|   13 |          -0.7581 |   meaningful | meaningful | declining |
|   24 |          -0.0374 |   variable |  -0.20 ish | -7.29e-05 |
|   36 |          -0.0879 |   variable |     -0.586 | -5.84e-05 |
|   40 |          -0.0168 |   variable |     -0.635 | -1.36e-04 |

Observed:

- Seed 0 (Phase D) and seed 2 held meaningful entropy through the
  middle of training, ending in the -0.02 to -0.64 range with
  intermittent `clip_fraction` spikes.
- Seed 3 collapsed to deterministic argmax by iter ~24
  (`entropy_loss` near -7e-5, `approx_kl = 0.0`,
  `clip_fraction = 0.0`) and stayed there for the final 16
  iterations. Seed 3 trained on a degenerate distribution for
  roughly half its budget.
- Three training trajectories are materially distinct.

## Per-eval-seed deterministic eval trajectories

Episode length and terminal cause per evaluation seed, across the
four train seeds:

| eval_seed | D s0 (len, term) | E s1            | E s2            | E s3            |
| --------- | ---------------- | --------------- | --------------- | --------------- |
|      1000 |  903, collision  |  333, collision | 1383, collision | 1383, collision |
|      1001 | 1800, timeout    |  273, collision |  483, collision |  483, collision |
|      1002 |  273, collision  |  843, collision | 1293, collision | 1293, collision |
|      1003 |  363, collision  |  963, collision |  603, collision |  603, collision |
|      1004 |  753, collision  | 1203, collision | 1443, collision | 1443, collision |
|      1005 | 1203, collision  | 1263, collision |  363, collision |  363, collision |
|      1006 |  183, collision  |  543, collision |  573, collision |  573, collision |
|      1007 |  693, collision  |  183, collision |  273, collision |  273, collision |
|      1008 |  423, collision  |  183, collision | 1800, timeout   | 1800, timeout   |
|      1009 |  303, collision  |  273, collision |  243, collision |  243, collision |

Three distinct per-seed trajectory vectors emerge across four train
seeds:

- Vector A: seed 0
- Vector B: seed 1
- Vector C: seed 2 = seed 3 (byte-equal per-eval-seed lengths and
  terminal causes; `elapsed_seconds` differs only by wall-clock
  noise)

## Per-train-seed aggregates (descriptive only)

| train seed | mean_reward | mean_length | collision_rate | timeout_rate | length_ratio |
| ---------- | ----------- | ----------- | -------------- | ------------ | ------------ |
| 0 (Phase D)|       688.8 |       689.7 |           0.9  |         0.1  |        0.383 |
| 1          |       605.0 |       606.0 |           1.0  |         0.0  |        0.337 |
| 2          |       844.8 |       845.7 |           0.9  |         0.1  |        0.470 |
| 3          |       844.8 |       845.7 |           0.9  |         0.1  |        0.470 |

Saturation gate (`length_ratio < 0.8` and `timeout_rate < 0.5`):
all four pass.

Seed 1 aggregate is identical to the best-negative-control aggregate
(605.0 reward, 606.0 length, 1.0 collision). Seed 1's trained policy
is empirically indistinguishable from the best negative control on
the 10-seed external eval set.

## Aggregate-across-train-seeds diagnostic (seeds 1, 2, 3)

Per the amended Phase E interpretation rule, the aggregate is
computed over the new sweep (seeds 1, 2, 3 pooled as 30 episodes)
WITHOUT silently pooling Phase D's seed 0. Seed 0 is shown
descriptively in the prior baseline column above.

Pooled over 30 episodes (seeds 1, 2, 3 x eval seeds 1000-1009):

- mean_reward: 764.87
- mean_length: 765.80
- collision_rate: 0.933 (28 collisions / 30 episodes)
- timeout_rate: 0.067 (2 timeouts / 30 episodes)

Gaps vs best negative control (605.0 reward, 606.0 length, 1.0
collision):

| criterion                              | aggregate | threshold | result |
| -------------------------------------- | --------- | --------- | ------ |
| mean_reward gap vs best neg control    |   +26.4%  |   +25%    | PASS   |
| mean_length gap vs best neg control    |   +26.4%  |   +25%    | PASS   |
| collision_rate reduction (pp)          |    6.7pp  |   20pp    | FAIL   |
| saturation gate                        |    pass   |   pass    | PASS   |
| same-seed pixel determinism            |  not run  |  PASS req |DEFERRED|

The reward and length bars are marginally cleared (~1pp above 25%
threshold). The collision bar fails by ~13pp. The agent is alive
longer on average across the sweep, but is not avoiding hazards at a
rate distinguishable from the negative-control regime (collision
rate 0.933 vs 1.0 = -6.7pp).

Per the amended outcome table, this is the "weak seed-sensitive
signal" row: seeds differ, aggregate is distinguishable from
controls, but is not closure-grade. The reward and length improvement
is being pulled up entirely by the seed 2 / seed 3 attractor.

## Verdicts

**Seed=0 fixed-point hypothesis: FALSIFIED.** Per-eval-seed
trajectories are NOT invariant across train seeds. Three distinct
trajectory vectors emerge across train seeds {0, 1, 2, 3}.

**Phase D conclusion ("eval policy is invariant under PPO knob
changes") refined, not overturned.** Under fixed `seed=0`, three
different recipes produced byte-equal trajectories. Under fixed
recipe (Phase D entropy), four different seeds produce three
distinct trajectories. So Phase D's invariance was a seed=0 fixed
point under that family of recipes; the policy space is not a single
global attractor.

**Attractor structure observed.** Seeds 2 and 3 converge to the
same eval-policy attractor despite materially different training
dynamics (seed 2 held high entropy with intermittent
`clip_fraction` spikes throughout; seed 3 collapsed entropy by
iter 24 and trained on a near-deterministic policy for the second
half). The eval-relevant argmax is robust to training-time
exploration in this case. Two of four observed train seeds produce
the same attractor.

**No H5 acceptance claim.** No individual train seed clears all
three quantitative bars. Seed 2 and seed 3 individually clear
reward (+39.6%) and length (+39.6%) but fail collision (-10pp vs
-20pp threshold). Per the diagnostic-not-selection rule, the best
individual seed is not promoted to candidate acceptance.

**Aggregate signal exists but is weak.** Reward and length bars
marginally cleared in aggregate (+26.4% over best neg control, ~1pp
above threshold). Collision bar fails badly (-6.7pp vs -20pp
threshold). Not closure-grade.

## Amended outcome table (Phase E exit rule)

| Phase E outcome                                                                                   | Interpretation                                       | Next move                                                                     |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| Seeds 1, 2, 3 byte-equal to Phase D                                                               | Structural invariance across recipes and train seeds | Frame stacking                                                                |
| Seeds differ, but all fail bars                                                                   | Seed affects argmax, but no learning-grade evidence  | Frame stacking, unless trajectory analysis points elsewhere                   |
| Seeds differ and aggregate over train seeds is distinguishable from controls but still below bars | Weak seed-sensitive signal                           | GPT decides Phase F, likely frame stacking or state comparator                |
| One seed clears bars but aggregate does not                                                       | Seed-fishing warning, not H5 evidence                | Do not close; require pre-registered replication or multi-train-seed protocol |
| Aggregate over train seeds clears bars                                                            | Candidate learning signal                            | Full 4-policy eval protocol and determinism checks before any H5 closure path |

**Match:** row 3 (weak seed-sensitive signal). Aggregate clears
reward and length bars by ~1pp; aggregate fails collision bar by
~13pp; no individual seed clears all three bars.

The amended rule is honored: no best-of-N seed selection. The seed 2
/ seed 3 attractor is reported descriptively, not promoted to a
candidate. The reward/length aggregate clearance is too marginal to
override the collision failure.

**Recommended next move:** GPT decides Phase F. Frame stacking
(`VecFrameStack(n=4)`) remains the most direct attack on the
remaining structural hypothesis: single-frame `(1, 84, 84)`
observations may not encode velocity. The seed sweep shows the
policy DOES vary, just not toward avoidance. That is consistent
with a perception bottleneck: the policy is selecting actions on
incomplete state. State-observation comparator (option 3 from
Phase D) is a secondary discriminator if frame stacking also stalls.

## Code changes and artifacts

- No code changes. Train CLI's existing `--seed` flag and eval
  CLI's existing `--train-run-dir` flag were sufficient.
- Run artifacts under
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed{1,2,3}_entropy_10k/`
  and
  `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_eval_phase_e_seed{1,2,3}_entropy_10k_trained_only/`.
- `runs/` remains gitignored. This doc is the durable on-disk
  summary.

## Variables now held constant across slices B, C, D, E

- SB3 NatureCNN under `CnnPolicy`
- `(1, 84, 84)` grayscale single-frame observation (no frame stack)
- sparse `+1`-per-step survival reward
- Signal Dodge H4 profile

Train seed and PPO hyperparameters have both been swept and produce
varied but failing policies. The remaining levers are the
observation contract (frame stack, state comparator), the reward
function (charter-amendment territory), and the architecture
(out-of-scope at H5).
