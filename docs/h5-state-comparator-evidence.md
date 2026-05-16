# H5 Phase G - State-Observation Comparator Evidence

Diagnostic-not-selection slice approved by GPT and Jeff to disambiguate
the H5 blocker after the Phase F frame-stack diagnostic closed as a
negative result. This document records what was run, what was measured,
and what the result implies for the next experiment lever. It does not
declare H5 closed; H5 requires the small CNN policy.

---

## Hypothesis under test

H5 Phase F (frame-stack diagnostic) closed with a negative result: the
pooled `trained_cnn` policy under the `(4, 84, 84)` frame-stack contract
scored worse than Phase E on reward/length and equal on collision, and
seed 1 exhibited entropy collapse from iteration 33. The Phase G
trigger fired no clear next lever among the four enumerated candidates
(`ent_coef`, more timesteps, reward shaping, different perception axis).

GPT proposed a fifth lever first: remove pixel perception from the
training loop entirely and re-run the recipe against a state-mode
observation (`obs_shape=(10,)`, `MlpPolicy`) so any outcome delta is
attributable to the observation contract rather than the recipe or the
CNN encoder. The diagnostic distinguishes three failure modes:

| State PPO outcome                                 | Implication                          |
| ------------------------------------------------- | ------------------------------------ |
| Clears reward/length, improves collision          | Pixel/perception is the H5 blocker   |
| Improves reward/length but not collision          | Reward/profile rewards survival, not avoidance |
| Collapses or fails to beat stay_only              | PPO/reward/local-optimum is the H5 blocker; pixel tuning is not the next lever |
| Mechanical failure (no run dir, eval errors, etc) | Plumbing slice required              |

---

## Recipe

Config: `configs/rl/signal_dodge_ppo_h5_state_comparator.yaml`
(committed at `e80614f`, config_hash differs per-run due to the
`run.seed` field being baked into the hash).

Hyperparameters inherited verbatim from the Phase D/F entropy recipe:

```
algo.policy = MlpPolicy
algo.device = cpu
hyperparams:
  n_steps: 256
  batch_size: 64
  n_epochs: 4
  ent_coef: 0.01
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
env.observation_mode = state
env.headless = false
train.total_timesteps = 10000
eval.eval_freq = 2048
eval.n_eval_episodes = 1
eval.deterministic = true
checkpoint.enabled = true
```

State observation contents verified by inspection of
`games/signal-dodge/scripts/main.gd::_h3_build_observation()`:

- `obs[0]` player x normalized to `[-1, 1]`
- `obs[1]` last move x action in `[-1, 1]`
- `obs[2..4]` hazard 0 (x_offset, y_offset, present_flag)
- `obs[5..7]` hazard 1 (x_offset, y_offset, present_flag)
- `obs[8..9]` hazard 2 (x_offset, y_offset); no present flag (10-dim
  ceiling)

Soundness narrowing: the 10-dim state encodes positions only, no
velocities. State PPO therefore tests "can position-only state PPO
learn avoidance," not "can velocity-rich state PPO learn avoidance."
This is acceptable for the diagnostic but constrains how the result is
read: a state failure cannot falsify a "missing velocity" hypothesis at
the pixel layer because state also lacks velocity. The diagnostic is
sharper as evidence for the PPO/reward/local-optimum hypothesis than
for the perception hypothesis.

---

## Smoke verification (seed 0, 512 timesteps)

`run_id`: `h5_train_state_comparator_smoke_seed0_512`

- `run_start.env_smoke.obs_shape = [10]`, `action_n = 3`
- Loaded `model.zip` observation space `Box(-1.0, 1.0, (10,), float32)`
- Loaded `model.zip` action space `Discrete(3)`
- Loaded `model.zip` policy class `ActorCriticPolicy`
- `summary.json status = ok`
- `config_hash 32d926cea81eb30b8685aeb4ca430c8921d1c34e89bd2dbf5b1345bdb112d90e`

---

## Train sweep (10k timesteps, seeds 1, 2, 3)

All three runs `status=ok`, `git_commit=5f4aa3e`.

| run_id                                       | seed | wallclock |
| -------------------------------------------- | ---- | --------- |
| `h5_train_state_comparator_seed1_10k`        | 1    | ~209s     |
| `h5_train_state_comparator_seed2_10k`        | 2    | ~194s     |
| `h5_train_state_comparator_seed3_10k`        | 3    | ~250s     |

Entropy trajectory across all three seeds: stable in the
`-1.07` to `-1.10` band across all 40 iterations. No collapse anywhere.
Sharp contrast to Phase F seed 1, which collapsed from iteration 33
onward to `entropy_loss <= -0.5`. This confirms that the H5 entropy
recipe is stable under state observations and that Phase F's collapse
was not a generic recipe failure mode.

---

## Eval (full mode, seeds 1000-1009, deterministic argmax)

Eval CLI: `sight_agent.rl.h5_baseline_cli --mode full --policies trained_cnn`.

Naming mismatch noted: the eval branch label `trained_cnn` is recorded
for an `MlpPolicy` run. The CLI loads `model.zip` through SB3 regardless
of the label string. The branch_metadata field in each summary records
`"ppo_cnnpolicy_loaded_from_disk"`; this is misleading on its face but
mechanically correct (SB3 dispatches to the policy class stored in the
archive). Do not interpret `trained_cnn` here as semantic evidence of
a CNN; it is mechanical evidence of a loaded SB3 PPO model. No code
alias was added because the existing path worked.

### Per-seed results

| seed | mean reward | mean length | collision rate | timeout rate | full survival episodes |
| ---- | ----------- | ----------- | -------------- | ------------ | ---------------------- |
| 1    | 518.10      | 519.10      | 1.000          | 0.000        | 0 (max length 1203)    |
| 2    | 733.80      | 734.70      | 0.900          | 0.100        | 1 (seed 1008, 1800)    |
| 3    | 688.80      | 689.70      | 0.900          | 0.100        | 1 (seed 1001, 1800)    |

### Pooled (mean of per-seed means)

| Metric     | State pooled | Phase E pixel | Phase F frame-stack | Phase F stay_only | GPT success bar |
| ---------- | ------------ | ------------- | ------------------- | ----------------- | --------------- |
| Reward     | **646.90**   | 764.87        | 712.87              | ~605              | >= 756.25       |
| Length     | **647.83**   | 765.80        | 713.80              | ~606              | >= 757.50       |
| Collision  | **0.933**    | 0.933         | 0.933               | tied at top       | <= 0.80         |
| Timeout    | **0.067**    | 0.067         | 0.067               | low               | n/a             |

State pooled **fails every GPT success-bar criterion**:

- Reward 646.90 < 756.25, and below Phase E 764.87 and Phase F 712.87.
- Length 647.83 < 757.50, same pattern as reward.
- Collision 0.933 > 0.80, identical to all pixel comparators.

The pooled collision rate exactly matches Phase E and Phase F at 0.933
under the per-metric strongest-comparator convention locked this
session. By that convention, state-mode does not improve over the
weakest defensible pixel baseline on collision.

---

## Reading the result

State PPO did NOT collapse to stay_only. Episode lengths span 183 to
1800; two seeds produced one full 1800-step survival each. The policy
is non-degenerate. But its avoidance behavior is no better than any
pixel comparator (collision 0.933 across all four), and on reward and
length it is meaningfully worse than every pixel comparator including
the Phase F frame-stack baseline that GPT classified as a negative
result.

Mapping to the diagnostic table from the slice plan: the closest match
is row 3 ("state policy also collapses to stay") but the match is
imperfect — state PPO did not collapse, it just failed to beat the
position-only ceiling. The strongest implication standing on the
evidence is: 10k timesteps of PPO with the H5 entropy recipe, position-
only state observations, and `+1/step` survival reward does not produce
hazard-avoiding behavior beyond what stay_only achieves on collision and
beyond what any pixel comparator achieves on reward/length.

This evidence points away from "pixel/perception is the blocker" as a
proximate cause. It points toward reward, optimizer dynamics, or
timestep budget as the proximate cause, in some combination. It is not
sufficient on its own to mandate reward shaping (which is the largest
scope change and requires a charter amendment per H5 plan section 7),
but it removes pixel-side tuning from the top of the lever queue.

### What this does NOT prove

- It does not prove velocity-augmented state would also fail.
- It does not prove longer training would not improve outcomes; the
  Phase D 50k single-seed run is the standing data point against
  expanded budget alone being sufficient, but state mode at 50k has
  not been tested.
- It does not prove the eval pipeline is sound for an MlpPolicy run.
  The eval used the `trained_cnn` branch which dispatches through SB3
  load and predicts via `model.predict(obs, deterministic=True)`. This
  is mechanically the same code path the pixel comparators use, so the
  comparison is fair, but no eval-pipeline-for-state-mode-specific
  invariant was independently verified.

---

## Operational notes

- Each train run wrote `runs\rl\signal_dodge_ppo_h5_state_comparator\h5_train_state_comparator_seed{N}_10k\` containing `events.ndjson`, `summary.json`, `config_effective.yaml`, `model.zip`, `godot-train\python.ndjson`, `godot-train\godot.ndjson`, `godot-train\godot-stdout.log`, `godot-train\godot-stderr.log`. Same for the eval runs under `h5_eval_state_comparator_seed{N}_10k_trained_only\evaluation\trained_cnn\`. None of these are tracked; `.gitignore:67 runs/` covers them.
- Operational learning: `start "" /B cmd /c <bat>` failed silently on this host once during the seed-1 launch attempt (Python alive with 0 stdout for 3+ minutes, no run dir created). The reliable pattern was inline `interact_with_process` invocation against a single persistent `cmd.exe` shell, accepting the MCP 4-minute false-timeout error, and recovering completion via subsequent `read_process_output`. The persistent shell PID was lost once between turns (PID 24364 vanished, presumably due to MCP server restart), so re-spawning shells and re-exporting `SIGHT_GODOT_EXE` / `PYTHONUNBUFFERED=1` on resume is a real consideration for sessions that span >30 minutes of wallclock.
- `branch_metadata="ppo_cnnpolicy_loaded_from_disk"` in the eval summaries is misleading but harmless for this diagnostic. If a state-mode comparator becomes a regular part of the experiment matrix, the eval CLI should grow a label that does not bake in `cnn` assumptions.

---

## Verdict

State-observation PPO under the H5 entropy recipe at 10k timesteps with
position-only state does not clear the GPT success bar. It does not
collapse. It is roughly mid-pack between stay_only and Phase E pixel
trained_cnn on reward/length, and indistinguishable from all pixel
comparators on collision. The diagnostic outcome removes
"pixel/perception is the H5 blocker" from the top of the lever queue.
The next experiment lever is GPT's call; the strongest signal in the
evidence is toward reward dynamics or optimization, not toward
encoder choice or perception channel.

Phase G remains NOT triggered. H5 remains open.
