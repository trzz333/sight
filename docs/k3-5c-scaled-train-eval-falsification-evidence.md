# K3.5c Scaled Train + Eval Falsification Evidence

**Date:** 2026-05-18
**Phase:** K (K3.5c direct eval falsification)
**Commit:** train-side commits 02bb809 (production reward-scaling train patch) plus this evidence commit
**Classification:** **Partial break** per GPT K3.5c contract row 2

---

## Verdict

K3.5c scaled training produced **eval results that DIFFER from canonical
Phase D** at the per-seed level, AND **two K3.5c checkpoints with
distinct model weights (2048 ts and 10000 ts) produce bit-identical
per-seed eval results to each other**.

This maps to the GPT K3.5c classification matrix row 2:

> Scaled eval differs from old Phase D, but 2048 and 10k scaled are
> identical to each other -> Partial break. K3.5 changed the argmax
> surface, but progression still not eval-visible. Scope K4 narrower.

The K3.5c reward-scaling intervention is therefore the **first K-series
intervention to move the deterministic-argmax eval surface** off the
Phase B/C/D fixed point. But the new surface that K3.5c lands on is
itself a fixed point: 8x additional training (2048 -> 10000 ts) does
not shift it, despite the two checkpoints having distinct SHA-256
hashes.

K4 (the deterministic-argmax wedge mechanism investigation) remains
warranted, narrowed to: why is the deterministic-argmax eval surface
effectively frozen on the panel within a single training regime,
even with reward scaling preventing value collapse?

---

## What was run

### Production reward-scaling train patch (commit 02bb809)

- `src/sight_agent/rl/reward_scaling.py`: `FixedRewardScaleVecEnv` and
  `maybe_wrap_train_env` ported from the K3.5-validated probe
  (`tools/h5_training_entropy_probe.py` lines 564-595).
- `src/sight_agent/rl/train.py`: new `--reward-scale-divisor FLOAT`
  CLI flag; train VecEnv wrapped only when divisor > 0 and != 1.0;
  `reward_scale_divisor` and `reward_scale_applied` recorded in
  `run_start` event and `summary.json`.
- `src/sight_agent/rl/config.py::apply_cli_overrides`: threads
  `reward_scale_divisor` into `cfg["train"]`.
- `tests/rl/test_reward_scaling.py`: 9 unit tests, all pass.
- Broader-test sweep across `test_rl_config`, `test_cartpole_smoke`,
  `test_h2_artifacts`, `test_h2_evaluate_smoke`, `test_h2_factories`,
  `test_h3_train_plumbing`, and `test_reward_scaling`: 67/67 pass
  in 11.37 s.

### Scaled training slices

Both slices used `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`,
seed 0, divisor 30, no value-bias init, otherwise the default H5
entropy recipe (`n_steps=256, batch_size=64, n_epochs=4,
ent_coef=0.01, learning_rate=0.0003`).

| Slice | Total ts | Run dir | Wall time | fps | Final n_updates |
|-------|----------|---------|-----------|-----|-----------------|
| K3.5c 2048  | 2048  | `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_2048/`  | 79 s   | 25-32 | 28  |
| K3.5c 10000 | 10000 | `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_10000/` | 409 s  | 24-32 | 156 |

Both `summary.json` entries record `reward_scale_divisor: 30.0`,
`reward_scale_applied: true`, `status: ok`.

### Trained-only H5 external eval

Both evals used `--config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`,
`--seeds 1000-1009`, `--mode full`, `--policies trained_cnn`. The eval
VecEnv is NOT wrapped in `FixedRewardScaleVecEnv`; eval reward stream
is the raw env reward (1.0 per surviving step), comparable to the
canonical Phase D baseline.

Eval artifacts:

- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_eval_scaled_div30_seed0_2048_trained_only/evaluation/trained_cnn/summary.json`
- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_eval_scaled_div30_seed0_10000_trained_only/evaluation/trained_cnn/summary.json`

Both eval CLI exit lines reported `passed=True saturated_negative_controls=[]`.

---

## Per-seed comparison table

Episode lengths and terminations, seeds 1000-1009. Phase D values are
the canonical bit-identical baseline carried in the handoff. K3.5c
columns are reproduced verbatim from the eval summary JSONs.

| Seed | Phase D len | Phase D term | K3.5c 2048 len | K3.5c 2048 term | K3.5c 10000 len | K3.5c 10000 term | 2048 == 10000 | 2048 == D | 10000 == D |
|------|-------------|--------------|----------------|-----------------|------------------|------------------|---------------|-----------|------------|
| 1000 |  903 | collision | 1383 | collision | 1383 | collision | YES | NO | NO |
| 1001 | 1800 | timeout   |  483 | collision |  483 | collision | YES | NO | NO |
| 1002 |  273 | collision | 1293 | collision | 1293 | collision | YES | NO | NO |
| 1003 |  363 | collision |  603 | collision |  603 | collision | YES | NO | NO |
| 1004 |  753 | collision | 1443 | collision | 1443 | collision | YES | NO | NO |
| 1005 | 1203 | collision |  363 | collision |  363 | collision | YES | NO | NO |
| 1006 |  183 | collision |  573 | collision |  573 | collision | YES | NO | NO |
| 1007 |  693 | collision |  273 | collision |  273 | collision | YES | NO | NO |
| 1008 |  423 | collision | 1800 | timeout   | 1800 | timeout   | YES | NO | NO |
| 1009 |  303 | collision |  243 | collision |  243 | collision | YES | NO | NO |

**Pairwise equality verdict:**

- K3.5c 2048 vs K3.5c 10000: bit-identical per-seed (10/10 length match,
  10/10 termination match).
- K3.5c 2048 vs Phase D: 0/10 length match. 9/10 termination match by
  rate, but the timeout slot has shifted from seed 1001 (Phase D) to
  seed 1008 (K3.5c) so per-seed termination match is 8/10.
- K3.5c 10000 vs Phase D: same as 2048 vs D (since 2048 == 10000).

### Aggregate comparison

| Config        | mean_reward | mean_length | collision_rate | timeout_rate |
|---------------|-------------|-------------|----------------|--------------|
| Phase D       | 688.8       | 689.7       | 0.9            | 0.1          |
| K3.5c 2048    | 844.8       | 845.7       | 0.9            | 0.1          |
| K3.5c 10000   | 844.8       | 845.7       | 0.9            | 0.1          |

Collision and timeout rates coincide at 0.9 / 0.1 across all three
configs purely because each happened to land on exactly one timeout
out of ten seeds. The timeout seed differs (1001 vs 1008), so the
rate coincidence is not evidence of behavioral coincidence.

K3.5c training shifts mean episode length from ~690 to ~846 steps
(+22 percent) but every K3.5c eval episode lands on the same trained
argmax sequence within a single training regime.

### Model artifact integrity

| Checkpoint            | model.zip SHA-256                                                  |
|-----------------------|--------------------------------------------------------------------|
| K3.5c 2048            | `E14D1A12ADD18B8FC6560A525B82B18BE65D538F5E89519DFC780C2ACA0900DB` |
| K3.5c 10000           | `5664A12E100B2FBE722E0A8FF0C7C81F140F11A73943FC5B46CC5CBFB0D0B6D8` |

Distinct model weights, identical per-seed eval outcomes. Rules out the
trivial confound where training silently no-ops past the 2048 mark.

---

## Mechanism status update

K3.5 + K3.5b previously established:

- Absolute reward magnitude is load-bearing under Adam at the value head.
- Fixed reward scaling at divisor 30 keeps `latent_vf_live_post = 128/128`
  across the 10k probe.
- The deterministic-argmax panel wedge is present at update 1 of a
  freshly-initialized CnnPolicy and survives every K3.x intervention.

K3.5c adds:

- K3.5 reward scaling at the production trainer **does** move the
  eval-time argmax surface off the Phase B/C/D fixed point. The original
  Phase K eval anomaly (Phase B/C/D bit-identical) is therefore not
  caused by something downstream of the value head being immune to
  training; reward scaling alone is sufficient to land on a different
  surface.
- However the new surface is itself a fixed point within the K3.5
  training regime. 8x more training produces zero per-seed eval change
  despite distinct model checkpoints.

This decomposes the anomaly into two layers:

1. **Cross-regime invariance** (Phase B/C/D bit-identical even across
   entropy and budget changes): partially broken by K3.5. Reward scaling
   is one viable lever.
2. **Within-regime invariance** (K3.5c 2048 == K3.5c 10000): still
   present. The deterministic-argmax wedge persists from update 1
   onward inside any given training recipe, even when value-shock
   collapse is prevented.

Layer 2 is the K4 target, narrower than before. Candidate mechanisms
remain: action-head initialization, CnnPolicy default weight scaling,
log-softmax tie-breaking at near-uniform initialization, and
features-extractor output uniformity producing identical logits
across the panel observation set. The K4 diagnostic slice should
inspect raw action-net logits on the panel at update 0 of a fresh
K3.5-scaled training, then again at u8 (K3.5c 2048 boundary) and
u156 (K3.5c 10000 boundary), and check whether the panel argmax
distribution actually changes or just stays pinned to the same
action.

---

## Pipeline integrity

- Eval pipeline loads the model from disk: `branch_metadata` is
  `ppo_cnnpolicy_loaded_from_disk` on both K3.5c eval summaries.
- Eval pipeline does NOT scale rewards: K3.5c per-seed `reward` values
  equal `episode_length - 1` (one less because the final terminating
  step on collision does not contribute a survival reward), the same
  raw 1.0-per-step structure Phase D used.
- The `SIGHT_TCP_IGNORE_DEATH` refusal guard was not exercised in this
  session; no death-tolerant eval branch was needed.
- `git_commit` field on both eval summaries reads `02bb809`, the
  production-patch commit hash. The trained checkpoints were produced
  under the patched trainer.

---

## Files added or changed this session (cumulative)

Code:

- `src/sight_agent/rl/reward_scaling.py` (new, 76 lines)
- `src/sight_agent/rl/train.py` (modified: CLI flag, train-env wrap, run_start
  and summary.json metadata)
- `src/sight_agent/rl/config.py` (modified: `apply_cli_overrides` threads
  `reward_scale_divisor`)
- `tests/rl/test_reward_scaling.py` (new, 9 tests)

Runs:

- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_2048/`
- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_scaled_div30_seed0_10000/`
- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_eval_scaled_div30_seed0_2048_trained_only/`
- `runs/rl/signal_dodge_ppo_h5_pixel_entropy/k3_5c_eval_scaled_div30_seed0_10000_trained_only/`

Docs:

- `docs/k3-5c-scaled-train-eval-falsification-evidence.md` (this file)
- `docs/sight-handoff.md` (refreshed in the chore commit)

---

## Reproducibility

```cmd
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe
set PYTHONPATH=C:\Projects\Sight\src
set PYTHONUNBUFFERED=1

REM Train slice 1
python -u -m sight_agent.rl.train --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml --seed 0 --total-timesteps 2048  --reward-scale-divisor 30 --run-id k3_5c_scaled_div30_seed0_2048

REM Train slice 2
python -u -m sight_agent.rl.train --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml --seed 0 --total-timesteps 10000 --reward-scale-divisor 30 --run-id k3_5c_scaled_div30_seed0_10000

REM Eval slice 1
python -u -m sight_agent.rl.h5_baseline_cli --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml --run-id k3_5c_eval_scaled_div30_seed0_2048_trained_only  --seeds 1000-1009 --mode full --policies trained_cnn --train-run-dir runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_2048

REM Eval slice 2
python -u -m sight_agent.rl.h5_baseline_cli --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml --run-id k3_5c_eval_scaled_div30_seed0_10000_trained_only --seeds 1000-1009 --mode full --policies trained_cnn --train-run-dir runs\rl\signal_dodge_ppo_h5_pixel_entropy\k3_5c_scaled_div30_seed0_10000
```

Background-launch note: `start "" /B cmd /c bat-with-stdout-redirect`
on Windows 11 26200 produced a pathological idle Python child (18 MB
WS, 0.16 CPU s over 3+ minutes) when stdout was redirected to a file
via cmd `>>`. The 10000-ts run in this session was launched via
PowerShell `Start-Process -RedirectStandardOutput -RedirectStandardError
-NoNewWindow` instead, which worked cleanly. Update the bat-sentinel
memory accordingly.
