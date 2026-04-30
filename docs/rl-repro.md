# Sight - RL Repro Guide

Reproducibility posture for H1 and H2 local runs. Same dependency and
hardware class is expected to produce equivalent trajectories under
deterministic seeding. Different platform or different package versions
are explicitly out of scope for bit-for-bit claims.

---

## 1. Environment

Older gaming-laptop class, 64 GB RAM, CPU-only PPO. Windows 10/11.

Python 3.14 (developed on `C:\Users\maste\AppData\Local\Python\bin\python.exe`
3.14.4).

Editable install with the H2 CPU constraints lockfile:

```
pip install -e ".[dev]" -c constraints/rl-cpu.txt
```

See `constraints/rl-cpu.txt` for the pinned set
(stable-baselines3, gymnasium, torch, numpy, pyyaml, pytest, cloudpickle,
typing_extensions, farama-notifications).

No TensorBoard, no W&B, no MLflow, no Comet, no cloud logging, no
network services. Local NDJSON only.

## 2. H1 train command

```
python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml
```

Writes, under `runs/rl/cartpole_ppo_h1/<run_id>/`:

- `events.ndjson`
- `summary.json` with `schema_version=2`, `kind=train`, `status=ok`,
  `git_commit`, and (retained for H1 backward compat) the
  `events_ndjson` field
- `config_effective.yaml`

H1 success bar is final-eval `mean_reward >= 475.0`. The fresh-clone
repro of original H1 commit `c958def` is recorded in
`docs/grok-h1-yellow-repro.md` (final mean_reward 500.0, deterministic
trajectory match).

## 3. H2 train command

```
python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h2.yaml
```

Same artifact layout as H1, plus a model checkpoint when
`checkpoint.enabled=true` in the config:

- `runs/rl/cartpole_ppo_h2/<run_id>/model.zip`

`summary.json` adds `config_path`, `config_hash`, and
`artifact_paths.{events,summary,config_effective,model}`. NDJSON event
schema matches H1 (`run_start`, `train_metrics`, `eval`, `run_end`).
`config_hash` is recorded in both `run_start` and `summary.json`.

## 4. H2 out-of-band eval command

```
python -m sight_agent.rl.evaluate --run <train_run_dir> --n-eval-episodes 5 --seed 0
```

Writes a separate eval artifact set under the train run:

- `<train_run_dir>/evals/<eval_id>/events.ndjson`
- `<train_run_dir>/evals/<eval_id>/summary.json` with
  `schema_version=2` and `kind=eval`

NDJSON event types are `eval_start`, `eval_episode` (one per episode),
and `eval_end`. The source train summary is embedded in the eval
summary as `source_train_summary`.

Optional flags: `--deterministic true|false` (default true),
`--eval-id <override>`.

## 5. Determinism posture

- `_set_global_seeds(seed)` sets Python `random`, NumPy, Torch, and
  CUDA (when present) seeds.
- SB3 PPO is constructed with `seed=seed`.
- Vec envs are constructed with the same seed via `make_env`.
- `git_commit` and `config_hash` are recorded in every run summary so
  downstream comparisons can verify identical source and config.

This is a posture, not a bit-for-bit guarantee. SB3 and torch can
introduce non-determinism via threading and certain ops. Same
dependency and hardware class is expected to reproduce identical
training curves at checkpoint resolution. Sub-step variance can occur.

## 6. Developer test gate

```
python -m pytest tests/rl -v --tb=short
```

This is the smoke gate run before any commit touching `src/sight_agent/rl/`.
It includes the H1 NDJSON contract tests
(`test_cartpole_smoke.py` invokes a short training run end-to-end), the
H2 artifact, factory, and evaluator smoke tests, and the NDJSON-logger
and config-loader unit tests.
