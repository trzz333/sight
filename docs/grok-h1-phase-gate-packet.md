# Sight - H1 Phase Gate Packet for Grok

Goal: independent sanity check before H1 is marked GREEN and GPT issues H2.

---

## 1. Repo state

```
HEAD          c958def chore: refresh handoff hash to 2a56e43
origin/main   c958def (fully pushed, working tree clean)
substantive   2a56e43 feat(rl): H1 PPO CartPole-v1 baseline with NDJSON logging
git status    (empty)
```

Recent log:

```
c958def chore: refresh handoff hash to 2a56e43
2a56e43 feat(rl): H1 PPO CartPole-v1 baseline with NDJSON logging
1b4c741 chore: refresh handoff hash to 3094655
3094655 handoff: post-merge state, hobby track active, ready for H1
d363eb0 merge: pivot Sight to hobby/research RL game-agent lab
```

`git ls-remote --heads origin main` matches local HEAD: `c958def08f8234e51d92b8935a2476f7d9fb1c8c`.

---

## 2. Test re-run (fresh, current HEAD)

Command:

```
python -m pytest tests/rl -v --tb=short
```

Result: **18 passed in 8.50s**. The smoke tests inside `tests/rl/test_cartpole_smoke.py` invoke a short training run and validate the NDJSON contract end-to-end, so this command is also the H1 repro/smoke gate.

Tests collected:

- `test_cartpole_smoke.py`: `test_h1_smoke_writes_required_events`, `test_h1_smoke_run_start_has_versions_and_effective_hparams`, `test_h1_smoke_run_end_has_artifact_paths` (all PASS)
- `test_ndjson_logger.py`: 7 tests covering one-object-per-line, common-field auto-fill, None step, numpy scalar coercion, nested NaN handling, fallback-to-str, context manager close (all PASS)
- `test_rl_config.py`: 8 tests covering YAML default load, missing-top-key rejection, non-NDJSON rejection, zero-timesteps rejection, non-dict hyperparams rejection, CLI override merge, None-skip, immutability (all PASS)

No long training run was triggered for this packet. The committed run artifact is sufficient for review.

---

## 3. Substantive commit `2a56e43` diff summary

```
commit 2a56e43f63093d1e0269b4e408cca34cfaae6a75
Author: trzz333
Date:   Wed Apr 29 16:16:29 2026 -0500

    feat(rl): H1 PPO CartPole-v1 baseline with NDJSON logging

 configs/rl/cartpole_ppo_h1.yaml     |  27 +
 pyproject.toml                      |   4 +
 src/sight_agent/rl/__init__.py      |   5 +
 src/sight_agent/rl/callbacks.py     | 170 +
 src/sight_agent/rl/config.py        |  90 +
 src/sight_agent/rl/envs.py          |  34 +
 src/sight_agent/rl/ndjson_logger.py | 129 +
 src/sight_agent/rl/train.py         | 241 +
 tests/rl/__init__.py                |   0
 tests/rl/test_cartpole_smoke.py     | 124 +
 tests/rl/test_ndjson_logger.py      | 138 +
 tests/rl/test_rl_config.py          |  99 +
 12 files changed, 1061 insertions(+)
```

Files touched:

- `configs/rl/cartpole_ppo_h1.yaml` (new)
- `pyproject.toml` (+4 lines: stable-baselines3>=2.0, gymnasium>=0.29, pyyaml>=6.0 deps)
- `src/sight_agent/rl/__init__.py`, `callbacks.py`, `config.py`, `envs.py`, `ndjson_logger.py`, `train.py` (all new)
- `tests/rl/__init__.py`, `test_cartpole_smoke.py`, `test_ndjson_logger.py`, `test_rl_config.py` (all new)

---

## 4. H1 implementation summary

- **Framework**: Stable-Baselines3 (selected over CleanRL).
- **Environment**: `CartPole-v1` via `gymnasium.make`. Single env, no vectorization.
- **Algorithm**: PPO (`MlpPolicy`), CPU device.
- **Seed handling**: seed 0 from config, applied to env + PPO + global RNG. Eval uses `deterministic=True`. `effective_hyperparams.seed=0` recorded in `run_start`.
- **Config path**: `configs/rl/cartpole_ppo_h1.yaml`. Validated by `sight_agent.rl.config` with explicit rejection of non-NDJSON logging, zero timesteps, non-dict hyperparams, missing top keys.
- **Entry point**: `python -m sight_agent.rl.train --config <path>`. CLI overrides supported via `apply_cli_overrides`.
- **Output path**: `runs/rl/<run_name>/<UTC_timestamp>_<run_name>_seed<N>_<git_short>/{events.ndjson,summary.json}`. `runs/` is gitignored at `.gitignore:67`.
- **NDJSON schema**: `schema_version=1`. Common fields auto-injected per line: `schema_version, run_id, phase, env_id, algo, framework, seed, git_commit, event, ts_utc, step`. Per-event fields layered on top.
- **Event types**: `run_start`, `train_metrics`, `eval`, `run_end`.
- **No external services**: no TensorBoard, no W&B, no MLflow, no Comet. Verified via `findstr` against `src/sight_agent/rl/*.py`, configs, and `pyproject.toml`. Only matches are explicit comments stating these are not used.

---

## 5. Run artifact

Run dir: `runs/rl/cartpole_ppo_h1/20260429T205656Z_cartpole_ppo_h1_seed0_1b4c741/`

### `summary.json` (verbatim)

```json
{
  "schema_version": 1,
  "run_id": "20260429T205656Z_cartpole_ppo_h1_seed0_1b4c741",
  "phase": "H1",
  "env_id": "CartPole-v1",
  "algo": "PPO",
  "framework": "stable-baselines3",
  "seed": 0,
  "total_timesteps": 25000,
  "eval_freq": 5000,
  "n_eval_episodes": 5,
  "deterministic_eval": true,
  "git_commit": "1b4c741",
  "versions": {
    "python": "3.14.4",
    "gymnasium": "1.2.3",
    "stable_baselines3": "2.8.0",
    "torch": "2.11.0+cpu"
  },
  "effective_hyperparams": {
    "learning_rate": 0.0003, "n_steps": 2048, "batch_size": 64,
    "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.2, "clip_range_vf": null, "normalize_advantage": true,
    "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "target_kl": null, "use_sde": false, "sde_sample_freq": -1,
    "seed": 0, "policy_class": "ActorCriticPolicy", "device": "cpu"
  },
  "status": "ok"
}
```

Note: `git_commit` is `1b4c741` because that was HEAD when training launched, prior to the substantive commit `2a56e43` that introduced the H1 code. The artifact pins the launch revision; the source of the code itself is `2a56e43`.

---

## 6. NDJSON validation

Programmatic line-by-line check (Python `json.loads` per line):

```
file size                13559 bytes
raw line count           20
parsed objects           20
malformed lines          0
event_counts             {'run_start': 1, 'train_metrics': 13, 'eval': 5, 'run_end': 1}
run_start present        True
run_end present          True
eval events              5  (steps 5000, 10000, 15000, 20000, 25000)
train_metrics events     13 (steps 2048, 4096, ..., 26624 at PPO rollout boundaries)
run_start has config     True
run_start has versions   True
run_start has hparams    True
seed                     0
git_commit               1b4c741
```

First 3 events (event, step):

- `(run_start, 0)` -- keys: `algo, config, config_path, effective_hyperparams, env_id, env_smoke, event, framework, git_commit, phase, provenance_note, run_id, schema_version, seed, step, ts_utc, versions`
- `(train_metrics, 2048)` -- metrics keys: `rollout/ep_len_mean, rollout/ep_rew_mean, time/fps, time/iterations, time/time_elapsed, time/total_timesteps`
- `(train_metrics, 4096)` -- additional metrics keys appear from iter 2: `train/approx_kl, train/clip_fraction, train/clip_range, train/entropy_loss, train/explained_variance, train/learning_rate, train/loss, train/n_updates, train/policy_gradient_loss, train/value_loss`

Last 3 events (event, step):

- `(eval, 25000)` -- `mean_reward=500.0, std=0.0, episode_rewards=[500.0, 500.0, 500.0, 500.0, 500.0]`
- `(train_metrics, 26624)` -- `rollout/ep_rew_mean=204.78, fps=700`
- `(run_end, 26624)` -- `status=ok, elapsed_seconds=39.256, total_timesteps=26624, artifact_paths={events_ndjson,summary_json}`

Train ep_rew_mean trajectory (rolling 100-episode mean from PPO logger):

```
step 2048   25.66
step 4096   29.94
step 6144   38.08
step 8192   48.07
step 10240  64.03
step 12288  79.84
step 14336  96.76
step 16384  115.80
step 18432  131.67
step 20480  148.59
step 22528  168.04
step 24576  187.05
step 26624  204.78
```

Eval (deterministic policy) trajectory:

```
step 5000   mean=468.4   std=35.95   episodes=[500.0, 500.0, 475.0, 402.0, 465.0]
step 10000  mean=457.2   std=85.60   episodes=[500.0, 500.0, 500.0, 500.0, 286.0]
step 15000  mean=228.2   std=33.26   episodes=[202.0, 218.0, 230.0, 200.0, 291.0]
step 20000  mean=453.4   std=57.53   episodes=[500.0, 500.0, 395.0, 500.0, 372.0]
step 25000  mean=500.0   std=0.00    episodes=[500.0, 500.0, 500.0, 500.0, 500.0]
```

Note the dip at step 15000 (228.2). The deterministic eval crashes mid-training while the stochastic train ep_rew_mean continues climbing through that window. By step 25000 the deterministic policy is fully converged. This is consistent with PPO's typical learning behavior on CartPole-v1 and is not a bug, but Grok should eyeball it.

---

## 7. Caveats

Honest about what was and was not verified:

- **Fresh-checkout reproduction from config alone: NOT verified this session.** The pytest smoke tests do invoke training and validate NDJSON, but a clean clone (or `git clean -xfd` followed by editable reinstall and `python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml`) was not exercised end-to-end at HEAD `c958def`. The committed run was authored under HEAD `1b4c741`. Recommend Grok or GPT explicitly request a clean-slate repro before H1 GREEN.
- **Trained policy was NOT re-evaluated independently after restart.** No model weights are persisted (no `.zip` or `.pt` checkpoint is written by the current `train.py`). The only evidence of policy quality is the in-run eval callback. There is no out-of-band eval pass.
- **Version pinning is lower-bound only.** `pyproject.toml` declares `stable-baselines3>=2.0`, `gymnasium>=0.29`, `pyyaml>=6.0`. Locally installed versions (captured in `summary.json`) are sb3 2.8.0, gymnasium 1.2.3, torch 2.11.0+cpu, python 3.14.4. There is no lock file. A different machine with different upper-bound resolutions could behave differently. H2 should add a lock file or pinned constraints.
- **`git_commit` field in artifact is `1b4c741`, not `2a56e43`.** This is the launch-time HEAD captured by the run, not the source commit of the H1 code. Both are pre-handoff (`c958def`). The full provenance chain is documented but slightly indirect.
- **18/18 rl tests pass in 8.50s on this machine.** Other test suites in the repo were not re-run. There is no top-level `pytest` gate exercised here.
- **Hardware**: single older gaming laptop, CPU-only training. No GPU was used. Charter explicitly approves CPU-only for H1.
- **Ethics scope**: `CartPole-v1` is a Gymnasium classic-control toy environment, fully on the approved target ladder. No commercial game, no online service, no automation of a third-party platform.

---

## 8. Recommended verdict (for Grok to evaluate, not Claude's final decision)

**Recommended: GREEN with one yellow flag.**

Strongest case for GREEN:

- H1 charter criterion is `local RL baseline on Gymnasium CartPole using SB3 or CleanRL with NDJSON training-metric logging`. All four parts satisfied: local, RL baseline, Gymnasium CartPole-v1 with SB3, NDJSON logging present and validated.
- Final eval at 500.0/500.0 is the maximum reward on CartPole-v1; the policy is genuinely solving the env, not pattern-matching luck.
- Tests pass (18/18) and the smoke tests gate the NDJSON contract programmatically, not just by spot-check.
- Run artifact is reproducibly structured: schema version, seed, env, algo, hyperparams, versions, and event types all in place. No malformed NDJSON lines.
- No leakage of charter non-goals: no TensorBoard, no W&B, no commercial game, no cloud service, no committed binaries.

Strongest case AGAINST GREEN (yellow flag):

- Fresh-checkout repro from config has not been demonstrated. The committed run is one execution, on one machine, at one launch-time HEAD. H1's success criterion implicitly leans on reproducibility; H2 explicitly tests reproducibility (`reproducible from a config file`). If H1 GREEN is intended to mean repro-capable, that has not been independently shown.
- The eval dip at step 15000 (228.2) is benign on CartPole but worth one-line acknowledgement so it is not a surprise later.

Strongest case for RED: none identified.

Suggested Grok action: GREEN if a single fresh-checkout `python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml` run on a clean clone of `c958def` produces a `summary.json` with `status==ok` and final eval `mean_reward >= 475.0`. YELLOW if Grok wants the repro evidence before signing off. RED only if the NDJSON schema is incompatible with the H2 plan, or a policy/test defect is found.

---

## 9. Bundle for Grok

All under `c958def` on `origin/main`:

- This packet: `docs/grok-h1-phase-gate-packet.md`
- Charter: `docs/sight-charter.md`
- Handoff: `docs/sight-handoff.md`
- Substantive diff: `git show 2a56e43` (12 files, 1061 insertions)
- Run artifact: `runs/rl/cartpole_ppo_h1/20260429T205656Z_cartpole_ppo_h1_seed0_1b4c741/{summary.json,events.ndjson}` (gitignored locally; bundle separately if Grok needs them)
