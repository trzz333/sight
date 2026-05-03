# Sight - H2 Phase Gate Packet (for Grok review)

Phase-gate packet for H2: reusable training and eval harness with
deterministic seeds, NDJSON logs, and reproducibility from a config
file. This packet collects the evidence needed for a Grok GREEN /
YELLOW / RED verdict on H2 closure.

H3 is not started. H1 closed GREEN per `docs/grok-h1-final-green.md`.

---

## 1. Prior phase state

H1 closed GREEN, recorded in `docs/grok-h1-final-green.md`.

- Substantive H1 commit: `2a56e43` feat(rl): H1 PPO CartPole-v1 baseline
  with NDJSON logging.
- H1 fresh-clone repro of `c958def` recorded in
  `docs/grok-h1-yellow-repro.md` (final-eval `mean_reward=500.0`,
  deterministic trajectory match), committed at `b5b4028`.
- H1 packet: `docs/grok-h1-phase-gate-packet.md` at `ce060a0`.

H2 implementation work was authorized at H1 closure. No H1 retro fixes
have landed since.

## 2. H2 scope and non-scope

In scope for H2:

- Reusable training entrypoint `python -m sight_agent.rl.train` driven
  by a YAML config file.
- Reusable evaluation entrypoint `python -m sight_agent.rl.evaluate`
  that loads a saved model checkpoint from a prior train run and writes
  its own eval artifact set.
- Artifact contract: `events.ndjson`, `summary.json` with
  `schema_version=2`, `kind=train|eval`, `git_commit`, `config_hash`,
  `artifact_paths`, `config_effective.yaml`, optional `model.zip` when
  `checkpoint.enabled=true` in the config.
- Deterministic seeding posture: Python `random`, NumPy, Torch, CUDA
  (when present), SB3 PPO seed, vec env seed.
- CPU-only PPO on Gymnasium CartPole-v1 on Windows / older
  gaming-laptop class. Local NDJSON only. No TensorBoard, W&B, MLflow,
  Comet, or network logging anywhere under `src/sight_agent/rl/`.
- Pinned constraints lockfile `constraints/rl-cpu.txt`.
- `docs/rl-repro.md` reproducibility guide.

Out of scope for H2 (deferred to H3+):

- Godot environments. The `make_env` factory rejects any `godot:`
  prefix with a clear error message (`tests/rl/test_h2_factories.py
  ::test_make_env_rejects_godot_prefix_with_clear_message`).
- Pixel observations.
- Non-PPO algorithms.
- Frameworks other than `stable-baselines3`.

Permanent non-goals (charter-pinned, unchanged):

- No offerwalls, no Freecash, no live commercial games, no bot-
  detection evasion, no account farming, no online multiplayer, no
  platforms where automation is prohibited.

## 3. Repo state at packet construction

- Substantive H2 commit: `ebb89b4` feat(rl): add H2 reusable train and
  eval harness.
- HEAD when the canonical runs in this packet were captured:
  `83d944a` chore: refresh handoff hash to ebb89b4. Doc-only chore
  commit on top of `ebb89b4`; no source, config, test, or constraints
  diff (`git diff ebb89b4..83d944a -- ":(exclude)docs"` is empty).
- Current `main` HEAD at the time this packet is committed: `6ff432a`
  chore: handoff update flagging CD crash blocker. Doc-only chore
  commit on top of `83d944a` (`git diff 83d944a..6ff432a --
  ":(exclude)docs"` is empty). Functionally equivalent to `ebb89b4`
  for code, config, and test purposes.
- All commits pushed to `origin/main` at
  `https://github.com/trzz333/sight.git`.

The recorded `git_commit` field inside every artifact in this packet is
`83d944a`, which is the HEAD when the runs executed. Functionally
equivalent to `ebb89b4` for code purposes.

## 4. Test gate

Command:

```
python -m pytest tests/rl -v --tb=short
```

Result on `83d944a`, run 2026-04-30:

- 48 passed, 0 failed, 0 errors, 0 warnings.
- Runtime: 13.36s.
- Test files exercised: `test_cartpole_smoke.py` (3),
  `test_h2_artifacts.py` (12), `test_h2_evaluate_smoke.py` (8),
  `test_h2_factories.py` (9), `test_ndjson_logger.py` (7),
  `test_rl_config.py` (8). One H1 NDJSON contract test inside
  `test_cartpole_smoke.py` invokes a short end-to-end training run.

Telemetry posture: no TensorBoard, W&B, MLflow, Comet, or network
imports anywhere under `src/sight_agent/rl/` (verified at the source
level for the H2 implementation commit).

## 5. Acceptance training run

Path:

```
C:\Projects\Sight\runs\rl\cartpole_ppo_h2\h2_acceptance_seed0\
```

Command:

```
python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h2.yaml
```

Files present:

- `summary.json`
- `events.ndjson`
- `config_effective.yaml`
- `model.zip` (141,863 bytes, SHA256 `3826b8ebeefe1cad2ca26b650a8da6d
  991c04dc352338e10bbaa808fe53f59c2`)

`summary.json` headline fields:

- `schema_version=2`, `kind=train`, `status=ok`
- `run_id=h2_acceptance_seed0`, `phase=H2`, `env_id=CartPole-v1`,
  `algo=PPO`, `framework=stable-baselines3`, `seed=0`
- `total_timesteps=25000`, `eval_freq=5000`, `n_eval_episodes=5`,
  `deterministic_eval=true`
- `git_commit=83d944a`
- `config_path=configs/rl/cartpole_ppo_h2.yaml`
- `config_hash=ebc35edbc747711d7ef701b6dea6ef64c4d7b961a5b75fc0dde20ea
  52b90d193`
- `versions={python=3.14.4, gymnasium=1.2.3, stable_baselines3=2.8.0,
  torch=2.11.0+cpu}`

Final in-run eval at step 25000: `mean_reward=500.0`, `std_reward=0.0`,
all five episode rewards 500.0. CartPole-v1 ceiling.

NDJSON validation: 20 lines total, 0 malformed. Event counts
`{run_start: 1, train_metrics: 13, eval: 5, run_end: 1}`. `run_end`:
`status=ok`, `elapsed_seconds=35.23`, `total_timesteps=26624`.

Per-eval trajectory (mean_reward at each eval step):

- step 5000: 468.4
- step 10000: 457.2
- step 15000: 228.2
- step 20000: 453.4
- step 25000: 500.0

## 6. Out-of-band evaluation against the acceptance run

Command:

```
python -m sight_agent.rl.evaluate --run runs\rl\cartpole_ppo_h2\h2_acceptance_seed0 --n-eval-episodes 5 --seed 0
```

Path:

```
C:\Projects\Sight\runs\rl\cartpole_ppo_h2\h2_acceptance_seed0\evals\eval_20260430T134046Z_seed0_n5_nceseed0\
```

Files present:

- `summary.json`
- `events.ndjson`

`summary.json` headline fields:

- `schema_version=2`, `kind=eval`, `status=ok`
- `phase=H2`, `env_id=CartPole-v1`, `algo=PPO`,
  `framework=stable-baselines3`, `seed=0`, `deterministic=true`,
  `n_eval_episodes=5`
- `git_commit=83d944a`
- `mean_reward=500.0`, `std_reward=0.0`, `episode_rewards=[500.0 x 5]`,
  `episode_lengths=[500 x 5]`
- `model_path=runs\rl\cartpole_ppo_h2\h2_acceptance_seed0\model.zip`
- `source_train_run_id=h2_acceptance_seed0`
- `source_train_summary` embedded (the acceptance summary verbatim)

NDJSON validation: 7 lines total, 0 malformed. Event counts
`{eval_start: 1, eval_episode: 5, eval_end: 1}`.

Confirms the model checkpoint can be loaded standalone and reproduces
the in-run final-eval result.

## 7. Fresh-clone reproducibility

Fresh clone of `https://github.com/trzz333/sight.git` checked out at
`83d944a`:

```
C:\Users\maste\AppData\Local\Temp\sight-h2-fresh-repro-83d944a\
```

`PYTHONPATH` set to the cloned `src/`. Same Python interpreter
(`C:\Users\maste\AppData\Local\Python\bin\python.exe` 3.14.4), same
package versions per `constraints/rl-cpu.txt`.

### 7.1 Fresh train

Command:

```
python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h2.yaml --run-id h2_fresh_repro_seed0
```

Path:

```
C:\Users\maste\AppData\Local\Temp\sight-h2-fresh-repro-83d944a\runs\rl\cartpole_ppo_h2\h2_fresh_repro_seed0\
```

Files present: `summary.json`, `events.ndjson`, `config_effective.yaml`,
`model.zip` (141,863 bytes, SHA256 `d9266649c8ed478493ae9887c9318bf741
dcdbfc87ad300e70d6d109bc84db16`).

`summary.json` headline fields:

- `schema_version=2`, `kind=train`, `status=ok`
- `run_id=h2_fresh_repro_seed0`, `git_commit=83d944a`
- `config_hash=4deb55a5d51198c039ef3b8844d36905d70bec7019a5cefa41541f6
  3294d06d0`
- `total_timesteps=25000`, `seed=0`, deterministic eval

Final in-run eval at step 25000: `mean_reward=500.0`, `std_reward=0.0`.

NDJSON validation: 20 lines, 0 malformed. Event counts identical to
the acceptance run `{run_start: 1, train_metrics: 13, eval: 5,
run_end: 1}`.

Per-eval trajectory at eval-checkpoint resolution (acceptance vs fresh):

- step 5000: 468.4 vs 468.4
- step 10000: 457.2 vs 457.2
- step 15000: 228.2 vs 228.2
- step 20000: 453.4 vs 453.4
- step 25000: 500.0 vs 500.0

Trajectories match at eval-checkpoint resolution. This is the
reproducibility posture stated in `docs/rl-repro.md` section 5.

### 7.2 Fresh out-of-band eval

Command:

```
python -m sight_agent.rl.evaluate --run runs\rl\cartpole_ppo_h2\h2_fresh_repro_seed0 --n-eval-episodes 5 --seed 0
```

Path:

```
C:\Users\maste\AppData\Local\Temp\sight-h2-fresh-repro-83d944a\runs\rl\cartpole_ppo_h2\h2_fresh_repro_seed0\evals\eval_20260430T135322Z_seed0_n5_proseed0\
```

`summary.json` headline fields: `schema_version=2`, `kind=eval`,
`status=ok`, `mean_reward=500.0`, `std_reward=0.0`,
`episode_rewards=[500.0 x 5]`, `episode_lengths=[500 x 5]`,
`model_path=runs\rl\cartpole_ppo_h2\h2_fresh_repro_seed0\model.zip`.

NDJSON validation: 7 lines, 0 malformed. Event counts
`{eval_start: 1, eval_episode: 5, eval_end: 1}`.

## 8. Dependency posture

`constraints/rl-cpu.txt` (committed at `ebb89b4`):

```
stable-baselines3==2.8.0
gymnasium==1.2.3
torch==2.11.0
numpy==2.4.4
pyyaml==6.0.3
pytest==9.0.3
cloudpickle==3.1.2
typing_extensions==4.15.0
farama-notifications==0.0.6
```

Install posture: `pip install -e ".[dev]" -c constraints/rl-cpu.txt`.

The constraints file header references "H2 acceptance and fresh-clone
repro runs". As of this packet those runs have landed and the header
is accurate.

## 9. Caveats and known nuances

1. The recorded `git_commit` in every artifact is `83d944a`. The
   substantive H2 implementation commit is `ebb89b4`. The diff between
   them outside `docs/` is empty, but a strict reviewer should be aware
   the runs were captured at HEAD `83d944a`.
2. `config_hash` differs between the acceptance run
   (`ebc35edbc747...b90d193`) and the fresh-clone run
   (`4deb55a5d511...294d06d0`) despite both using the same source
   YAML. The fresh-clone command applied the CLI override
   `--run-id h2_fresh_repro_seed0`, which feeds into the effective
   config dict that `config_hash` covers. This is a feature of the
   hash, not a defect: it correctly distinguishes effective configs
   that differ in run-identifying fields. Hyperparameters,
   `total_timesteps`, `seed`, `eval_freq`, `n_eval_episodes`, and the
   resulting trajectory all match.
3. Model checkpoint bytes (SHA256) differ between the acceptance and
   fresh-clone runs. The eval-checkpoint trajectory matches step for
   step. This matches the documented posture in `docs/rl-repro.md`
   section 5: same dependency / hardware class is expected to
   reproduce identical training curves at checkpoint resolution; sub-
   step variance and serialization-order differences in `model.zip`
   are not claimed to be bit-identical.
4. `events.ndjson` event objects carry their own `schema_version=1` per
   event. The `schema_version=2` field at the run level lives in
   `summary.json`. This is intentional and matches the H2 design but
   is worth noting for any reviewer parsing both schemas.
5. The H1 backward-compat field `events_ndjson` (a string path)
   remains in `summary.json` alongside the H2 `artifact_paths` dict.
   Intentional, documented, and exercised by H1 regression tests.

## 10. What is NOT in this packet

- No verbatim Grok review text. Grok closure for any phase is recorded
  as a verdict only, per the H1 closure pattern in
  `docs/grok-h1-final-green.md`.
- No H3 work. Godot environment integration, pixel observations, and
  Signal Dodge policy training are all out of scope for H2.
- No bit-for-bit cross-machine reproducibility claim. CPU-only same-
  machine reproducibility was tested and held at eval-checkpoint
  resolution.

## 11. Recommended verdict scope for Grok

Grok is asked to evaluate H2 against the following criteria:

1. Reusable train and eval harness driven by a config file: yes / no.
2. Deterministic seeding posture across Python `random`, NumPy, Torch,
   SB3 PPO, and vec envs: defensible / not.
3. NDJSON contract for both train and eval (event types, line-by-line
   validity, expected counts): clean / not.
4. Artifact set complete (`summary.json`, `events.ndjson`,
   `config_effective.yaml`, `model.zip`, eval set under `evals/`):
   yes / no.
5. Acceptance train hits CartPole-v1 success bar
   (`mean_reward >= 475.0`) and out-of-band eval reproduces it:
   yes / no.
6. Fresh-clone repro of `83d944a` produces matching eval-checkpoint
   trajectories under the same dependency / hardware class:
   yes / no.
7. Dependency posture (`constraints/rl-cpu.txt`,
   `docs/rl-repro.md`): adequate / not for a hobby-track local-first
   reproducibility claim.
8. Test suite: 48/48 passing on `83d944a`: confirmed.

If all eight read clean, the recommended verdict is GREEN and H2 is
closed. H3 (tiny Godot environment exposed as a Gym-style env, state
observations only, no pixels) becomes the next phase.

If Grok flags YELLOW caveats, those will be addressed and a YELLOW
closure doc will be added in the H1 pattern.

## 12. Pointers

- Charter: `docs/sight-charter.md`
- Repro guide: `docs/rl-repro.md`
- H1 GREEN closure: `docs/grok-h1-final-green.md`
- H1 packet: `docs/grok-h1-phase-gate-packet.md`
- H1 YELLOW repro: `docs/grok-h1-yellow-repro.md`
- Handoff: `docs/sight-handoff.md`
- H2 implementation commit: `ebb89b4`
- Recovery / current commit: `83d944a` (handoff-only above `ebb89b4`)
