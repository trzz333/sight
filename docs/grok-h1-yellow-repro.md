# Sight - H1 Fresh-Checkout Repro Evidence

Closes the YELLOW caveat from Grok's H1 phase-gate review. No source code changed; only this doc and the handoff are touched in the docs commit.

---

## 1. Grok YELLOW request

Verbatim from Grok:

> If YELLOW: the single exact command or check Claude should run: on a fresh clone of c958def run `python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml` and confirm summary.json has status=ok plus final eval mean_reward >=475.0
>
> What evidence would change the verdict: successful completion of the above command producing status=ok and mean_reward >=475.0

## 2. Fresh clone

- Origin: `https://github.com/trzz333/sight.git`
- Clone path: `C:\Users\maste\AppData\Local\Temp\sight-h1-fresh-repro-c958def`
- Checked-out commit: `c958def08f8234e51d92b8935a2476f7d9fb1c8c` (short `c958def`)
- `git status --short` after checkout and after the training run: empty (`runs/` is gitignored)

## 3. Exact command

```
python -m sight_agent.rl.train --config configs/rl/cartpole_ppo_h1.yaml
```

`PYTHONPATH` was set to `<scratch>\src` for the duration of the invocation so module resolution used the fresh-clone source tree, not any global editable install. No other environment changes.

Wall-clock elapsed: 46.25s. Process returncode: 0.

## 4. Run artifact paths

- summary: `C:\Users\maste\AppData\Local\Temp\sight-h1-fresh-repro-c958def\runs\rl\cartpole_ppo_h1\20260429T222543Z_cartpole_ppo_h1_seed0_c958def\summary.json`
- events: `C:\Users\maste\AppData\Local\Temp\sight-h1-fresh-repro-c958def\runs\rl\cartpole_ppo_h1\20260429T222543Z_cartpole_ppo_h1_seed0_c958def\events.ndjson`
- run_id: `20260429T222543Z_cartpole_ppo_h1_seed0_c958def`
- artifact `git_commit`: `c958def` (matches checked-out HEAD; the original packet artifact captured `1b4c741` because that was HEAD at original launch)

## 5. summary.json key fields

- `status`: `ok`
- `git_commit`: `c958def`
- `phase`: `H1`
- `env_id`: `CartPole-v1`
- `algo`: `PPO`
- `framework`: `stable-baselines3`
- `seed`: `0`
- `total_timesteps`: `25000`

## 6. Eval trajectory from events.ndjson

```
step 5000   mean=468.40   std=35.95
step 10000  mean=457.20   std=85.60
step 15000  mean=228.20   std=33.26
step 20000  mean=453.40   std=57.53
step 25000  mean=500.00   std=0.00
```

Final eval (step 25000): `mean_reward=500.0`, `std_reward=0.0`, `episode_rewards=[500.0, 500.0, 500.0, 500.0, 500.0]`.

NDJSON event counts: `run_start=1, train_metrics=13, eval=5, run_end=1` (20 total lines, no malformed). Schema and event types match the original packet artifact.

This trajectory matches the original packet artifact (`20260429T205656Z_..._1b4c741`) at every checkpoint, including the transient dip at step 15000. Deterministic-seed handling reproduces the same training curve on a clean clone.

## 7. Pass/fail

- `summary.status == "ok"`: True
- `final_eval.mean_reward (500.0) >= 475.0`: True
- Result: PASS

## 8. Source-code change note

No source code was modified. The repro was run on a fresh clone of `c958def` with no patches. The only files touched in the working repo (`C:\Projects\Sight`) for this gate are:

- `docs/grok-h1-yellow-repro.md` (this file, new)
- `docs/sight-handoff.md` (status update)

Both are docs. Substantive H1 code (commit `2a56e43`) and the H1 packet (`ce060a0`) are untouched.
