# ViZDoom PPO teacher: defend_the_center findings

Date: 2026-07-11. Run of record: `runs\vzd\ppo_defend` (summary.json,
model.zip, gameplay.mp4). Trainer: `tools\vzd_ppo_train.py` at 9ad5bc6+.

## Result

PPO CnnPolicy from pixels on `VizdoomDefendCenter-v1`, trained on the
RTX 4080 Laptop GPU: **mean 12.17, IQM 12.75 kills per episode** over 30
deterministic eval episodes (range 7 to 15; 15, the effective episode
ceiling, hit in 13 of 30). An untrained policy scores approximately 0.
Reward here is +1 per kill, -1 on death, so double digits means the
agent is aiming and conserving its limited ammo, not spraying.

## Recipe

- Obs: gray stride-4 downsample of the 240x320 RGB screen (60x80),
  frame-skip 4, frame-stack 4.
- PPO: n_steps 256, batch 512, lr 2.5e-4, gamma 0.99, gae_lambda 0.95,
  clip 0.1, ent_coef 0.01, 4 epochs, 8 SubprocVecEnv workers, CUDA.
- Throughput: 121.5 env steps/s wall (each is 4 game tics).

## Steps accounting (honest)

The run was killed once at 772k by our own stale-PID kill_process
(postmortem in the handoff; rule since: never kill_process near a live
run). It was resumed from the 750k checkpoint with `--steps 1500000`,
intending 1.5M total. SB3's `reset_num_timesteps=False` treats the
argument as ADDITIONAL steps, so the model actually trained to
**2.25M total steps** (3h26m for the resumed leg). The docstring now
documents this. All reported eval numbers are from the 2.25M model.

## Why gamma 0.99 was locked in

The Signal Dodge K-phase produced the project's main transferable
lesson: at gamma 0.999 (effective horizon ~1000 steps) the PPO critic's
targets were too high-variance to fit, explained_variance sat near 0,
and every from-scratch run failed regardless of exploration or
capacity. Cutting gamma to 0.99 (horizon ~100) fixed the critic
(EV 0.94+) and was the single load-bearing change. This run inherited
gamma 0.99 from day one and the critic was healthy throughout
(EV 0.90 to 0.944 across the resumed leg).

## Artifacts

- `runs\vzd\ppo_defend\summary.json`: eval of record.
- `runs\vzd\ppo_defend\gameplay.mp4`: 30s, 640x480, deterministic
  policy, recorded via `tools\vzd_ppo_watch.py` (the recorded episode
  scored 15).
- Checkpoints every 250k under the same dir.

## Next

BC-from-demos pipeline (vzd_bc_*) is validated end to end; this model
can serve as a demonstration teacher if the imitation track proceeds,
or the human-demo path via the desktop recorder remains open.
