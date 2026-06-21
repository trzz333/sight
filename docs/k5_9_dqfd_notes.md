# K5.9 DQfD-lite design notes

Reference for building K5.9 (demonstration-seeded replay) if K5.8 NoisyNet fails its in-env eval. Verified on disk this session; not yet implemented.

## Why K5.9

Cross-method single-direction collapse (PPO constant, DQN right-only, QR-DQN left-only) in a provably L/R-symmetric env is exploration collapse over shared exploration/replay plumbing. DQfD (Hester 2018) is the textbook cure when a good policy provably exists: prefill the replay buffer with expert transitions and keep them at a fixed expert:agent sampling ratio, so the value function sees high-return two-sided play from step 0 instead of bootstrapping off a one-sided-collapsed buffer. A good policy provably exists here (BC = 1737.3 through the same env path).

## Demo asset (VERIFIED this session)

`C:\Projects\Sight\runs\phase_k\k5_6_bc\dataset_2000_2035.npz`

- Keys: `X` (10-dim float32 states), `y` (int actions 0/1/2 = Left/Stay/Right), `ep_lengths`, `seeds`.
- 64800 transitions, 36 episodes, every episode exactly 1800 steps (all timeouts, zero collisions). Gold-standard full-survival trajectories.
- Action mix is two-sided and near-symmetric: L 25.3% / Stay 49.8% / R 24.9% (16398 / 32299 / 16103). This is precisely the two-sided dodging signal the from-scratch runs collapse away from.
- Seeds 2000-2035, disjoint from the held-out eval seeds 1000-1009, so no eval leakage.

## Transition reconstruction (BC pairs -> replay tuples)

The npz is a BC supervised set (obs, action) with episode boundaries in `ep_lengths`. Rebuild (s, a, r, s', done) per episode:

- `s = X[i]`, `a = y[i]`.
- `r = +1.0` for every transition (reward_shaping "none" = +1 per surviving step; every demo step survives).
- `s' = X[i+1]` within an episode.
- Episode boundary (every 1800 steps): the final step is a TIMEOUT, not a collision. Treat as truncation, not termination: bootstrap should continue, so set the SB3 replay `done`/`timeout` flags so the target does NOT zero the next-state value. Cleanest: drop the single boundary transition per episode (lose 36 of 64800, negligible) to avoid a cross-episode `s'`.

## The one real gotcha: obs-normalization parity

K5.7/K5.8 wrap the env in VecNormalize(norm_obs=True) with running mean/std that EVOLVE during training. Prefilling the buffer with RAW oracle `X` while the agent stores running-normalized obs is a distribution mismatch that breaks the value targets.

Decision for K5.9: drop running VecNormalize. Compute FIXED normalization stats from the oracle `X` (mean/std over the 64800 rows), apply them as a static transform to both demo obs and live agent obs. Demo and agent transitions then share one fixed normalized space. Save the fixed stats alongside the model for eval parity (the eval already reads mean/var off a saved object; point it at the fixed stats instead of a VecNormalize pickle).

## Build checklist (next session, only if K5.8 fails)

1. `tools\k5_9_dqfd_train.py`: load npz, compute fixed obs stats, reconstruct transitions, prefill an SB3 QRDQN replay buffer, set a fixed expert:agent sampling ratio (start 1:4), epsilon-greedy or NoisyNet exploration on the agent half, reward "none", net_arch [128,128], same eval bar.
2. Self-test the buffer prefill (shapes, reward all +1, no cross-episode s') before any train.
3. Smoke 8000 steps against live Godot, confirm exit 0 + finite ep_len_mean.
4. Full 200k detached via .bat + sentinel; eval `tools\k5_9_*_eval_inenv.py` on seeds 1000-1009. PASS only if mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97.
