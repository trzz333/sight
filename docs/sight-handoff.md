# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (Phase D entropy slice complete; NOT closure-grade; entropy-collapse hypothesis falsified)

**Last commit:** `3e803b1` docs(h5): phase D entropy recipe evidence slice

**Current task:** Phase D 10000-timestep PPO CnnPolicy slice with the learning-grade entropy recipe (`ent_coef=0.01`, `n_steps=256`, `batch_size=64`, `n_epochs=4`, `eval_freq=2048`) on the new `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`. Trained_cnn aggregate (10 seeds 1000-1009): 688.8 reward, 689.7 length, 0.9 collision rate, 0.383 length_ratio. **Per-seed eval trajectories are bit-identical to Phase C** despite materially different training dynamics. All three H5 section 6 quantitative bars FAIL at the same magnitude as Phase B and C (+13.9% reward, +13.8% length, -10 pp collision). Saturation gate passes. Full 4-policy eval skipped per Phase D decision rule (gaps below 20% on reward/length, collision_rate > 0.8). Findings written to `docs/h5-trained-policy-phase-d-entropy-10k-evidence.md`.

**Next action:** GPT picks the next experiment from a now-narrower hypothesis space. The "smoke-cheap hyperparameters are the blocker" framing (Phase B/C diagnosis) is empirically dead: three different recipes produce byte-equal eval trajectories. Candidates ranked in the evidence doc by how directly they break the invariance: (1) seed sweep (cheapest discriminator of fixed-point vs structural), (2) frame stacking via `VecFrameStack(n=4)` (single-frame may not encode velocity), (3) state-observation comparator (isolates perception path vs PPO), (4) dense reward shaping (charter-amendment territory per H5 plan section 7). Further PPO hyperparameter iteration is explicitly not on the list.

**Blockers:** none operational. Open: next-experiment recipe call belongs to GPT.

**Notes:**

- Exact commands: training `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml --total-timesteps 10000 --run-id h5_train_phase_d_entropy_10k`; eval `python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml --run-id h5_eval_phase_d_entropy_10k_trained_only --seeds 1000-1009 --mode full --policies trained_cnn --train-run-dir runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_d_entropy_10k`. Artifacts under `runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_d_entropy_10k/` and `.../h5_eval_phase_d_entropy_10k_trained_only/evaluation/trained_cnn/`.
- Entropy budget verified IN training (entropy_loss -1.09 at iter 2, > -0.7 through iter ~13, > -0.15 through iter ~23; `approx_kl` 0.01-0.12 with substantial `clip_fraction` 0.1-0.5 in same window). In-training deterministic eval reward locked at 212.0 from step 2048 (iter 8, deep inside high-entropy phase) and stayed flat across 8000 more timesteps. Value function did learn (`explained_variance` 0.001 -> 0.455). The eval-relevant argmax is invariant to action-distribution entropy.
- Training wall 338.91 s (~5m39s) per `run_end` event, 40 PPO iterations, `train/n_updates=156`, `time/fps=30`. Eval wall ~191 s aggregate per-seed `elapsed_seconds`. No code changes this session; YAML-only addition.
- Variables held constant across the three failing slices (B, C, D): `seed=0`, SB3 NatureCNN under `CnnPolicy`, `(1, 84, 84)` grayscale single-frame observation, sparse `+1`-per-step survival reward, Signal Dodge H4 profile. These are the candidate next levers; everything inside the PPO knob set has been varied across slices already.
- Claude Desktop crashed twice during this session AFTER training and trained-only eval had both completed cleanly but BEFORE any evidence doc or handoff edits landed. State recovered by `git status`, live-process inspection (no orphaned Godot/Python found), and on-disk artifact inventory (train `summary.json status=ok`, `run_end` event present, eval `summary.json` valid). No retraining performed; existing run dirs reused.
