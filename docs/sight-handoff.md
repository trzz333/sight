# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (Phase E seed sweep complete; NOT closure-grade; seed=0 fixed-point hypothesis falsified; aggregate weak seed-sensitive signal)

**Last commit:** `13e7b08` chore: refresh handoff after phase D entropy slice

**Current task:** Phase E diagnostic train-seed sweep of the Phase D entropy recipe at train seeds 1, 2, 3 (10000 timesteps each), eval seeds 1000-1009 trained_cnn only. Three distinct per-eval-seed trajectory vectors observed across train seeds {0, 1, 2, 3}: A=seed 0 (688.8 reward), B=seed 1 (605.0 reward, byte-equal aggregates to best neg control), C=seed 2 = seed 3 (844.8 reward, byte-equal per-eval-seed lengths and terminal causes despite seed 2 holding entropy and seed 3 collapsing to ~0 entropy by iter 24). Aggregate over seeds {1, 2, 3} pooled: 764.87 reward (+26.4%), 765.80 length (+26.4%), 0.933 collision (-6.7pp), 0.067 timeout. Reward and length bars marginally cleared by ~1pp; collision bar fails by ~13pp. Per the amended diagnostic-not-selection rule, no individual seed is promoted to candidate. Findings in `docs/h5-trained-policy-phase-e-seed-sweep-evidence.md`.

**Next action:** GPT picks Phase F. The Phase D candidate list narrows: seed sweep is now spent (Phase E executed), and the result is row 3 of the amended outcome table (weak seed-sensitive signal). Ranked remaining attacks on the structural hypothesis: (1) frame stacking via `VecFrameStack(n=4)` (single-frame may not encode velocity; policy IS varying with seed but not toward avoidance, consistent with perception bottleneck), (2) state-observation comparator (isolates perception path vs PPO; H3 state-mode pipeline already exists), (3) dense reward shaping (charter-amendment territory per H5 plan section 7). Increasing timestep budget within the current recipe family remains off the list.

**Blockers:** none operational. Open: Phase F recipe call belongs to GPT.

**Notes:**

- Exact commands: trains `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml --seed <N> --total-timesteps 10000 --run-id h5_train_phase_e_seed<N>_entropy_10k` for N in {1, 2, 3}; evals `python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml --run-id h5_eval_phase_e_seed<N>_entropy_10k_trained_only --seeds 1000-1009 --mode full --policies trained_cnn --train-run-dir runs/rl/signal_dodge_ppo_h5_pixel_entropy/h5_train_phase_e_seed<N>_entropy_10k`. Artifacts under those `runs/` paths.
- Phase D's conclusion ("eval policy is invariant under PPO knob changes") is refined: under fixed `seed=0`, three different recipes produced byte-equal trajectories; under fixed Phase D entropy recipe, four different seeds produce three distinct trajectories. Invariance was a seed=0 fixed point under that recipe family, not a single global attractor. Seeds 2 and 3 converge to the same attractor despite materially different training-time entropy regimes; the eval-relevant argmax is robust to training exploration in this attractor.
- Seed 1 aggregate is byte-equal to best-negative-control aggregate (605.0 reward, 606.0 length, 1.0 collision). One in three trained seeds produced a policy indistinguishable from no-learning at the 10-seed external eval set.
- Phase E was executed strictly diagnostically per the amended interpretation rule. No best-of-N seed selection. The seed 2 / seed 3 attractor's reward and length clearance is reported descriptively only; it does not constitute H5 evidence. Aggregate-over-{1,2,3} marginally clears reward and length bars but fails collision bar, matching row 3 of the amended outcome table (weak seed-sensitive signal).
- MCP timeout discovery operational note: Claude Desktop's MCP layer caps `interact_with_process` waits at ~4 min regardless of requested `timeout_ms`. Workaround: poll with empty input after the trip-out; the underlying process continues uninterrupted. Each ~5m39s train and ~3m eval survives this with one or two polls.
