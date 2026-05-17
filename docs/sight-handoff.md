# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 amended; Phase K K0 training-time entropy-collapse probe complete. Verdict K-C.

**Last commit:** `a0c3b29` Phase K K0 evidence + tool: training-time entropy-collapse probe at 2048 timesteps.

**Current task:** K0 ran one instrumented 2048-timestep training session (train_seed=2, `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`, 8 PPO updates, 64.79 s wall) via `tools/h5_training_entropy_probe.py` (InstrumentedPPO subclass mirroring SB3 2.8.0 `train()` with per-minibatch grad norms, pre/post-update entropy and raw-logit margin on rollout obs, advantage/value stats, action_net deltas, K-A/K-B/K-C/K-D auto-classification). None of the three collapse thresholds crossed: mean entropy stayed in `[1.025, 1.083]` (vs `ln(3)=1.0986` ceiling), rollout top-action fraction in `[0.395, 0.523]` (vs 0.95), raw margin in `[0.082, 0.599]` (vs 4.0). Evidence at `docs/h5-phase-k-training-entropy-probe-evidence.md`. NDJSON and summary JSON at `runs/phase_k/entropy_probe_seed2.{ndjson,summary.json}` (gitignored).

**Next action:** Rerun the same probe at `--total-timesteps 10000` (full YAML default), same seed, same config, no other changes. Invocation: `python tools\h5_training_entropy_probe.py --config configs\rl\signal_dodge_ppo_h5_pixel_entropy.yaml --seed 2 --total-timesteps 10000 --out-dir runs\phase_k --label entropy_probe_seed2_10k`. Expected ~39 PPO updates, ~5-6 min wall time. K1 architecture probe and K2 train-seed asymmetry probe held behind this rerun per the K-C clause pre-agreed in the prior planning round.

**Blockers:** none.

**Notes:**

- At 2048 timesteps the trained raw-logit margin (0.116) is 14% of Phase E seed 2's converged 0.83 (Phase I left tape). The wedge-commitment basin Phase J seed-1008 falsified does not exist at this budget; it forms in the 2048-10000 timestep window. Action_net row norms at update 8 are 55-74% of Phase E seed 2's final magnitudes.
- K0 rollout argmax oscillates left (updates 1, 2, 7, 8) and stay (3-6) across 8 updates. The Class B "train_seed=2 picks left" identity from Phase H is not yet locked at 2048 timesteps. Useful data point for the eventual seed-asymmetry probe option (3).
- K-B precedence rule uses OR semantics on advantage-std-or-explained-variance, which would fire on any early-training run because explained_variance is near zero in the first ~2k timesteps. Did not engage here because no entropy collapse to compare against. Tighten to AND semantics or stricter explained_variance ceiling before the 10000-timestep rerun.
- Detached `start /B cmd /c "...phase_k_driver.bat"` from a Desktop Commander shell does not survive parent-shell exit; first K0 attempt logged only its start header before the parent died. Inline invocation under `interact_with_process` succeeded in 64.79 s. For runs under ~5 min, prefer inline; for longer runs, use `start /MIN` (not `/B`).
- `InstrumentedPPO.train()` mirrors SB3 2.8.0 `stable_baselines3/ppo/ppo.py` exactly with three additions: per-minibatch pre-clip grad norms between `loss.backward()` and `clip_grad_norm_`, pre/post-update policy-state and action_net snapshots, and rollout-buffer aggregate statistics. SB3 logger calls preserved with identical keys so downstream callbacks see no shape change.
