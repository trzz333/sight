# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Mission env (Signal Dodge, 930.27 bar) reopened via a fast-replica budget experiment. MinAtar adoption banked (3/3 seeds clear).

**Last commit:** `SUBST_HASH` sd-fast replica + budget-at-speed experiment (from-scratch PPO 5M, constant-left collapse)

**Current task:** MinAtar seed-2 collected (held-out mean 10.967, std 1.741, clears 9.4); 3/3 Breakout seeds clear (11.5 / 14.7 / 10.967, mean-of-means 12.39), recorded in `docs\minatar-adopt-spike-findings.md`. Mission port executed. Godot Signal Dodge measured at 59.8 steps/s (`tools\sd_throughput_probe.py`), which is why every prior from-scratch run capped at ~10k-1M steps. Built a fidelity-validated pure-Python replica `src\sight_agent\rl\sd_fast.py` at 237,910 steps/s (~4000x): replica constant_stay 518.8 matches the geometric analytic ~524; Godot is ~10-15% more collision-forgiving (safe transfer direction). Ran one from-scratch PPO seed at 5M steps (MinAtar recipe verbatim, reward "none") on the replica: collapsed to constant-left (mean 669.93, actions L 1.0 / S 0 / R 0, diversity_ok false), explained_variance ~0 throughout. Corrected read: this is not a clean budget isolation because the MinAtar recipe dropped VecNormalize and used gamma 0.99, both of which Phase M2.1 needed on this env. Evidence: `docs\sd-fast-replica-budget-findings.md`, `runs\sd_fast\*_summary.json` (gitignored).

**Next action:** Run the clean budget isolation with `tools\sd_fast_ppo.py` extended to Phase M2.1's exact recipe (reward none, gamma 0.999, VecNormalize(norm_obs+norm_reward), ent_coef 0.01, 8 envs, MlpPolicy [64,64]) at 5M steps, one seed, eval greedy on replica held-out seeds 5000-5029. If it clears the replica dodging bar with diverse actions, port to a Godot 5M run for the eval of record. If it reproduces M2.1's diverse sub-baseline plateau at 5x budget, budget is refuted and the wall is the exploration/credit structure (critic blind to death-timing from the 3-hazard obs); redirect off budget.

**Blockers:** None requiring Jeff.

**Notes:**

- The fast replica is a validated same-game reimplementation, not a new target environment (no Jeff gate). Eval of record stays Godot vs the 930.27 bar.
- Refuted this session: "a constant action already survives ~930." Best constant is 845.7 (Godot, K5.2); 930.27 = 845.7 x 1.10, set above best-constant so clearing it requires real dodging. Dense reward shaping was already tried (K5.5 alpha 0.30) and backfired by rewarding wall-hugging; reward stays "none".
- Prior from-scratch ceiling: Phase M2.1 (Godot, 1M, VecNormalize, healthy critic EV 0.85-0.94, diverse actions) IQM 418, CI [314,670] < bar. My 5M replica run's collapse is a critic/gamma artifact, not a budget verdict; the M2.1-recipe 5M control settles it.
- Imitation still clears reliably (BC 1737.3, PPO-finetune 1710.5). Mission is met by imitation; the open problem is from-scratch reliability.
- Anchors: `sd_fast_s0_5M_summary.json` (mean 669.93, L1.0), throughput probes in session log, `docs\sd-fast-replica-budget-findings.md`. HEAD after chore commit below.
