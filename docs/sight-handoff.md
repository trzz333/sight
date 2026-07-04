# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Mission env (Signal Dodge, 930.27 bar) open. Fast replica: budget isolation closed (1/N seeds clear at 5M); reward-geometry lever (potential-based shaping) under test, sweep in flight.

**Last commit:** `a9ab412` sd-fast: /found-art fix, external search; eval methodology ADAPT (rliable), control arm n=5

**Current task:** Matched-seed none-vs-shaped 5M sweep RUNNING DETACHED on the replica (launcher `tools\sd_fast_shaped_sweep.py`, spawned windowless PID 16080; 7 runs sequential, ~85 min). Control arm done: reward=none 5M mean_len s0 1119 (CLEAR), s1 598, s2 888, s3 670, s4 643, so none clears ~1/5. Shaped arm (potential-based reward, seeds 0-4, `sd_fast_m21sh_s{0..4}_5M`) in flight; at handoff shaped s0 was training, no shaped summary yet, `runs\sd_fast\shaped_sweep.sentinel` not yet written. found-art run this session (external, /found-art): Ng 1999 PBRS confirmed canonical/policy-invariant (shaping ADAPT stands, implementation correct); rliable (Agarwal 2021) is the standard reliability eval but its dep `arch` needs an MSVC compiler absent here, so the methodology is adopted, not the package.

**Next action:** When `runs\sd_fast\shaped_sweep.sentinel` reads CHAIN_DONE, write `tools\sd_fast_reliability.py`: reload-eval all 10 models (none + shaped, seeds 0-4) greedily on held-out seeds 5000-5029, then compute per-arm IQM, a percentile bootstrap 95% CI over seeds, and P(shaped IQM > none IQM), in numpy/scipy, citing Agarwal 2021 (single env, so stratified bootstrap reduces to seed-level resampling). Record the none-vs-shaped comparison in `docs\sd-fast-replica-budget-findings.md`. Decision rule: shaped beats none on clear-rate / IQM / P(improve) -> tune coef 1.0 or port the shaped recipe to a Godot 5M eval-of-record (build SB3 CheckpointCallback+resume first, Phase M saw Godot worker crashes). Shaped no better than none -> PBRS refuted at 5M on this env and the Jeff-owned scope call goes live. Do NOT launch the Godot eval-of-record until the replica clears reproducibly.

**Blockers:** None requiring Jeff. One parked Jeff-owned scope call goes live ONLY if the shaped sweep fails to lift reliability: keep pursuing from-scratch reliability, or accept imitation (BC 1737, PPO-finetune 1710, both clear reliably) as the standing mission solution.

**Notes:**

- 3-seed budget spread (reload-eval, `tools\sd_fast_iqm_spread.py`): s0 IQM 1148.7 CLEAR, s1 482.1 collapse, s2 750.1 diverse-sub-bar. Two failure modes (basin collapse + mediocre dodging), not one. Budget lifts ceiling, not reliability.
- Correction held from last session (disk beats memory): final EV does NOT track clearing (clear s0 EV 0.18; sub-bar s1/s2 0.885/0.907). Basin collapse is a MINORITY mode (2/3 diverse).
- Reward shaping: `SignalDodgeFast(reward_mode="shaped", shape_coef=0.5, shape_gamma=0.999)`, potential-based (Ng 1999), Phi = imminence-weighted horizontal clearance to nearest hazard. `reward_mode="none"` byte-identical (dynamics verified 20 seeds). Eval is reward-agnostic.
- Eval methodology is rliable's (Agarwal 2021: IQM + bootstrap CI + P-improvement), adapted to numpy because the package will not build here (arch C-ext needs MSVC). Do not retry installing rliable without a compiler.
- runs\ is gitignored; summaries/logs/models on disk only. Detached launch pattern: CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP via `tools\_spawn_shaped_sweep.py`.
