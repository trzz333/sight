# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Signal Dodge (930.27 bar) open. Scope call RULED (Jeff): keep pursuing from-scratch. First from-scratch reliable clear achieved via a start-state curriculum.

**Last commit:** `<PENDING>` sd-fast: start-state curriculum lever, from-scratch seed 0 clears the bar (1743.07)

**Current task:** New from-scratch lever is live and positive on seed 0. `tools\sd_fast_ppo_curriculum.py` runs the m21 recipe verbatim plus a start-state curriculum: a `CurriculumSDF` subclass of `SignalDodgeFast` (base env byte-identical, eval harness and imitation number untouched) injects `curriculum_n_init` hazards above the player at reset (headroom 100px, no reset collision), and `AnnealCurriculum` anneals that count 6 -> 0 over the first 70% of training so the run ends on the true clean-start distribution. Reward "none", eval unchanged (greedy, held-out seeds 5000-5029, via `sd_fast_ppo.evaluate`), so numbers are directly comparable to the m21 none arm. Matched seed-0 result (anchor `runs\sd_fast\sd_fast_m21curr_s0_5M_summary.json`, read this session): eval mean 1743.07 vs m21 none s0 1119.4, std 287.5 vs 593.3, 28/30 seeds at the 1800 cap, 29/30 clear the bar, action fracs 0.396/0.215/0.389 (diverse three-way dodging). This is the first from-scratch Signal Dodge policy in the project to clear at imitation grade (BC replica 1764.6, PPO-ft 1571.2). INTERIM: one seed. found-art ADAPT (curriculum learning Bengio 2009 / reset-state Go-Explore Ecoffet 2019 / arXiv 2410.16790; web search named in `docs\sd-fast-curriculum-findings.md`).

**Next action:** Collect `runs\sd_fast\sd_fast_m21curr_s1_5M_summary.json` (seed 1 was launched detached in the same chain, trainer pid 25072 live at handoff, matched to the WEAK m21 none s1 baseline 598.0, so it is the real variance-reduction test). If seed 1 also clears, chain curriculum seeds 2-4 for a 5-seed arm and run the rliable IQM/CI/POI vs the m21 none arm (add a curriculum run list to `tools\sd_fast_reliability.py`). If the 5-seed curriculum arm clears reliably, that recipe is the one to port to a Godot 5M eval-of-record (bar 930.27), previously blocked because no from-scratch recipe cleared reproducibly.

**Blockers:** None requiring Jeff. The scope call is resolved (keep from-scratch). Do not open a second new lever while the curriculum arm is being completed; finish the 5-seed arm before judging the method.

**Notes:**

- Curriculum lever: `tools\sd_fast_ppo_curriculum.py`, findings `docs\sd-fast-curriculum-findings.md`. Run-ids `sd_fast_m21curr_s{0,1}_5M`. Detached chain launched via throwaway `tools\_curr_spike_chain.py` + `_launch_curr.py` (untracked, delete next session); log `runs\sd_fast\curr_chain.log`. Seed 1 lands on disk when done.
- Eval of record is rliable (Agarwal 2021) ported to numpy. `tools\sd_fast_reliability.py` (from-scratch none/shaped arms); `tools\sd_fast_imitation_reliability.py` (BC/PPO-ft on the same replica block). Caches under `runs\sd_fast\*.json` (gitignored).
- Imitation reference on the replica (seeds 5000-5029): BC IQM 1800.0 30/30, PPO-ft IQM 1738.5 28/30; from-scratch none/shaped clear ~31-40%. Committed 80efb01 last session.
- Twice-failed methods, do NOT retry harder: reward geometry (K5.5 dense shaping + PBRS), plus the pre-registered from-scratch levers (CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet exploration).
- `runs\` is gitignored; summaries, logs, models, eval caches live on disk only. Leftover `-i` REPL pids and the status server (localhost:8767) may be running; harmless, clean up as needed.
