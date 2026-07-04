# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Signal Dodge (930.27 bar) open. Scope call RULED (Jeff): keep pursuing from-scratch. Start-state curriculum lever is judged: a large real improvement over the none arm, but NOT yet reliable enough to port.

**Last commit:** `b7f81f9` sd-fast: curriculum 5-seed arm judged, better than none but not port-reliable (+ this chore refresh on top).

**Current task:** DONE this session. The 5-seed curriculum arm (`sd_fast_m21curr_s{0..4}_5M`, m21-verbatim + start-curriculum, 5M, reward none) is complete and judged by rliable vs the m21 none arm. Per-seed curriculum means: s0 1743.1, s1 1704.3, s2 887.7, s3 669.9, s4 1591.6 (3/5 clear the bar). rliable: curr IQM 1394.5 [742.5, 1730.1] vs none IQM 733.6 [613.0, 1042.2]; diff +661.0; P(IQM_curr>IQM_none)=0.970; POI=0.743; clears 3/5 vs 1/5. Verdict: BETTER THAN NONE, HOLD THE GODOT PORT. Two seeds (s2, s3) collapse below the bar and the curr IQM CI lower bound (742.5) is under 930.27, so a Godot port would likely be a coin-flip. Root cause found: a shared constant-collapse attractor. Byte-identical held-out length arrays across genuinely different models (none-s3, curr-s3, shaped-s0, shaped-s1 all mean 669.9), verified element-wise in `reliability_eval_cache.json`; from-scratch PPO collapses on a subset of seeds into one near-constant policy. Failing seeds are low-entropy (af ~0.66 on one action); succeeding seeds sit near 0.40/0.20/0.39. The wall is now VARIANCE, not mean. Full writeup: `docs\sd-fast-curriculum-findings.md`.

**Next action:** Variance-reduction on the SAME curriculum lever (found-art ADAPT, not a new lever, not a twice-failed method). Un-tried anti-collapse knob first: raise/anneal `ent_coef` above the m21 default 0.01 in `tools\sd_fast_ppo_curriculum.py` (CLI knob already exists) to hold policy entropy through the anneal window so laggard seeds don't collapse when the scaffold is pulled. Pre-register: one knob (ent_coef), same 5-seed arm, judge by whether clears goes 3/5 -> 5/5 AND the curriculum IQM 95% CI lower bound clears 930.27. Run the arm detached (windowless chain pattern), then `.venv-c1\Scripts\python.exe tools\sd_fast_reliability.py` (the curriculum arm is already wired in; you may want to add the new run-ids or a third variant list). Only if that clears reliably, port the recipe to a Godot 5M eval-of-record (bar 930.27). Secondary un-tried knobs if ent_coef is insufficient: `anneal_frac` 0.7 -> 0.9, or higher `n_init_max`.

**Blockers:** None requiring Jeff. Scope call resolved (keep from-scratch). The curriculum lever is judged, so the "don't open a second lever" freeze is lifted for the ent_coef variance knob (same lever, tuning). Do NOT open a structurally new lever until the ent_coef arm is judged.

**Notes:**

- Curriculum lever: `tools\sd_fast_ppo_curriculum.py` (CLI: --seed --steps --run-id --n-init-max --anneal-frac --ent-coef, m21 defaults verbatim). Findings `docs\sd-fast-curriculum-findings.md`. Run-ids `sd_fast_m21curr_s{0..4}_5M`, all 5 summaries + models on disk. Each seed ~12.5 min at ~6.7k steps/s.
- Eval of record is rliable (Agarwal 2021) ported to numpy. `tools\sd_fast_reliability.py` now carries three arms: none / shaped / curr, headline comparison curr-vs-none, port gate P(IQM)>=0.975. Reload-evals each arm's saved model greedily on held-out seeds 5000-5029; caches per-model length arrays in `runs\sd_fast\reliability_eval_cache.json` (gitignored) so re-runs are instant. It independently reproduced all 5 curriculum summary means this session (cross-check passed).
- Shaped vs none stays refuted-ish: IQM diff +7.4, P=0.607, POI=0.569 (PBRS is not the lever). Imitation still clears reliably (BC IQM 1800 30/30, PPO-ft IQM 1738.5 28/30) but the scope call is to stay from-scratch.
- Twice-failed methods, do NOT retry harder: reward geometry (K5.5 dense shaping + PBRS), and the pre-registered from-scratch levers (CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet). The curriculum is NOT in this list; it is positive-but-noisy, so tune it.
- `runs\` is gitignored; summaries, logs, models, eval caches live on disk only. Detached chain log this session: `runs\sd_fast\curr_chain_234.log` (CHAIN_234_DONE). Leftover `-i` REPL pids and the status server (localhost:8767 / pid ~22984) may be running; harmless, clean up as needed. One `-i` REPL (was pid 41724) got a stuck buffered sleep this session; kill it if lingering.
