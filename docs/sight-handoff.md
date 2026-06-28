# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `e68aefe` Phase N C1: fix detached env (venv), add ES gate eval + screen, save CMA mean

**Current task:** Detached-launch env bug root-caused and fixed. The WMI-detached process cannot read user-site; cma lived only there (numpy/torch/sb3 resolve from system site, which is why only cma crashed). Both user-site fixes failed (batch PYTHONPATH, in-script sys.path). Durable fix: `.venv-c1` (`--system-site-packages`, inherits system numpy/torch/sb3, owns a readable site-packages with cma 4.4.4); batches use the venv python + `PYTHONNOUSERSITE=1`. Verified under a cleared-APPDATA/PYTHONPATH proxy and a live detached run. C1 trend run (12 gens, seeds_per_gen 2, EXIT 0) verdict POSITIVE: shaped fitness climbed off gen-0 873 into a 700-1350 band, running-best 1351.5, diverse actions. ES learns on Signal Dodge. BUT held-out eval of the trend best (`tools\c1_es_eval.py`, gate on 1000-1009) gave mean 537.1, C1-FAIL: seeds_per_gen=2 fitness overfits (train 1351 vs held-out 537) and saving the argmax-of-noisy best candidate exploits noise. Fixes shipped: trainer also saves `best_mean_vec.npy` (es.result.xfavorite, CMA's noise-robust recommendation); screen raised to seeds_per_gen=4, gens=100. Screen seed 0 is RUNNING detached (PID 30072, ~286s/gen, ~8h; gen 0 len 618, gen 1 len 430; `best_mean_vec.npy` written every gen so a partial run is evaluable).

**Next action:** Poll `runs\phase_n\c1_screen_s0\c1_screen.sentinel` and `es_history.json`. When the screen has meaningful gens (or completes), eval BOTH `runs\phase_n\c1_screen_s0\best_mean_vec.npy` and `best_actor_vec.npy` through `tools\c1_es_eval.py` on held-out 1000-1009 (venv python, `PYTHONNOUSERSITE=1`). If the CMA-mean vector trends toward or clears the 930.27 gate, stage seeds 1 then 2 (`tools\launch_c1_screen.ps1 -Seed N`) for the 3-seed screen, then the full N=10 sweep only if the screen clears. If the mean plateaus sub-bar across 100 gens, record C1 ES screen NEGATIVE (do not retry harder) and drop to linear-policy + ARS (Mania 2018) as C1's structurally-different second attempt per the plan fallback.

**Blockers:** None requiring Jeff. Slate cap (3 paradigms, one honest shot at the gate each) is the only Jeff lever; the default proceeds without asking.

**Notes:**

- DETACHED ENV (corrects prior note): WMI-detached processes cannot read user-site `C:\Users\maste\AppData\Roaming\Python\Python314\site-packages`; they resolve only system site `C:\Python314\Lib\site-packages` (not writeable without elevation). Durable pattern: a `.venv-c1` created with `--system-site-packages` + `PYTHONNOUSERSITE=1`, batch calls the venv python. PYTHONPATH-to-user-site was never actually proven (M2.1 deps all live in system site). `.venv-c1/` and `runs/` are gitignored.
- Throughput real rate: seeds_per_gen=2 ~165s/gen, seeds_per_gen=4 ~286s/gen (not the old 93s/gen claim). 100-gen screen ~8h per seed.
- Noisy-fitness lesson: a k=2 seed mean over episode lengths (180-1400 range) has huge variance; argmax of it over-selects lucky candidates. Gate the CMA distribution mean (es.result.xfavorite), not just the best sampled candidate. Screen now uses k=4 and saves both vectors.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. `tools\c1_es_eval.py` loads an ES vector into the same MlpPolicy (5059 params), single bare env, reset(seed=s) per seed (sidesteps broken VecEnv.seed). Validated end-to-end (90.5s).
- Trend POSITIVE / screen pending: ES clears the "does it learn" bar; the open question is reliability on held-out seeds, which the seeds_per_gen=4 screen + CMA-mean gate now tests honestly.
