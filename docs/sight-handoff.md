# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE (M2 critic-broken, M2.1 critic-fixed-but-sub-baseline, IQM 418.25 CI [314.44, 670.50] below the 930.27 bar).

**Last commit:** `e5b832f` Phase N C1: ES trainer (sep-CMA-ES) built, candidate-packing throughput fix. Foreground-validated; detached-launch env bug open.

**Current task:** C1 ES trainer built at `tools\c1_es_train.py` and validated end-to-end in the foreground (2-gen real run drives es.tell, persists history + best_vec, fitness climbed). The naive design hit the throughput wall the plan predicted (5.6 gens/hour); rebuilt the evaluator to PACK 8 different candidates per lockstep wave with common-random-numbers, re-measured 38.6 gens/hour (6.9x), one popsize-29 generation now ~93s. A 2-gen run exposed a fitness flaw: raw mean-length rewards single-action collapse (gen-0 best was left=1.0 surviving 1203), which the eval gate rejects; added `shaped_fitness` penalizing max_frac>0.90 and lateral<0.05 to steer CMA toward gate-passing behavior. OPEN BUG: the WMI-detached trend run crashes instantly at `import cma` (ModuleNotFoundError) even though cma 4.4.4 is present in user-site and foreground imports work; the batch PYTHONPATH is not reaching the detached process.

**Next action:** Add a launch-env-independent fix to `tools\c1_es_train.py`: inject the user-site dir (`C:\Users\maste\AppData\Roaming\Python\Python314\site-packages`) onto `sys.path` at the top of the script so cma/numpy/torch resolve regardless of how the process was launched. Then re-run the 12-gen seeds_per_gen=2 trend check detached (`tools\run_c1_trend.bat` via WMI) and confirm shaped fitness climbs across generations. If it climbs, launch the 3-seed screen staged (seed 0 first, ~120 gens at seeds_per_gen=2 ~= 6h, abort early if flat), eval the best vector through the M2 gate (`tools\m2_state_ppo_eval.py` logic, held-out 1000-1009). If fitness does not climb, drop C1 to linear-policy + ARS (Mania 2018, far fewer params and generations) per the plan fallback.

**Blockers:** None Jeff-owned. The slate cap is the only Jeff lever: Phase N defaults to THREE distinct paradigms, one honest shot at the reliability gate each, then FINAL NEGATIVE and close if all fail. Raising the cap is Jeff's call; the default proceeds without asking.

**Notes:**

- PACKAGE LOCATION REALITY (corrects the prior system-wide note): all packages live in user-site `C:\Users\maste\AppData\Roaming\Python\Python314\site-packages`, NOT `C:\Python314\Lib\site-packages` (system site is not writeable; pip silently used user-site). The proven detached pattern is PYTHONPATH injection to user-site (see `tools\run_m2_1_multiseed.bat`), not a system-wide install. cma 4.4.4 installed there. The current detached crash means PYTHONPATH alone is not reaching the WMI process; the durable fix is in-script sys.path injection.
- Throughput lever is candidates-per-wave, not eps/sec: per-episode wall-time is fixed by the TCP round-trip, so packing 8 distinct candidates into one lockstep wave (common random numbers within a generation for fair CMA ranking) is the 6.9x win. Random-init policies already average ~700 steps and one hit 1203 (>bar), so bar-clearing policies exist near random init: ES has real headroom.
- Fitness MUST be diversity-shaped. Raw length rewards single-action survival that the gate rejects. `shaped_fitness` multiplies length by a collapse penalty (ramps in past max_frac 0.90, 0 at full collapse) and a lateral penalty (below 0.05 min of frac_left/frac_right). CMA optimizes the shaped value; reports keep raw mean_length for gate readability.
- Policy binding verified on disk: SB3 actor is exactly 5059 params (`mlp_extractor.policy_net.*` + `action_net.*`), flatten/load round-trips to 0.0, argmax forward matches the M2 deterministic eval. SB3 2.8.0 `VecEnv.seed(list)` is broken and plain reset leaves sub-envs unseeded; per-env seeds go through `env_method("reset", seed=s, indices=[i])` (verified).
- Eval gate unchanged: held-out seeds 1000-1009, PASS = mean>=930.27 AND frac_left>=0.03 AND frac_right>=0.03 AND max(frac)<0.97. `tools\m2_state_ppo_eval.py` is the reference; the C1 best vector loads into the same MlpPolicy for an apples-to-apples gate run. Env: GodotSignalDodgeEnv, state obs, reward "none", max_steps 1800, headless, one Godot subproc + one TCP port per worker.

---
