# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `06da707` Phase N C1: crash root-cause + foolproof infra (resume, wall-budget, env fix)

**Current task:** A BOUNDED ES screen is RUNNING detached (seed 0, WMI console pattern, cmd PID 8420; `--max-wall-s 5400` ~= 18 gens, `--resume`, into `runs\phase_n\c1_screen_s0`). It self-stops cleanly with a resumable `es_state.pkl` and writes `c1_screen.sentinel`. Launched bounded because Jeff is gaming on this machine in a few hours: do NOT start any run longer than that window. Verified healthy at launch (8 Godot + python pool alive, past START, no error).

Prior full screen (PID 30072) crashed at gen 6/100 — root-caused to Intel-runtime `forrtl: error (200) window-CLOSE event` (MKL-linked worker aborted on an external console event, broke SubprocVecEnv pipes). NOT a plateau. Held-out eval (1000-1009) of the gen-5 vectors: `best_mean_vec` 845.7, `best_actor_vec` 879.6 (saved `eval_mean_g5.json` / `eval_actor_g5.json`), both gate-FAIL on mean only (~50-85 short), action spread fine, trend still climbing. ES is learning; the screen was cut off by infra, so it continues.

**Next action:** Poll `runs\phase_n\c1_screen_s0\c1_screen.sentinel` + `es_history.json`. When the bounded run finishes (or gaming is done), eval BOTH `best_mean_vec.npy` and `best_actor_vec.npy` via `tools\c1_es_eval.py` on 1000-1009 (venv python, `PYTHONNOUSERSITE=1`) to read the gen-~18 trend. Then (post-gaming) stand up NSSM for seed 0 and `--resume` to the full 100 gens unattended (per `docs\phase-n-foolproof-design.md`): crash auto-restart + `--resume` = foolproof. Run the NSSM account/ACL check (LOCALSYSTEM read of the WinGet Godot exe; else run-as `maste`, which needs Jeff's password = Jeff-owned credential step). If the full screen plateaus sub-bar, ADAPT to pyribs CMA-MAE (NOT ARS) — same pycma ask-tell interface, attacks the diagnosed behavioral collapse directly.

**Blockers:** None requiring Jeff now. Two Jeff-owned levers on the horizon: (a) the NSSM run-as-maste password IF the ACL check fails; (b) slate cap (3 paradigms). Default proceeds without asking.

**Notes:**

- FOOLPROOF INFRA shipped this turn: `--resume` (atomic per-gen pickle of full CMA state -> crash resumes not restarts), `--max-wall-s` (clean checkpointed stop, maxiter stays --gens), `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` in batches (kills the forrtl-200 abort). Launcher reverted to proven WMI console pattern.
- DETACHED_PROCESS (WMI CreateFlags=8) is REJECTED with on-disk evidence (`runs\phase_n\c1_smoke_detach`: START only, no gens, no sentinel, procs dead). No console = no valid std handles for multiprocessing/Godot workers = silent death. Do not retry it. NSSM is the correct console-less home (valid I/O redirection).
- found-art AXIS A (supervision): ADOPT NSSM for the full 8h run (session-independent, AppExit 0 Exit + default Restart, AppThrottle). AXIS B (exploration "treats"): ADAPT pyribs CMA-MAE if ES plateaus; QD archive resists the action-collapse (gen-5 L .91/R .09/S 0). Full writeup: `docs\phase-n-foolproof-design.md`.
- Godot runs `headless=True` (trainer line 174) -> session-0 service is viable, no desktop needed.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. All detached runs use `.venv-c1\Scripts\python.exe` + `PYTHONNOUSERSITE=1`; never `C:\Python314\python.exe` (cma only in the venv).
