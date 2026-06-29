# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `06da707` Phase N C1: crash root-cause + foolproof infra (resume, wall-budget, env fix)

**Current task:** The full C1 ES overnight screen is RUNNING detached (seed 0, WMI console pattern, cmd PID 25928), `--resume`d from the gen-18 checkpoint toward gen 100. Verified healthy: log shows `resumed_from_gen 18` (elapsed offset preserved), 8 Godot + python pool alive. `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` + per-gen `es_state.pkl` checkpoint = crash-resistant and resumable; ETA ~6.5h. KEY FINDING this session: held-out generalization is REGRESSING while train climbs. Held-out eval (1000-1009) of the CMA-mean vector fell 845.7 (gen 5) to 519.0 (gen 18); best-actor vector fell 879.6 to 632.7; meanwhile train running-best rose to 1172.2. Train-up / held-out-down. Peak held-out so far (879.6) was early at gen 5 and both vectors have since regressed, all still sub-gate.

**Next action:** When the screen reaches gen 100 (or `runs\phase_n\c1_screen_s0\c1_screen.sentinel` appears), eval BOTH `best_mean_vec.npy` and `best_actor_vec.npy` on 1000-1009 (`.venv-c1\Scripts\python.exe`, `PYTHONNOUSERSITE=1`). Pre-registered decision rule: if gen-100 held-out is sub-bar AND not above the gen-5 peak (879.6), record C1 ES screen NEGATIVE and ADAPT to pyribs CMA-MAE (NOT ARS) per `docs\phase-n-foolproof-design.md` (QD archive attacks the train-overfit/collapse directly, same pycma ask-tell interface). If held-out climbs toward/clears 930.27, stage seeds 1 then 2 via `tools\launch_c1_screen.ps1 -Seed N`. If the run died (no sentinel, no procs), relaunch `tools\run_c1_screen.bat 0` (it `--resume`s from `es_state.pkl`).

**Blockers:** None requiring Jeff. On the horizon: NSSM run-as-maste needs Jeff's password IF the session-0 ACL check on the WinGet Godot exe fails; slate cap (3 paradigms). Default proceeds without asking.

**Notes:**

- Held-out regression is THE signal to watch: gens 5->18 the held-out CMA-mean dropped 845.7->519.0 and best-actor 879.6->632.7 while train shaped-fitness rose. More gens may not fix a generalization gap; the gen-100 result decides ES vs the CMA-MAE pivot. 10-seed evals are noisy, so the 100-gen verdict (not gen-18) is the honest call.
- Foolproof infra VALIDATED end-to-end this session: `--resume` continued a real gen-18 checkpoint (`resumed_from_gen 18`); bounded run (90-min wall) self-stopped clean (sentinel EXIT 0, 18 gens) then resumed seamlessly. DETACHED_PROCESS (CreateFlags=8) REJECTED (silently killed the worker pool, no valid std handles). Evidence in `runs\phase_n\c1_smoke_detach`.
- found-art (`docs\phase-n-foolproof-design.md`): ADOPT NSSM for unattended auto-restart on the long run (the one gap left: crash currently sits dead until a human relaunches; NSSM `default Restart` + `AppExit 0 Exit` + `--resume` closes it). ADAPT pyribs CMA-MAE if ES goes NEGATIVE; it replaces the weaker ARS fallback.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. All detached runs use the venv python + `PYTHONNOUSERSITE=1`; never `C:\Python314\python.exe` (cma only in the venv). Godot `headless=True` (trainer line 174).
