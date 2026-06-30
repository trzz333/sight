# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `5b8ceec` Phase N C1: seeds 0+1 evaluated sub-bar, console-less infra + stoplight server

**Current task:** C1 ES screen, gen-100 held-out results across seeds. Seed 0 (gen 100): CMA-mean 906.4, best-actor 845.4. Seed 1 (gen 100): CMA-mean 591.0, best-actor 707.7. All four sub-bar (gate 930.27). Seed 0's 906.4 cleared the gen-5 peak (879.6) and every diversity sub-gate, which is why seeds 1+2 were staged per the pre-registered rule; seed 1 then collapsed to 591.0, showing high between-seed variance and non-reproducibility of seed 0's near-miss. Seed 2 reached gen 8 then was KILLED to stop window spam, INCOMPLETE. Two of three seeds done, both sub-bar. KEY INFRA FINDING this session: WMI cmd-console detached processes die on console CLOSE/Ctrl events when interactive shells churn, NOT reboot; this killed seed 2 at gen 5, the supervisor mid-launch, and the status server repeatedly. Seeds 0/1 only survived because nothing touched the machine while running. Eval metric values anchored to this session's eval-program stdout; the four eval JSONs are currently permission-locked for re-read (lingering lock/ACL artifact).

**Next action:** Bring seed 2 to gen 100 windowlessly. Launch the Python supervisor via `C:\Python314\python.exe tools\launch_supervisor_detached.py 2` and verify three things with NO visible console window appearing: (a) the 8-Godot pool survives CREATE_NO_WINDOW, (b) supervisor pid survives shell churn, (c) `es_history.json` advances past gen 8. If CREATE_NO_WINDOW kills the pool, fall back to a Scheduled Task "run whether user is logged on or not" or NSSM (needs Jeff password, see Blockers). When seed 2 hits gen 100, eval `best_mean_vec.npy` + `best_actor_vec.npy` on 1000-1009 (`.venv-c1\Scripts\python.exe`, `PYTHONNOUSERSITE=1`), then record the 3-seed C1 verdict: all three sub-bar means C1 ES screen NEGATIVE, ADAPT to pyribs CMA-MAE per `docs\phase-n-foolproof-design.md`.

**Blockers:** Windowless durable overnight execution is UNPROVEN. If CREATE_NO_WINDOW kills the Godot pool, NSSM/Scheduled-Task run-as-maste needs Jeff's Windows password (service account); Jeff-owned. Also: Workbench scheduled tasks `WBReach` and `Workbench-ClaudeDesktopBootRecover` were DISABLED this session at Jeff's instruction to stop cross-project window spam; re-enabling is Jeff's call via `Enable-ScheduledTask -TaskName '<name>' -TaskPath '\'`.

**Notes:**

- C1 two-seed result, all sub-bar (930.27): seed 0 CMA-mean 906.4 / best-actor 845.4; seed 1 591.0 / 707.7. Seed 0 cleared the gen-5 peak 879.6 and all diversity sub-gates (left .67 stay .03 right .30); seed 1 collapsed to 591. High between-seed variance, near-miss not reproducible. Confidence HIGH (eval stdout this session); JSONs permission-locked for re-read right now.
- Root infra cause: WMI cmd-console detached procs die on console-close events from shell churn. Fix is console-less launch: status server via DETACHED_PROCESS, trainer pool via CREATE_NO_WINDOW (gives a console for Godot handles but no window). DETACHED_PROCESS still kills the trainer pool (no console at all), so the trainer needs CREATE_NO_WINDOW specifically, untested.
- Status server RUNNING now: pid 16076, console-less, http://localhost:8765, no window, safe to leave up. Auto-follows the freshest c1_screen_s* run; GREEN/YELLOW/RED/DONE from es_history+es_state mtime and Godot worker count. Restart: `C:\Python314\python.exe tools\launch_status_detached.py`.
- New tools committed this session: `tools\sight_status_server.py`, `tools\run_c1_supervised.py` (Python auto-restart supervisor, CREATE_NO_WINDOW), `tools\launch_supervisor_detached.py`, `tools\launch_status_detached.py`. Obsolete `tools\supervise_c1.ps1` (PowerShell supervisor that wedged on an orphaned-worker pipe handle) deleted.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. Detached trainer must use `.venv-c1` python + `PYTHONNOUSERSITE=1` (cma only in the venv). Godot headless.
