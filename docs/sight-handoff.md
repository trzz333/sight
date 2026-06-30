# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = Evolution Strategies (separable CMA-ES). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `ab87c7b` Phase N C1: eliminate terminal popups in monitoring + supervisor (substantive). This handoff chore commits on top.

**Current task:** C1 ES screen, seed 2 to gen 100, running WINDOWLESSLY and UNSUPERVISED. At 10:46 it was gen 9/100, 8 Godot workers, monitor GREEN, train running_best 1239 (TRAIN fitness, not held-out), avg ~323 s/gen, ETA ~489 min (~8 h). Supervisor pid 31404, status server pid 46780 (both this-session pids). Seeds 0 and 1 already evaluated sub-bar at gen 100 (gate 930.27): s0 CMA-mean 906.4 / best-actor 845.4; s1 591.0 / 707.7.

**Next action:** When `runs\phase_n\c1_screen_s2\c1_screen.sentinel` appears (or es_history reaches gen 100), eval `best_mean_vec.npy` + `best_actor_vec.npy` on held-out 1000-1009 with `.venv-c1\Scripts\python.exe` and `PYTHONNOUSERSITE=1` (Godot headless). Then record the 3-seed C1 verdict. Seeds 0 and 1 are already sub-bar, so unless seed 2's held-out clears 930.27 (train peaks did NOT reproduce on held-out for s0/s1, so do not expect it), C1 ES screen = NEGATIVE -> ADAPT to pyribs CMA-MAE per `docs\phase-n-foolproof-design.md`. Do NOT relaunch any trainer with CREATE_NEW_CONSOLE.

**Blockers:** Windowless durable execution is now PROVEN against shell churn (see note 1) but NOT against a reboot. If the box reboots mid-run, auto-restart-on-boot via NSSM or a Scheduled Task "run whether user is logged on or not" needs Jeff's Windows password (service account); Jeff-owned. Workbench scheduled tasks `WBReach` and `Workbench-ClaudeDesktopBootRecover` remain DISABLED (Jeff's call to re-enable via `Enable-ScheduledTask`).

**Notes:**

- WINDOWLESS DURABLE EXECUTION PROVEN this session (was UNPROVEN). Trainer launched 10:37, survived ~8 of my own cmd.exe tool-call shells AND advanced gen 8->9 with its 8-Godot pool intact. Prior crash loop was console-control kills: exit code 3221225786 = 0xC000013A = STATUS_CONTROL_C_EXIT. Fix = fully console-less chain: supervisor DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP, trainer CREATE_NO_WINDOW (hidden console gives Godot pool valid handles), kill_godot + status-poll also CREATE_NO_WINDOW, plus FOR_DISABLE_CONSOLE_CTRL_HANDLER=1. Confidence HIGH.
- Popup root cause was PLURAL, not the single source last session assumed. (a) status_server tasklist poll flashed a console every cycle -> dev-team added CREATE_NO_WINDOW; (b) supervisor kill_godot() PowerShell flashed on every restart -> I added CREATE_NO_WINDOW. Also fixed a stale comment in launch_supervisor_detached.py that wrongly claimed CREATE_NEW_CONSOLE. All in commit ab87c7b, pushed.
- Two-seed prior result unchanged, all sub-bar (930.27): s0 906.4/845.4, s1 591.0/707.7. Seed 2's gen-9 train running_best 1239 is TRAIN fitness; for s0/s1 train peaks did not reproduce on held-out, so it implies nothing about the gate. No C1 verdict until the seed-2 held-out eval.
- Status server: pid 46780, console-less, http://localhost:8765, correctly GREEN now. Jeff's "RED for a while" = the server had been DOWN (port 8765 not listening) and the prior run had crashed (workers 0 -> RED). Restart if needed: `C:\Python314\python.exe tools\launch_status_detached.py`. The `/` page shows static red lamp markup; live state is at `/status.json`.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. Detached trainer uses `.venv-c1` python + `PYTHONNOUSERSITE=1` (cma only in venv). Resume launchers: `tools\launch_supervisor_detached.py <seed>`, `tools\launch_status_detached.py`.
