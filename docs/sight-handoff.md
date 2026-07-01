# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = CMA-ES CLOSED NEGATIVE. C2 = pyribs CMA-MAE is next. On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `842c21d` Phase N C1 (CMA-ES) screen: NEGATIVE, all 3 seeds sub-bar

**Current task:** C1 ES screen complete and NEGATIVE. Seed 2 ran gen 8 -> 100 windowlessly (rc=0, sentinel EXIT 0, ~8 h) then evaluated held-out on 1000-1009. Three-seed held-out result, both vectors, all sub-bar (gate 930.27), anchored to c1_eval_summary.json on disk: s0 CMA-mean 906.4 / best-actor 845.4; s1 591.0 / 707.7; s2 564.0 / 663.0. Best number in the whole screen (s0 gen-100 mean 906.4) did not reproduce across seeds. Confidence HIGH. Windowless durable overnight execution is now PROVEN (was the open blocker).

**Next action:** Stand up C2 = pyribs CMA-MAE per `docs\phase-n-foolproof-design.md` Axis B and `docs\phase-n-c1-es-findings.md`. found-art ADAPT, do NOT build from scratch. Concretely: `pip install ribs` into `.venv-c1`; replace the CMA optimizer with a pyribs CMA-MAE scheduler over the SAME 5059-param SB3 actor, reusing the existing pycma-style ask/tell + Godot rollout + eval/gate infra; define a 1-2 dim behavior descriptor (action-fraction simplex, or trajectory mean x); run the screen seed 0 to gen 100 windowless (DETACHED_PROCESS supervisor + CREATE_NO_WINDOW trainer, the proven pattern); eval on 1000-1009 with the unchanged gate. Do NOT relaunch any trainer with CREATE_NEW_CONSOLE.

**Blockers:** None requiring Jeff for C2. (Optional: unattended reboot-recovery via NSSM or a "run whether logged on or not" Scheduled Task as `maste` needs Jeff's Windows password; the no-reboot overnight case is already solved and needs no Jeff action. Workbench tasks `WBReach` and `Workbench-ClaudeDesktopBootRecover` remain DISABLED, Jeff's call to re-enable via `Enable-ScheduledTask`.)

**Notes:**

- C1 NEGATIVE, full detail in `docs\phase-n-c1-es-findings.md`. All 12 C1 eval summaries (3 seeds x {mean,actor} x checkpoints) are C1-FAIL. Seed 2 did not collapse (diverse fracs, all diversity sub-gates passed); it is simply sub-bar on the mean. The decisive lever stays exploration, not optimizer tuning, so we change method, not retry CMA-ES.
- WINDOWLESS DURABLE EXECUTION PROVEN. Seed 2 gen 8 -> 100 rc=0 over ~8 h, no window, survived active shell churn. Pattern: supervisor DETACHED_PROCESS (pool-less, immune to console Ctrl/CLOSE) + trainer CREATE_NO_WINDOW (hidden console = valid Godot pool handles) + kill_godot/status-poll CREATE_NO_WINDOW + FOR_DISABLE_CONSOLE_CTRL_HANDLER=1. Prior crash loop was console-control kills (exit 0xC000013A = STATUS_CONTROL_C_EXIT). NSSM now optional (reboot only), not required.
- C2 = CMA-MAE counts as a distinct paradigm (QD archive illumination, not single-point optimization). found-art ADAPT verdict (reuse pyribs `ribs` + infra) and "distinct-paradigm shot" are separate axes; both hold. Phase N budget: C1 spent NEGATIVE, two shots remain (C2 CMA-MAE, C3 TBD).
- Popup fix shipped this session (commit `ab87c7b`): status-server tasklist poll (dev-team) + supervisor kill_godot PowerShell call both given CREATE_NO_WINDOW; stale CREATE_NEW_CONSOLE comment corrected. Status server pid 46780, console-less, http://localhost:8765, shows DONE. Restart: `C:\Python314\python.exe tools\launch_status_detached.py`. Live state is `/status.json`; `/` shows static red lamp markup.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. Detached work uses `.venv-c1\Scripts\python.exe` + `PYTHONNOUSERSITE=1` (cma/ribs in the venv), Godot headless. Launchers: `tools\launch_supervisor_detached.py <seed>`, `tools\launch_status_detached.py`.
