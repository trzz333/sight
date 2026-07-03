# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). C1 = CMA-ES CLOSED NEGATIVE. C2 = pyribs CMA-MAE IN PROGRESS (seed 0 done + FAIL, seeds 1+2 running unattended). On NLDC (hostname MSI, user maste). Phase M closed FINAL NEGATIVE.

**Last commit:** `1242d6a` Phase N C2: seed-0 held-out FAIL both vectors; stage seeds 1+2 per screen rule

**Current task:** C2 CMA-MAE screen. Seed 0 completed 100 gens clean (sentinel EXIT 0, ~66 min, archive 166 cells, train archive_best 1741.5). Held-out gate on 1000-1009 both vectors FAIL: best_actor_vec mean 523.0 (frac R 0.0); best_mean_vec mean 845.7 (frac R 0.007 < 0.03). Best held-out 845.7 is below C1 seed-0's 906.4 and below the 930.27 bar, same collapse signature. Per the screen rule (all seeds complete before NEGATIVE), staged seeds 1 and 2. Seed 1 launched detached (~66 min); chain_c2_seed2.py (detached waiter pid was 38816) auto-launches seed 2 on seed-1's sentinel, sequential so pools never contend.

**Next action:** When seeds 1 and 2 finish (check `runs\phase_n\c2_screen_s1\c2_screen.sentinel` and `..._s2\c2_screen.sentinel` = EXIT 0), run held-out eval on BOTH vectors for EACH seed with the unchanged gate: `.venv-c1\Scripts\python.exe tools\c1_es_eval.py --vec runs\phase_n\c2_screen_sN\best_{actor,mean}_vec.npy --label C2-sN-{actor,mean} --seeds 1000-1009 --out runs\phase_n\c2_screen_sN\eval_{actor,mean}`. Then write `docs\phase-n-c2-findings.md` (mirror c1 findings structure), call the C2 verdict (expected NEGATIVE, all seeds sub-bar; if any seed clears 930.27 AND passes diversity, that is a genuine result, verify hard before claiming). Update paradigm accounting: C1 spent, C2 spent, one shot remains (C3 TBD). If C2 NEGATIVE, identify C3 candidate (structurally distinct 3rd paradigm) or call Phase N FINAL NEGATIVE per the 3-paradigm stopping rule.

**Blockers:** None requiring Jeff. Seeds 1+2 run unattended via the proven windowless pattern. (Optional, unchanged: NSSM/Scheduled-Task reboot-recovery needs Jeff's Windows password; no-reboot overnight case already solved.)

**Notes:**

- C2 seed-0 verdict is FAIL on BOTH vectors, anchored to `runs\phase_n\c2_screen_s0\eval_{actor,mean}\c1_eval_summary.json` on disk. Diversity in the archive (166 cells, diverse fracs on TRAIN seeds) did NOT transfer to held-out competence: the eval-seed policies still starve a lateral action. Confidence HIGH that seed 0 failed; UNKNOWN on seeds 1/2 until evaluated; expectation LOW that either clears the bar (seed 0 missed by 9% with the collapse signature).
- C2 infra all committed and working: `tools\c2_mae_train.py` (CMA-MAE, objective = raw mean length, measures = 2D action-fraction simplex, 20x20 archive lr 0.01 threshold_min 0.0 ranker "imp"), `tools\run_c2_supervised.py`, `tools\launch_supervisor_c2_detached.py`, `tools\chain_c2_seed2.py`. ribs 0.11.0 in `.venv-c1` (numba 0.65.1 + llvmlite 0.47.0 cp314 wheels, no conflicts). Emits best_actor_vec.npy (archive.best_elite["solution"]) + best_mean_vec.npy (emitter._opt.mean); both feed the UNCHANGED c1_es_eval gate.
- Method-fails-twice watch: C1 (CMA-ES) and now C2 seed-0 (CMA-MAE) both fail the same way, lateral-action collapse on held-out despite train-seed competence. The recurring project finding holds: exploration/generalization is the decisive lever, and pure black-box weight search over this actor has not cracked held-out reliability. If C2 closes NEGATIVE, C3 should attack held-out generalization directly (candidate not yet chosen), not be a third weight-search variant.
- WINDOWLESS DURABLE EXECUTION PROVEN (C1 + C2 seed-0 both ran ~66 min clean, sentinel EXIT 0). Pattern: supervisor DETACHED_PROCESS (pool-less) + trainer CREATE_NO_WINDOW (hidden console = valid Godot pool handles) + kill_godot/status CREATE_NO_WINDOW + FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 + PYTHONNOUSERSITE=1. Do NOT relaunch any trainer with CREATE_NEW_CONSOLE.
- Eval gate unchanged: held-out 1000-1009, PASS = mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. Status server (if needed): `C:\Python314\python.exe tools\launch_status_detached.py`, http://localhost:8765.
