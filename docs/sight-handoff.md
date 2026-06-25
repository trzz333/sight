# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K. From-scratch N=10 reliability verdict FINAL (negative). K6 self-supervised on/off comparison COMPLETE and FINAL (negative).

**Last commit:** `45941fd` K6 self-supervised on/off comparison FINAL (negative); sequential WMI-detached finisher. (Reliability numbers below live in prose because `runs/` is gitignored.)

**Current task:** K6 on-vs-off comparison is DONE. The sequential finisher (`tools\run_k6_finish.bat`, launched detached via WMI Win32_Process.Create) ran ~14.5h and completed cleanly, surviving the full run with no cascade-kill. All 10 runs (5 seeds x 2 arms, 200k) have real eval reports (10 episodes each). Reliability comparison computed per arm via `tools\k5_8_reliability_report.py`.

K6 VERDICT (FINAL, negative): the self-supervised next-state-prediction auxiliary does NOT lift from-scratch reliability.
- OFF (dyn_beta 0.0, baseline): IQM 660.4, 95% CI [516.9, 1019.0]; mean 719.5; median 606.0; frac_above_bar 0.20 (1/5, off_s2=1108.5 but degenerate); frac_nondegenerate 0.20.
- ON (dyn_beta 1.0, self-sup): IQM 626.1, 95% CI [570.4, 793.7]; mean 661.3; median 630.0; frac_above_bar 0.00 (0/5); frac_nondegenerate 0.40.
- IQM point estimate slightly favors the BASELINE; CIs overlap heavily (n=5/arm). Both arms' CIs sit entirely below bar 930.27. Neither arm produced a single PASS (above-bar AND nondegenerate). The auxiliary buys no dependability at 200k; reinforces the N=10 from-scratch negative. Reports: runs\phase_k\k6_off_reliability_report.json, k6_on_reliability_report.json.

**Next action:** (1) Begin the d3rlpy offline-RL pivot (DiscreteCQL/IQL/BCQ on fixed datasets); stop hand-rolling RL. (2) Found-art fix for reliability stats: replace the hand-rolled boot_ci loop in `k5_8_reliability_report.py` with `scipy.stats.bootstrap` (zero new deps; scipy already installed); RE-CHECK whether arch 8.0.0 has a Windows cp314 wheel so rliable can be adopted wholesale (arch shipped cp314 Linux/macOS wheels 2025-10-21).

**Blockers:** None requiring Jeff. (Desktop Commander MCP went unresponsive mid-session for ~8 min then recovered; commit completed after recovery.)

**Notes:**

- HANDOFF-WAS-WRONG, reconfirmed: prior handoff claimed "NO eval reports for any K6 seed" and prescribed re-running all 10. On-disk evidence falsified it: off_s0/s1/s2 were fully complete (train+eval), on_s0/s1/s2 trained to 200k but eval had failed on a Godot transport timeout under arm-concurrency, only s3 was partial. Verify handoff claims against artifacts before acting. Each K6 training is ~3.5-4.1h wall; re-running good seeds would have burned ~24h for zero evidential gain.
- METHOD CHANGED and VALIDATED: concurrent two-arm detached-bat (failed twice: external cascade-kill + eval transport timeout under contention) replaced with a sequential single-headless-Godot finisher launched via WMI Win32_Process.Create. Survived the full ~14.5h detached run. Sequential execution fixed the on-arm eval timeouts (contention, not a model bug). Use this pattern for future long detached runs.
- FOUND-ART forward rule: reliability stats (IQM, bootstrap CI, perf profiles) are an off-the-shelf problem (`scipy.stats.bootstrap` + `trim_mean`; or rliable). Do not hand-roll statistical machinery; search first. arch cp314 wheels now exist (Linux/macOS 2025-10-21); confirm a Windows wheel before assuming rliable is uninstallable.
- AU guard `NoAutoRebootWithLoggedOnUsers` set to 1 via gsudo this session (verified 0x1) to protect the detached run. Machine rebooted 06/24 17:53 (staged update applied). Revert the AU key via gsudo when no long run is pending or before the next planned reboot. Claude handles this elevation directly; NOT a Jeff action.
- BC 1737.3 and PPO-finetune 1710.5 still MEDIUM, not re-verified. Re-confirm from their eval artifacts before any external use. Portfolio framing: two clean honest-negative results now (N=10 from-scratch; K6 auxiliary) alongside BC as the reliable above-bar policy.

---
