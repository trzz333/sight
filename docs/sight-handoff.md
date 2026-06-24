# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K. From-scratch N=10 reliability verdict FINAL (negative). K6 self-supervised on/off comparison sweep INTERRUPTED at seed 3.

**Last commit:** `580a18b` K6 trainer+eval+multiseed: DynQRDQN on/off (dyn_beta 1.0 vs 0.0), seeds 0-4, 200k, pipeline smoke-verified

**Current task:** The N=10 from-scratch reliability sweep (K5.8 NoisyNet QR-DQN, seeds 0-9, 200k) is COMPLETE and the dependability verdict is settled negative. Re-run this session via `tools\k5_8_reliability_report.py`: IQM 729.2, 95% bootstrap CI [607.9, 835.8], bar 930.27, frac_above_bar 0.10 (1 of 10), frac_nondegenerate 0.50, MAX 980.7, MIN 476.5, MEDIAN 715.4. The entire CI sits below the bar, so from-scratch RL does not dependably clear the constant-action baseline at the 200k budget; only the single best seed cleared it. Imitation (BC 1737.3) and PPO-finetune (1710.5) remain the reliable above-bar policies (both carried from prior sessions, NOT re-verified this session, treat as MEDIUM). Separately, the K6 self-supervised on/off comparison (DynQRDQN dyn_beta=1.0 vs dyn_beta=0.0, seeds 0-4, 200k, two arms concurrent) ran overnight and was externally terminated about 10 minutes before this handoff, mid seed-3. Both arms completed seeds 0, 1, 2 to 200000 steps; off_s3 stopped at 141144 and on_s3 at 144308; seed 4 never started; NO eval reports were generated for any K6 seed (eval is the post-training stage and was not reached). Train logs end cleanly with no traceback, uptime is 63.3h with zero shutdown events, so the kill was neither a reboot nor a code crash. Exact cause UNKNOWN.

**Next action:** Relaunch the K6 dyn on/off multiseed sweep clean (`tools\run_k6_dyn_multiseed.bat`, seeds 0-4 both arms, 200k), discarding the partial/mixed s3 dirs. Harden detachment so a closing parent shell cannot cascade-kill it (launch via `start ""` detached or a scheduled task, write a per-seed done-sentinel, lean-poll <=45s). Confirm no python is running before launch. When all 10 runs finish, eval each through the existing eval + reliability machinery and compare frac_above_bar and IQM on vs off to test whether the next-state-prediction auxiliary lifts from-scratch reliability.

**Blockers:** None requiring Jeff. The N=10 portfolio-framing question is resolved: Jeff adopted the honest-negative-result headline this session.

**Notes:**

- K6 sweep died mid seed-3 (~10 min before handoff), both arms together. External kill, NOT a reboot (uptime 63.3h, no shutdown events 6/21-6/23) and NOT a code crash (clean SB3 logs, no traceback). Cause UNKNOWN; the detached `.bat` pattern did not survive whatever happened. Discard partial s3 dirs and harden detachment on relaunch. Method has now failed once this way; if a hardened relaunch dies the same way, change the launch method, do not retry it harder.
- N=10 verdict FINAL and negative: IQM 729.2, CI [607.9, 835.8] entirely below bar 930.27, 1/10 above bar, frac_nondegenerate 0.50. Honest-negative result is the adopted portfolio headline. Resume-bullet language fix flagged: "fixed exploration collapse" overstates given 50% of seeds still degenerate; scope to the controlled K5.7-vs-K5.8 comparison or soften to "substantially reduced."
- BC 1737.3 and PPO-finetune 1710.5 are carried from prior sessions, NOT re-verified this session (MEDIUM). Re-confirm from their eval artifacts before either number goes external.
- The K6 reliability comparison (does the dynamics-prediction auxiliary lift from-scratch reliability) is still UNKNOWN. No K6 eval report exists. The pillar's central question stays untested until a clean sweep + eval completes.
- Operational, off-charter but load-bearing for the next overnight run: a Windows update reboot is staged (pending). Auto-reboot was deferred this session (`NoAutoRebootWithLoggedOnUsers=1`, confirmed set under HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU). Jeff reboots manually after training. Relaunched runs have NO auto-resume, so a reboot mid-run loses progress. Revert the AU key after the planned reboot.

---
