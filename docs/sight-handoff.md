# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase M (from-scratch on-policy PPO). M2 and M2.1 both FINAL NEGATIVE. Migrated StrongerJr -> NLDC (hostname MSI) this session.

**Last commit:** `0e0d9a9` Phase M findings: M2.1 FINAL NEGATIVE (from-scratch PPO + VecNormalize).

**Current task:** M2.1 (PPO + VecNormalize) is FINAL NEGATIVE (HIGH, `runs\phase_m\m2_1_eval3\m2_eval_summary.json` + `m2_1_s{0,1,2}\m2_train_report.json`, all confirmed this session). Clean 3-seed re-run on NLDC, all clean (learn_error null). VecNormalize fixed the M2 value-fitting defect: explained_variance 0.933/0.939/0.854 (was ~0), value_loss ~0.02 (was ~311). But the policy did not clear the bar: greedy eval over held-out seeds 1000-1009 gave means 417.3/696.8/778.1, verdict M2.1-FAIL, 0/3. Action distributions diverse, not collapsed (e.g. s2 L .35/stay .34/R .31). The policy can survive full episodes occasionally (s2: 3/10 eval seeds hit the 1800 cap) but not reliably. Net: the failure mode moved from "critic cannot fit returns" (M2) to "policy converges to high-variance sub-baseline behavior" (M2.1). From-scratch on-policy PPO has now failed twice in two distinct ways. Findings written to `docs\phase-m-from-scratch-ppo-findings.md`.

**Next action:** Compute IQM + 95% stratified-bootstrap CI (Agarwal et al. 2021) over the M2.1 eval lengths in `runs\phase_m\m2_1_eval3\m2_eval_summary.json` and append the aggregate statistic to `docs\phase-m-from-scratch-ppo-findings.md`, then commit. This finalizes the M2.1 statistical record independent of the direction call below. Do not launch M2.2 / more PPO knob-twiddling; the method failed twice.

**Blockers:** Jeff direction call. From-scratch on-policy PPO failed twice (M2 critic-broken, M2.1 critic-fixed-but-sub-baseline). Decide between (a) accept the 3-seed negative as the portfolio finding and close Phase M, (b) run the N=10 reliability sweep (~6.5h compute) for a CI-lower-bound claim, or (c) attempt a structurally different from-scratch method. Direction/scope = Jeff. Everything below that line is Claude's.

**Notes:**

- Migration StrongerJr -> NLDC (hostname MSI, user maste) complete. Python now at `C:\Python314`; packages installed SYSTEM-WIDE into `C:\Python314\Lib\site-packages` (a --user-only install is invisible to WMI-detached training, no APPDATA in that minimal env, crashed every seed at `import numpy`; system-wide is the durable fix). Godot 4.6.2-stable at the winget path (matches scripts' DEFAULT_EXE). Bat path fixes in `b19cf06`.
- M2.1 critic fix worked but bought no competence: explained_variance 0 -> ~0.92 across seeds, yet eval still 0/3 below the 930.27 bar. Fixing the diagnosed defect did not lift the policy. Do not retry PPO harder.
- Portfolio context for the direction call: imitation clears the bar reliably and wide (BC 1737.3, PPO-finetune-from-BC 1710.5). The unsolved problem is from-scratch RL reliability, not imitation. Eval gate rejects single-action survival via `max(frac)<0.97` (proven). Reward "none" best-constant 845.7 < bar forces dodging; do not add reward shaping for M2.x.
- M-phase from-scratch PPO runs the GLOBAL interp (SB3 2.8.0, gymnasium 1.2.3, torch 2.11.0, numpy 2.4.4). `.venv-d3rlpy` offline stack is off the active path; never import d3rlpy globally if revisited. `requirements-lock.txt` reproduces the env (`pip install -r`, strip the self-referential `sight_agent` git line).
- noautoreboot: LEAVE set (Jeff, this session). The Dell/StrongerJr keeps it for legal-corpus access; the old "revert before reboot" instruction is retired. Not a Sight concern post-migration.

---
