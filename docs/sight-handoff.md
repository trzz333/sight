# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase L (d3rlpy offline-RL). DiscreteCQL FINAL NEGATIVE. Value-RL thread CLOSED. Claimable findings doc committed.

**Last commit:** `49a7aa2` Phase L findings: offline-RL synthesis doc (CQL FAIL vs filtered-BC PASS).

**Current task:** Produced the claimable Phase L findings writeup as a committed evidence doc: `docs\phase-l-offline-rl-findings.md` (HIGH). Every number anchored to an on-disk artifact re-read this session, not memory: CQL 606.0 FAIL stay-collapse [0,1,0] and filtered-BC 1299.7 PASS [0.212,0.307,0.481] from `runs\phase_k\k7_offline\real\k7_eval_report.json`; from-scratch history K5.8 NoisyNet QR-DQN IQM 729.2 (best seed 980.7, 1/10 above bar) from `k5_8_reliability_report.json`, K6 aux-head OFF IQM 660.4 / ON IQM 626.1 from `k6_off/on_reliability_report.json`; imitation history BC 1737.3 / PPO-ft 1710.5 from the K5.6 docs. Doc structure ADOPTs the existing `-evidence.md` shape. Git verified clean and in sync at session start (HEAD was `fb61710`), substantive commit `49a7aa2`, then this chore refresh.

**Next action:** Value-RL is closed and the claim is now documented and committed. The open fork is Jeff-owned DIRECTION: does Phase L continue on the imitation/BC side (the only family that clears the bar), or does the project pivot to a new env/method? The substantive claim in hand: value-based RL (online from-scratch and offline conservative DiscreteCQL) collapses to the modal action on this CartPole-tier env where a competent policy provably exists, while imitation on competent/filtered demonstrations clears the bar every time; the decisive lever is the quality of the action signal, not the RL algorithm's conservative/exploration knobs. Await Jeff direction before opening a new thread. Do not open new training scope unprompted.

**Blockers:** Continue-vs-pivot is a Jeff-owned direction call (not a technical blocker). Nothing else blocking.

**Notes:**

- Interpreter split load-bearing: collection + in-env eval run in the GLOBAL interp (SB3, gymnasium 1.2.3, Godot); d3rlpy training runs only in `.venv-d3rlpy` via subprocess; policy crosses back as TorchScript. Never import d3rlpy globally.
- scipy 1.18.0 confirmed resolving from `.venv-d3rlpy\Lib\site-packages` even under a stripped env (APPDATA/USERPROFILE blanked, PYTHONNOUSERSITE=1). WMI-detached ModuleNotFoundError blocker is fixed; keep scipy in the venv, never user-site.
- DiscreteCQL collapse is structural, not a tuning artifact: equals a constant-Stay policy at both alpha=1.0/n_critics=1 and alpha=0.5/n_critics=3. Discrete-CQL penalty == BC-NLL toward the modal dataset action (Stay), and the full mixed set is Stay-dominant. Do not reopen with more CQL knobs. Working lever for any continuation is trajectory filtering / data quality (filtered-BC PASS proves it on the same npz).
- Findings doc records the K5.8 single-seed peak (980.7) as a single seed only; the from-scratch headline is the across-seed reliability (IQM 729.2, 1/10 above bar). Never headline single-seed peaks.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
