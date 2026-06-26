# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase M (from-scratch on-policy PPO, take 2). Offline value-RL (Phase L) CLOSED. Harness control PASS. Signal Dodge training not yet run.

**Last commit:** `5d2d6ca` Phase M open: found-art pivot to on-policy PPO + CartPole harness control PASS.

**Current task:** Ran found-art on the from-scratch wall and pivoted (HIGH, `docs\phase-m-found-art-and-control-evidence.md`). Two verified anchors: (1) offline value-RL stays closed because Kumar et al. 2022 (arXiv 2204.05618) says offline RL cannot beat BC on near-expert, dense-reward, no-stitching data, which is the Signal Dodge regime, so K7 was expected; (2) prior state-PPO collapse (K5.5) had four named deficiencies (shaped reward, 10k budget, single env, no control), and its own routing blamed reward geometry. M1 infra-loop audit ran the RL-Zoo tuned CartPole-v1 PPO config on SB3 on StrongerJr: mean reward 500.0, std 0.0, PASS (`runs\phase_m\m1_cartpole_control_report.json`, exit 0). Harness is proven known-good; prior Sight RL collapses are localized to the Signal Dodge env/reward wiring, not algorithm or install.

**Next action (M2):** Point the same SB3 PPO trainer (`tools\m1_cartpole_ppo_control.py` is the template) at state-obs `GodotSignalDodgeEnv` with reward_shaping "none", reference-grade budget + vectorized envs, gamma raised toward 0.99+ (episodes run to 1800; CartPole's 0.98 horizon is likely too short), and a nonzero entropy coefficient as the anti-collapse lever. Requires `SIGHT_GODOT_EXE` set. Multi-seed, eval greedy on held-out seeds 1000-1009 vs bar 930.27. PASS = mean >= 930.27 AND frac_L >= 0.03 AND frac_R >= 0.03 AND max(frac) < 0.97. Diagnostic value either way: if Signal Dodge collapses where CartPole solved, the bug is definitively in the env/reward, which becomes the next target.

**Blockers:** None technical. Continue-vs-pivot was resolved by Jeff ("try a different approach"); Phase M is the approved direction.

**Notes:**

- Reward-geometry key fact: under reward "none" (+1/surviving step), the best constant action caps at 845.7 (`constant_left`, K5.2), below the 930.27 bar, so survival reward provably forces dodging. K5.5's shaped reward (alpha 0.30) had a degenerate constant-action optimum; do not use shaping for M2.
- Offline value-RL is closed for theory reasons, not just empirics. Do not reopen DiscreteCQL or add CQL knobs. The Phase L synthesis is `docs\phase-l-offline-rl-findings.md`.
- Interpreter note: M-phase from-scratch PPO runs in the GLOBAL interp (SB3 2.8.0, gymnasium 1.2.3, Godot). The `.venv-d3rlpy` offline stack is no longer on the active path; never import d3rlpy globally if revisited.
- BC (1737.3) and PPO-finetune-from-BC (1710.5) remain the only policies that clear the bar; both are imitation-derived. Phase M is the attempt to get a genuinely from-scratch RL policy above the bar reliably.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
