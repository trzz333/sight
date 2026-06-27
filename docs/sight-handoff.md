# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase M (from-scratch on-policy PPO, take 2). M2 baseline run COMPLETE and FAILED (0/3 seeds clear bar). Pivoting to M2.1 (VecNormalize).

**Last commit:** `1ba52c8` M2 toolchain: from-scratch PPO trainer + eval gate on Godot Signal Dodge.

**Current task:** M2 from-scratch PPO baseline is FINAL NEGATIVE (HIGH, `runs\phase_m\m2_eval\m2_eval_summary.json`). 3 seeds x 1e6 steps, gamma 0.999, ent_coef 0.01, reward none, n_envs 8. Greedy eval on seeds 1000-1009: verdict M2-FAIL, 0/3 clear the gate. s0 mean 472.0 with balanced bidirectional motion (L .22/stay .55/R .23, max_frac .55) but incompetent survival; s1 mean 663.1 near-collapsed right (R .945); s2 mean 579.0 collapsed to stay (.992). All below the 930.27 bar. Root cause is confirmed by training telemetry: explained_variance ~= 0 throughout (critic never fit the large high-gamma dense returns), value_loss ~310-324, so PPO had no usable advantage signal. ent_coef partially worked (held s0 bidirectional) but did not prevent s1/s2 degeneracy and could not buy competence. The wall is value-learning geometry, not the policy or entropy. Trainer/eval/driver are committed and proven (`1ba52c8`); smokes passed (dummy ep_len 451, subproc 278 steps/s, eval gate self-test rejected a constant-right collapse).

**Next action (M2.1):** Add `VecNormalize(norm_obs=True, norm_reward=True, gamma=0.999)` around the train vec env in `tools\m2_state_ppo_train.py` (the diagnosed fix: normalized returns bring value targets to ~unit scale so the critic can fit; explained_variance~=0 is the named defect). Save the normalize stats next to model.zip (`vecnormalize.pkl`); the eval must load them and roll with `training=False, norm_reward=False`. found-art: VecNormalize is the SB3 standard, ADOPT not build. Keep reward none, gamma 0.999, ent_coef 0.01 fixed so M2.1 isolates the normalization effect. Re-run the 3-seed batch (reuse `run_m2_multiseed.bat`), eval seeds 1000-1009 against the same gate, record M2.1 verdict. If M2.1 still plateaus below bar with explained_variance now healthy, the next lever is lowering gamma toward 0.997 and/or value-function return clipping; if explained_variance stays ~0 even normalized, the wall is the env/reward and that becomes the target.

**Blockers:** None requiring Jeff.

**Notes:**

- M2 confirmed the critic is the wall: explained_variance ~= 0 across all 3 seeds, mirroring the 150k-step early warning. The from-scratch policy never gets a usable advantage signal. M2.1 VecNormalize is the targeted fix; this is a structurally different change to the named defect, not retry-harder.
- Eval gate works as designed: it rejects single-action survival via `max(frac)<0.97` (proven on a constant-right smoke model that survived 1351.5 yet failed). Keep the gate; a real dodger uses stay + both directions.
- Reward-geometry key fact: under reward "none" (+1/surviving step), the best constant action caps at 845.7 (10-seed MEAN; per-seed varies widely), below the 930.27 bar, so survival reward provably forces dodging. Do not add reward shaping for M2.x; K5.5's shaped reward had a degenerate constant-action optimum.
- Interpreter note: M-phase from-scratch PPO runs in the GLOBAL interp (SB3 2.8.0, gymnasium 1.2.3, Godot). The `.venv-d3rlpy` offline stack is no longer on the active path; never import d3rlpy globally if revisited.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
