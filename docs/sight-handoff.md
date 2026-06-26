# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase M (from-scratch on-policy PPO, take 2). M2 trainer + eval gate built and smoke-passed on the real Godot env. 3-seed real run IN FLIGHT.

**Last commit:** `1ba52c8` M2 toolchain: from-scratch PPO trainer + eval gate on Godot Signal Dodge.

**Current task:** M2 wired end-to-end and proven (HIGH). `tools\m2_state_ppo_train.py` runs SB3 PPO MlpPolicy on state-obs `GodotSignalDodgeEnv`, reward_shaping none, gamma 0.999, ent_coef 0.01, standard PPO geometry (n_steps 512, gae 0.95, n_epochs 10, lr 3e-4, clip 0.2). Vectorizes Godot directly (one subprocess + one TCP port per env), bypassing the factory's H3-era n_envs=1 cap. Smokes: DummyVecEnv n_envs=1 exit 0 ep_len 451 reward-none confirmed; SubprocVecEnv n_envs=8 exit 0 278 steps/s under Windows spawn. Eval `tools\m2_state_ppo_eval.py` self-tested: gate correctly FAILED the smoke model's constant-right collapse (survived 1351.5, max_frac 1.0 >= 0.97). Detached 3-seed batch launched via WMI (`tools\run_m2_multiseed.bat`, PID 34496 at launch): seeds 0/1/2 sequential, 1e6 steps each, n_envs 8, into `runs\phase_m\m2_s0|s1|s2`. At last check seed 0 was at 151k/1e6, fps 390 (~43 min/seed, ~2.1h total).

**Next action (M2 verdict):** When `runs\phase_m\m2_multiseed.done` exists (and each `m2_s{0,1,2}.sentinel` reads EXIT 0), run `tools\m2_state_ppo_eval.py --runs s0=runs\phase_m\m2_s0,s1=runs\phase_m\m2_s1,s2=runs\phase_m\m2_s2 --seeds 1000-1009 --out runs\phase_m\m2_eval`. Read `m2_eval_summary.json`; verdict = how many of 3 train seeds clear the full gate (mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97). Record the M2 verdict, commit + push + refresh this handoff. If a seed sentinel reads nonzero or the run is still going, read `runs\phase_m\m2_s{N}\train.log` tail before acting.

**Blockers:** None. Batch runs detached; just needs time (~2h from 11:16 launch) and an eval pass.

**Notes:**

- M2 critic-health watch-item (MEDIUM): at 150k steps seed 0 showed explained_variance ~= 0, value_loss ~311, clip_fraction 0, approx_kl ~0.001 (policy barely moving, ep_len creeping 451->496). Classic high-gamma + large-magnitude dense-return signature. If the 3-seed batch plateaus below bar, the prepared next lever is SB3 `VecNormalize` (normalize obs + returns) wrapping the vec env, which directly targets the unfittable critic. found-art: VecNormalize is the SB3 standard, ADOPT not build. Do not interrupt the in-flight run to apply it; measure against the completed baseline first.
- Constant-action nuance: the 845.7 cap below is a 10-seed MEAN; per-seed survival under a constant action varies widely (eval self-test saw constant-right survive 903 and 1800 on seeds 1000/1001, mean 1351.5). This does not break the bar logic; the M2 gate's `max(frac)<0.97` clause is what rejects single-direction survival regardless of how high it scores.

- Reward-geometry key fact: under reward "none" (+1/surviving step), the best constant action caps at 845.7 (`constant_left`, K5.2), below the 930.27 bar, so survival reward provably forces dodging. K5.5's shaped reward (alpha 0.30) had a degenerate constant-action optimum; do not use shaping for M2.
- Offline value-RL is closed for theory reasons, not just empirics. Do not reopen DiscreteCQL or add CQL knobs. The Phase L synthesis is `docs\phase-l-offline-rl-findings.md`.
- Interpreter note: M-phase from-scratch PPO runs in the GLOBAL interp (SB3 2.8.0, gymnasium 1.2.3, Godot). The `.venv-d3rlpy` offline stack is no longer on the active path; never import d3rlpy globally if revisited.
- BC (1737.3) and PPO-finetune-from-BC (1710.5) remain the only policies that clear the bar; both are imitation-derived. Phase M is the attempt to get a genuinely from-scratch RL policy above the bar reliably.
- AU key `NoAutoRebootWithLoggedOnUsers` = 1 still SET. Revert via gsudo before the next reboot. Claude handles this elevation; NOT a Jeff action.

---
