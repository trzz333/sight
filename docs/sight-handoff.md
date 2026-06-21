# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.7 OPEN. From-scratch RL must itself beat the 930.27 constant-action baseline. Vanilla DQN rejected (sub-baseline). QR-DQN + n-step (sb3-contrib) is the active escalation: 50k checkpoint FAILs (degenerate), 200k final eval still pending. PPO rejected K5.1-K5.5; PPO finetune rejected K5.6. Single-voice governance.

**Last commit:** `<PENDING-SUBSTANTIVE>`

**Current task:** QR-DQN smoke test PASSED (8000 steps, exit 0, train_metrics.ndjson populates finite ep_len_mean once episodes complete; loss finite past learning_starts). Full 200k launched detached via runs\phase_k\run_qrdqn_200k.bat (python -u, --out runs\phase_k\k5_7_qrdqn) and STILL RUNNING at session close: sentinel runs\phase_k\k5_7_qrdqn_200k.sentinel not yet written, last observed ~61k steps, fps drifted 58->28 under CPU load, loss climbing monotonically 11->25->113->169->459 (watch flag for value divergence; the in-env greedy eval is the verdict, not loss). The 50k checkpoint was staged to runs\phase_k\k5_7_qrdqn_ckpt50k and greedy-evaled early on held-out seeds 1000-1009: FAIL. Mean episode length 736.3 (bar 930.27, delta -193.97; below best-constant 845.7 by -109.4), 9/10 die by collision, pooled action fractions L/S/R 0.832/0.168/0.000, nondegenerate false because frac_R = 0.000. Anchored to runs\phase_k\k5_7_qrdqn_ckpt50k\eval_inenv\dqn_eval_inenv_report.json. This is the MIRROR of vanilla DQN's right-collapse (frac_L approx 0): distributional value + n-step flipped the collapse direction, did not remove it, evidence the failure is directional/exploration collapse, not the value-overestimation or credit-assignment that QR-DQN+n-step target.

**Next action:** On resume, check runs\phase_k\k5_7_qrdqn_200k.sentinel. If DONE, eval the final: set SIGHT_GODOT_EXE inline, then `python tools\k5_7_qrdqn_eval_inenv.py --run runs\phase_k\k5_7_qrdqn --seeds 1000-1009`. PASS only if mean >= 930.27 AND non-degenerate (frac_L >= 0.03, frac_R >= 0.03, max(frac) < 0.97). Given the 50k checkpoint was degenerate (frac_R 0.000) and loss was climbing, expect the final likely also degenerate or sub-bar. If the final FAILs, escalate to CleanRL Rainbow-lite (charter permits CleanRL) but configure it against the ACTUAL failure (directional/exploration collapse, not value-estimation): sustained exploration (longer exploration_fraction or higher final_eps) and/or inspect the env for a left/right symmetry that makes one lateral direction a basin, on top of dueling + PER + distributional + n-step. Do not retry QR-DQN harder. If the sentinel is not yet present, poll it in short lean checks.

**Blockers:** None requiring Jeff. (Repo `github.com/trzz333/sight` still PUBLIC; Jeff decided acceptable.)

**Notes:**

- The 200k QR-DQN run saturates the i7-10750H; MCP shell calls with in-call sleeps over ~60s time out past the 4-min watchdog. Poll with sleeps <= 45s or instant no-sleep checks. The run is detached and survives regardless of poll timeouts.
- Directional collapse is now the cross-method pattern in this env+pipeline: PPO -> constant action (K5.1-5.5); vanilla DQN -> right-only (frac_L approx 0, mean 695.7); QR-DQN 50k -> left-only (frac_R 0.000, mean 736.3). The wall is a single-direction basin, to be attacked as an exploration/symmetry problem, not a value problem.
- The only above-baseline policies remain BC (1737.3) and PPO-finetune-from-BC (1710.5), both warm-started. The honest from-scratch-RL demo still does not exist.
- reward_shaping "none" stays load-bearing: return = episode length, best constant 845.7 < bar 930.27. Do not reintroduce shaping unless a method measurably collapses first.
- runs/ is gitignored; QR-DQN run/eval/ckpt artifacts and the .bat/.log/.sentinel files are not committed. Only the tracked k5_7_* tools and this handoff persist.
