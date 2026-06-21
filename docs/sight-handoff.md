# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.7 OPEN. From-scratch reinforcement learning must itself produce a Signal Dodge policy above the 930.27 constant-action baseline. Vanilla DQN rejected (K5.7 first method, sub-baseline). QR-DQN + n-step (sb3-contrib) is the active escalation. PPO rejected K5.1-K5.5; PPO finetune rejected K5.6. Single-voice governance.

**Last commit:** `PLACEHOLDER` K5.7 QR-DQN: sb3-contrib trainer + eval (n-step + distributional), untested in-env

**Current task:** Vanilla DQN (K5.7 first method) FAILed and was stopped. The 50k checkpoint greedy eval on held-out seeds 1000-1009 returned mean episode length 695.7 (bar 930.27, delta -234.57; below best-constant 845.7 by 150), 9/10 episodes die by collision, action fractions L/S/R 0.052/0.05/0.899: a right-biased evader that is past the degeneracy floor but scores worse than a dumb constant action, so its learned Q-values steer it wrong (mis-trained, not merely under-trained). Anchored to runs\phase_k\k5_7_dqn_ckpt50k_eval\eval_inenv\dqn_eval_inenv_report.json. The live 200k run was killed at ~80k: training ep_len_mean sat ~420-470 from 22k through 80k with exploration floored since 60k and nothing structural left to change before 200k, so finishing it was the forbidden "retry harder". The kill returned control to the .bat which then wrote a misleading DONE-1 sentinel; that sentinel was deleted (a kill is not a clean completion). Escalation chosen (found-art ADOPT, searched the SB3 ecosystem): sb3-contrib QR-DQN, a true drop-in on the existing SB3 plumbing whose distributional target attacks the misleading-Q failure, and whose native n_steps param lets one run also attack the delayed-consequence credit-assignment hypothesis. sb3-contrib 2.8.0 installed (SB3 2.8.0 untouched). New tools tools\k5_7_qrdqn_train.py and tools\k5_7_qrdqn_eval_inenv.py written by adapting the verified DQN tools (QRDQN, n_step 3, n_quantiles 200, lr 2.3e-4; env/vecnorm/eval plumbing reused). Both py_compile clean. NOT smoke-tested in-env, NOT run. UNKNOWN whether QR-DQN clears the bar.

**Next action:** Smoke-test the QR-DQN trainer before any long run: set SIGHT_GODOT_EXE inline, then `python -u tools\k5_7_qrdqn_train.py --timesteps 8000 --out runs\phase_k\k5_7_qrdqn_smoke` and confirm runs\phase_k\k5_7_qrdqn_smoke\train_metrics.ndjson populates with finite ep_len_mean (no NaN). If clean, launch the full 200k detached (.bat + done-sentinel + poll, `python -u`, --out runs\phase_k\k5_7_qrdqn) and poll under ~3 min per wait. Eval the 50k ckpt early then the final: `python tools\k5_7_qrdqn_eval_inenv.py --run runs\phase_k\k5_7_qrdqn --seeds 1000-1009`. PASS only if mean >= 930.27 AND non-degenerate (frac_L >= 0.03, frac_R >= 0.03, max(frac) < 0.97). If QR-DQN is also sub-bar, escalate to CleanRL Rainbow-lite (n-step + distributional + dueling + PER together; charter permits CleanRL); do not retry QR-DQN harder.

**Blockers:** None requiring Jeff. (Repo `github.com/trzz333/sight` still PUBLIC; Jeff decided acceptable.)

**Notes:**

- The only above-baseline policy that exists is BC (1737.3) and the PPO-finetune-from-BC (1710.5), both warm-started, NOT from-scratch RL. The honest from-scratch-RL demo does not exist yet: PPO failed K5.1-K5.5, PPO finetune K5.6 only preserved BC, vanilla DQN K5.7 is sub-baseline. A demo claiming "a learned ML policy beats the baseline" is honest today via BC; a demo claiming "from-scratch RL beats the baseline" is not yet earned.
- Vanilla DQN greedy (695.7) scoring BELOW best-constant (845.7) is the diagnostic: mis-trained, not under-trained. Distributional value (QR-DQN) plus n-step target this directly; Double DQN (overestimation only) was deferred as off-target for this failure.
- reward_shaping "none" stays load-bearing: return = episode length, no constant-action optimum (best constant 845.7 < bar 930.27). Do not reintroduce shaping unless a method measurably collapses first.
- Detached-pattern caveat learned this session: killing the python hands control back to the .bat, which writes its DONE sentinel. Sentinel presence is NOT proof of clean completion after a manual kill. Trust the eval and train_metrics, not just the sentinel.
- runs/ is gitignored; only the tracked k5_7_* tools persist. QR-DQN run artifacts will not be committed.
