# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.7 OPEN. From-scratch reinforcement learning must itself produce a Signal Dodge policy above the 930.27 constant-action baseline. DQN method (PPO rejected K5.1-K5.5; PPO finetune rejected K5.6). Single-voice governance.

**Last commit:** `483a0f7` K5.7 DQN: wrap env in VecMonitor so ep_rew_mean/ep_len_mean log

**Current task:** From-scratch DQN loop validated end-to-end under `python -u`. Found and fixed a real bug (483a0f7): the env was never wrapped in a Monitor, so SB3's ep_info_buffer stayed empty and the entire training curve logged NaN for ep_rew_mean/ep_len_mean. Episodes still terminated and counted via the dones array, so training itself was unaffected, but the run was unauditable. Fix is VecMonitor(venv) innermost under VecNormalize(norm_reward=False) so return stays in episode-length units. Confirmed on the 8k smoke: curve now populates, final ep_len_mean 410.0 in `runs\phase_k\k5_7_dqn_smoke\train_metrics.ndjson` (was NaN pre-fix). The full 200k DQN run is LIVE and detached as of this handoff (`runs\phase_k\run_k5_7_dqn_full.bat`, log/sentinel siblings under `runs\phase_k\`). Last observed: 65,811 timesteps, exploration_rate floored at 0.05 (decay completed at 60k as scheduled), loss finite, sentinel absent (RUNNING). ep_len_mean has sat ~420-470 since 22k steps and did NOT rise as epsilon decayed to the floor: MEDIUM evidence the vanilla DQN config is not learning to extend survival (below best-constant 845.7, well below the 930.27 bar). This is training-episode length with 5% residual exploration, NOT the held-out greedy eval, so it is suggestive, not the verdict. 50k checkpoint exists (`runs\phase_k\k5_7_dqn\ckpts\dqn_50000.zip` + `vecnorm_50000.pkl`). Eval tool `tools\k5_7_dqn_eval_inenv.py` verified correct this session: loads model + frozen VecNormalize stats, greedy argmax on seeds 1000-1009, PASS gate requires mean >= 930.27 AND non-degenerate (frac_L >= 0.03 and frac_R >= 0.03 and max(frac) < 0.97), so a constant-action policy that happens to score high is correctly FAILed.

**Next action:** Check the run: if `runs\phase_k\k5_7_dqn_full.sentinel` exists, run the in-env eval on the final model (set SIGHT_GODOT_EXE inline, then `python tools\k5_7_dqn_eval_inenv.py --run runs\phase_k\k5_7_dqn --seeds 1000-1009`). If still RUNNING, eval the 50k checkpoint first for an early read: copy `ckpts\dqn_50000.zip` -> a scratch dir as `dqn_sb3.zip` and `ckpts\vecnorm_50000.pkl` -> `vecnormalize.pkl`, then eval that scratch dir. PASS only if mean >= 930.27 AND non-degenerate. If sub-bar, escalate per Notes (Double DQN / n-step / QR-DQN); do not retry vanilla DQN harder.

**Blockers:** None requiring Jeff. (Repo `github.com/trzz333/sight` still PUBLIC; Jeff decided acceptable.)

**Notes:**

- found-art correction this session: `reward_shaping "none"` is a DENSE reward (+1 per surviving step), not sparse. HER is the wrong escalation lever (it targets sparse goal-conditioned rewards) and is demoted. The real difficulty is delayed-consequence credit assignment (the fatal move precedes the collision by several frames).
- Escalation levers if the eval confirms vanilla DQN sub-bar, in order (found-art ADAPT; searched DQN overestimation + multi-step plateau): Double DQN / Maxmin to cut overestimation, then n-step returns / sb3-contrib QR-DQN (distributional + overestimation control), then CleanRL Rainbow-lite (charter permits CleanRL). No BUILD.
- Always launch training with `python -u`; redirected stdout block-buffers otherwise and the run looks wedged (flat CPU, empty log). Detached pattern is .bat + done-sentinel + poll; keep each poll wait under ~3 min (a 200s+overhead poll tripped the 4-min MCP wall this session).
- `reward_shaping "none"` is load-bearing: return = episode length, which has no constant-action optimum (best constant 845.7 < bar 930.27). Do not reintroduce shaping unless DQN measurably collapses first.
- Success is unambiguous and must not be relabeled: from-scratch DQN clears 930.27 in-env on seeds 1000-1009 with a non-degenerate policy. Matching BC (1737.3) or a high-scoring constant-action policy is a FAIL. `runs/` is gitignored; only the `k5_7_dqn_*` tools are tracked.
