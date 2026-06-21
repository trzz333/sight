# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.7 OPEN. Goal hardened: reinforcement learning must ITSELF produce a Signal Dodge policy above the 930.27 constant-action baseline. The K5.6 "literal RL PASS" (PPO finetune, mean 1710.5) is REJECTED as contrived. Single-voice governance.

**Last commit:** `2177e14` K5.7 from-scratch DQN: trainer + eval written (untested end-to-end)

**Current task:** Method changed from PPO to DQN. PPO cold-start collapsed to constant-action across K5.1-K5.5 (verified: `docs\k5-5-state-observation-control-evidence.md`, all 3 state seeds degenerate, but confounded by 10k budget and shaped reward alpha030 whose argmax is a constant action). K5.6 PPO finetune rejected (verified: `docs\k5-6-ppo-finetune-evidence.md`, clip_fraction ~0.008, mean 1710.5 <= BC 1737.3, RL added zero competence). Wrote `tools\k5_7_dqn_train.py` (SB3 DQN, MlpPolicy [128,128], state mode, reward_shaping "none" so return = episode length with no constant-action optimum since best constant 845.7 < bar 930.27, VecNormalize obs-only for self-contained from-scratch stats, epsilon-greedy) and `tools\k5_7_dqn_eval_inenv.py` (greedy in-env eval seeds 1000-1009, same raw-env loop as the BC eval for number parity vs 930.27 / 845.7 / 1737.3). found-art ADOPT verified this session: policy collapse is a named failure in both PG and value methods, DQN documented more robust than PPO and more sample-efficient on discrete low-dim control (fits the single-Godot-env ~55 fps, charter n_envs=1, throughput ceiling). NOT YET VALIDATED: DQN.learn() has not completed once. Env stepping is verified (raw VecEnv steps clean, ~10-24 ms/step) and SB3 2.8.0 DQN imports, but the training loop is unrun. No RL has learned anything this session; there is no result yet.

**Next action:** Relaunch the DQN smoke with `python -u` (unbuffered) so SB3 output flushes: `set SIGHT_GODOT_EXE=...` then `python tools\k5_7_dqn_train.py --out runs\phase_k\k5_7_dqn_smoke --timesteps 8000 --learning-starts 2000 --ckpt-every 999999`. Confirm loss finite, exploration_rate decays, ep_rew_mean rises off the floor. Then launch the full run detached (.bat + sentinel + poll), `--timesteps 200000`, and run `tools\k5_7_dqn_eval_inenv.py --run runs\phase_k\k5_7_dqn`. Report the in-env mean and action distribution honestly: PASS only if mean >= 930.27 AND non-degenerate (not constant-action).

**Blockers:** None requiring Jeff. (Repo `github.com/trzz333/sight` still PUBLIC; Jeff decided acceptable. Flipping private is Jeff's manual GitHub action if ever wanted.)

**Notes:**

- The earlier detached smoke logged EMPTY and looked wedged (flat CPU). Cause: Python block-buffers stdout when redirected to a file, so `verbose=1` never flushed; the process was killed before any rollout. Not an env bug. `_probe_step.py` proved the env steps cleanly. Always launch training with `python -u`.
- `reward_shaping "none"` is load-bearing, not a default: it makes return = episode length, which has NO constant-action optimum (best constant 845.7 < 930.27). K5.5's shaped alpha030 reward was satisfiable by a constant action and produced the degenerate optimum. Do not reintroduce shaping unless DQN measurably collapses first.
- Escalation levers if vanilla DQN also collapses (found-art, in order): Double DQN, larger net + capacity-preserving regularizer (arXiv 2204.09560), Adam beta-match + L2 (OpenReview policy-collapse paper), QR-DQN or Rainbow via sb3-contrib, then HER for the sparse-survival credit assignment.
- `runs/` is gitignored. Tracked this session: the two `k5_7_dqn_*` tools only. The DQN model/vecnorm/metrics will land under `runs\phase_k\k5_7_dqn\` (untracked), matching the BC/PPO pattern.
- Success is unambiguous and must not be relabeled: EITHER from-scratch DQN clears 930.27 in-env on seeds 1000-1009 with a non-degenerate policy, OR a finetune measurably beats BC (mean materially > 1737.3 and/or collision < 0.20) with the gain attributable to RL updates. Matching BC is a FAIL. Never call a non-result a win again.
