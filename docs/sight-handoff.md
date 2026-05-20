# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.4 replay-derived logit/obs probe classified the K5.1 alpha=0.30 checkpoint as STAY-BIASED-MISRANKING; the argmax is structurally pinned to stay and blind to hazard geometry)

**Last commit:** `40857f5` K5.4 replay-derived logit/obs probe -> STAY-BIASED-MISRANKING

**Current task:** K5.4 evidence on disk at `docs/k5-4-logit-obs-probe-evidence.md` (226 lines) and pushed. New tool `tools/k5_4_logit_obs_probe.py` drives the production Godot H3 pixel env with six collector policies (stay, left, right, oracle, seeded_random, sweep), labels every aligned post-step reward_state with the K5.2 hazard-reactive oracle, queries the K5.1 alpha=0.30 CnnPolicy on the post-step pixel obs, and emits per-geometry-bucket stats, a confusion matrix, and a five-way K5.4 classification. Run: 60 episodes (6 collectors x 10 seeds), 26079 aligned samples, coverage exceeded all minimums (no expansion pass needed). Result: argmax = stay on 100.000% of all 26079 production-pixel observations; confusion matrix collapses to three cells all in the argmax_stay column; per-oracle-label top-1 accuracy stay 1.000, left 0.000, right 0.000; overall accuracy 0.6395 equals the oracle stay-label fraction and is a constant-stay-predictor artifact not a competence signal; accuracy decays monotonically with hazard proximity (arrival none 1.000, le15 0.533, le30 0.490, le60 0.271); entropy mean 0.685 nats so the distribution is soft not collapsed; fixed probability ordering stay > right > left independent of hazard direction. The K5.2/K5.4 hazard-reactive oracle collector survived all 10 seeds to the 600-step cap, confirming a hazard-conditioned policy is reachable in env and function class. Verdict STAY-BIASED-MISRANKING: K5.1 learned a real soft distribution but its top action is stuck at stay even when geometry calls for motion.

**Next action:** GPT scopes K5.5 as a state-observation PPO control run: train PPO on the same Signal Dodge task, same 10k-step budget, same alpha=0.30 shaping, but with a low-dimensional state-vector observation (player_x, player_y, nearest-hazard relative geometry) instead of single-frame pixels. This isolates one variable, whether the stay-pinned argmax is caused by the single-frame pixel representation or by the PPO objective/budget failing to learn the mapping even when geometry is handed over directly. K5.5 is a new training run and a scope change, so GPT scopes it and Grok phase-gates it before Claude executes.

**Blockers:** None requiring Jeff. K5.5 is a training run (scope change); the charter routes scope changes through GPT planning then Grok phase-gate, which is the normal flow, not a blocker.

**Notes:**

- K5.4 confirms and explains K5.3. K5.3 SOFT-BAD-POLICY said the soft off-argmax mass does not improve survival; K5.4 shows why: the argmax never leaves stay and the residual mass is undirected (not conditioned on hazard position), so sampling produces motion uncorrelated with the hazard.
- reward_state path: the Godot wire info is forwarded nested under `info["godot_info"]["reward_state"]` by `src/sight_agent/rl/godot_env.py` _build_info. The first K5.4 launch read a flat `info["reward_state"]` path, recorded 0 rows per episode, was killed, and the nested path was confirmed by a standalone Godot smoke test before the reported run.
- The K5.3 evidence doc described its player_x extraction as opportunistic; K5.3 was actually reading the dead flat path and player_x was always None. K5.3 verdict does not depend on player_x and stands; no retroactive amendment needed.
- Overall oracle top-1 accuracy is not a competence metric. For a constant-stay predictor it equals the oracle stay-label fraction. Per-label accuracy is the load-bearing number; future readers should not cite the overall figure.
- K5.5 routing depends on the state-observation control result: if it learns a hazard-conditioned policy at 10k steps the pixel representation is implicated (K5.6 frame_stack=4 or CNN feature-extractor change); if it also collapses to stay-pinned argmax the representation is exonerated and K5.6 is a longer-budget or PPO-hyperparameter slice.
