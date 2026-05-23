# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.5 state-observation PPO control classified STATE-CONTROL-FAIL-ACTIVE-BAD; state observation does not rescue the agent, single-frame pixels exonerated as the sole blocker)

**Last commit:** `ebb8329` K5.5 state-observation PPO control -> STATE-CONTROL-FAIL-ACTIVE-BAD

**Current task:** K5.5 evidence on disk at `docs/k5-5-state-observation-control-evidence.md` (125 lines) and pushed. New config `configs/rl/signal_dodge_ppo_h5_state_shaped_alpha030.yaml` (production 10-dim state vector, MlpPolicy, K5.1-matched alpha=0.30 shaping, reward-scale-divisor 30.0). New tool `tools/k5_5_state_control_eval.py` loads the three train-seed checkpoints, rolls deterministic model.predict on eval seeds 1000-1009, records per-step action distributions, classifies into the five K5.5 buckets. Trained seeds 0/1/2 at 10k each (all summary.json status ok, reward_scale_divisor 30.0, reward_scale_applied true). Result: 0 of 3 seeds clear the 930.27 survival bar; pooled collision 0.967; all three seeds converged to degenerate single-action policies (seed0 constant-left mean 845.7, seed1 and seed2 constant-stay mean 606.0). seed0 scores exactly the K5.2 best constant 845.7; seed1/seed2 reproduce exactly the K5.1 stay-pinned 606.0 / collision 1.00. Verdict STATE-CONTROL-FAIL-ACTIVE-BAD: state geometry handed directly to the policy produces the same constant-policy failure, so the pixel representation is not the primary blocker.

**Next action:** GPT scopes K5.6 as a PPO objective, reward-geometry, and credit-assignment audit. NOT frame_stack, NOT a CNN feature-extractor change (state obs has exonerated single-frame pixels as the sole blocker). Leading scoping question: whether the next lever is reward geometry (threat_weighted_clearance shaping may be satisfiable by a constant action, which would explain constant-policy convergence), the 10k budget, or the PPO credit-assignment path. seed0 converging to exactly the K5.2 best constant suggests the objective is being optimized correctly toward a degenerate optimum, which points at reward geometry over budget. K5.6 is a scope change; GPT scopes, Grok phase-gates, then Claude executes.

**Blockers:** None requiring Jeff. K5.6 is the normal GPT-scope then Grok-gate flow.

**Notes:**

- The bit-identical-eval anomaly from the prior handoff is resolved by K5.5. seed1 and seed2 are distinct trained networks (distinct SHA-256) and produce bit-identical per-eval-seed lengths because both converged to the same deterministic-argmax policy (constant-stay). seed0 (constant-left) is a distinct argmax behavior and produces a distinct length set. Distinct weights yield identical eval iff they converge to the same deterministic-argmax policy. The eval pipeline is not defective; bit-identical eval is a symptom of degenerate convergence to identical constant policies.
- Production state obs is 10-dim and does NOT include raw player_y. Schema: obs[0] player_x_norm, obs[1] last applied move x, obs[2..4] nearest hazard dx/dy/present, obs[5..7] second hazard dx/dy/present, obs[8..9] third hazard dx/dy. The prior handoff's "player_x, player_y" phrasing for the K5.5 schema was loose; K5.5 used the existing 10-dim production vector with no new schema.
- "ACTIVE" in the bucket name overstates the result. All three seeds are constant-action policies, not moving-but-misaligned. Pooled non-stay 0.411 is entirely seed0's constant left; pooled right fraction is 0.0. No seed produces hazard-conditioned or bidirectional motion. K5.6 scoping must not assume an active policy needing steering correction.
- Grok gated K5.5 YELLOW with no blocking defects; the sole required condition (config comment block documenting differences from the prior state comparator and confirming the 10-dim schema) is present in the committed config.
- Detached `start /b` batch launch of training stalled before artifact creation in a prior session attempt; foreground `python -u` launch worked cleanly (cold start torch 22s, sb3 29s, train ~260s per 10k seed). Prefer foreground per-seed launches over the detached-batch pattern for short Godot training runs.
