# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.5 state-observation PPO control classified STATE-CONTROL-FAIL-ACTIVE-BAD; state observation does not rescue the agent, single-frame pixels exonerated as the sole blocker). Governance changed to single-voice this session.

**Last commit:** `a58ebe5` governance: single-voice handoff skill, self-audit + found-art + layman-summary contract

**Current task:** K5.5 closed and pushed (evidence `docs/k5-5-state-observation-control-evidence.md`). Three seeds at 10k each, 0 of 3 clear the 930.27 survival bar, pooled collision 0.967, all three converged to degenerate single-action policies (seed0 constant-left mean 845.7 = K5.2 best constant; seed1/seed2 constant-stay mean 606.0 = K5.1 stay-pinned / collision 1.00). Leading hypothesis on record: `threat_weighted_clearance` shaping is satisfiable by a constant action, which explains constant-policy convergence. This session was governance only: GPT and Grok removed (Jeff unsubscribed), project is now single-voice with Claude as architect, executor, self-auditor, and lateral-auditor; `/sight-handoff` skill rewritten to match; new per-turn operating contract added (found-art reflex, evidence-anchored self-audit, lateral/infra-loop audit, one-paragraph layman summary, earned-agreement sycophancy posture, fluid rules). No training run this session.

**Next action:** Claude scopes and executes K5.6 (no GPT scope step, no Grok gate, single-voice). K5.6 = run `/found-art` on the PPO constant-action / policy-entropy-collapse / value-shock failure class (known, heavily documented PPO pathology, not novel), then test the reward-geometry hypothesis directly (is `threat_weighted_clearance` fully satisfiable by a constant action) and pull the documented levers (entropy coefficient, advantage and return normalization, reward rescaling, movement-incentive geometry) rather than continue pure diagnosis. NOT frame_stack, NOT a CNN feature-extractor change (state obs already exonerated single-frame pixels).

**Blockers:** None requiring Jeff. Single-voice means Claude scopes K5.6 directly; Jeff approves only if scope, direction, money, legal, IP, or a new target environment is touched, none of which K5.6 does.

**Notes:**

- Governance: GPT and Grok are gone (Jeff unsubscribed, 2026-06). Single-voice. Claude is architect + executor + self-auditor + lateral-auditor. Jeff approves direction, scope, money, legal, IP, target environments only. No model gate, no "A or B, Jeff picks." Rules are fluid and serve the mission: a small trained policy that plays Signal Dodge above the constant-action baseline.
- Self-audit method (replaces GPT review): evidence-anchored, adapted from factored Chain-of-Verification (Dhuliawala et al., ACL Findings 2024). Decompose each load-bearing claim into a verification question, answer it against an external anchor (re-read file, re-run eval, diff hash, recompute from summary.json). Introspection alone is unreliable and can degrade accuracy (Huang et al., ICLR 2024). Tool output over memory.
- The bit-identical-eval anomaly is resolved: distinct trained networks (distinct SHA-256) produce bit-identical per-seed eval lengths only because they converged to the same deterministic-argmax constant policy. Eval pipeline is not defective.
- Production state obs is 10-dim, no raw player_y. Schema: obs[0] player_x_norm, obs[1] last applied move x, obs[2..4] nearest hazard dx/dy/present, obs[5..7] second hazard dx/dy/present, obs[8..9] third hazard dx/dy.
- Foreground `python -u` per-seed launch works (cold start torch 22s, sb3 29s, train ~260s per 10k seed). Detached `start /b` batch stalls before artifact creation. Prefer foreground for short Godot runs.
