# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.5 state-observation PPO control classified STATE-CONTROL-FAIL-ACTIVE-BAD; state observation does not rescue the agent, single-frame pixels exonerated as the sole blocker). Single-voice governance.

**Last commit:** `d50fd87` docs: self-audit re-run is discretionary (anchor on committed artifact; re-run only when stale)

**Current task:** K5.5 closed and pushed (evidence `docs/k5-5-state-observation-control-evidence.md`). All three 10k seeds collapse to degenerate single-action policies (seed0 constant-left 845.7 = K5.2 best constant; seed1/seed2 constant-stay 606.0 = K5.1 stay-pinned), 0 of 3 clear the 930.27 bar, pooled collision 0.967; leading hypothesis is `threat_weighted_clearance` shaping satisfiable by a constant action. This session was non-training meta-work: governance switched to single-voice (GPT and Grok removed); `/sight-handoff` skill rewritten and packaged as a distributable `sight-handoff.skill` with a bundled save-button UI asset (`assets/save-bootstrap.html`, rendered every handoff via Section 3c); self-audit re-run made discretionary (anchor on committed artifact, re-run only when stale); Handshake game-dev form answered from verified repo facts (Godot 4 GDScript, Godot work since 2026-04-24, no other engines, repo public). No training run.

**Next action:** Next session, build the demo. Run K5.6 to break the constant-action collapse: found-art on the PPO constant-action / value-collapse failure class, then measure how often a constant action survives across many seeds (the demo0 seed-1008 clip showed a constant policy surviving to timeout, which would mean the task underpunishes not-dodging) BEFORE tuning the learner, then pull reward-geometry, entropy, and normalization levers. Goal artifact is a gameplay clip of an agent dodging above the 930.27 baseline as the public work sample. NOT frame_stack, NOT a CNN feature-extractor change.

**Blockers:** Repo visibility is Jeff's call and pending. `github.com/trzz333/sight` is currently PUBLIC (verified HTTP 200 unauthenticated, no token set, this session). Decide whether to flip it private before sharing the Handshake portfolio link; the form item-1 answer depends on that choice.

**Notes:**

- Governance: GPT and Grok are gone (Jeff unsubscribed, 2026-06). Single-voice. Claude is architect + executor + self-auditor + lateral-auditor. Jeff approves direction, scope, money, legal, IP, target environments only. No model gate, no "A or B, Jeff picks." Skill packaged as `sight-handoff.skill` (SKILL.md + save-button asset). Decision record `docs/governance-2026-06-single-voice.md`.
- Self-audit (replaces GPT review): evidence-anchored, adapted from factored Chain-of-Verification (Dhuliawala et al., ACL Findings 2024). Decompose each load-bearing claim into a verification question, answer against an external anchor. A committed artifact re-read is a valid anchor; re-run an eval only when freshness is in doubt. Introspection alone is unreliable (Huang et al., ICLR 2024).
- Bit-identical eval resolved: distinct trained networks (distinct SHA-256) produce identical per-seed eval lengths only because they converged to the same deterministic-argmax constant policy. Eval pipeline is not defective.
- Production state obs is 10-dim, no raw player_y. obs[0] player_x_norm, obs[1] last applied move x, obs[2..4] nearest hazard dx/dy/present, obs[5..7] second hazard dx/dy/present, obs[8..9] third hazard dx/dy.
- Foreground `python -u` per-seed launch works (cold start torch 22s, sb3 29s, train ~260s per 10k seed). Detached `start /b` batch stalls before artifact creation. Prefer foreground for short Godot runs.
