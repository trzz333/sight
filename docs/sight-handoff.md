# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** hobby RL/game-agent lab. Pivot merged to main. Hobby charter and target backlog live on main. Pre-pivot Python harness WIP archived unverified on `pivot-preserve-p3-wip` at a29beb3.

**Last commit:** 3094655 on main. Merge commit d363eb0 lands the pivot. Substantive pivot commits ef40f43 (recharter) and 909c81c (target backlog). Pre-pivot WIP archived at a29beb3 on `pivot-preserve-p3-wip`.

**Current task:** Pivot merged. Hobby charter active on main. Ready for H1.

**Next action:** Implement H1: local RL baseline on Gymnasium CartPole using Stable-Baselines3 or CleanRL with NDJSON training-metric logging. GPT issues the H1 implementation prompt next. Claude executes against that prompt.

**Blockers:** Stable-Baselines3 vs CleanRL choice not made (defer to GPT's H1 prompt). Pre-pivot Python harness WIP at a29beb3 is unverified, archive-only.

**Notes:**

- Jeff wants a hobby project, not a product validation track. Buyer-discovery gates retired. Success metric is learning progress and reproducible local training, not buyers or revenue.
- Approved target ladder: Gymnasium -> custom Godot state env -> Godot pixels -> Signal Dodge/successor. Open-source single-player games allowed only after explicit license and automation review. No proprietary commercial game (Diablo II and similar) until legal/ToS posture is verified for that specific game and offline single-player use.
- Future target candidates tracked in `docs/target-backlog.md`. No backlog target is approved. Backlog cannot displace H1-H5. Promotion requires Jeff-only approval after a written license/ToS/community/technical review.
- Existing product-era code on main (Signal Dodge GDScript, TCP transport, controller, logger, evaluator metrics core) is reusable as Phase H3-H5 target environment. No code deletion in this pivot, only doc rewrite.
- Pre-pivot Python harness WIP archived on `pivot-preserve-p3-wip` at a29beb3, unverified. Do not revive under the hobby track without an explicit decision and re-validation, since the slice was authored against product-era framing.
