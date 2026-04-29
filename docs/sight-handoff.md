# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** pivot to hobby RL/game-agent lab. Pivot branch `pivot/hobby-rl-lab` open from main at 2b08a0b. Main untouched. Pre-pivot Python harness WIP preserved unverified on `pivot-preserve-p3-wip` at a29beb3.

**Last commit:** PENDING_BACKLOG_COMMIT on pivot/hobby-rl-lab. Previous pivot doc commit ef40f43 already on this branch. Pre-pivot WIP at a29beb3 on pivot-preserve-p3-wip. Main still at 2b08a0b handoff: P3 GDScript foundation slice landed.

**Current task:** Recharter Sight from product track to hobby and research lab. Charter rewritten with hobby-track mission, H1-H5 phase plan, expanded non-goals, 64 GB RAM weaker-GPU hardware profile, retired P4-P6 product gates listed explicitly. Future target exploration backlog added at `docs/target-backlog.md` (Flare ARPG, RTS/economy candidates, OpenRA caution bucket, proprietary nostalgia bucket). Backlog explicitly cannot displace H1-H5.

**Next action:** Merge `pivot/hobby-rl-lab` to main (in-place recharter recommended over new repo, reasoning in notes). Then implement H1: local RL baseline on Gymnasium CartPole using Stable-Baselines3 or CleanRL with NDJSON training-metric logging.

**Blockers:** `pivot/hobby-rl-lab` unmerged. Main still carries product-era charter (mission text, P4-P6 product gates, success criteria of paying customer / contract / inbound). Pre-pivot Python harness WIP on `pivot-preserve-p3-wip` is unverified. Choice of Stable-Baselines3 versus CleanRL not made.

**Notes:**

- Jeff wants a hobby project, not a product validation track. Buyer-discovery gates retired. Success metric is learning progress and reproducible local training, not buyers or revenue.
- Approved target ladder: Gymnasium -> custom Godot state env -> Godot pixels -> Signal Dodge/successor. Open-source single-player games allowed only after explicit license and automation review. No proprietary commercial game (Diablo II and similar) until legal/ToS posture is verified for that specific game and offline single-player use.
- Future target candidates tracked in `docs/target-backlog.md`. No backlog target is approved. Backlog cannot displace H1-H5. Promotion requires Jeff-only approval after a written license/ToS/community/technical review.
- Existing product-era code on main (Signal Dodge GDScript, TCP transport, controller, logger, evaluator metrics core) is reusable as Phase H3-H5 target environment. No code deletion in this pivot, only doc rewrite.
- In-place recharter recommended over new repo: trzz333/sight is private, product-era history is informative not noise, GDScript and evaluator code carry over to H3-H5, two-repo bookkeeping compounds. Pivot branch and safety branch already mark the discontinuity in history.
