# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** pivot to hobby RL/game-agent lab. Pivot branch `pivot/hobby-rl-lab` open from main at 2b08a0b. Main untouched. Pre-pivot Python harness WIP preserved unverified on `pivot-preserve-p3-wip` at a29beb3.

**Last commit:** ef40f43 on pivot/hobby-rl-lab. Pre-pivot WIP at a29beb3 on pivot-preserve-p3-wip. Main still at 2b08a0b handoff: P3 GDScript foundation slice landed.

**Current task:** Recharter Sight from product track to hobby and research lab. Done on this branch: charter rewritten with hobby-track mission, H1-H5 phase plan, expanded non-goals (offerwalls, Freecash, bot-detection evasion, account farming, online multiplayer, live-service games, proprietary commercial games until explicit legal/ToS posture verified), 64 GB RAM weaker-GPU hardware profile, retired P4-P6 product gates listed explicitly so they cannot quietly return. Handoff rewritten to match.

**Next action:** Decide in-place recharter (merge `pivot/hobby-rl-lab` to main) versus new repo. Claude's recommendation: in-place. Reasoning at end of this doc. After decision, next session implements H1 (local RL baseline on Gymnasium CartPole with Stable-Baselines3 or CleanRL and NDJSON training-metric logging).

**Blockers:** `pivot/hobby-rl-lab` unmerged. Main still carries product-era charter (mission text, P4-P6 product gates, success criteria of paying customer / contract / inbound). Pre-pivot Python harness WIP on `pivot-preserve-p3-wip` is unverified (not compiled, not tested this session). Choice of Stable-Baselines3 versus CleanRL not made.

**Notes:**

- Jeff wants a hobby project, not a product validation track. Buyer-discovery gates retired. Success metric is learning progress and reproducible local training, not buyers or revenue.
- Do not use Diablo II or any proprietary game until legal and ToS posture is verified for that specific game and that specific use.
- Start with Gymnasium classic control or custom Godot microgames only. Open-source single-player games allowed only after explicit license and automation review.
- Existing product-era code on main (Signal Dodge GDScript, TCP transport, controller, logger, evaluator metrics core) is reusable as a Phase H3-H5 target environment under the hobby track. No code deletion in this pivot, only doc rewrite.
- Recommendation reasoning for in-place recharter: trzz333/sight is private, history of product-era attempts is informative not noise, GDScript and evaluator code carry over directly as H3-H5 target environment, two-repo bookkeeping costs more than one labeled pivot branch, and the pivot branch plus safety branch already mark the discontinuity in history.
