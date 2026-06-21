# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.6 literal-RL CLOSED. A learned Signal Dodge policy beats the constant-action baseline as BOTH imitation (BC) and reinforcement learning (PPO finetune). Mission bar met. Single-voice governance.

**Last commit:** `d71ae4e` K5.6 literal RL: PPO finetune from BC warm-start clears 930.27 (mean 1710.5)

**Current task:** Done this session. PPO-finetuned an SB3 MLP actor warm-started from `bc_policy.pt` (state, no CNN) for 20480 steps, exported it back into the BCPolicyNet schema, and ran the existing harnesses on it. In-env eval seeds 1000-1009: mean 1710.5 vs bar 930.27 (PASS, +780.23), 8/10 to the 1800 cap, collision 0.20, action counts non-degenerate on every seed. Value head learned (explained_variance -0.03 -> 0.56) with no collapse and no actor corruption (clip_fraction ~0.008): separate actor/critic heads over a parameter-free state extractor insulate the warm-started actor from value-shock, the structural opposite of the K5.1-K5.5 shared-CNN cold starts. Clip rendered on seed 1009 (`runs\phase_k\k5_6_bc\ppo_ft\demo\seed1009\demo.mp4`, action counts bit-identical to the eval episode). Also corrected the sight-handoff skill (removed orient-and-stop and the appended save/print widget) and delivered the fixed SKILL.md to Jeff. Evidence: `docs\k5-6-ppo-finetune-evidence.md`.

**Next action:** No queued execution task. K5.6 literal-RL is closed and the above-baseline mission bar is met. Hold for Jeff direction on scope (see Blockers); do not open new scope unprompted.

**Blockers:** Jeff-owned scope decision (surface only, not blocking execution): the mission bar is met, so the next direction is Jeff's, lower collision via DAgger (Ross 2011) on covariate-shift states, advance the phase, or wrap Phase K. Also: Jeff to reinstall the corrected sight-handoff skill (downloaded this session) in claude.ai Settings for the new orient-then-execute bootstrap to take effect. Repo `github.com/trzz333/sight` still PUBLIC; flipping private is Jeff's manual GitHub action.

**Notes:**

- K5.6 literal-RL PASS is the milestone. Warm-started PPO actor clears 930.27 (mean 1710.5); finetune PRESERVED BC (1737.3) rather than improving it. The win is "literal RL that holds above baseline," not a new high score.
- Root cause of K5.1-K5.5 collapse confirmed by contrast: shared CNN trunk let value-head collapse drag perception. Separate MLP actor/critic has no shared trunk, so the actor is immune; value EV climbed to 0.56.
- Collision rate 0.20 unchanged from BC. Lever to lower it is DAgger, NOT frame_stack and NOT a CNN change.
- `runs/` is gitignored; ckpt (`ppo_ft_policy.pt`), mp4, eval report, metrics, and `ppo_ft_sb3.zip` live on disk under `runs\phase_k\k5_6_bc\ppo_ft\`. Tracked in git: `tools/k5_6_ppo_finetune.py`, the render-note fix, and the evidence doc. Matches the BC pattern.
- Harness reuse works because the finetuned actor is exported in BCPolicyNet schema with the BC mu/sd; `k5_6_bc_eval_inenv.py` and `k5_6_bc_render_demo.py` run on `ppo_ft_policy.pt` unchanged.
