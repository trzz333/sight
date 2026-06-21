# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.6 method pivot to behavioral cloning; env and task geometry exonerated, the PPO learning pipeline is the wall). Single-voice governance.

**Last commit:** `06f161a` K5.6: pivot to behavioral cloning; env+task exonerated, learner is the wall

**Current task:** Diagnosis settled from committed K5.2 evidence (re-read this session, `docs/k5-2-env-dynamics-sanity-evidence.md`, ENV-PASS): env mechanics and task geometry are not the cause. constant_left mean 845.7 (collision 0.9, timeout 0.1); hazard_reactive_oracle mean 1762.8 (collision 0.1, timeout 0.9); delta 917.1 over the 84.57 threshold. The 930.27 bar = 845.7 + 84.57 (best constant + 10% margin). A constant action dies by collision 90% of the time, so the task does NOT underpunish standing still; the demo0 seed-1008 timeout-survival was the 10% lucky tail. The K5.1-K5.5 constant-action collapse is the PPO pipeline alone. No reward-geometry surgery. Method changed (PPO failed far past twice): clone the working K5.2 oracle into a small 10-dim state MLP via supervised behavioral cloning, which structurally cannot hit the value-head collapse and respects NOT frame_stack / NOT CNN. Two tools added and smoke-validated (commit 06f161a): `tools/k5_6_bc_dataset.py` (oracle rollout to (obs, action) npz; smoke 2 seeds = 3600 samples, both timeout, L/S/R 0.247/0.507/0.247) and `tools/k5_6_bc_train.py` (class-weighted MLP, smoke 97.5% val acc, flattered by within-seed leakage). Full 36-seed expert dataset (seeds 2000-2035) was generating in background at handoff (PID 52312, ~2031/2035 done, sentinel `runs\phase_k\k5_6_bc\dataset_gen.sentinel`).

**Next action:** Confirm `runs\phase_k\k5_6_bc\dataset_2000_2035.npz` exists, then train: `python tools\k5_6_bc_train.py --dataset runs\phase_k\k5_6_bc\dataset_2000_2035.npz`. Then write the in-env greedy eval (load `bc_policy.pt`, run eval seeds 1000-1009, mean episode length vs 930.27) as the REAL verdict, since BC val accuracy does not guarantee survival under covariate shift. If it clears 930.27, render the demo clip via the `tools/demo0_visible_play.py` path and optionally PPO-finetune from the BC weights to keep the deliverable genuinely RL. If BC alone misses in-env, DAgger is the next lever.

**Blockers:** Repo `github.com/trzz333/sight` is PUBLIC (verified prior session). Flipping it private is Jeff's call and a manual GitHub Settings -> Danger Zone action; Claude cannot change repo access controls via browser automation. Does not block the demo build.

**Notes:**

- Diagnosis closed: env, reward, and task geometry are exonerated (K5.2 ENV-PASS, re-read this session). The wall is the PPO learning pipeline. Do NOT pursue reward-geometry edits.
- 930.27 provenance: 845.7 (K5.2 best constant, constant_left) + 84.57 (10% margin). "Above baseline" means beat the best constant action by 10%.
- Lateral move per "method fails twice, change it": PPO collapsed K5.1-K5.5; BC from the existing oracle is supervised, so it cannot reach the value-head collapse. found-art verdict ADAPT (imitation learning; DAgger, Ross 2011), plain PyTorch BC chosen over the `imitation` library for lowest dependency.
- BC eval-in-env is the verdict, NOT val accuracy. BC errors compound under covariate shift; if survival misses 930.27, DAgger is next.
- Scope flag (not yet a blocker): mission says "RL policy"; BC is imitation. Plan keeps it RL via BC-pretrain then PPO-finetune. If finetune re-collapses, the BC checkpoint is the demo, labeled imitation-learned. Jeff owns the public-sample labeling if it comes to that.
