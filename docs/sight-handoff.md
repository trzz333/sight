# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K, K5.6 COMPLETE. The BC policy clears the above-baseline bar in-env. First learned Signal Dodge policy in this project to beat the constant-action baseline. Single-voice governance.

**Last commit:** `81ceb0b` K5.6: BC clears 930.27 in-env (mean 1737.3); env+task were exonerated, learner was the wall

**Current task:** Done this session. Trained the BC MLP (val_acc 0.9903, leakage-inflated, not the verdict), then ran the real test: in-env greedy eval on held-out seeds 1000-1009 (disjoint from training seeds 2000-2035). Mean episode length 1737.3 vs bar 930.27 (+807), vs best constant 845.7 (+892). 8/10 survive to the 1800 cap, action counts non-degenerate on every seed (real three-way dodging, NOT the K5.1-K5.5 constant-action collapse). BC clone within 1.4% of the K5.2 oracle it imitated (1737.3 vs 1762.8). Rendered a faithful demo clip: `runs\phase_k\k5_6_bc\demo\seed1009\demo.mp4` (1801 frames, 30 fps; seed-1009 action counts bit-identical to the eval run, so the clip IS the evaluated episode). Evidence: `docs/k5-6-bc-evidence.md`. Tools committed: `tools/k5_6_bc_eval_inenv.py` (the verdict), `tools/k5_6_bc_render_demo.py` (geometry mp4).

**Next action:** APPROVED plan (Jeff, 2026-06-20): PPO-finetune to literal RL, then render the literal-RL clip. (1) Warm-start: load `bc_policy.pt` weights into an SB3 MLP-policy actor (state mode, NOT pixel/CNN) and PPO-finetune. This also tests the sharper question: does warm-starting from a non-collapsed policy avoid the value-head collapse that killed K5.1-K5.5 cold-start? (2) Eval the finetuned policy with the existing `tools/k5_6_bc_eval_inenv.py` harness (same seeds 1000-1009, same 930.27 bar). (3) Render the literal-RL clip by adapting the `tools/k5_6_bc_render_demo.py` action selector to drive the finetuned SB3 policy. If finetune sustains >= 930.27 the mission is satisfied as literal RL; if it re-collapses, that isolates the failure to PPO cold-start value learning and the BC checkpoint stands as the labeled imitation deliverable. DAgger is the lever if the 0.20 collision rate needs lowering (NOT frame_stack, NOT a CNN change).

**Blockers:** Repo `github.com/trzz333/sight` is PUBLIC. Flipping private is Jeff's manual GitHub Settings -> Danger Zone action; Claude cannot change repo access via automation. Non-blocking. No open scope question: Jeff approved pursuing literal RL (finetune + RL clip) this session.

**Notes:**

- K5.6 PASS is the milestone: env, reward, and task geometry were exonerated in K5.2 (ENV-PASS); the wall was the PPO learning pipeline alone, and supervised BC structurally sidesteps it (no value head to collapse).
- 930.27 provenance: 845.7 (K5.2 best constant, constant_left) + 84.57 (10% margin). "Above baseline" = beat the best constant action by 10%.
- BC eval-in-env is the verdict, NOT val accuracy: the train val split leaks across episodes (random permutation over flattened pairs), so 0.9903 is inflated. The 1737.3 in-env mean on held-out seeds is the real number.
- Greedy argmax is deterministic; the eval is 10 fixed trajectories over 10 seeds, computed identically to the 845.7/1762.8 baselines. The 0.20 collision rate is BC covariate-shift error; it does not pull the mean under the bar.
- found-art on the BC route: ADAPT (imitation learning; DAgger, Ross 2011). Eval harness ADAPTs K5.2 layer-6 `_run_one_episode`; renderer ADAPTs demo0's cv2 VideoWriter. SIGHT_GODOT_EXE path confirmed valid this session (the earlier GODOT_MISSING was a cmd same-line `%VAR%` expansion artifact).
