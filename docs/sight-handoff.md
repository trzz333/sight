# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K K2 baseline shared-head train-seed asymmetry probe complete (verdict K2-A, four-seed shared-head wedge confirmed). GPT scoping refined this turn: next slice is a value-head capacity sweep gated on a fixed observation-conditioning panel and a non-constant-action success contract. K1-extended remains parked. Phase nomenclature unchanged (H5 amended).

**Last commit:** `05896bd` docs: refine K-series next-action with observation-conditioned success contract.

**Current task:** K2 evidence and verdict are recorded at `docs/h5-phase-k-training-entropy-probe-evidence.md` sections 25-31 and committed at `fa13801`. GPT has now refined the next-slice contract: failure is redefined as any final policy whose deterministic argmax is one action for all sampled observations (all-left, all-stay, and all-right are equivalent failures); entropy collapse is no longer the central success metric; the gating metric is whether action and logit responses vary with the input. Two prerequisites for the next architectural slice are queued here.

**Next action:** Extend `tools/h5_training_entropy_probe.py` (or add a sibling diagnostic) to evaluate the in-training policy against a fixed observation-conditioning panel of deliberately varied player and hazard configurations and report per-update: det_argmax counts across panel, top det_argmax fraction across panel, mean logits per action, logit std per action, mean policy probs per action, per-action probability ranges. With that panel in place, run a first value-head capacity sweep slice (start with `policy_kwargs.net_arch = dict(pi=[64], vf=[128])` and `vf=[256]`, shared-head feature extractor retained, entropy recipe otherwise unchanged, train_seed 3 as the hard-wedge comparator), judged against GPT's success contract. Minimum non-wedge bar: top det_argmax fraction across fixed panel < 0.95 AND at least two actions selected by deterministic argmax AND EV crosses positive. Better bar: top fraction < 0.80, all three actions with nonzero support somewhere on the panel, and deployment eval shows at least one action change before timeout or collision.

**Blockers:** None.

**Notes:**

- Success contract reframe: a policy whose deterministic argmax is constant *independent of observation* is the failure, regardless of which action it is or how high EV climbs. A policy whose deterministic argmax is deterministic *conditional on observation* with at least two actions in use across the fixed panel is the candidate mechanism win. The observation-conditioning panel exists because rollout observations are biased by whatever the current policy does and a constant-action policy can create a narrow state distribution that looks stable for the wrong reason.
- Future hard-wedge comparator is train_seed 3 or 4, not seed 2. Seed 2 is the mildest shared-head wedge across the four tested (max margin 3.62 vs 8.15-13.07; final entropy 0.82 vs ~0.00-0.01) and is retained only as the K1 continuity anchor. Anchoring new architectural comparisons on seed 2 would understate the phenomenon they are trying to break.
- K1-extended (rerun K1 at 30k or 50k) stays parked. K2 makes the architectural mechanism more central than the training-budget mechanism; K1's value head shows no learning signal at 10k, and the budget-extension question is downstream of the capacity question.
- Basin-choice mechanism (why seed 1 picks `stay` while seeds 2/3/4 pick `left`) is explicitly deprioritized this cycle. The primary failure is now characterized clearly enough as a high-confidence constant-action attractor across train seeds; the per-seed basin identity is secondary.
- K2 launch pattern reusable: serial bat-with-sentinel at `C:\Users\maste\AppData\Local\Temp\sight_k2\run_k2_serial.bat` with `SIGHT_GODOT_EXE` inline. `ping -n N 127.0.0.1 > nul` is the safe delay primitive inside `interact_with_process`; `timeout /t N /nobreak > nul` causes the cmd shell to exit with "Input redirection is not supported."
