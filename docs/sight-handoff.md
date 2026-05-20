# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.3 stochastic eval classified the K5.1 alpha=0.30 checkpoint as SOFT-BAD-POLICY; the learned distribution is soft but the action ranking is wrong relative to env solvability geometry)

**Last commit:** `cbe7ce9` K5.3 stochastic eval -> SOFT-BAD-POLICY

**Current task:** K5.3 evidence on disk at `docs/k5-3-stochastic-eval-evidence.md` (173 lines) and pushed. Tool patched at `tools/h5_stochastic_eval.py`: step-weighted sampled and argmax action fractions, sampled-differs-from-argmax step fraction, entropy_nats stats, top1-top2 probability margin stats, top1_top2_margin_lt_0p05_fraction, per-action probability percentiles, four-way ordered primary bucket classifier, NEAR-TIE-BIAS overlay computed independently per the GPT amendment to Grok's GREEN thresholds. K5.1 alpha=0.30 seed=0 10k-step deterministic baseline (mean 606.0 frames, 10/10 collisions) hardcoded under DETERMINISTIC_BASELINES. Classifier smoke-tested on five synthetic distributions including the overlay-coexists-with-ARGMAX-ARTIFACT case. K5.3 ran 50 episodes (10 seeds x 5 replicates, 27150 evaluated steps): collision_rate 1.00, timeout_rate 0.00, sampled mean episode length 543.0 (below K5.1 deterministic baseline 606.0 and material survival bar 930.27), sampled_differs_from_argmax_step_fraction 0.189, entropy_nats mean 0.607, top1_top2_margin mean 0.704 and lt_0p05_fraction 0.000, argmax 100% stay on all 27150 steps. Primary bucket SOFT-BAD-POLICY, near_tie_bias overlay false.

**Next action:** K5.4 logit/obs probe on hazard-relative states. GPT K5.3 decision tree branch for sampled-differs-but-survival-does-not-improve points to misranked or poorly conditioned action probabilities. Build a controlled obs set (synthetic or replay-derived) where the hazard-relative-x is known by construction and the K5.2 layer-6 oracle's optimal action is the ground truth, then quantify whether the CnnPolicy's logit ordering correlates with that ground truth. Candidate harness: `tools/h5_logit_compare.py` already has margin/entropy aggregation; the missing slice is the controlled obs generator. Grok phase-gate sanity check on the K5.3 SOFT-BAD-POLICY classification before scoping K5.4 in detail. frame_stack=4 retrain, CnnPolicy width sweep, longer-budget retrain, and reward-shape revision are all premature relative to the logit/obs probe.

**Blockers:** None requiring Jeff.

**Notes:**

- SOFT-BAD-POLICY rules out the K5.3 ARGMAX-ARTIFACT branch (the deterministic-argmax fixed point hypothesis) and the POLICY-DIST-COLLAPSE branch. The learned distribution has substantial entropy (0.607 nats vs the 0.10 collapse threshold and ~1.099 nats uniform) and the soft tail off stay is small but real on every evaluated step (left 0.078 +- 0.010, stay 0.813 +- 0.021, right 0.109 +- 0.011).
- Sampling makes survival worse on average (543.0 vs 606.0 deterministic baseline), which is the diagnostic signal for misranked probabilities: there is real non-stay mass but it is not aligned with hazard kinematics, so drawing from it produces worse choices than always staying.
- Three trained networks across the project so far show three distinct action-distribution shapes (K3.5c constant-left under argmax, K5.1 alpha=0.30 constant-stay under argmax with soft right-over-left tail, earlier phases varied). Argmax is reading off structure PPO learned, not flattening it. The pathology is in the learning pipeline, not the eval protocol.
- K5.2 layer-6 hazard-reactive 1-step oracle reaches 1762.8 mean frames on the same 10 seeds. A much better policy exists in the function class on the shaped surface; the alpha=0.30 shaping is not the root cause.
- K5.3 wall time 759.9 s for 50 episodes; .bat + sentinel pattern at `C:\Users\maste\AppData\Local\Temp\k5_3_run_stochastic_eval.bat` and `k5_3_stochastic_eval.done` worked cleanly. Classifier and helpers are now in `tools/h5_stochastic_eval.py` and reusable for any future stochastic-eval slice.
