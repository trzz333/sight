# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K3.2 Stage 1 complete: returns-distribution evidence shows value-MLP collapse is pathological, not L2-correct). K1-extended remains parked.

**Last commit:** `50c1f78` Phase K K3.2 Stage 1 returns-distribution evidence: value-MLP collapse is pathological, not L2-correct.

**Current task:** K3.2 Stage 1 no-code extraction from `runs\phase_k\k3_1_seed3_pi64_vf128_feature_chain_10k.ndjson` is complete. All 40 updates classify as `real_returns_variance + model_not_beating_constant + advantage_nondegenerate`. `model_to_constant_mse_ratio` 7.56 to 53.4 across the run, EV essentially zero throughout. Mechanism: at init `latent_vf` has 128 live dims with rollout `val_std=0.020`; update 1 produces a 69 to 158 norm gradient on the scalar value head and 41 norm on the shared `mlp_extractor`, which collapses `latent_vf` from 128 to 70 to 2 to 0 live dims by update 3. From update 4 onward only the scalar value-head bias can update, and it chases a growing `returns_mean` (15 -> 44) with a persistent 10-unit lag. Persistent positive advantages (96 to 100% positive fraction) drive the constant-action attractor. Evidence in `docs\k3-2-stage1-returns-distribution-evidence.md`. CSV in `runs\phase_k\k3_2_stage1_returns_table.csv`.

**Next action:** GPT scoping needed on whether to (a) execute the planned Stage 2 patch of `tools\h5_training_entropy_probe.py` per the K3.2 contract (add `returns_std`, `constant_baseline_mse`, `model_to_constant_mse_ratio`, per-submodule grad norms, CNN-to-returns CV R^2, latent_pi/latent_vf-to-returns CV R^2) and run seed 3, 2048 ts, pi=[64]/vf=[128], or (b) skip directly to K3.3 value-side interventions (reduced `vf_coef` 0.1 or 0.05, value-head init, return normalization) since the Stage 1 mechanism is already sharper than GPT contract Branch B. Recommended: do both in parallel, Stage 2 patch as pure diagnostic confirmation, K3.3 vf_coef sweep as the actual intervention. Action-net-gain stays parked.

**Blockers:** None requiring Jeff.

**Notes:**

- Stage 1 evidence falsifies the "value MLP correctly collapsed to a constant under near-constant returns" hypothesis. Returns have `ret_std` 1.81 to 5.30 every update, `ret_range` 10.5 to 26.2. Returns are not near-constant and not low-effective.
- Value head learned a tiny `val_std=0.0197` at update 1 before collapse, so observations are predictive enough at init for the value head to start learning. The collapse interrupted learning, it did not happen because there was nothing to learn.
- Naming caveat per GPT: `grad_norms_preclip.value_net_mean` in the NDJSON is the final scalar `policy.value_net`, not `mlp_extractor.value_net`. `grad_norms_preclip.mlp_extractor_mean` is mixed pi and vf branches. Stage 2 patch contract specifies adding per-submodule MLP grad norms to disambiguate.
- The first-update value loss is 111 and `vf_coef=0.5`, producing a scalar-value-head gradient of 69.1 and a shared-MLP-extractor gradient of 41.2. By update 2 the scalar-value-head gradient is 157.7. This is the optimization-shock signature. Standard remediation is reducing `vf_coef`, adding value normalization (e.g., return clipping or running mean/std), or initializing the value head so its outputs are in the right magnitude range at init.
- Returns grow over training (15 at update 1, 44 at update 40) because episodes get longer as the policy converges to `stay`, accumulating more per-step survival reward. This makes the bias-only chase strictly impossible to converge: the bias is gradient-descending against a target that itself moves up roughly as fast as the bias does.
