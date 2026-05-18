# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K3.3 complete: vf_coef sweep falsifies vf_coef as a value-shock intervention; Adam normalization renders vf_coef approximately ineffective). K1-extended remains parked.

**Last commit:** `eade7ff` Phase K K3.3 vf_coef sweep falsifies vf_coef as a value-shock intervention under Adam.

**Current task:** K3.3 patched `tools\h5_training_entropy_probe.py` with `--vf-coef`, per-update `value_fit` (pre/post-rollout model_mse, ratio, EV), per-submodule grad norms (`mlp_policy_net`, `mlp_value_net`, `mlp_value_net_weights`, `scalar_value_head`), and the weights-only L2 helper. Three 2048-ts seed-3 pi=[64]/vf=[128] slices at `vf_coef` 0.5, 0.1, 0.05 all hit `latent_vf_live_post_update_3 == 0`. Primary mechanism gate FAIL, failure gate fires per GPT K3.3 contract. The `mlp_value_net_mean` grad norm at update 1 scales linearly with vf_coef (41.18 / 8.24 / 4.12, a 10x range), yet `post_val_mean` delta across runs is at most 0.01 over 8 updates. That is the Adam-normalization signature: the running second moment v_hat scales with squared gradient so the effective step is approximately scale-invariant. SB3 PPO uses Adam by default. Evidence in `docs\k3-3-vfcoef-sweep-evidence.md`. CSV at `runs\phase_k\k3_3_vfcoef_sweep_table.csv`.

**Next action:** GPT scoping needed on K3.4 intervention choice. Per K3.3 contract, the next lever is return/value scaling (return normalization, reward rescaling, value-head bias init, separate value-head optimizer); action-net gain stays parked. The Adam finding rules out anything that only changes value-loss-side scalars under Adam optimization, so the surgical candidates are (a) value-head scalar bias init at `mean(returns_first_rollout)`, (b) return normalization wrapper that divides returns by a running std, (c) reward rescaling in `games/signal-dodge/scripts/main.gd` (charter-relevant change, separate commit). Recommended K3.4 scope: option (a) alone first as smallest patch and cleanest test; option (b) if (a) fails; option (c) deferred until the value-side hypothesis is fully exhausted.

**Blockers:** None requiring Jeff.

**Notes:**

- vf_coef IS a valid knob to control gradient magnitude (the 10:2:1 grad scaling is exact). It is NOT a valid knob to control parameter-update magnitude under Adam optimization. This is the K3.3 contribution beyond the contract: distinguishing gradient-scale interventions from parameter-update-scale interventions.
- All three K3.3 runs reproduce the K3.1/K3.2 collapse trajectory faithfully on the fixed panel: cnn_features stable around 129 live dims, latent_pi and latent_vf both collapse by update 3, post-update constant-action attractor flag is True for every update. The mechanism identified in K3.1 is robust across vf_coef.
- Reward shape from Signal Dodge appears to be per-step survival (returns grow with episode length: 15 at update 1 to 25 at update 8 in this 2048-ts window, to 44 at update 40 in the K3.1 10k run). If K3.4 (a) and (b) fail, the deeper pathology is the reward shape itself, in which case K3.5 will need to coordinate with the Godot game spec.
- The K3.3 patch preserves all legacy `grad_norms_preclip` keys for back-compat with K3.x downstream readers. New keys are additive. Smoke validated at seed 3, 512 ts, `--vf-coef 0.1`.
- New evidence and CSV layout: `docs\k3-2-stage1-returns-distribution-evidence.md` (K3.2), `docs\k3-3-vfcoef-sweep-evidence.md` (K3.3), `runs\phase_k\k3_2_stage1_returns_table.csv` (K3.2 derived), `runs\phase_k\k3_3_vfcoef_sweep_table.csv` (K3.3 derived). CSVs are gitignored under `runs/`.
