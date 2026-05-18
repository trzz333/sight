# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K3.4 complete: scalar value-head bias init at `mean(rollout_buffer.returns)` of the first rollout rescues update 1 fully, partially rescues update 2-3, collapse trajectory rejoins K3.3 by update 4. Classified TRANSIENT RESCUE per revised contract). K1-extended remains parked.

**Last commit:** TBD substantive Phase K K3.4 commit; this handoff bump follows it.

**Current task:** K3.4 patched `tools\h5_training_entropy_probe.py` with `--value-bias-init {none,first-rollout-mean}` CLI flag, `InstrumentedPPO.value_bias_init_mode` kwarg, in-`train()` mutation of `policy.value_net.bias` before pre-snapshot and first optimizer step (mutation fires exactly once at update 1), additive `value_fit.pre_rollout_after_value_bias_init` schema field, and a peer `value_bias_init` record block. Header now carries `value_bias_init_mode` and `value_bias_init_applied`. Smoke (seed 3, 512 ts) verified mutation fires once (`old_bias=0.0 -> new_bias=15.7140`) and additive fields populate. Real slice (seed 3, 2048 ts, pi=[64], vf=[128], default `vf_coef=0.5`) ran 8 updates. Primary mechanism gate FAIL on `latent_vf_live_post_update_3 = 10 < 16` (other two subgates PASS: vp_std_post_u3 = 2.16e-4 > 1e-6, post_ratio_u3 = 2.60 vs K3.3 baseline 22.84). Strong win gate FAIL on all three subgates at update 8 (lvf=0, vp_std=8.26e-7, post_ratio=30.61 vs baseline 6.18). Failure gate does not fire. Update 1 alone is a total rescue: lvf_post = 128/128, vp_std_post = 4.95e-2 (81x K3.3 baseline), v_loss = 7.54 (vs baseline 111.21), |g|_total = 4.45 (vs baseline 135.32). Update 2 returns_mean climbs to 28.97 driving a fresh ~13.3-unit bias gap; the value head must chase again and collapse resumes. Evidence in `docs\k3-4-value-bias-init-evidence.md`. CSV at `runs\phase_k\k3_4_value_bias_first_rollout_mean_table.csv`.

**Next action:** GPT scoping needed on K3.5 Python-side reward-scaling wrapper. K3.4 mechanistically validates the K3.2 update-1 value-shock hypothesis (eliminate the bias gap, update 1 stops killing latent_vf) and identifies the secondary mechanism (the per-step survival reward shape grows `returns_mean` monotonically with episode length, so the value head chases a moving target). Per the revised decision rule, primary gate did not pass and strong gate failed at update 8, so K3.4 is classified as TRANSIENT RESCUE; do not run 10k confirmation. K3.5 should be a `gym.Wrapper` that scales per-step reward by a fixed or running scalar so `returns` live in O(1) range. Reward-shape change in `games\signal-dodge\scripts\main.gd` (terminal-only or episode-normalized rewards) is the K3.6 escalation if the wrapper fails. Action-net gain stays parked.

**Blockers:** None requiring Jeff.

**Notes:**

- The bias init is mechanistically validated: when the scalar value-target gap is removed at update 1, latent_vf is fully preserved (128 live dims) and observation conditioning rises 81x on the fixed panel. K3.2 framing of the collapse mechanism is now directly confirmed, not just inferred from K3.3 falsification.
- The bias init is operationally insufficient because returns drift upward with surviving episode length. `ret_mean` climbs 15.71 -> 28.97 -> 33.63 -> ... -> 39.91 across 8 updates. The value head's bias chases this moving target every rollout; each catch-up is a fresh value shock comparable in magnitude to the original (v_loss_u2 = 110 vs v_loss_u1_baseline = 111). Bias init delays the killer shock by exactly one rollout.
- `pre_rollout_after_value_bias_init` (additive `value_fit` field from GPT revision 2) confirms the bias-mutation effect in isolation: `model_to_constant_mse_ratio = 0.998`, `value_pred_std = 0.0197`. The bias-initialized policy matches the constant-mean baseline almost exactly and preserves observation conditioning at the pre-update magnitude.
- The K3.4 patch preserves all K3.3 fields and adds two new fields (`value_fit.pre_rollout_after_value_bias_init`, top-level `value_bias_init`). Both are `null` for non-bias-init runs; downstream K3.x readers that ignore unknown keys are unaffected. Smoke validated at seed 3, 512 ts, `--value-bias-init first-rollout-mean`.
- Evidence and CSV layout: `docs\k3-3-vfcoef-sweep-evidence.md` (K3.3), `docs\k3-4-value-bias-init-evidence.md` (K3.4), `runs\phase_k\k3_3_vfcoef_sweep_table.csv` (K3.3 derived), `runs\phase_k\k3_4_value_bias_first_rollout_mean_table.csv` (K3.4 derived). CSVs are gitignored under `runs/`.
