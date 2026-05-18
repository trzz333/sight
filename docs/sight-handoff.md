# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K K3.1 feature-chain diagnostic landed and ran. `snapshot_policy_state` in `tools/h5_training_entropy_probe.py` now captures cross-panel diversity at six points along the forward pass (raw_obs, cnn_features, latent_pi, latent_vf, action_logits, value_predictions) on every pre/post snapshot of the 32-item fixed observation-conditioning panel. Smoke at 2048 ts seed 3 pi=[64]/vf=[128] passed. One 10k seed-3 pi=[64]/vf=[128] slice completed and the diagnostic decisively localized the variance-collapse mechanism: the value-MLP submodule of `mlp_extractor` flattens latent_vf to zero cross-panel variance within 3 PPO updates, value_predictions become bit-identical across all 32 panel items from update 3 onward, while cnn_features stay diverse (~129 live dims, max_std 0.046) through all 40 updates. K1-extended remains parked.

**Last commit:** `ad44979` Phase K K3.1 feature-chain diagnostic: latent_vf collapses inside mlp_extractor.value_net within 3 updates, cnn_features stable.

**Current task:** K3.1 is complete. Evidence and reproduction in `docs/k3-1-feature-chain-diagnostic-evidence.md`. The collapse locus is inside `mlp_extractor.value_net`, not the shared CNN and not the value-net Linear. cnn_features hold their variance through training; raw_obs are diverse with 96 live pixel dims. Pixel-pipeline audit (GPT branch a) and unshared-CNN refactor (GPT branch b) are both ruled out by this evidence. The recommended next direction is upstream: probe the return and advantage distributions across the rollout buffer to determine whether the L2-optimal value function is genuinely a constant given current Signal Dodge rewards, or whether the value loss is over-collapsing despite real return variance.

**Next action:** GPT scope decision. Two diagnostics, both cheap, that would discriminate the next-best fix without another arch sweep:

  1. Returns-distribution probe. Add cross-rollout std and per-update min/max of `rollout_buffer.returns` and `rollout_buffer.advantages` to the per_update_digest (or to a separate ndjson event). If returns std is near zero from update 0, the value head is correctly degenerate and the reward shaping or episode-length distribution is the upstream cause. If returns have real variance but the value loss still drives latent_vf to zero, the optimizer or value loss weighting is at fault.

  2. Action_net init gain. SB3 default `ortho_init` uses gain 0.01 for the action head. latent_pi keeps ~40 live dims throughout training but action_net compresses cross-panel logit std to O(1e-5). Test: bump the action_net init gain (or remove the small-gain init) and rerun. One-line `policy_kwargs` change.

Recommend GPT pick 1 first; it's the cleaner upstream diagnostic and is architecture-independent.

**Blockers:** None requiring Jeff. Scoping is a GPT/Claude technical decision.

**Notes:**

- Decisive K3.1 finding: at update 0 pre-update the network is fully observation-conditioning (all 64 latent_pi dims and all 128 latent_vf dims live across the 32 panel items; value_predictions span [-0.074, -0.050]). By update 3 latent_vf has 0 live dims and value_predictions std is exactly 0.0. The collapse is downstream of cnn_features (which never collapse: 129 live dims, max_std 0.046, stable through all 40 updates).
- `snapshot_policy_state` schema delta: returns `feature_chain_diversity` with subkeys `share_features_extractor`, `raw_obs`, `cnn_features`, `latent_pi`, `latent_vf`, `action_logits` (each a per-dim std summary dict) and `value_predictions` (n, mean, std, min, max). Field appears on every pre/post fixed_panel_policy_state and on summary.{initial,final}_fixed_panel_policy_state.
- The instrumentation assumes `share_features_extractor=True` (default for `ActorCriticCnnPolicy`). The flag is reported in the diagnostic for audit. If a future run sets it False, the existing extract_features-then-mlp_extractor chain inside `snapshot_policy_state` will need adapting.
- 10k run gates: `min_bar=false`, `better_bar=false`, top_argmax_action=stay, top_argmax_fraction=1.0, num_det_actions=1, constant_action_attractor=true, max EV=0.0022. Failure under the K3 contract, as expected given K3 already established the constant-action attractor for this config; the K3.1 contribution is mechanism, not contract clearance.
- Smoke-validated launch pattern retained: bat-with-sentinel at `C:\Users\maste\AppData\Local\Temp\sight_k3_1_smoke\run_smoke.bat` and `sight_k3_1_10k\run.bat` with `SIGHT_GODOT_EXE` set inline, stdout/stderr redirected to a `.log`, `%ERRORLEVEL%` written to `.done` on exit. Detached launch via `start "" /b cmd /c <bat>`. Poll via short ping waits inside `interact_with_process`.
