# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K3.1 complete: feature-chain diagnostic localized value-MLP collapse mechanism). K1-extended remains parked.

**Last commit:** `ad44979` Phase K K3.1 feature-chain diagnostic: latent_vf collapses inside mlp_extractor.value_net within 3 updates, cnn_features stable.

**Current task:** K3.1 instrumentation and the seed-3 pi=[64]/vf=[128] 10k diagnostic slice are complete and pushed. The variance-collapse locus is inside the value-MLP submodule of `mlp_extractor`: latent_vf goes from 128 live dims at init to 0 live dims by update 3 and stays there. cnn_features do not collapse (stable ~129 live dims, max_std 0.046 across all 40 updates). Pixel-pipeline audit and unshared-CNN refactor are both ruled out by this evidence. Evidence is in `docs\k3-1-feature-chain-diagnostic-evidence.md`.

**Next action:** Returns-distribution probe. Add cross-rollout std and per-update min/max of `rollout_buffer.returns` and `rollout_buffer.advantages` to the per_update_digest, then re-run a single 10k seed-3 slice on the same config. If returns std is near zero from update 0, the value head is correctly degenerate under current Signal Dodge rewards and the upstream cause is the reward shaping or episode-length distribution under the initial policy. The action_net-init-gain experiment is a fallback only if returns have real variance but latent_vf still collapses.

**Blockers:** None requiring Jeff.

**Notes:**

- At update 0 pre-update the network is fully observation-conditioning across all 64 latent_pi and 128 latent_vf dims; value_predictions span [-0.074, -0.050] across the 32 panel items. The collapse is learned, not init.
- `snapshot_policy_state` schema delta: returns a new `feature_chain_diversity` key with subkeys `share_features_extractor`, `raw_obs` / `cnn_features` / `latent_pi` / `latent_vf` / `action_logits` (each a per-dim std summary dict) and `value_predictions` (n, mean, std, min, max). Present on every pre/post fixed_panel_policy_state and on summary.{initial,final}_fixed_panel_policy_state.
- The K3.1 10k slice failed the K3 contract gates as expected (min_bar=false, better_bar=false, top_argmax_action=stay, top_argmax_fraction=1.0, num_det_actions=1, constant_action_attractor=true, max EV=0.0022). K3.1 contribution is mechanism, not contract clearance.
- Instrumentation assumes `share_features_extractor=True` (default for `ActorCriticCnnPolicy`). Flag is reported in the diagnostic so a future unshared run will be obvious. The existing `extract_features` then `mlp_extractor` chain in `snapshot_policy_state` already assumed it; K3.1 does not introduce new dependence on the assumption.
- Smoke-validated launch pattern reused: bat-with-sentinel at `C:\Users\maste\AppData\Local\Temp\sight_k3_1_smoke\run_smoke.bat` and `sight_k3_1_10k\run.bat` with `SIGHT_GODOT_EXE` set inline, stdout/stderr redirected to a `.log`, `%ERRORLEVEL%` written to `.done` on exit. Detached launch via `start "" /b cmd /c <bat>`. Poll via short ping waits inside `interact_with_process`.
