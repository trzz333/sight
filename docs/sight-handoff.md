# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K K3 capacity sweep complete. Two 10k seed-3 slices at pi=[64], vf=[128] then vf=[256] both classified as Failure under the K3 contract: `fixed_panel_constant_action_attractor=true` on the final update, `top_argmax_fraction=1.0`, `num_det_actions=1` (constant `stay`), `min_bar=false`, `better_bar=false`. The value-head capacity hypothesis is falsified by this slice. Meta finding from the fixed-panel logit_std: O(1e-5) on vf128 and O(1e-7) on vf256. The policy network is observation-blind across the 32-item panel, and increasing value-head capacity made cross-panel variance worse, not better. K1-extended remains parked.

**Last commit:** `e39988d` Phase K K3 capacity sweep evidence: vf128/vf256 both Failure, policy network observation-blind.

**Current task:** K3 sweep is closed. The next investigation is upstream of the value head, not another capacity sweep. Evidence and reproduction are recorded in `docs/k3-value-head-capacity-sweep-evidence.md` including the schema-mapping note that GPT's flat `fixed_panel_*_post` field names actually live nested at `post_update.fixed_panel_policy_state.*` in the K3 instrumentation.

**Next action:** GPT scope decision required. Recommended direction: a diagnostic slice that interrogates the CNN feature extractor on the same 32-item fixed panel rather than another arch sweep. Concretely, instrument the K3 panel to also snapshot the CNN feature-extractor output (penultimate features) per panel item, so a single short run can show whether features themselves are constant across the panel or whether features differ but the policy head squashes them to a constant logit. If features are constant, the problem is the CNN / observation pipeline (frame stack, normalization, downsampling, color channel handling). If features differ, the problem is the head's initialization or the optimization regime collapsing to the constant-`stay` attractor within the first few updates.

**Blockers:** None requiring Jeff. Scoping the upstream diagnostic is a GPT/Claude technical decision.

**Notes:**

- Schema mapping: GPT's contract referenced `fixed_panel_det_argmax_counts_post`, `fixed_panel_logit_std_post`, `fixed_panel_prob_ranges_post` as flat keys on per_update_digest rows. Actual K3 schema has only `fixed_panel_constant_action_attractor` flat; the other three live at `post_update.fixed_panel_policy_state.{det_argmax_counts,logit_std,prob_ranges}` (and analogously at `pre_update.fixed_panel_policy_state.*`). All values exist; only the path differs. The summary's `final_fixed_panel_policy_state` is the equivalent of the last digest row's `post_update.fixed_panel_policy_state`.
- Self-correction logged in evidence doc: the in-session log-tail showed `ev=0.0000` across all 40 updates because the probe's PPO digest line prints with `%.4f` precision. True max EV is 0.0022 (vf128) and 0.00021 (vf256). Both effectively zero (min EV is negative on both), classification unchanged.
- vf=[256] produced strictly worse cross-panel variance than vf=[128]: logit_std O(1e-7) versus O(1e-5). Capacity is not the bottleneck and may be actively making the policy head flatten faster via shared-feature feedback.
- Deployment eval was not run. K3 contract requires `min_bar` clearance on at least one variant first; neither cleared.
- Smoke-validated launch pattern was reused: bat-with-sentinel at `C:\Users\maste\AppData\Local\Temp\sight_k3_vf128\` and `sight_k3_vf256\` with `SIGHT_GODOT_EXE` set inline, stdout/stderr redirected to a `.log`, `%ERRORLEVEL%` written to `.done` on exit. Launched detached via `start "" /b cmd /c <bat>`. Polled via short ping waits inside `interact_with_process`.
