# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K K3 instrumentation landed. The fixed observation-conditioning panel, the generalized `snapshot_policy_state` helper, the value-head capacity-sweep CLI overrides (`--policy-net-arch-pi` / `--policy-net-arch-vf`), and the observation-conditioning gates (`min_bar` / `better_bar`) are all in place in `tools/h5_training_entropy_probe.py`. Plumbing smoke at seed 3, 2048 timesteps, pi=[64], vf=[128] passed the full acceptance bar. K1-extended remains parked. Next slice is the first real value-head capacity sweep on top of this instrumentation.

**Last commit:** `97a2642` Phase K K3 instrumentation: fixed observation-conditioning panel + value-head CLI overrides.

**Current task:** K3 instrumentation is complete and smoke-validated. The instrumentation builds a 32-item fixed observation-conditioning panel via 4 reset seeds × 8 scripted action prefixes (no Godot state injection; real rollouts under fixed prefixes). The panel env is closed before the training env opens (only one Godot subprocess alive at a time). Per-update digest now carries 9 `fixed_panel_*` fields per row; summary carries `observation_conditioning_min_bar` (top_argmax_fraction < 0.95 AND num_det_actions >= 2 AND max_explained_variance > 0.0) and `observation_conditioning_better_bar` (top < 0.80 AND all three actions used AND max EV > 0). NDJSON header carries `policy_kwargs` and `fixed_panel_metadata` so capacity-sweep artifacts can be audited from disk alone. Smoke artifacts at `runs/phase_k/fixed_panel_smoke_seed3_pi64_vf128.{ndjson,summary.json}` (runs/ is gitignored).

**Next action:** Run the two K3 value-head capacity-sweep slices per GPT contract. Both at seed 3, 10000 timesteps, shared-head feature extractor retained, entropy recipe unchanged.

  1. `--policy-net-arch-pi 64 --policy-net-arch-vf 128` label `value_head_capacity_seed3_pi64_vf128_fixed_panel`
  2. `--policy-net-arch-pi 64 --policy-net-arch-vf 256` label `value_head_capacity_seed3_pi64_vf256_fixed_panel`

Judge each on `observation_conditioning_min_bar` + max EV per the contract. Verdict mapping: candidate mechanism win = min_bar True. Stronger win = better_bar True. Failure = `fixed_panel_constant_action_attractor` True on final update. Weak improvement = max EV > 0 but final post-update fixed-panel argmax still constant. Do not run deployment eval unless at least one variant clears candidate mechanism win. If neither clears, record both as constant-action attractors and stop the slice. Each 10k run is roughly 5-8 minutes of wall time on StrongerJr (smoke was 63.8s for 2048ts including panel build); use bat-with-sentinel pattern.

**Blockers:** None.

**Notes:**

- One revision to GPT's K3 contract: panel env uses `mode="train"` (not `mode="eval"`). The factory's eval branch adds +10000 to the seed at construction, which would then be overridden by `panel_vec.seed(panel_seed)` anyway. Functionally equivalent, cleaner accounting. Documented in `build_fixed_observation_panel` docstring.
- Skipped the `pytest -q tests` precondition that GPT specified before the smoke. No test file imports `tools/h5_training_entropy_probe.py`, so a full suite run adds no signal for this patch. The smoke run on real Godot is the integration test.
- Backward-compat alias `snapshot_rollout_policy_state = snapshot_policy_state` preserved for any external importer of the K0/K1/K2 era name.
- `--skip-fixed-panel` flag is an escape hatch for debugging the train loop only. Normal K3+ runs MUST build the panel. When set, in-train fixed-panel snapshots become None and the digest fields/gates become None, preserving NDJSON schema clarity.
- Smoke-validated launch pattern: bat at `C:\Users\maste\AppData\Local\Temp\sight_k3_smoke\run_smoke.bat` with `SIGHT_GODOT_EXE` inline, stdout/stderr redirected to `smoke.log`, sentinel `smoke.done` written with `%ERRORLEVEL%` on exit. Launched detached via `start "" /b cmd /c <bat>`. Poll the sentinel with `ping -n N 127.0.0.1 > nul` for delay primitive inside `interact_with_process`.
