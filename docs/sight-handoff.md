# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.1 high-force clearance reward alpha=0.30 FAIL: constant-action regime shift from K3.5c constant-LEFT to K5.1 constant-STAY; no hazard-responsive lateral motion; per GPT scope route to K5.2-grok env-dynamics sanity check)

**Last commit:** `1db19a9` Phase K K5.1 alpha=0.30 high-force clearance reward: constant-action regime shift, FAIL

**Current task:** K5.1 evidence on disk in `docs/k5-1-clearance-alpha030-visible-behavior-evidence.md` (271 lines) and pushed. Trained checkpoint at `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k/model.zip`. Trained-eval at `runs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030/k5_1_alpha030_seed0_10k_trained_only/`. Verdict: deterministic-argmax eval emits `action=0 (stay)` on 6060/6060 reached steps across 10 seeds with `player_x = 360.0` on every step. K3.5c baseline at the same regime emits `action=-1 (left)` on 8457/8457 reached steps with player jammed at `x=16` on 92.55% of steps. Both are deterministic-argmax fixed points; the shaped reward at alpha=0.30 shifted the basin from wall-left to center-stay but did not produce non-constant behavior. 32 reward-shaping tests green. Smoke aggregate `frac_active_threat_saturated_norm` = 0.416 < 0.50 over the representative sample; the per-episode ep-000005 FAIL is the same alpha-invariant structural artifact disposed in Phase G under "Claude revises GPT's decisions on evidence." Phase G monitoring requirement met in K5.1 training (trained-policy aggregate `frac_sat_active` = 0.263 over 4972 active-threat steps).

**Next action:** Grok scopes K5.2 env-dynamics sanity check per GPT K5.1 scope item 7 and the K5.0 evidence doc section 7 ladder. Two anchored facts to explain at the env layer: K3.5c-divisor30 seed 0 with `reward_shaping=none` converges to constant-left at the left wall; the same regime with `reward_shaping=threat_weighted_clearance` alpha=0.30 converges to constant-stay at center. Both deterministic-argmax fixed points at 10k steps with moderate entropy decay (-0.858 to -0.697). Layers Grok must interrogate: action timing per Godot physics tick, hazard kinematics, observation freshness across the H3 transport boundary, frame-stack contract, and player kinematics. Do not run further coefficient sweeps, safe-lateral tweaks, entropy tuning, or alternative reward formulations until env-dynamics is cleared.

**Blockers:** None requiring Jeff. K5.1 was a Claude-executed GPT scope under charter role; the FAIL routing is per GPT's pre-declared decision tree.

**Notes:**

- The K3.5c-vs-K5.1 attractor comparison is the most useful diagnostic surface to date: same env, same algo, same seed, same training budget, same divisor; only reward shaping toggles, and the policy collapses to a different constant action with no lateral motion. Whatever drives single-action collapse is not the reward shape.
- K5.1 training used `--reward-scale-divisor 30` (Claude tactical fill-in) so the run would match the K4.1 reference regime that GPT named in the scope. Without divisor=30 the run would have landed in an untested intermediate.
- K4.1-style logit-margin probe was deferred. Behavioral evidence (constant action, zero player motion) is binary FAIL without needing margin telemetry. Probe can be added if K5.2-grok findings prompt re-examination.
- `tools/h5_smoke_parse.py` is now alpha-parametric via `--alpha` (default 0.05 preserved). Saturation threshold normalized to `0.98 * alpha`; mean-bonus range gates scaled by alpha.
- Driver bats under `runs/smoke/` and `runs/rl/` (`run_k5_1_smoke.bat`, `run_k5_1_train.bat`, `run_k5_1_eval.bat`) are gitignored but follow the documented Phase G smoke pattern with `SIGHT_GODOT_EXE` set inline, log + sentinel files, and `start /min cmd.exe /c <bat>` launch pattern for MCP-ceiling-exceeding wall times.
