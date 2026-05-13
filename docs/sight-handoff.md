# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 implementation in progress. Baseline / evaluation harness landed in Step 1; H5 baseline CLI and 2-seed live negative-control non-saturation pre-check landed and ran in Step 2A-lite. **Current Signal Dodge pixel profile FAILed the non-saturation gate**: all three negative controls saturated at `timeout_rate=1.0`, `mean_episode_length=1800/1800` on seeds 1000, 1001. Nothing collides under stay-only, seeded-random, or untrained CnnPolicy. H4 pixel YAML stays unchanged (H4 closure record stands); H5 cannot train on this profile as learning evidence per `docs/sight-h5-plan.md` section 5. Next slice is an additive H5 hard profile YAML before training.

**Last commit:** `be320b0` feat(rl): add h5 baseline cli and live gate path

**Current task:** Author H5 hard profile (additive Signal Dodge difficulty knob YAML + Godot wire-up) so the H5 non-saturation gate has a measurable gap between negative controls and a learnable surface.

**Next action:** Add `configs/rl/signal_dodge_ppo_h5_hard_pixel.yaml` as a sibling to `configs/rl/signal_dodge_ppo_h4_pixel.yaml` (do NOT replace the H4 YAML; H4 reproducibility evidence stays intact). In-scope difficulty knobs per Step 2A-lite hardening note: shorter spawn interval, faster hazard speed, higher hazard density. GPT should scope the knob set (one knob at a time is the disciplined posture so isolated effect can be measured; pulling all three at once risks overshoot into impossibility). Any tuned values must be mirrored between Python constants/config plumbing and Godot GDScript with drift-detection tests. After the new profile lands, re-run the 2-seed pre-check via `python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h5_hard_pixel.yaml --run-id h5_negative_controls_hard_smoke --seeds 1000,1001 --mode negative-controls`. If that smoke passes, run the full 16-seed gate before training. If it still fails, increase difficulty further before training.

**Blockers:** none. Live Godot pipeline confirmed working end-to-end this session on StrongerJr (3 sequential Godot launches across 3 policies × 2 seeds in 5m17s wall time via inline `SIGHT_GODOT_EXE` in a `.bat` runner).

**Notes:**

- 2-seed live smoke artifact at `runs/rl/signal_dodge_ppo_h4_pixel/h5_negative_controls_smoke/evaluation/`; `index.json` records the canonical thresholds (`timeout_rate_threshold=0.50`, `length_ratio_threshold=0.80`) and the FAIL decision. Wall-time 5m17s for 3 policies × 2 seeds, useful timing reference for the 16-seed budget once the profile is hardened.
- H5 baseline CLI landed at `src/sight_agent/rl/h5_baseline_cli.py` with `--mode {negative-controls,full}`, seed spec parser supporting `1000,1001` and `1000-1015` and mixes, and a hard reject of `trained_cnn` in negative-controls mode. 28 new CLI tests in `tests/rl/test_h5_baseline_cli.py`; `pytest tests/rl` is 286 passed / 2 deselected (was 258 / 2).
- PPO.load dry check on `runs/rl/signal_dodge_ppo_h4_pixel/20260511T022255Z_..._0e96bfe/model.zip` (20,237,552 bytes) passed: `build_trained_cnn_policy(env=None)` constructs and `model.predict` on `(1,1,84,84)` uint8 returns a valid action. The "trained branch unit-tested for path resolution only" gap from H5 Step 1 is now closed in practice.
- Handoff and `docs/sight-h5-plan.md` wording reconfirmed precise: the pre-training non-saturation gate evaluates three negative controls only (`stay_only`, `seeded_random`, `untrained_cnn`); `trained_cnn` belongs to the post-training acceptance suite. The handoff test in `test_h5_baseline_cli.py` enforces this so the imprecise wording cannot regress silently.
- Operational find this session: `SIGHT_GODOT_EXE` inline via a `.bat` runner at `C:\Users\maste\AppData\Local\Temp\h5_smoke.bat` plus a sibling shell to poll the log/done sentinel is the reliable Desktop Commander pattern for live Godot runs that exceed the MCP 4-minute command timeout. The same pattern carries forward to the future 16-seed gate.


---
