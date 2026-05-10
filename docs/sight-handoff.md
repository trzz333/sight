# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H3 closed via Grok GREEN on 2026-05-09 (closure artifact `docs/grok-h3-final-green.md`). H4 Steps 1 through 5 are landed, tested, and pushed. Default test gate at 222 passed, 2 deselected on `tests/rl` (up by 1 from H4 Step 4; live_godot pool unchanged at 2). Live smoke gate at 1 passed (H4 pixel same-seed step-by-step trajectory equality). Next slice is H4 Step 6: PPO `CnnPolicy` config YAML per `docs/sight-h4-plan.md` section 8 and implementation sequence step 7.

**Last commit:** `dbf248d` test(rl): h4 cnn policy construction smoke

**Current task:** H4 Step 5 complete on main. The default-tier `tests/rl/test_h4_cnn_policy_construct.py` builds an SB3 PPO model with `CnnPolicy` over a fresh Gymnasium stub env exposing the H4 pixel observation contract `Box(0, 255, (1, 84, 84), uint8)` and `Discrete(3)`. The stub env is intentionally not a `GodotSignalDodgeEnv` subclass so failure attribution stays inside SB3 policy construction rather than transport. The test runs `model.predict(obs, deterministic=True)` for a valid-action check (action in `{0,1,2}`), then `model.learn(total_timesteps=8)` with `n_steps=8, batch_size=4, n_epochs=1` so the rollout enters PPO's train path. Acceptance: `env.reset_count >= 1`, `env.step_count >= 8`, and a wrapped `policy.optimizer.step` fires at least once. Episode horizon is 4 so the 8-step rollout exercises SB3's mid-rollout reset path. CPU-only. No live Godot, no CUDA dependency, no production env code touched.

**Next action:** H4 Step 6 (= plan implementation sequence step 7). Add `configs/rl/signal_dodge_ppo_h4_pixel.yaml`. CPU PPO `CnnPolicy` smoke config consumed by `sight_agent.rl.train`. Mirror the H3 config plumbing pattern. Implementation sequence step 8 (optional `--total-timesteps 128` live training smoke run on StrongerJr) remains opt-in and is gated behind Godot launch, not behind config landing.

**Blockers:**

- None for H4 Step 6 implementation. Pixel obs path is end-to-end green from Python env through Godot capture and back; SB3 `CnnPolicy` construction surface is now proven on the H4 pixel space.
- H4 windowed-mode capture pops a real OS window during live runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 Step 5 acceptance gate.** Acceptance is "constructs and runs while writing artifacts," not "learns." Plan section 8 ("Smoke test" + implementation sequence step 6). The stub env contract is documented inline in `tests/rl/test_h4_cnn_policy_construct.py`: episode horizon 4 forces a mid-rollout reset during the 8-step `learn()` so SB3's reset path is exercised, and observations are non-blank deterministic uint8 patterns (column stripe + row band keyed on a frame counter) so the conv stack is not fed degenerate all-zero images.
- **Stub env, not Godot env subclass.** The H4 plan separates "Stub env first, then live opt-in." Subclassing `GodotSignalDodgeEnv` would couple this slice to fake transport lifecycle, reset semantics, and wrapper behavior, blurring failure attribution. H4 Step 4 already proved the live pixel transport at byte-equal trajectory granularity.
- **H4 Step 4 invariants intact.** Same-seed plus same scripted-action pixel trajectory equality (`np.array_equal` on every observation) is the binding determinism criterion from `docs/sight-h4-plan.md` section 9 and section 10 criterion 6. Live smoke verifies this at 1 reset + 10 steps; H4 Step 7 acceptance run will scale this up.
- **H4 reset capture invariants (load-bearing).** Synchronous `remove_child` for hazards must precede `queue_free` in `_h3_perform_soft_reset`. Survival label must be set to the deterministic `"RESET seed=N ep=ID"` string before the pixel capture awaits. State-mode does not render; the changes are no-ops in state mode but mandatory for pixel-mode determinism.
- **H4 windowed-mode launch is canonical.** Pixel mode rejects `headless=True` at env construction (the H4 spike proved Godot 4.6.2 `--headless` does not emit `frame_post_draw`). Live smoke pops two real OS windows in sequence; runs are intentionally serial, not concurrent, so they cannot collide on TCP port allocation.
