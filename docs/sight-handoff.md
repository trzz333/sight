# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H3 closed via Grok GREEN on 2026-05-09 (closure artifact `docs/grok-h3-final-green.md`). H4 Steps 1 through 4 are landed, tested, and pushed. Default test gate at 221 passed, 2 deselected (live_godot pool now includes the H3 100-step smoke and the new H4 pixel smoke). Live smoke gate at 1 passed (H4 pixel same-seed step-by-step trajectory equality). Next slice is H4 Step 5: CnnPolicy construction test per `docs/sight-h4-plan.md` section 8 and implementation sequence step 6.

**Last commit:** `1ad79e1` test(rl,gd): h4 live godot pixel smoke + reset capture invariants

**Current task:** H4 Step 4 complete on main. The live `tests/rl/test_h4_godot_pixel_smoke.py` test launches the real Godot Signal Dodge build twice in pixel mode (windowed, not headless), runs `seed=0` with a fixed scripted action sequence `[1,0,2,1,0,2,1,0,2,1]`, and asserts `np.array_equal` on every (1,84,84) uint8 observation across the two runs (post-reset plus per-step). Live smoke runs in ~5s on StrongerJr. Production code received two surgical fixes that the live smoke exposed as real H4 Step 3 defects: synchronous `remove_child` of pre-reset hazards before `queue_free` (queue_free defers to idle, leaving lingering hazards in the scene tree at capture time, count varying with pre-mode-lock duration); and ordering `_survival_label.text` to a deterministic post-reset string BEFORE the pixel capture awaits `frame_post_draw` (else the captured frame shows the last pre-mode-lock legacy-loop label text whose elapsed-time component is wall-clock-derived and varies across runs). State-mode behavior is unchanged.

**Next action:** H4 Step 5 (= plan implementation sequence step 6). Add `tests/rl/test_h4_cnn_policy_construct.py`. Build a SB3 PPO model with `CnnPolicy` over a stub env that exposes the H4 pixel observation space `Box(0, 255, (1, 84, 84), uint8)`. Run at least one rollout step plus one optimizer step. No live Godot required. Default-tier test, not live_godot-marked. CPU-first; CUDA optional and not part of the acceptance bar.

**Blockers:**

- None for H4 Step 5 implementation. Pixel obs path is end-to-end green from Python env through Godot capture and back.
- H4 windowed-mode capture pops a real OS window during runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 Step 4 acceptance gate.** Same-seed plus same scripted-action pixel trajectory equality (`np.array_equal` on every observation, NOT first-pixel only) is the binding determinism criterion from `docs/sight-h4-plan.md` section 9 and section 10 criterion 6. Live smoke verifies this at 1 reset + 10 steps scale; H4 Step 7 acceptance run will scale this up. Approximate-equality matching was explicitly rejected; the criterion stands as byte-equality.
- **H4 reset capture invariants (load-bearing).** Synchronous `remove_child` for hazards must precede `queue_free` in `_h3_perform_soft_reset`. Survival label must be set to the deterministic `"RESET seed=N ep=ID"` string before the pixel capture awaits. State-mode does not render; the changes are no-ops in state mode but mandatory for pixel-mode determinism.
- **H4 windowed-mode launch is canonical.** Pixel mode rejects `headless=True` at env construction (the H4 spike proved Godot 4.6.2 `--headless` does not emit `frame_post_draw`). Live smoke pops two real OS windows in sequence; runs are intentionally serial, not concurrent, so they cannot collide on TCP port allocation.
- **H3 invariants preserved.** Default gate did not regress (221 passed). State-mode reset/step path is byte-for-byte unchanged. `_launch_godot` redirects stdout/stderr to per-run files when `run_dir` set else `subprocess.DEVNULL` (PIPE deadlocks Godot 4.6.2 on Windows; verified 2026-05-09).
- **Pre-mode-lock variance is permitted by the H3 closure caveat carried into H4.** H4 trajectory equality applies only to observations returned through the locked pixel-mode transport (post-handshake). The pre-mode-lock period covers the moments between Godot start and the first reset arrival; physics-tick scheduling jitter in that window is acknowledged and contained by the reset-time invariants above.
