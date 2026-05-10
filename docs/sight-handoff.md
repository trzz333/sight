# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H3 closed via Grok GREEN on 2026-05-09 (closure artifact `docs/grok-h3-final-green.md`). H4 Step 1 (Python observation_mode plumbing), H4 Step 2 (TCP protocol extension with pixel-obs response validation), and H4 Step 3 (windowed Godot viewport pixel source) are landed, tested, and pushed. Default test gate at 221 passed, 1 deselected (live_godot smoke). Next slice is H4 Step 4: live Godot pixel smoke under `pytest -m live_godot` with same-seed scripted step-by-step trajectory equality per `docs/sight-h4-plan.md` sections 9 and 10.

**Last commit:** `fcbcf7e` feat(gd): h4 step 3 - windowed godot viewport pixel source

**Current task:** H4 Step 3 complete on main. Real pixel source wired end-to-end from Python env down to GDScript viewport capture. `tcp_controller.gd` no longer refuses `observation_mode="pixel"` at reset; the blanket `bad_request` is replaced by per-condition rejection (both mode still deferred, channels != 1 still deferred). New `send_reset_ok_pixel` and `send_step_result_pixel` helpers carry the H4 obs Dictionary on the wire. `main.gd::_h3_perform_soft_reset` and `_h3_perform_step` branch on `_tcp.h3_observation_mode()` at the send site; pixel path awaits `RenderingServer.frame_post_draw`, reads `Viewport.get_texture().get_image()`, converts to `FORMAT_L8`, resizes to locked dims via `INTERPOLATE_NEAREST`, packs to a flat int array, and emits the full H4 payload (`mode/shape/dtype/encoding/data/pixel_source=godot_windowed_viewport/capture_point=RenderingServer.frame_post_draw/headless_allowed=false/viewport_width/viewport_height`). Capture failure sends `ERROR_INTERNAL`; no synthetic raster fallback. State mode is byte-for-byte unchanged. Sanity: `godot --headless --path games\signal-dodge --quit-after 2` exits 0 with empty stderr; scripts parse and load cleanly.

**Next action:** H4 Step 4 (= plan implementation sequence step 5). Live Godot pixel smoke test behind `pytest -m live_godot` at `tests/rl/test_h4_godot_pixel_smoke.py`. Required acceptance bar from `docs/sight-h4-plan.md` section 10 criterion 6: same-seed plus same scripted action sequence produces matching pixel observations at every post-mode-lock step across two runs (step-by-step trajectory equality, NOT merely first-pixel equality). Concrete slice: launch real Godot windowed against `games/signal-dodge`, run a fixed scripted action sequence (e.g., `[1, 0, 2, 1, 0, 2, 1, 0, 2, 1]`) at seed=0 twice, assert byte-equality on each step's pixel observation. Pre-mode-lock physics-tick variance permitted per H3 closure caveat. Live smoke must run windowed (not headless); the H4 spike confirmed `--headless` does not emit `frame_post_draw`.

**Blockers:**

- None for H4 Step 4 implementation. Python validation, transport, and Godot capture path are all in place.
- H4 windowed-mode capture pops a real OS window during runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 Step 3 wire contract.** Pixel-mode `reset_ok` and `step_result` carry `obs` as a Dictionary with the schema in `protocol.REQUIRED_FIELDS_PIXEL_OBS`. State-mode helpers and the H3 length-10 array path are unchanged. The two paths share envelope structure (run_id/episode_id/protocol_version/frame/info) and differ only in the obs field's JSON type.
- **H4 Step 3 capture invariants.** Synchronization barrier is `await RenderingServer.frame_post_draw` inside the consumed reset/step path. `INTERPOLATE_NEAREST` is mandatory: bilinear/cubic introduce subpixel float math that the H4 plan section 9 same-seed step-by-step equality criterion treats as a regression risk. Channels = 1 (grayscale L8) is the only implemented format; `tcp_controller.gd` rejects c != 1 at reset with explicit `bad_request`. Both mode remains rejected at reset.
- **H3 invariants preserved.** `_launch_godot` redirects stdout/stderr to per-run files when `run_dir` is set else `subprocess.DEVNULL` (PIPE deadlocks Godot 4.6.2 on Windows; verified 2026-05-09). `tcp_controller.gd::_h3_dispatch` accepts both TYPE_INT and TYPE_FLOAT for protocol_version and pixel dims. Default test gate did not regress (221 passed, 1 deselected, identical to Step 2 baseline).
- **H4 plan amendments still apply (commit `2a4d1c6`).** Section 1 enforces explicit headless rejection at env construction. Decision 4 wire payload schema with pixel_source / capture_point / headless_allowed / viewport dims. Section 9 and section 10 criterion 6 require step-by-step scripted trajectory equality, not first-pixel only. Step 4 inherits these directly as the live-smoke acceptance bar.
- **Tooling reminder.** `cmd.exe`-style `&` chaining is sequential, not background; `%VAR%` expansion inside chained commands can fail silently. Prefer absolute paths in quotes or two separate `interact_with_process` calls when running Godot binaries via Desktop Commander, not `&`-chained one-liners.
