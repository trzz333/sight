# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H3 closed via Grok GREEN on 2026-05-09 (closure artifact `docs/grok-h3-final-green.md` with verbatim verdict). H4 Step 1 (Python observation_mode plumbing on `GodotSignalDodgeEnv`) is landed, tested, and pushed; default tests pass 169/169 (1 deselected live_godot). Next slice is the Godot-side protocol extension for observation_mode and pixel dimensions plus default-transport protocol-error tests, mapping to `docs/sight-h4-plan.md` implementation sequence step 3.

**Last commit:** `b8b7fba` feat(rl): h4 step 1 - observation_mode plumbing on GodotSignalDodgeEnv

**Current task:** H4 Step 1 complete on main. `GodotSignalDodgeEnv` constructor now accepts `observation_mode in {"state","pixel","both"}` (default `"state"`), `pixel_width`/`pixel_height`/`pixel_channels` (defaults 84/84/1), validates inputs with `ValueError`, rejects `headless=True` for pixel/both modes per the Grok closure caveat, and dispatches `observation_space` accordingly: state -> `Box(-1,1,(10,),float32)`, pixel -> `Box(0,255,(channels,height,width),uint8)`, both -> `Dict({state, pixel})`. Test module `tests/rl/test_h4_godot_env_construct.py` adds ~50 parametrized cases (default, valid/invalid modes, pixel-dim validation, headless rejection, observation_space dispatch, H3-invariant preservation). H3 default and live gates untouched.

**Next action:** H4 Step 2 (= plan implementation sequence step 3). Extend the Godot-side TCP contract in `games/signal-dodge/scripts/tcp_controller.gd` to accept `observation_mode`, `pixel_width`, `pixel_height`, `pixel_channels` on `reset` and to return the H4 wire payload schema (`obs.mode`, `obs.shape`, `obs.dtype`, `obs.encoding`, `obs.data`, plus the H4-plan section-4 metadata: `pixel_source="godot_windowed_viewport"`, `capture_point="RenderingServer.frame_post_draw"`, `headless_allowed=false`, `viewport_width`, `viewport_height`). Add Python default-transport tests for the new request/response shape and protocol errors on unknown mode and pixel-dimension mismatch. No Godot-side viewport capture yet; that is plan implementation step 4. No live Godot pixel smoke yet; that is step 5.

**Blockers:**

- None for H4 Step 2 implementation. Plan amendments and Step 1 plumbing are landed; protocol extension is the natural next slice.
- H4 windowed-mode capture pops a real OS window during runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H3 closure record.** `docs/grok-h3-final-green.md` captures the verbatim Grok verdict in section 2 (deviation from H1 closure pattern, which captured verdict-only). Closure unblocked H4 implementation per `docs/sight-h4-plan.md` "Phase gates first".
- **H4 plan amendments live.** Section 1 enforces explicit headless rejection at construction. Decision 4 (wire payload schema) extends pixel obs with `pixel_source`, `capture_point`, `headless_allowed`, viewport dimensions. Section 9 and section 10 criterion 6 require step-by-step scripted trajectory equality, not first-pixel only. Step 2 must implement the wire payload extensions.
- **H4 Step 1 default-test count: 169 passed, 1 deselected (live_godot smoke).** Net delta from H3 baseline (121 passed) is +48 tests, all from the new `test_h4_godot_env_construct.py` parametrized cases. H3 default tests pass unchanged; observation_mode default of "state" preserves H3 byte-for-byte behavior.
- **H3 invariants to preserve in H4.** `_launch_godot` redirects stdout/stderr to per-run files when `run_dir` is set else `subprocess.DEVNULL` (`subprocess.PIPE` deadlocks Godot 4.6.2 on Windows; verified by 6-cell matrix on 2026-05-09). `tcp_controller.gd::_h3_dispatch` accepts both `TYPE_INT` and `TYPE_FLOAT` for `protocol_version` because Godot 4.6.2's `JSON.parse_string` widens JSON integers to `TYPE_FLOAT`. Both must not regress in H4 work.
- **H4 spike verdict locked.** `docs/sight-h4-spike.md` is the durable record. Decision 2 default is option 2 (windowed Godot viewport API), byte-deterministic across 3 runs at SHA-256 `6a2caeb8993a8d54...`. Option 1 (headless) blocked by `RenderingServer.frame_post_draw` non-emission under dummy display server. Option 3 (synthetic raster) last-resort with explicit Jeff approval. Option 4 (MSS) rejected.
