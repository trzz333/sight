# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 CLOSED via Grok GREEN verdict on 2026-05-09; closure artifact at `docs/grok-h3-final-green.md` with verbatim verdict text. H4 implementation authorized. H4 plan amended in this docs commit with three caveat-driven changes (headless rejection at construction, pixel metadata schema, step-by-step trajectory equality). H4 Step 1 (env constructor extension) is the immediate next code slice.

**Last commit:** `__SUBSTANTIVE_HASH__` docs(h3,h4): record h3 grok green closure and amend h4 plan with caveats

**Current task:** H3 closure recorded. H4 plan amendments landed (sections 1, 4 wire-payload schema, 9 determinism, 10 criterion 6). Repo `main` is the docs-only checkpoint before H4 code. Next slice is H4 Step 1: extend `GodotSignalDodgeEnv` with `observation_mode` and pixel constructor args, default `state` mode unchanged, reject `headless=True` for `pixel`/`both` modes per the new plan section 1 rule.

**Next action:** Implement H4 Step 1 in `src/sight_agent/rl/godot_env.py`: add `observation_mode: Literal["state","pixel","both"]="state"`, `pixel_width=84`, `pixel_height=84`, `pixel_channels=1` constructor params; raise `ValueError` on invalid mode; raise `ValueError` when `observation_mode in {"pixel","both"}` and resolved `headless=True`; select `observation_space` per mode (Box(-1,1,(10,),float32) for state, Box(0,255,(1,84,84),uint8) for pixel, Dict for both). Add tests: state-mode default unchanged, constructor validation for valid/invalid modes, pixel-mode headless rejection, observation-space shape/dtype assertions for pixel and both. Run `pytest tests/rl -v --tb=short`. Commit and push if green.

**Blockers:**

- None for H4 Step 1. H3 closure GREEN unblocks H4 implementation per plan section "Phase gates first".
- H4 windowed-mode capture pops a real OS window during runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H3 closure artifact pattern.** `docs/grok-h3-final-green.md` follows `docs/grok-h1-final-green.md` structure with one deviation: H3 captures the verbatim Grok verdict text in section 2 because Jeff explicitly relayed it for audit. H1 closure recorded the verdict-only decision per its own pattern.
- **H4 plan amendments (this commit).** Section 1 adds explicit headless-rejection rule for `pixel`/`both` modes. Section 4 (Decision 4: transport, wire payload schema) extends the pixel observation payload with `pixel_source`, `capture_point`, `headless_allowed`, and explicit viewport dimensions. Section 9 (Determinism posture) and section 10 criterion 6 are tightened from first-pixel equality to step-by-step scripted trajectory equality. These amendments are docs-only and land before any H4 production code per Grok's execution instructions.
- **H3 invariants to preserve in H4.** `_launch_godot` redirects stdout/stderr to per-run files when `run_dir` is set else `subprocess.DEVNULL` (`subprocess.PIPE` deadlocks Godot 4.6.2 on Windows; verified by 6-cell matrix on 2026-05-09). `tcp_controller.gd::_h3_dispatch` accepts both `TYPE_INT` and `TYPE_FLOAT` for `protocol_version` because Godot 4.6.2's `JSON.parse_string` widens JSON integers to `TYPE_FLOAT`. Both load-bearing for the live gate and must not regress in H4 work.
- **H4 spike verdict locked.** `docs/sight-h4-spike.md` is the durable record. Decision 2 default is option 2 (windowed Godot viewport API), byte-deterministic across 3 runs at SHA-256 `6a2caeb8993a8d54...`. Option 1 (headless) blocked by `RenderingServer.frame_post_draw` non-emission under dummy display server. Option 3 (synthetic raster) last-resort with explicit Jeff approval. Option 4 (MSS) rejected.
- **H3 acceptance runs durable.** `runs/eval/h3_acceptance/run{1,2}/test_live_godot_reset_and_100_0/` each carry `python.ndjson`, `godot.ndjson`, `godot-stdout.log`, `godot-stderr.log`. All H3 plan section 7 minimum event types present, `h3_step` count=100, no collisions, same-seed first-step state matches. `runs/` is gitignored; artifacts regenerable via `pytest tests/rl/test_h3_godot_smoke.py -m live_godot`.
