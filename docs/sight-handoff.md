# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 closure pending Grok GREEN verdict on `docs/grok-h3-phase-gate-packet.md`. H3 implementation complete at the technical-acceptance tier. H4 plan drafted at `docs/sight-h4-plan.md`. H4 pre-implementation viewport spike complete; verdict and durable record at `docs/sight-h4-spike.md`. H4 implementation remains gated on H3 closure GREEN.

**Last commit:** `11398ab` docs: record h4 viewport spike verdict (option 2, windowed)

**Current task:** Awaiting Grok verdict on the H3 phase-gate packet. The self-contained review bundle is at `runs/handoff/sight-h3-grok-review-bundle.md` (46098 bytes, gitignored). H4 spike resolved Decision 2: option 2 (windowed Godot viewport API) is the default pixel source. Option 1 (headless) blocked because Godot 4.6.2's `--headless` dummy display server does not emit `RenderingServer.frame_post_draw`. Option 3 (synthetic raster) remains a last-resort fallback, not invoked. Option 4 (MSS) remains rejected.

**Next action:** Jeff hands the H3 packet to Grok and relays the verdict. On GREEN, Claude updates this handoff to record H3 closure with verbatim Grok verdict text, then begins H4 implementation per `docs/sight-h4-plan.md` sections 1-10 with the spike-resolved Decision 2: windowed mode pixel capture, viewport size set deliberately at env construction, Signal Dodge scene drawn to fit. On YELLOW or RED, address only the flagged items.

**Blockers:**

- Grok review of H3 phase-gate packet is the only gate for H3 closure.
- H4 implementation gate is H3 closure GREEN.
- H4 windowed-mode capture pops up a real OS window during runs. Not a blocker for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 spike verdict.** `docs/sight-h4-spike.md` is the durable record. Three windowed runs of the quarantined probe each produced byte-identical 7056-byte grayscale captures across frames 0/5/50; SHA-256 `6a2caeb8993a8d54...`. Three headless runs each timed out because `await RenderingServer.frame_post_draw` never returns under the dummy display server. Probe meta confirms `display_driver=headless` under `--headless` and Vulkan `Forward+ - NVIDIA GeForce RTX 2060` under windowed.
- **H4 plan locks pixel-source ranking.** Default for H4 implementation: option 2 (windowed Godot viewport API). H4 env contract must enforce windowed launch when `observation_mode=pixel`. Option 1 (headless root viewport) blocked by spike evidence; SubViewport-with-explicit-render-target headless path is parked for H5+ if needed. Option 3 (synthetic raster) is last-resort with explicit Jeff approval and H5 contamination disclosure. Option 4 (MSS) rejected.
- **H3 acceptance runs.** `runs/eval/h3_acceptance/run{1,2}/test_live_godot_reset_and_100_0/` each carry `python.ndjson`, `godot.ndjson`, `godot-stdout.log`, `godot-stderr.log`. All H3 plan section 7 minimum event types present, `h3_step` count=100, no collisions, same-seed first-step state matches. `runs/` is gitignored.
- **`SIGHT_GODOT_EXE` durable.** Set at User scope to the WinGet console build path. Persists across shells. Console build choice was orthogonal to the H3 PIPE deadlock per the 6-cell matrix.
- **H3 invariants to preserve in H4.** `_launch_godot` redirects stdout/stderr to per-run files when `run_dir` is set else `DEVNULL` (subprocess.PIPE deadlocks Godot 4.6.2 on Windows). `tcp_controller.gd::_h3_dispatch` accepts both `TYPE_INT` and `TYPE_FLOAT` for `protocol_version` because Godot 4.6.2's `JSON.parse_string` widens JSON integers to `TYPE_FLOAT`. Both load-bearing for the live gate.
