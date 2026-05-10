# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 closure pending Grok GREEN verdict on `docs/grok-h3-phase-gate-packet.md`. H3 implementation complete at the technical-acceptance tier (live and default gates green on StrongerJr; two acceptance runs captured with same-seed first-step reproducibility). H4 plan drafted at `docs/sight-h4-plan.md` but H4 implementation is gated on H3 closure.

**Last commit:** `181c104` docs: add h4 pixel observation plan

**Current task:** Awaiting Grok verdict on the H3 phase-gate packet. The self-contained review bundle for Grok is at `runs/handoff/sight-h3-grok-review-bundle.md` (46098 bytes, gitignored). On GREEN, Claude updates this handoff to record H3 closure with verbatim Grok verdict text or a committed verdict artifact, then begins the H4 pre-implementation spike per `docs/sight-h4-plan.md` section 0. On YELLOW, address only the flagged items. On RED, reopen `docs/sight-h3-plan.md` section 10 acceptance criteria.

**Next action:** Jeff hands the H3 packet to Grok. Grok can read source files directly from the public GitHub repo at https://github.com/trzz333/sight/ if its browsing is enabled, or work from `runs/handoff/sight-h3-grok-review-bundle.md` if not. Claude does not record an H3 verdict in this handoff without verbatim Grok verdict text in session or a committed verdict artifact in the repo.

**Blockers:**

- Grok review of H3 phase-gate packet is the only gate for H3 closure.
- H4 implementation gate is H3 closure GREEN. The H4 pre-implementation spike (`docs/sight-h4-plan.md` section 0) does not land production code and may run in parallel to Grok review if Jeff prefers, but no production H4 code lands on `main` before H3 closes.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 plan locks pixel-source ranking before any code.** Preferred is Godot viewport texture capture under `--headless`. Second is windowed viewport API. Last resort is deterministic Godot-side synthetic raster, only with explicit Jeff approval and an H5 plan amendment acknowledging that synthetic raster does not exercise rendered-pixel learning. MSS / OS screen capture rejected outright. Pre-implementation spike (`docs/sight-h4-plan.md` section 0) selects which option becomes default. Spike artifacts go under `runs/eval/h4_spike/`.
- **H3 acceptance runs.** `runs/eval/h3_acceptance/run{1,2}/test_live_godot_reset_and_100_0/` each carry `python.ndjson`, `godot.ndjson`, `godot-stdout.log`, `godot-stderr.log`. All H3 plan section 7 minimum event types present, `h3_step` count=100, no collisions, same-seed first-step state matches across both runs. `runs/` is gitignored; artifacts live durably on disk on StrongerJr.
- **H3 review bundle.** `runs/handoff/sight-h3-grok-review-bundle.md` (46098 bytes, gitignored under `runs/`). Self-contained export of `docs/sight-charter.md`, `docs/sight-h3-plan.md`, `docs/grok-h3-phase-gate-packet.md`, and `docs/sight-handoff.md` with header asking for GREEN/YELLOW/RED verdict. Use this if Grok cannot browse public GitHub.
- **`SIGHT_GODOT_EXE` durable.** Set at User scope to `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe`. Persists across shells. Console build choice was orthogonal to the H3 PIPE deadlock per the 6-cell matrix on 2026-05-09.
- **H3 invariants to preserve in H4.** `_launch_godot` redirects stdout/stderr to `<run_dir>/godot-stdout.log` / `godot-stderr.log` when `run_dir` is set else `DEVNULL` (subprocess.PIPE deadlocks Godot 4.6.2 on Windows). `tcp_controller.gd::_h3_dispatch` accepts both `TYPE_INT` and `TYPE_FLOAT` for `protocol_version` because Godot 4.6.2's `JSON.parse_string` widens JSON integers to `TYPE_FLOAT`. Both load-bearing for the live gate.
