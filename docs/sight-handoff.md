# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation complete at the technical-acceptance tier. Live gate green on StrongerJr; both default and live gates pass; H3 phase-gate packet written. H3 closure pending Grok GREEN verdict on `docs/grok-h3-phase-gate-packet.md`.

**Last commit:** `7e4f23f` fix(rl,gd): unblock h3 live gate and ship phase-gate packet

**Current task:** Live gate `pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short` is green: 1 passed, 7 deselected, 2.6s. Default gate `pytest tests/rl -v --tb=short` 121 passed, 1 deselected, 11.67s (net +3 regression tests vs the post-step-9 baseline of 118). Two acceptance runs captured under `runs/eval/h3_acceptance/run{1,2}/test_live_godot_reset_and_100_0/` with full NDJSON evidence and same-seed first-step reproducibility verified. H3 phase-gate packet at `docs/grok-h3-phase-gate-packet.md`.

**Next action:** Hand the H3 phase-gate packet to Grok for GREEN/YELLOW/RED verdict. Acceptance criteria 1-10 from `docs/sight-h3-plan.md` section 10 all hold per the packet. On Grok GREEN, H3 closes and H4 (pixel observations on Signal Dodge with a small CNN policy) becomes the next phase under the existing charter.

**Blockers:**

- Grok review of H3 phase-gate packet is the only gate.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only; Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **Bug 1 fixed: `subprocess.PIPE` deadlocks Godot 4.6.2 on Windows.** `_launch_godot` now redirects stdout/stderr to `<run_dir>/godot-stdout.log` and `<run_dir>/godot-stderr.log` when `run_dir` is set, else `subprocess.DEVNULL`. File handles tracked on `self._godot_stdout_file` / `self._godot_stderr_file` and closed in `close()`. Three regression tests added in `tests/rl/test_h3_godot_env.py`. Verified by a 6-cell matrix on 2026-05-09 (PIPE hangs both windowed and console builds; DEVNULL and file redirection both bind in <1s; `CREATE_NO_WINDOW` does not help). Pipe-fill is ruled out (only the 73-byte engine banner is ever written).
- **Bug 2 fixed: GDScript `protocol_version` strict-int check.** Godot 4.6.2's `JSON.parse_string` widens JSON integers to `TYPE_FLOAT`, so Python's `json.dumps({"protocol_version": 2})` arrives as `2.0`. `tcp_controller.gd::_h3_dispatch` now accepts both `TYPE_INT` and `TYPE_FLOAT` and compares via `int()` coercion. JSON has no int/float distinction at the wire, so the relaxed check is the correct contract regardless of parser behavior. No GDScript test framework in this round; covered by the live gate end-to-end.
- **Acceptance runs.** `runs/eval/h3_acceptance/run1/` and `run2/` each contain `python.ndjson`, `godot.ndjson`, `godot-stdout.log`, `godot-stderr.log`. Run 1: 132 godot lines, 103 python lines, all H3 plan section 7 minimum event types present, `h3_step` count=100, no collisions. Run 2: 133 godot lines (+1 controller_connected/+1 controller_disconnect, -1 player_tick from pre-mode-lock physics-tick variance), same first-step state under seed=0. Same-seed first-step reproducibility holds. `runs/` is gitignored; artifacts live durably on disk on StrongerJr.
- **`SIGHT_GODOT_EXE` durable.** Set at User scope to `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe`. Persists across shells. Console build choice was orthogonal to the deadlock bug per the matrix; switching to the windowed build is unnecessary.
- **Packet pattern follows H2.** `docs/grok-h3-phase-gate-packet.md` mirrors the H2 packet structure: scope/non-scope, repo state, test gate, acceptance runs, reproducibility, caveats, what's not included, recommended verdict scope, pointers. Section 3 hash placeholder `7e4f23f` is patched with the substantive commit hash post-commit by the same `re.subn` pattern used for the handoff hash refresh.
