# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Step 10 (live Godot smoke behind the `live_godot` pytest marker) closed at the implementation tier. End-to-end live launch on StrongerJr is gated on `SIGHT_GODOT_EXE` being set; the marker is registered, the live test is wired, and the default gate is green at 118 passed.

**Last commit:** `b5b3fad` test(rl): add live_godot marker and live h3 godot smoke

**Current task:** Step 10 implementation done. `live_godot` marker registered in `pyproject.toml` `[tool.pytest.ini_options]`; `addopts` now excludes both `live_mss` and `live_godot` by default. Live test landed in `tests/rl/test_h3_godot_smoke.py` (not a sibling file) so the documented opt-in command stays true. Default gate `pytest tests/rl -v --tb=short` is 118 passed, 1 deselected, 11.90s on StrongerJr. Opt-in `pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short` correctly selects 1 (deselects 7) and fails fast with the intended actionable message because `SIGHT_GODOT_EXE` is unset; this proves the marker is registered, the test is collected, and the env-var-missing fail path triggers.

**Next action:** Jeff sets `SIGHT_GODOT_EXE` at User scope on StrongerJr (Claude did not auto-set; charter says no install-path guessing on opt-in acceptance gates). Discovered candidate path: `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe` (windowed) or the `_console.exe` sibling (surfaces Godot stderr to stdout; useful for the first live run). After the env var is set, rerun the opt-in command. On green, H3 acceptance criterion 9 is met; then write the H3 phase-gate packet and close H3.

**Blockers:**

- `SIGHT_GODOT_EXE` unset on StrongerJr. Verified empty at User, Machine, and Process scope. Operational gate only; not an evidence problem.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- `pyproject.toml`: `addopts = "-ra -m 'not live_mss and not live_godot'"`; `markers` extended with `"live_godot: live Godot Signal Dodge H3 smoke test (requires SIGHT_GODOT_EXE)"`. No `pytest.ini` introduced. `live_mss` behavior preserved verbatim. `[tool.pytest.ini_options]` was the existing block; reused per GPT's pre-plan decision.
- New `test_live_godot_reset_and_100_step_smoke` in `tests/rl/test_h3_godot_smoke.py`: real Popen + real loopback TCP via the env's default factories, isolated port from a bind-to-0 helper, `tmp_path` as `run_dir`, `max_steps=120` so the env-level clamp does not truncate the rollout, action 1 (stay) for up to 100 steps or until terminated/truncated, per-step assertions on obs/reward/term/trunc/info shape and types, transport/protocol/remote errors propagate (no malformed protocol assertion), `env.close()` in `finally`. Post-close reads `<run_dir>/godot.ndjson` and asserts the H3 plan-section-7 minimum event-type set: `run_start`, `controller_connected`, `controller_hello`, `controller_reset_received`, `episode_start`, `h3_step`. `collision`/`death`/`run_end` deliberately not required at this tier (terminal-contingent + shutdown-timing-sensitive). If `<run_dir>/python.ndjson` exists, asserts no event has `type == "error"`. Module docstring rewritten to reflect both tiers; the prior "No live Godot. No real subprocess. No `live_godot` marker." line was step-9 load-bearing and is now stale.
- Production code untouched. `factories.py`, `godot_env.py`, `godot_transport.py`, `train.py`, `evaluate.py`, `godot_config.py`, all GDScript under `games/signal-dodge/scripts/` byte-for-byte unchanged. Step 10 is purely test infrastructure plus marker plumbing.
- The `userMemories` claim that Godot 4.3+ is not yet installed on StrongerJr is stale. Godot 4.6.2 windowed and console binaries exist at the WinGet path above. The handoff's Notes is the right place for this until the next memory refresh.
- Convention reminder for next session: by the userMemories two-commit pattern, "Last commit" tracks the substantive commit hash, not literal HEAD. After this session HEAD will be the chore: refresh handoff hash commit, while "Last commit" above is the step-10 substantive. That mismatch is the convention, not staleness.
