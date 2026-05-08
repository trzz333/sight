# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Implementation Sequence step 4 (Signal Dodge state observation builder + H3 step action wiring + pause-world cadence in main.gd) complete and Godot-side parse-validated under 4.6.2. Step 5 (Python transport request-response client and `GodotSignalDodgeEnv`) is next.

**Last commit:** `937f0f8` feat(godot): H3 step 4 real obs builder + step wiring + pause-world cadence

**Current task:** H3 step 4 closed and validated. Ready for step 5 of `docs/sight-h3-plan.md` Implementation Sequence.

**Next action:** Step 5: implement Python-side bidirectional TCP transport (request-response client, newline-delimited JSON, loopback only) under `src/sight_agent/`. Wire the `hello` -> `reset` -> `step` -> `step_result` exchange against the Godot server already implemented in `tcp_controller.gd` + `main.gd`. After transport lands, step 6 is `GodotSignalDodgeEnv` under `src/sight_agent/rl/godot_env.py` and step 7 is the `godot:signal-dodge-v0` factory branch in `factories.py`. The H3 wire contract (section 7 of the plan) and the cadence assumption (one Python `step` request -> exactly one Godot physics frame -> one `step_result`) are both load-bearing for the Python client design. Reward is sparse-survival; terminal_reason is `"collision"` / `"timeout"` / `""` per main.gd.

**Blockers:**

- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only, not Sight evidence blocker. Sight sessions run on standalone DC remote MCP (deviceId 64416a67-1bdb-42fc-bf1a-48f988e6901d).

**Notes:**

- Cadence (locked, GPT step-4 directive): H3 mode (locked once tcp_controller sees a `protocol_version` field) takes a fast-path branch at the top of `_physics_process` that bypasses the legacy autonomous loop and the `_alive` gate. `_h3_physics_tick` polls every physics tick (60 Hz) but only advances state when it consumes a valid reset/step request. Idle ticks advance nothing (no frame counter, no hazards, no spawn, no player movement, no reward, no `player_tick` log). Exactly one consumed `step` request advances exactly one Godot physics frame and produces one `step_result`. Legacy v1 TCP and non-TCP in-Godot agent loops are unchanged.
- Mode-locking transition tick: when `_tcp.poll()` first locks H3 inside the legacy block, a parked `reset` is dispatched inline (state-clobber wipes the legacy world advance that already happened that tick). A `step` arriving before `reset` is a Python protocol error and gets `error bad_request` via `TCP_CONTROLLER.ERROR_BAD_REQUEST`; subsequent ticks take the H3 fast path. Step on a `_h3_episode_done` episode also replies `bad_request`.
- Observation contract (`_h3_build_observation`): 10-element float `Array`, all values `clampf(x, -1.0, 1.0)`. Layout per plan section 2: `[player_x_norm, player_vx_norm, h0_dx, h0_dy, h0_present, h1_dx, h1_dy, h1_present, h2_dx, h2_dy]`. Third hazard has no explicit present flag per spec; absence signaled by both dx and dy being 0.0. `player_vx_norm` = last applied wire-mapped action (-1/0/+1). Hazard order is deterministic: filter to `hazard.y <= player.y`, sort primary by smallest positive (player.y - hazard.y), secondary by smallest `abs(hazard.x - player.x)`, tertiary by `h3_spawn_id` meta stamped in `_spawn_hazard`. `h3_spawn_id` resets to 0 on each soft reset.
- Collision in H3 mode flips `_h3_step_terminated` / `_h3_terminal_reason="collision"` and records `_h3_collision_info`; `_h3_perform_step` sees the flag synchronously after `_player.move_action(...)` and emits a terminated `step_result` with reward 0.0. TCP is NOT stopped, `get_tree().quit()` is NOT called. Reset re-arms the episode in place. Timeout uses `max_steps` from reset and produces `truncated=true`, `terminal_reason="timeout"`. Both terminals set `_h3_episode_done=true` until the next reset.
- Validation gate this round: `Godot_v4.6.2-stable_win64_console.exe --headless --path C:\Projects\Sight\games\signal-dodge --quit-after 5` exit 0 with banner only and no error/parse/warning lines; verbose `--quit-after 60` confirms `Loading resource:` for logger.gd, main.gd, hazard.gd, player.gd, agent.gd (tcp_controller.gd parses transitively via main.gd's preload const). `pytest tests/rl -v --tb=short` 48 passed (Python untouched, step-2 baseline preserved across step 3 and step 4). Live Python<->Godot smoke deferred to step 5+.
- HEAD progression: `5293605` (handoff post Godot install) -> `937f0f8` (H3 step 4 code) -> handoff refresh (this commit).
