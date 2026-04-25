# Signal Dodge

Minimal Godot 4 test harness for the Sight agent loop. Not a game. Not polished.
Square player dodges falling square hazards. Collision ends the run. Events stream to NDJSON.

As of commit after d30c323: movement is driven by an in-Godot rule agent, not the keyboard.
The loop is end-to-end autonomous.

## Parameters (locked for P2)

| Parameter | Value |
| --- | --- |
| Physics tick rate | 60 Hz |
| Spawn interval | 30 frames (0.5 s) |
| Spawn edge | Top only |
| Hazard speed | 200 px/sec (downward) |
| Player speed | 300 px/sec (horizontal only) |
| Screen | 720 x 540 |
| Player size | 32 x 32 |
| Hazard size | 24 x 24 |
| Agent RNG seed | 42 (global `seed()`) |
| Alignment threshold | 28 px (player_half + hazard_half) |

All movement, spawning, and logging run in `_physics_process`.
`Main._physics_process` is the single authoritative loop driver.
Player and Hazard no longer have their own `_physics_process`; Main calls `player.move_action()` and `hazard.step()` explicitly.

## Loop order (per physics tick)

1. Advance hazards: `hazard.step(delta)`, cull offscreen
2. Agent capture: read player + hazard positions into a plain dict
3. Agent perceive: find nearest aligned hazard above player (within 28 px column)
4. Agent decide: return action in `{-1, 0, +1}`; dodge away from threat, stay if safe
5. Controller: `player.move_action(action, delta)`
6. Log `agent_tick` with frame, player_x, action, threat info
7. Spawn hazard if `frame % 30 == 0`
8. Update survival label

## Scene tree

```
Main (Node2D)                    [scripts/main.gd]
  Player (Area2D)                [scripts/player.gd]
    CollisionShape2D (32x32 rect)
  Agent (Node)                   [scripts/agent.gd]
  SurvivalLabel (Label)
  <Hazard>... (spawned at runtime, each Area2D)   [scripts/hazard.gd]
    CollisionShape2D (24x24 rect)
```

## Files

```
games/signal-dodge/
  project.godot
  .gitignore
  README.md
  scenes/
    main.tscn
    player.tscn
    hazard.tscn
  scripts/
    main.gd       # orchestrator, loop order, spawn timer, end-of-run
    player.gd     # move_action(action, delta), collision signal
    hazard.gd     # step(delta), offscreen cull
    agent.gd      # capture / perceive / decide (pure-ish pipeline)
    logger.gd     # autoload, NDJSON writer
```

## Run

1. Install Godot 4.3 or newer
2. Open `games/signal-dodge/project.godot` in the editor, or `godot --path games/signal-dodge`
3. Press F5 to run (editor) or it runs as the main scene headlessly if launched directly
4. Process quits automatically on collision
5. Logs: `user://runs/run_<timestamp>.ndjson`
   - Windows resolves `user://` to `%APPDATA%\Godot\app_userdata\Signal Dodge\`

Determinism: `seed(42)` makes `randf_range` calls repeatable. Same hazard spawn sequence every run. Agent is stateless rule-based. Under a fixed tick rate the run is reproducible modulo floating-point tick timing.

## NDJSON event schema

Every record has `t` (seconds since `run_start`) and `type`. Additional fields vary.

Event types:

| type | fields |
| --- | --- |
| `run_start` | `path`, `seed` |
| `agent_tick` | `frame`, `player_x`, `action` (-1/0/+1), `threat` (bool), and if threat: `threat_x`, `threat_y`, `threat_dist` |
| `spawn` | `x`, `y` |
| `collision` | `player_x`, `player_y`, `hazard_x`, `hazard_y` |
| `death` | `survival_time` |
| `run_end` | (none) |

Sample (hand-written to emitter format; first real run on Godot-equipped host will confirm):

```json
{"t":0.0,"type":"run_start","path":"C:/Users/Jeff/AppData/Roaming/Godot/app_userdata/Signal Dodge/runs/run_2026-04-24T15-02-11.ndjson","seed":42}
{"t":0.017,"type":"agent_tick","frame":1,"player_x":360.0,"action":0,"threat":false}
{"t":0.5,"type":"agent_tick","frame":30,"player_x":360.0,"action":0,"threat":false}
{"t":0.5,"type":"spawn","x":214.33,"y":-24.0}
{"t":1.3,"type":"agent_tick","frame":78,"player_x":360.0,"action":0,"threat":false}
{"t":1.0,"type":"spawn","x":355.11,"y":-24.0}
{"t":1.5,"type":"agent_tick","frame":90,"player_x":360.0,"action":-1,"threat":true,"threat_x":355.11,"threat_y":176.0,"threat_dist":332.0}
{"t":1.517,"type":"agent_tick","frame":91,"player_x":355.0,"action":-1,"threat":true,"threat_x":355.11,"threat_y":179.3,"threat_dist":328.7}
{"t":2.4,"type":"collision","player_x":340.2,"player_y":508.0,"hazard_x":340.1,"hazard_y":502.0}
{"t":2.4,"type":"death","survival_time":2.4}
{"t":2.4,"type":"run_end"}
```

## Known limitations (by design, for P2)

- `agent_tick` logs every physics frame (60 Hz). ~1800 lines per 30 s run. Evaluator will downsample in P3.
- Rule policy is trivial: dodge nearest aligned hazard. No look-ahead, no multi-threat balancing, no edge avoidance. Expected to die within a few seconds under dense spawns.
- No reset mechanic. Process exits on collision. Re-run by relaunching.
- Hazards that exit offscreen without collision are not logged. Add `hazard_exit` in P3 if needed.
- Python agent layer (capture via screenshot + OpenCV perception) is deferred. This in-Godot agent is the determinism and wiring proof.
- UIDs omitted from .tscn files. Godot regenerates on first editor open; commit the resulting diff.


## TCP controller mode (optional, P2 Python layer)

The default loop runs the in-Godot rule agent. To drive the player from an external Python
agent instead, launch Godot with `SIGHT_TCP_MODE=1`:

```powershell
$env:SIGHT_TCP_MODE = "1"
# optional: $env:SIGHT_TCP_PORT = "8765"
godot --path C:\Projects\Sight\games\signal-dodge
```

In TCP mode `scripts/tcp_controller.gd` listens on `127.0.0.1:8765`. The Python client (see
`src/sight_agent/controller/tcp_client.py`) sends a `hello` and then one `action` per
decision. Godot holds the previous action if no new command arrived that frame, and emits a
`controller_cmd_applied` event with `seq`, `frame`, `action`, `move_x` for reconciler join.

Loopback only. No external network surface.
