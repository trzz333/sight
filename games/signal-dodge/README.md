# Signal Dodge

Minimal Godot 4 test harness for the Sight agent loop. Not a game. Not polished.
Square player dodges falling square hazards. Collision ends the run. Events stream to NDJSON.

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
| Controls | Arrow keys or WASD |

All movement, spawning, and logging run in `_physics_process` for cross-machine consistency.

## Scene tree

```
Main (Node2D)                    [scripts/main.gd]
  Player (Area2D)                [scripts/player.gd]
    CollisionShape2D (32x32 rect)
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
    main.gd       # orchestrator, spawn timer, end-of-run
    player.gd     # input, movement, clamp, collision
    hazard.gd     # fall, offscreen cleanup
    logger.gd     # autoload, NDJSON writer
```

## Run

1. Open the project in Godot 4.3 or newer (`godot --editor project.godot`)
2. Press F5 to run, or `godot project.godot`
3. Quits automatically on collision
4. Logs written to `user://runs/run_<timestamp>.ndjson`
   - Windows path: `%APPDATA%\Godot\app_userdata\Signal Dodge\runs\`

## NDJSON event schema

Every record has `t` (seconds since `run_start`) and `type`. Additional fields vary.

Sample (hand-written, matching emitter output format):

```json
{"t":0.0,"type":"run_start","path":"C:/Users/Jeff/AppData/Roaming/Godot/app_userdata/Signal Dodge/runs/run_2026-04-24T14-32-10.ndjson"}
{"t":0.017,"type":"player_tick","x":360.0,"y":508.0}
{"t":0.033,"type":"player_tick","x":360.0,"y":508.0}
{"t":0.5,"type":"spawn","x":214.3,"y":-24.0}
{"t":0.517,"type":"player_tick","x":365.0,"y":508.0}
{"t":1.0,"type":"spawn","x":487.9,"y":-24.0}
{"t":2.4,"type":"player_tick","x":378.2,"y":508.0}
{"t":2.4,"type":"collision","player_x":378.2,"player_y":508.0,"hazard_x":378.2,"hazard_y":502.0}
{"t":2.4,"type":"death","survival_time":2.4}
{"t":2.4,"type":"run_end"}
```

## Known limitations (by design, for P2)

- `player_tick` logs at 60 Hz. High volume (~1800 lines / 30 s run). Evaluator will downsample.
- No difficulty scaling, no multiple hazard types, no reset mechanic. Close window or relaunch.
- No animation, no sound, no menu. Harness only.
- `user://` path is OS-dependent. Evaluator will resolve at read time.
- Hazards that exit offscreen without collision are not logged as events. Add `hazard_exit` in P3 if evaluator needs it.
- UID directives omitted from scenes. Godot will generate on first editor open and rewrite the files. Commit the regenerated UIDs when that happens.
