# Signal Dodge - Game Specification

First Sight target. Custom Godot micro-game owned by Jeff.

## Scene

- Player (square)
- Hazard spawner
- Boundaries

## Controls

- Arrow keys or WASD

## Hazards

- Spawn every N frames
- Move in a straight line toward the screen

## End condition

- Collision between player and hazard

## UI

- Survival timer

## Omitted (intentional)

- No animations
- No sound
- No menus

## Open for P2 tuning (values needed before first build)

- `N` (hazard spawn interval, frames)
- Spawn edge(s): top only, or all four
- Hazard velocity (fixed scalar, or range)
- Play area dimensions
- Player movement speed
- Player square size, hazard square size

## Why this shape works for Sight

- Square-on-solid-color perception reduces to HSV range or template match; no vision model needed for P2
- Straight-line trajectories are deterministic, so policy improvement shows cleanly in survival-time deltas
- Single terminal condition (collision) yields unambiguous episode boundaries for the logger
- No animations or audio means no frame-timing or state-machine confounds in the capture loop
