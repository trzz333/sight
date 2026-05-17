# Sight Demo-0 - Visible AI Playing Signal Dodge

Evidence for the first end-to-end watchable artifact of a trained CNN policy controlling Signal Dodge. This is a demo deliverable, **not** H5 acceptance evidence and not a product claim.

## TL;DR

- Tool: `tools\demo0_visible_play.py` (new this session).
- Model: `runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed2_entropy_10k\model.zip` (Phase E seed=2, entropy recipe, 10000 timesteps).
- Eval seed: 1008. Episodes: 1. Deterministic argmax. Visible (windowed) Godot, pixel mode 1x84x84 uint8.
- Result: 1800 steps, total_reward = 1800.0, terminated=false, truncated=true, terminal_reason="timeout", wall = 50.0 s.
- Behavior: every action across 1800 steps was `left`. No deviation. The episode survived only because seed 1008's hazard pattern happens to be survivable by a constant-left agent on this map.
- Artifact: 1801 frames captured, MP4 (336x336 grayscale, 20 fps) and PNG sequence written, plus per-step NDJSON sidecar and manifest.

This is the wedge / constant-action behavior K0-10k diagnosed at training time. The agent commits to one basin and rides it. Survival here is luck of the seed, not learned avoidance competence.

## Reproduction

Exact command run from `C:\Projects\Sight` in cmd.exe:

```
set SIGHT_GODOT_EXE=C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe

"C:\Users\maste\AppData\Local\Python\bin\python.exe" -u tools\demo0_visible_play.py ^
  --train-run-dir runs\rl\signal_dodge_ppo_h5_pixel_entropy\h5_train_phase_e_seed2_entropy_10k ^
  --seed 1008 ^
  --out-dir runs\demo0\trained_seed2_eval1008 ^
  --episodes 1 ^
  --fps 20 ^
  --upscale 4
```

`SIGHT_GODOT_EXE` must be set in the active shell prior to invocation; user-scope env vars are not reliably inherited by Godot launches under this project's transport pipeline.

## Tooling

`tools\demo0_visible_play.py` (this session):

- Loads `config_effective.yaml` and `model.zip` from a completed H5 train run.
- Builds `sight_agent.rl.godot_env.GodotSignalDodgeEnv` directly with `headless=False` (windowed Godot is required for pixel mode per the H4 spike).
- Steps the env with `model.predict(obs, deterministic=True)`.
- Captures every step's pixel observation (shape `(1,84,84)`, uint8) as a grayscale frame.
- Writes upscaled frame PNGs, an MP4 via OpenCV `VideoWriter` (mp4v codec), a `steps.ndjson` per-step log, and a `manifest.json`.
- No new evaluator infrastructure; the existing `sight_agent.rl.evaluate` CLI writes NDJSON metrics only, with no frame or video output, so a small dedicated tool was the right scope.

Compile-checked. Imports verified. Env reset/step 5-tuple Gymnasium API confirmed against `src\sight_agent\rl\godot_env.py`.

## Artifact paths

`runs\demo0\trained_seed2_eval1008\` (all paths relative to repo root):

| File | Size | Notes |
|---|---|---|
| `manifest.json` | small | model path, seed, git commit, terminals, caveat |
| `steps.ndjson` | 244 KB | 1800 lines, one per step: action, reward, terminated, truncated, terminal_reason |
| `demo0.mp4` | 456 KB | 1801 frames, 336x336 grayscale, 20 fps, mp4v codec |
| `frames\frame_NNNNN.png` | 1801 files | 336x336 grayscale PNGs (nearest-neighbor 4x upscale of the 84x84 obs) |
| `godot\godot.ndjson`, `godot\godot-stdout.log`, `godot\godot-stderr.log` | varies | Godot-side per-step log and process stdout/stderr from the launched windowed Godot |

The PNGs and MP4 frames are exactly what the policy saw: 84x84 grayscale, nearest-neighbor upscaled 4x for legibility. No overlay, no annotation, no telemetry burned in. The MP4 is the watchable artifact.

Artifact sha256:

- `demo0.mp4`         `3a61c098f22e99ec3b110714005a6d79babaf4c94054beb404b98db960e7e6b8`
- `steps.ndjson`      `33b7a43280151021eebd2247018e42dee3c0e380eabdbd057a65db93e9eeb893`
- `manifest.json`     `0e341ccaaa4dc33495a2207a5d7f10282459c1bd5433e60a540ce95b67e7f559`

## Per-step behavior

From `steps.ndjson` aggregation:

```
n_steps          = 1800
action_counts    = {'left': 1800}
total_reward     = 1800.0
first action change index = none
last 5 actions   = ['left', 'left', 'left', 'left', 'left']
```

Every single step picked action `left`. This matches the K0-10k training-time finding: this train run's deterministic argmax locks to `left` at update 9 (timestep 2304) and never reverses through update 40. The K0-10k handoff documents this as steady PPO commitment, not value/advantage degeneration. Demo-0 confirms that at deployment time, with `deterministic=True`, that lock-in persists end-to-end across an entire episode regardless of pixel observation content.

`terminal_reason="timeout"` (truncated, not terminated) means the agent did not collide. The 1800-step cap fired first. This is not evidence the policy avoids hazards. It is evidence that seed 1008's hazard placement is survivable by a constant-left strategy on this map. Other seeds would not be expected to behave the same way.

## Watchability

The MP4 is 90 seconds of playback at 20 fps. A human opening `demo0.mp4` in any media player sees:

- Signal Dodge in 84x84 grayscale, upscaled to 336x336.
- A player paddle pinned to the left side of the play area for the entire duration.
- Hazards scrolling past. None hit the player on this seed.

Success criterion per GPT's Demo-0 spec ("a human can open an artifact on disk and watch Signal Dodge being controlled by a trained CNN policy"): **met**.

## Caveat

The Phase E seed=2 entropy-recipe policy is the best currently available trained checkpoint and is also a known wedge attractor. Demo-0 documents the policy's actual decisions on a live pixel observation stream with full fidelity. It does not certify learning closure, acceptance-grade behavior, or generalization. The artifact's "successful" 1800-step survival is a coincidence of seed selection, not an indicator of avoidance competence.

## Next technical slice after demo

K1 architecture probe (`policy_kwargs.net_arch = dict(pi=[64], vf=[64])`, train_seed=2, 10000 timesteps, same entropy YAML otherwise), pending Jeff's explicit go signal. K1 should be labeled a mechanism probe, not an acceptance eval. Demo-0 does not change K1's framing or comparator anchors.
