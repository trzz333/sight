# Sight H4 Pre-Implementation Spike — Summary

Recorded outside `runs/` for durability. The spike's raw artifacts live under `runs/eval/h4_spike/` (gitignored). This document is the durable verdict.

## Goal

Resolve `docs/sight-h4-plan.md` Decision 2 (pixel source) before any H4 production code lands. Test, in order: option 1 (Godot viewport texture capture under `--headless`) and option 2 (Godot viewport API in windowed mode). Do not test option 4 (MSS/screen capture). Escalate before option 3 (synthetic raster) if both viewport options fail.

## Method

Quarantined Godot probe project under `runs/eval/h4_spike/godot_probe/` (no imports from `sight_agent.rl.envs`, no edits to production code). The probe draws four deterministic shapes via `_draw()`, then captures the root viewport at frames 0, 5, 50 by `await RenderingServer.frame_post_draw`, `Viewport.get_texture().get_image()`, `convert(FORMAT_L8)`, `resize(84, 84, INTERPOLATE_NEAREST)`, and `get_data()` to a 7056-byte buffer. Per-run artifacts under `runs/eval/h4_spike/runs_{headless,windowed}/run{1,2,3}/`. SHA-256 of frame bytes used for byte-equality comparison.

Godot binary: `4.6.2.stable.official.71f334935`. Console build at `SIGHT_GODOT_EXE`. StrongerJr (RTX 2060 + Intel UHD 630 hybrid GPU). Subprocess pattern follows the H3 invariant: real file handles for stdout/stderr, never `subprocess.PIPE`.

## Findings

### Option 1: headless viewport capture — BLOCKED

Three independent runs under `--headless` with a 10-second per-run timeout. All three timed out and were killed. Per-run artifacts confirm the failure mode:

- `meta.txt` shows `display_driver=headless`, `vp_size=(64.0, 64.0)`. Godot launches and `_ready()` runs.
- stdout shows `[h4_spike] ready ...` then nothing further. `_process()` never advances past frame 0.
- The dummy display server installed by `--headless` does not emit `RenderingServer.frame_post_draw`. The probe's `await RenderingServer.frame_post_draw` never returns, so `_capture()` is never called and frames 0/5/50 are never written.

This is a fundamental Godot 4 behavior under the default `--headless` flag, not a probe bug. The probe ran cleanly through `_ready()` and meta capture.

The spike did NOT test the SubViewport-with-explicit-render-target-mode path that some Godot CI screenshot pipelines use. That path may or may not work under `--headless`. Out of scope for this spike given option 2 produced a byte-deterministic result.

### Option 2: windowed viewport capture — WORKS, byte-deterministic

Three independent runs under windowed mode with a 30-second per-run timeout. All three completed cleanly:

- run1: rc=0 elapsed=2.896s
- run2: rc=0 elapsed=2.073s
- run3: rc=0 elapsed=2.083s

Vulkan banner: `Vulkan 1.4.325 - Forward+ - Using Device #0: NVIDIA - NVIDIA GeForce RTX 2060`.

All 9 captured frames (frames 0, 5, 50 across runs 1, 2, 3) hashed to the same SHA-256: `6a2caeb8993a8d54...`. The frame content is structured and non-trivial: 5 distinct grayscale values, 6963 of 7056 pixels at value 76 with minor clusters at 18, 54, 182, 237. Min/max byte values 18/237. The image is not blank, not zeroed, not random.

Byte-equality across same-seed same-mode runs holds. Capture is reproducible.

### Determinism note

Within a single run, frames 0, 5, and 50 also hashed identically. With a static `_draw()` scene (no game logic, no animation), this is expected and not evidence of frozen capture. The capture is reproducible across runs, which is the property H4 requires.

For H4 implementation, Signal Dodge's actual scene state changes per frame, so frames 0/5/50 in a real episode will differ. The spike does not test that; it tests the capture mechanism on a deterministic scene.

## Recommendation

**Option 2: Godot viewport API in windowed mode** is the H4 default pixel source.

H4 implementation must explicitly document the display dependency in the env contract: pixel mode requires a windowed Godot launch, not `--headless`. The default H3 launch posture (windowed-or-headless via the existing env config) must be extended to enforce windowed when `observation_mode=pixel`.

Option 1 (headless) is parked. If H5 or beyond needs offscreen pixel rendering, the SubViewport-with-explicit-render-target path is the candidate to revisit. Out of H4 scope.

Option 3 (synthetic raster) remains a last-resort fallback per the H4 plan. Not invoked. Jeff approval not needed.

Option 4 (MSS/screen capture) remains rejected.

## Caveats and known unknowns

- Default root viewport size on this probe was `(64, 64)`, smaller than expected. The drawn shapes were positioned outside the visible rect, so the captured 84x84 image is mostly the background fill color. This is a probe-construction artifact and not a Godot capture issue. H4 implementation will set viewport size deliberately to fit the Signal Dodge scene.
- Frame-queue / capture-timing characterization is partial. The probe awaits `frame_post_draw` before capture, which is the correct synchronization barrier. Determinism held across 3 runs at 3 frames each with a static scene. The spike did not stress dynamic-scene capture under physics-tick variance; that becomes an H4 implementation acceptance question, not a spike question.
- Vulkan windowed mode pops up a real OS window during capture. For H4 acceptance runs this is acceptable. For unattended CI it would need a virtual display, which is out of scope.
- The 84x84 grayscale 7056-byte JSON payload per step is the wire format implied by the H4 plan. The spike does not exercise the TCP transport.

## Artifacts

Spike raw artifacts under `runs/eval/h4_spike/`:
- `godot_probe/` — quarantined Godot project (project.godot, main.tscn, main.gd)
- `orchestrate.py` — Python orchestrator
- `runs_headless/run{1,2,3}/` — stdout, stderr, exit, elapsed, note (TIMEOUT_KILLED), meta, start_us
- `runs_windowed/run{1,2,3}/` — stdout, stderr, exit, elapsed, meta, start_us, frame_{0,5,50}.bin, frame_{0,5,50}_us.txt
- `summary.md` — orchestrator-generated summary, identical verdict to this document at the time of writing

`runs/` is gitignored. This `docs/sight-h4-spike.md` is the durable record.
