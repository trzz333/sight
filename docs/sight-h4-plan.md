# Sight H4 plan: pixel observations on Signal Dodge with CNN policy smoke

H4 extends the same Godot Signal Dodge environment to emit pixel observations and proves a small CNN policy can be constructed and run end-to-end through the existing SB3 train and eval seam. H4 is a boundary and smoke test, not a learning-quality test.

## Phase gates first

H4 implementation does not begin until H3 closes GREEN by Grok verdict on `docs/grok-h3-phase-gate-packet.md`. The H3 packet is the gate; the handoff records closure status. Until H3 closes, this plan and the pre-implementation spike (section 0) are the only H4-tagged artifacts that may land on `main`.

H4 closes only when all of the following hold simultaneously on StrongerJr:

1. `GodotSignalDodgeEnv` supports `observation_mode = "pixel"` while preserving H3 `state` mode unchanged.
2. `reset(seed=0)` and `step(action)` return valid pixel observations through the Gymnasium 5-tuple boundary.
3. SB3 PPO with `CnnPolicy` can be constructed and run for a short smoke cycle without crashing.
4. Same-seed plus same scripted action sequence produces a matching first pixel observation across two runs.
5. The pixel source is one of the accepted sources from "Decision 2: pixel source ranking" and the choice is documented with spike evidence.
6. Local artifacts (Python NDJSON, Godot NDJSON, config or config hash, summary, pixel metadata) are written.
7. No new commercial-game surface, no platform automation, no bot-evasion surface, no product framing.

H4 does not close on the basis of any one of: an image file capture, a Godot viewport rendering in isolation, Python downsampling working in isolation, or `CnnPolicy` constructing without running. The boundary must be exercised end-to-end.

## Open design decisions

### Decision 1: env class shape

One env class, not a sibling. Extend `GodotSignalDodgeEnv` with a constructor field `observation_mode` taking values `state`, `pixel`, or `both`.

`state` remains the default. H3 configs and tests must not change. `pixel` is the H4 mode. `both` is optional for diagnostics and not required for H4 closure.

A separate env class would duplicate Godot launch, soft reset, TCP framing, NDJSON capture, terminal handling, and factory wiring. The cost of those duplicates exceeds the cost of mode-branching inside the existing class. The existing class already owns the lifecycle and is the natural seam.

### Decision 2: pixel source ranking

Pixel sources are ranked from preferred to last-resort. The spike in section 0 selects which one becomes default for H4. No decision is committed before the spike.

1. Godot viewport texture capture under `--headless`. Preferred path because H4 is intended to test rendered pixel observations through the same engine that will train H5. Headless capture removes display, compositor, and DPI dependencies. Reproducibility under same seed and scripted actions is the binding question.
2. Godot viewport API in windowed mode. Falls back here if headless viewport readback fails or is nondeterministic. Still uses Godot's own viewport API, not OS screen capture. Reproducibility risk increases due to display compositor and frame-pacing variance. Document the display dependency explicitly if this becomes default.
3. Deterministic Godot-side synthetic raster. Last-resort fallback only if both viewport paths fail. If selected, the H4 acceptance packet must explicitly label the observation as state-encoder-shaped image data, not equivalent to rendered pixels, and the H5 plan must acknowledge that H4 did not exercise rendered-pixel learning. The CNN policy in H5 will need to redo the pipeline against actual rendered frames.
4. MSS or OS screen capture. Rejected. Depends on desktop state, window placement, DPI scaling, monitor configuration, and active foreground application. Outside the H4 boundary contract.

### Decision 3: image shape and dtype

`gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)`.

Channel-first matches SB3 `NatureCNN` convolution input expectations. `uint8` avoids normalization ambiguity at the env boundary; SB3 image wrappers normalize internally. 84x84 is the smallest standard size that reliably passes the default `NatureCNN` convolution stack. Smaller targets (32x32, 64x64) may work but are not the H4 default.

If the spike or the local SB3 build proves a different shape is required, this decision is revisited before implementation, not at acceptance time.

### Decision 4: transport

Request/response TCP, same shape as H3. No async streaming. `step` is one request and one response.

The reset response and step response carry pixel observations only when the active mode requires it. Initial encoding is a flat `uint8` array on the wire to keep the protocol obvious for the smoke gate. Base64 of raw bytes is the small optimization to defer until after the boundary is green.

The wire payload schema for pixel observations:

- `obs.mode` is `"pixel"`.
- `obs.shape` is `[1, 84, 84]`.
- `obs.dtype` is `"uint8"`.
- `obs.encoding` is `"flat_uint8"` initially, with room to grow to `"base64"` later.
- `obs.data` is an array of integers in `[0, 255]` of length 7056 for `flat_uint8`.

Pixel-mode reset and step responses MUST also carry source metadata so reviewers can audit the capture path:

- `obs.pixel_source` is `"godot_windowed_viewport"` for the H4 default. Any other value indicates a fallback (option 3 synthetic raster or a future SubViewport path) and requires explicit Jeff approval before landing.
- `obs.capture_point` is `"RenderingServer.frame_post_draw"` for the H4 default, matching the spike-resolved synchronization barrier.
- `obs.headless_allowed` is `false` for pixel and both modes. Set explicitly so artifact consumers do not need to infer the windowed requirement.
- `obs.viewport_width` and `obs.viewport_height` are integers recording the viewport size set at env construction (separate from the resized 84x84 observation; the spike caveat about default 64x64 viewport is closed by setting this deliberately).

### Decision 5: policy

PPO + `CnnPolicy` from SB3. CPU-first. RTX 2060 may be tried after H4 closes, but H4 acceptance does not depend on CUDA.

H4 trains for a smoke-sized number of timesteps. The acceptance bar is "constructs and runs without crashing while writing artifacts," not "learns." Learning quality is H5.

## 0. Required pre-implementation spike

Before any H4 production code lands on `main`, the following spike must be executed and recorded.

Goal: prove or disprove whether Godot 4.6.2 on StrongerJr can produce stable viewport pixel readback under `--headless`, and characterize reproducibility.

Steps:

1. Standalone minimal Godot scene with a viewport rendering one or two known shapes at known positions. No game logic.
2. Run under `--headless` with `--rendering-driver` defaults.
3. Read the viewport texture via Godot's viewport API. Encode to grayscale 84x84 uint8.
4. Capture frame N over multiple identical runs with the same seed and same scripted scene state.
5. Compare frame bytes across runs. Record exact equality, near equality (per-pixel hamming or L1), or divergence.
6. Repeat under windowed mode for comparison.
7. Record capture latency per frame. Record whether capture is bound to render thread, physics tick, or process tick.

Artifacts to capture under `runs/eval/h4_spike/`:

- Frame bytes for N in {0, 5, 50} from at least three independent runs per mode.
- Per-run timing trace.
- Console output and any Godot warnings.
- A short markdown summary recording outcomes and the recommended pixel source for H4.

Spike outcomes drive the decision:

- Headless viewport capture passes byte-equality, default becomes option 1.
- Headless capture fails byte-equality but windowed capture passes, default becomes option 2 with documented display dependency.
- Both fail byte-equality, escalate to Jeff before falling back to option 3. Synthetic raster requires explicit Jeff approval and an H5 plan amendment acknowledging the contamination.
- Both fail to produce any image, H4 is blocked. Open a Grok escalation packet.

The spike does not produce production code. Spike is a quarantined probe under a feature branch or under `runs/eval/h4_spike/`. Spike code does not import from `sight_agent.rl.envs`.

## 1. Env class extension

The existing `GodotSignalDodgeEnv` gains:

- Constructor parameter `observation_mode: Literal["state", "pixel", "both"] = "state"`.
- Constructor parameters `pixel_width: int = 84`, `pixel_height: int = 84`, `pixel_channels: int = 1`.
- Validation. Any other value for `observation_mode` raises `ValueError` at construction.
- Headless rejection. `observation_mode` in `{"pixel", "both"}` MUST reject a resolved `headless=True` configuration at construction time by raising `ValueError`. Per the H3-to-H4 closure caveats, caller intent must be honored or rejected, not silently transformed. The env does not auto-flip `headless` to `False` for pixel modes.
- `observation_space` selection.
  - `state` is unchanged from H3.
  - `pixel` is `Box(0, 255, (pixel_channels, pixel_height, pixel_width), uint8)`.
  - `both` is `gym.spaces.Dict({"state": <H3 state space>, "pixel": <pixel space>})`.

H3 state mode is the byte-for-byte default. All H3 tests must continue to pass without modification.

## 2. Observation space

State mode is unchanged from H3 plan section 2.

Pixel mode is `Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)`.

Both mode is `gym.spaces.Dict({"state": ..., "pixel": ...})`. Optional, not required for H4 closure.

The Python side validates pixel observation shape, dtype, and value range on every receive. Out-of-range, wrong-shape, or wrong-dtype payloads raise a protocol error, not a silent clip.

## 3. Action space

Unchanged from H3. `Discrete(3)`.

## 4. Reward function

Unchanged from H3. No reward shaping in H4. H4 is a boundary and smoke gate.

## 5. Terminal conditions

Unchanged from H3. Same failure, success, and crash terminals.

## 6. Reset semantics

Same lifecycle as H3. The reset request body is extended to carry observation mode and pixel dimensions when applicable. Soft reset semantics are unchanged.

## 7. Godot side and TCP/IPC contract

Existing H3 protocol with additions:

- `reset` request gains optional fields `observation_mode`, `pixel_width`, `pixel_height`, `pixel_channels`. Defaults match H3 state mode.
- `reset_ok` response gains an `obs` payload field as defined in Decision 4 when active mode is `pixel` or `both`.
- `step_result` response gains an `obs` payload field on the same shape contract.
- Unknown observation mode returns a protocol error. Mismatched pixel dimensions return a protocol error. Both are tested.

The Godot side selects the active observation mode at reset time and holds it for the episode. Mid-episode mode changes are not allowed in H4.

## 8. Smoke test

Default fake-transport tests:

- State mode round-trip unchanged from H3.
- Pixel mode round-trip with a stub Godot transport returning a known pixel payload.
- Both mode round-trip if implemented.
- Invalid `observation_mode` raises at construction.
- Pixel shape mismatch raises a protocol error.

Live Godot smoke test, behind `pytest -m live_godot`:

- `tests/rl/test_h4_godot_pixel_smoke.py`.
- One reset and N steps under `pixel` mode against the live Godot binary on StrongerJr.
- Asserts pixel obs shape, dtype, value range, and same-seed reproducibility for the first observation.

CnnPolicy construction test:

- `tests/rl/test_h4_cnn_policy_construct.py`.
- Builds a PPO model with `CnnPolicy` over a stub env that exposes the H4 pixel observation space.
- Runs at least one rollout step and one optimizer step.
- No live Godot required.

Optional live training smoke:

- `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --total-timesteps 128`.
- Writes artifacts under `runs/train/`.

## 9. Determinism posture

Same-seed step-by-step scripted trajectory equality is the binding determinism criterion for H4. Per the H3-to-H4 closure caveats, first-pixel equality is necessary but not sufficient: H4 must verify that two runs at the same seed under the same scripted action sequence produce matching pixel observations at every step, not merely on the first observation.

H3-style pre-mode-lock physics-tick variance is permitted in the pre-handshake window only; trajectory comparison applies to post-mode-lock observations exclusively, consistent with the H3 closure caveat carried forward by the Grok GREEN verdict.

The viewport-readback path can introduce nondeterminism through render-thread ordering, frame-queue depth, and compositor timing. The spike characterizes this. If frame-queue-sensitive timing is observed, H4 must specify the capture point, for example immediately after physics tick with a forced render flush, and verify reproducibility under that capture point before declaring closure.

Synthetic raster is deterministic by construction but is the last-resort path for the reasons in Decision 2.

## 10. Acceptance criteria for H4 close

H4 technical acceptance is GREEN only if all of the following are true:

1. H3 default and live gates pass unchanged.
2. `GodotSignalDodgeEnv` accepts `observation_mode = "pixel"` and the H4 config sets it.
3. The pixel observation space is `Box(0, 255, (1, 84, 84), uint8)`, or a documented alternative that passes installed-SB3 evidence.
4. `reset(seed=0)` returns valid pixel obs.
5. `step(action)` returns valid pixel obs in the Gymnasium 5-tuple.
6. Same seed plus same scripted action sequence produces matching pixel observations at every post-mode-lock step across two runs (step-by-step trajectory equality, not merely first-pixel equality). Pre-mode-lock physics-tick variance is permitted per the H3 closure caveat.
7. Pixel source matches one of options 1, 2, or 3 from Decision 2, documented with spike evidence and acceptance-run evidence.
8. SB3 PPO `CnnPolicy` constructs successfully from `configs/rl/signal_dodge_ppo_h4_pixel.yaml`.
9. A short CNN smoke run completes without crash and writes local artifacts.
10. Acceptance run artifacts include Python NDJSON, Godot NDJSON, config or config hash, summary, and pixel-source metadata sufficient to audit shape, dtype, source path, and capture point.
11. No new network telemetry, no commercial-game surface, no platform automation, no bot-evasion surface, no product framing.

## Non-goals

- No new game.
- No vectorized parallel Godot envs.
- No frame skip unless explicitly justified by a throughput problem after the boundary is green.
- No reward shaping.
- No GPU dependency at acceptance time.
- No general Godot env framework.
- No external target environments.
- No MSS or OS screen capture as primary observation source.
- No product or commercial framing.

## Known risks

- Headless viewport readback may not be reliable on Godot 4.6.2 / Windows / Intel UHD + RTX 2060 hybrid. Spike result determines fallback.
- Capture timing may be frame-queue-sensitive even within a single mode. Same-seed reproducibility is the binding test, not "approximately equal."
- Flat uint8 JSON payloads cost about 7 KB per step. Acceptable for smoke. May force base64 if H4 step throughput becomes a problem post-acceptance.
- Default `NatureCNN` may reject non-(1,84,84) shapes. Smaller-image experiments are deferred until after H4 closes.

## Implementation sequence

1. Pre-implementation spike (section 0). Spike artifacts under `runs/eval/h4_spike/`. Summary committed to `docs/sight-h4-spike.md` after the spike completes.
2. Python observation-mode plumbing on `GodotSignalDodgeEnv`. Default-transport tests for state, pixel, both, and invalid mode.
3. Godot protocol extension for observation mode and pixel dimensions. Default-transport tests for protocol errors.
4. Pixel source implementation per the spike's recommended option.
5. Live Godot pixel smoke test behind `live_godot`.
6. CnnPolicy construction test. Stub env first, then live opt-in.
7. `configs/rl/signal_dodge_ppo_h4_pixel.yaml`. CPU PPO `CnnPolicy` smoke.
8. Optional `--total-timesteps 128` smoke run on StrongerJr.
9. Acceptance runs with NDJSON evidence, repeated for same-seed reproducibility.
10. `docs/grok-h4-phase-gate-packet.md`, modeled on the H3 packet.
11. Grok review.

## Claude execution boundary

Claude executes the spike, the implementation, the tests, and the acceptance runs. Claude commits docs-only changes during planning. Claude commits implementation changes only after H3 closes GREEN by Grok verdict.

Claude does not record an H3 GREEN verdict in `docs/sight-handoff.md` without verbatim Grok verdict text in the session or a committed verdict artifact in the repo.

Claude does not pivot to synthetic raster (option 3) without explicit Jeff approval.

Claude does not begin H4 implementation while H3 closure is pending.

## Fallback authorization

If the spike falls through option 1, then option 2, then escalates to option 3, Claude pauses and requests Jeff's explicit approval before any H4 code lands. The fallback authorization request includes the spike artifacts, the recommended option, and the H5 plan amendment text required to acknowledge synthetic-raster contamination.

If both viewport options fail and Jeff does not approve synthetic raster, H4 is blocked. Open a Grok escalation packet documenting the blockage.
