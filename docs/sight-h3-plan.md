# H3 Plan: Godot Signal Dodge as Gym-Style State Environment

## Phase gates first

H3 closes only when Sight has a minimal, reproducible, state-observation Godot environment exposed through the existing RL stack as a Gymnasium-style env.

H3 does not close merely because Python can launch Godot, send actions, or parse logs. H3 closes when `env.reset()` and `env.step()` work through the real Godot process and the RL factory seam can construct the env by ID.

H3 remains inside the existing ethical boundary:

- local-only
- Signal Dodge only
- no commercial games
- no live-service games
- no platform automation
- no bot-detection evasion
- no account farming
- no pixels for H3
- no telemetry beyond local NDJSON

Acceptance must mirror the H2 pattern:

1. implementation landed
2. tests green
3. deterministic metadata captured
4. smoke run produces artifacts
5. phase-gate packet written
6. Grok phase-gate review only if charter requires or Claude/GPT disagreement remains load-bearing

## Open design questions and decisions

### Decision 1: observation transport

Use bidirectional TCP.

Reject NDJSON log tailing as the primary observation path. Logs are for evidence and reconciliation, not for Gym step semantics. Tailing NDJSON would make `step()` depend on file timing, buffering, and parsing order. That is avoidable complexity.

H3 should extend the existing loopback JSON-line contract so Python can send `reset` and `step` messages and Godot can reply with state observations.

### Decision 2: reset model

Use in-process soft reset.

Reject subprocess-per-episode as the default. Relaunching Godot for every episode is slower, harder to train against, and turns process startup into part of environment dynamics. Keep subprocess relaunch as a fallback for hard crashes only.

Soft reset means:

- clear hazards
- reset frame counter
- reset death state
- reset score/survival counters
- reposition player
- reseed Godot RNG from the Python-provided episode seed
- emit `run_start` or `episode_start`
- return initial observation

### Decision 3: step granularity

Use one Gym `step()` per Godot physics frame for H3.

Do not add frame skip in H3. Frame skip is a later performance optimization. H3's job is to establish the correct boundary, not optimize training throughput.

### Decision 4: reward shaping

Use sparse/simple survival reward only.

Do not add alignment penalties, hazard-distance shaping, or heuristic curriculum in H3. Reward shaping would make it harder to distinguish "the env works" from "we smuggled in a policy prior."

H3 reward:

- `+1.0` per non-terminal step survived
- `0.0` on collision terminal step
- timeout uses truncation, not failure

Optional later phases can add shaped reward behind an explicit config flag.

## 1. Env class shape and module path

Add:

`src/sight_agent/rl/godot_env.py`

Primary class:

`GodotSignalDodgeEnv`

Shape:

- Gymnasium-compatible `Env`
- state observations only
- `render_mode=None`
- no image capture
- owns Godot subprocess lifecycle
- owns TCP connection lifecycle
- writes Python-side local NDJSON evidence
- points Godot to a run-specific NDJSON log path
- closes child process and socket cleanly

Public methods:

- `__init__(...)`
- `reset(seed=None, options=None)`
- `step(action)`
- `close()`

Constructor inputs should be minimal:

- `godot_executable`
- `project_path`
- `tcp_host`
- `tcp_port`
- `run_dir`
- `max_steps`
- `connect_timeout_s`
- `step_timeout_s`
- `seed`
- `headless`

Default env ID:

`godot:signal-dodge-v0`

Factory seam:

Update `src/sight_agent/rl/factories.py::make_env` to route `godot:signal-dodge-v0` to `GodotSignalDodgeEnv`.

Use lazy import so normal Gymnasium envs remain unaffected.

Config:

Add:

`configs/rl/signal_dodge_ppo_h3.yaml`

Minimum config posture:

- `env.id: godot:signal-dodge-v0`
- `env.n_envs: 1`
- SB3 PPO
- MlpPolicy
- CPU-safe defaults
- short smoke-oriented training values only if training is included at all

H3 does not need meaningful training performance. It needs a correct env boundary.

## 2. Observation space

Observation type:

`numpy.float32`

Observation space:

`gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)`

Vector:

1. `player_x_norm`
2. `player_vx_norm`
3. `nearest_hazard_dx_norm`
4. `nearest_hazard_dy_norm`
5. `nearest_hazard_present`
6. `second_hazard_dx_norm`
7. `second_hazard_dy_norm`
8. `second_hazard_present`
9. `third_hazard_dx_norm`
10. `third_hazard_dy_norm`

Semantics:

- `player_x_norm`: player center X normalized to `[-1, 1]`
- `player_vx_norm`: current horizontal intent or velocity normalized to `[-1, 1]`
- hazard `dx_norm`: hazard center X minus player center X, divided by screen width or half-width, clipped to `[-1, 1]`
- hazard `dy_norm`: hazard center Y minus player center Y, normalized to `[-1, 1]`
- present flags: `1.0` if that hazard slot exists, `0.0` otherwise
- missing hazard coordinate slots: `0.0`

Hazard ordering:

Sort active hazards by descending threat priority:

1. hazards below or near top and moving toward player
2. smallest positive vertical distance to player
3. then absolute horizontal distance

For H3, simpler acceptable ordering:

- top three hazards by largest Y value that are still above or near the player
- stable tie-break by instance ID or spawn order

The key is determinism, not perfect threat modeling.

Why three hazards rather than one:

A one-hazard observation is too brittle once multiple hazards exist. Three hazards is still tiny, state-only, and MLP-friendly without requiring pixels or object arrays.

## 3. Action space

Use:

`gymnasium.spaces.Discrete(3)`

Mapping:

- `0`: left
- `1`: stay
- `2`: right

Wire mapping to existing action semantics:

- `left` maps to `-1`
- `stay` maps to `0`
- `right` maps to `+1`

Reject continuous action for H3. Signal Dodge only needs horizontal intent. Continuous action adds an unnecessary control problem and makes the first Godot env harder to debug.

## 4. Reward function and shaping policy

Reward:

- alive after step: `+1.0`
- collision/death terminal step: `0.0`
- timeout/truncation: no bonus, final step still receives alive reward if no collision occurred

No shaping in H3.

No distance-to-hazard penalty.

No centerline bias.

No expert imitation reward.

No reward based on action smoothness.

Reason:

H3 is a boundary-integration phase. Reward shaping should not be introduced until the env is proven stable and measurable.

Log reward components anyway:

- `reward_survival`
- `reward_total`
- `terminated`
- `truncated`
- `terminal_reason`

## 5. Terminal conditions and Godot signaling

Use Gymnasium split:

- `terminated`: environment-defined terminal state
- `truncated`: timeout or administrative cutoff

### Failure terminal

Collision with hazard:

- `terminated=True`
- `truncated=False`
- `terminal_reason="collision"`

Godot should signal this in the step response, not only in logs.

### Success terminal

No true success terminal in H3.

Signal Dodge is an endurance task. "Success" for acceptance is surviving to timeout.

If max steps reached:

- `terminated=False`
- `truncated=True`
- `terminal_reason="timeout"`

### Crash or protocol failure

If Godot exits unexpectedly, socket dies, malformed response arrives, or step timeout occurs:

- raise a Python exception
- close subprocess/socket
- write local diagnostic artifact
- do not convert crash into normal `terminated=True`

A broken environment is not a terminal state.

## 6. Reset semantics and seedability

`reset(seed=None, options=None)` must:

1. call Gymnasium parent seed handling
2. derive episode seed
3. ensure Godot process is running
4. ensure TCP is connected
5. send `reset` message with:
   - protocol version
   - run ID
   - episode ID
   - seed
   - max steps
6. Godot soft-resets state
7. Godot returns initial observation
8. Python returns `(obs, info)`

`info` should include:

- `run_id`
- `episode_id`
- `seed`
- `godot_pid`
- `tcp_port`
- `frame`
- `git_commit` if available
- `config_hash` if available

Seed posture:

- Python global seed remains governed by H2 conventions
- per-env seed is threaded into `reset(seed=...)`
- Godot applies seed during soft reset
- spawn sequence should be reproducible for same seed, same Godot version, same dependency/hardware class
- bit-for-bit cross-machine determinism is not promised

## 7. Godot side and TCP/IPC contract

Use existing game:

`games/signal-dodge`

No new game for H3.

Reuse H2/pre-pivot transport foundation, but extend it.

Do not stand up a separate IPC channel. H3 should evolve the existing loopback JSON-line TCP channel because:

- host/port constants already exist
- controller plumbing already exists
- Godot TCP-mode already exists
- logs already capture controller events
- factories already anticipate Godot env IDs

### Protocol

Transport:

- loopback TCP
- newline-delimited UTF-8 JSON
- one request, one response for `reset` and `step`
- local only

Required Python-to-Godot message types:

#### `hello`

Purpose:

- handshake
- protocol version check
- run ID binding

Required fields:

- `type`
- `protocol_version`
- `run_id`

#### `reset`

Purpose:

- soft reset current Godot scene into a new episode

Required fields:

- `type`
- `protocol_version`
- `run_id`
- `episode_id`
- `seed`
- `max_steps`

Expected response:

`reset_ok`

Response fields:

- `type`
- `protocol_version`
- `run_id`
- `episode_id`
- `frame`
- `obs`
- `terminated`
- `truncated`
- `info`

#### `step`

Purpose:

- apply action for one physics frame and return the resulting observation

Required fields:

- `type`
- `protocol_version`
- `run_id`
- `episode_id`
- `seq`
- `action`

Expected response:

`step_result`

Response fields:

- `type`
- `protocol_version`
- `run_id`
- `episode_id`
- `seq`
- `frame`
- `obs`
- `reward`
- `terminated`
- `truncated`
- `terminal_reason`
- `info`

### Godot implementation posture

Godot should be the source of truth for:

- player position
- hazard positions
- collision state
- frame count
- terminal state

Python should be the source of truth for:

- action selection
- seed passed to episode
- max step cutoff
- run directory
- artifact naming
- training/evaluation config

Godot should continue logging NDJSON for evidence:

- `run_start` or `episode_start`
- `controller_hello`
- `reset`
- `step`
- `controller_cmd_applied`
- `collision`
- `death`
- `run_end` or `episode_end`

But Gym semantics must come from TCP responses, not from log parsing.

## 8. Smoke test

Add:

`tests/rl/test_h3_godot_smoke.py`

Use two tiers.

### Default fast test

No Godot binary required.

Use a fake or stub transport that simulates the Godot protocol.

Assertions:

- `GodotSignalDodgeEnv` declares observation space shape `(10,)`
- observation dtype is `float32`
- action space is `Discrete(3)`
- `reset(seed=0)` returns `(obs, info)`
- returned obs is inside observation space
- `step(1)` returns Gymnasium 5-tuple:
  - obs
  - reward
  - terminated
  - truncated
  - info
- 10 steps can run without protocol drift
- forced collision from stub produces `terminated=True`
- forced timeout from stub produces `truncated=True`
- `close()` is idempotent

This test should be in the default `tests/rl` gate.

### Live Godot smoke test

Add marker:

`live_godot`

Default pytest should exclude it unless explicitly requested.

Live test should:

1. locate Godot executable by env var or configured path
2. launch `games/signal-dodge`
3. connect via TCP on an isolated port
4. call `reset(seed=0)`
5. call `step(1)` for 100 steps or until terminal
6. assert obs shape and dtype on every step
7. assert no malformed protocol messages
8. assert Godot NDJSON contains expected event mix
9. close cleanly

Live test command:

`pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short`

Default H3 test command:

`pytest tests/rl -v --tb=short`

Before H3 close, run both default RL tests and live Godot smoke on Strongerjr.

## 9. Determinism posture

Reproducible:

- Python config hash
- git commit
- run ID
- episode ID
- Python seed
- env seed
- Godot reset seed
- action sequence
- observation sequence on same dependency/hardware class
- local NDJSON artifacts

Best effort:

- exact Godot physics timing across machines
- process startup timing
- socket scheduling
- wall-clock timestamps

Not promised:

- bit-for-bit cross-machine equivalence
- identical behavior across Godot versions
- identical behavior if frame skip is added later

H3 artifact metadata must include:

- `git_commit`
- `config_hash`
- `python_version`
- `godot_version` if cheaply available
- `protocol_version`
- `env_id`
- `seed`
- `max_steps`
- `run_id`
- `episode_id`

## 10. Acceptance criteria for H3 close

H3 technical acceptance is GREEN only if all are true:

1. `GodotSignalDodgeEnv` exists under `src/sight_agent/rl/godot_env.py`
2. factory seam constructs it through `env.id="godot:signal-dodge-v0"`
3. observation space is state-only, no pixels
4. action space is `Discrete(3)`
5. bidirectional TCP supports `hello`, `reset`, and `step`
6. Signal Dodge supports in-process soft reset
7. same seed produces same initial observation and same short scripted rollout on same machine
8. default RL tests pass:
   - `pytest tests/rl -v --tb=short`
9. live Godot smoke passes on Strongerjr:
   - `pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short`
10. local artifacts are written:
   - Python NDJSON
   - Godot NDJSON
   - config copy or config hash
   - run metadata

Required closure checks, not technical acceptance criteria:

1. Charter invariants are explicitly checked in the H3 phase-gate packet:
   - no network telemetry added
   - no pixel path used
   - no commercial or platform automation scope added
2. H3 phase-gate packet is written using the H2 packet pattern.
3. `docs/sight-handoff.md` is updated with phase, last commit, current task, next action, blockers, and no more than five notes.

## Non-goals

Do not solve PPO performance in H3.

Do not add frame skip.

Do not add vectorized parallel Godot envs.

Do not add visual observations.

Do not clean all pre-pivot debt unless it blocks tests.

Do not build a generalized Godot env framework.

Do not add reward shaping.

Do not add new game mechanics.

## Known risks

### Pre-pivot tests may confuse closure

The inspection found old pre-pivot tests still in the default test tree. H2 validated `tests/rl`, not necessarily the full default test collection. H3 should keep its acceptance gate explicit as `pytest tests/rl -v --tb=short`.

If default `pytest` fails because of stale pre-pivot tests, do not treat that as an H3 regression unless H3 touched those paths. Record it separately.

### Soft reset may reveal Godot state coupling

If clearing hazards and resetting frame/player state is messy, keep the soft reset small and explicit. Do not refactor the game broadly.

### Bidirectional protocol can grow too fast

Only implement messages needed for H3:

- `hello`
- `reset`
- `step`
- error response

Do not add general RPC machinery.

## Implementation sequence

1. Add protocol notes or constants for H3 message types.
2. Extend Godot TCP controller to parse `reset` and `step`.
3. Add soft reset support in Signal Dodge main scene.
4. Add state observation builder in Godot.
5. Add Python transport support for request-response.
6. Add `GodotSignalDodgeEnv`.
7. Add factory branch for `godot:signal-dodge-v0`.
8. Add config `configs/rl/signal_dodge_ppo_h3.yaml`.
9. Add stub transport unit tests.
10. Add live Godot smoke test behind `live_godot`.
11. Run `pytest tests/rl -v --tb=short`.
12. Run live Godot smoke on Strongerjr.
13. Write H3 phase-gate packet.
14. Update handoff.
15. Commit and push.

## Claude execution boundary

Implement the smallest H3 env boundary that satisfies this plan.

If implementation pressure pushes toward broader cleanup, stop and report instead of expanding scope.

If bidirectional TCP or soft reset proves materially harder than expected, report the smallest failing point and propose the fallback. Do not silently switch to NDJSON tailing or subprocess-per-episode.
