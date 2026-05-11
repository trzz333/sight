# Sight - H4 Phase Gate Packet (for Grok review)

Phase-gate packet for H4: pixel observations on the same Godot Signal
Dodge environment with a small CNN policy. Boundary and smoke gate
only; learning quality is H5's concern. This packet collects the
evidence needed for a Grok GREEN / YELLOW / RED verdict on H4 closure.

H5 (learning evaluation of the small CNN policy on Signal Dodge or
its successor microgame) is not started. H3 closed GREEN per
`docs/grok-h3-phase-gate-packet.md`.

---

## 1. Prior phase state

H3 closed via the H3 phase-gate packet. H4 implementation work was
authorized at H3 closure. Substantive H4 commits up to and including
this packet:

- `b7a3e72` feat(rl): add h4 pixel cnn config plumbing
- `dbf248d` test(rl): h4 cnn policy construction smoke
- `1ad79e1` test(rl,gd): h4 live godot pixel smoke + reset capture invariants
- `fcbcf7e` feat(gd): h4 step 3 - windowed godot viewport pixel source
- `0567fec` fix(rl): isolate godot tcp port and absolute godot log path for h4 live train

The intermediate `chore: refresh handoff hash` commits between
substantive landings are doc-only.

HEAD at packet construction: `0e96bfe` (chore handoff refresh on top
of `0567fec`). This packet's commit will append to that line; the hash
patches into section 3 when the commit lands.

## 2. H4 scope and non-scope

In scope for H4 (per `docs/sight-h4-plan.md`):

- `GodotSignalDodgeEnv` accepts `observation_mode = "pixel"` while
  preserving H3 `state` mode unchanged.
- Pixel observation space `Box(0, 255, (1, 84, 84), uint8)`.
- Pixel source: windowed Godot viewport readback, capture point
  `RenderingServer.frame_post_draw`.
- TCP wire payload carries `obs.{mode, shape, dtype, encoding, data,
  pixel_source, capture_point, headless_allowed, viewport_width,
  viewport_height}` on reset_ok and step_result.
- SB3 PPO `CnnPolicy` constructs from
  `configs/rl/signal_dodge_ppo_h4_pixel.yaml`.
- Live 128-step CPU PPO `CnnPolicy` smoke against the H4 pixel config
  on StrongerJr writes the full artifact set (summary.json,
  events.ndjson, config_effective.yaml, model.zip, godot-train/*,
  godot-eval/*).
- Same-seed step-by-step pixel observation equality across two
  consecutive scripted-action rollouts under live windowed Godot.

Out of scope for H4 (deferred to H5+):

- Learning quality / policy performance on Signal Dodge.
- Headless pixel capture (rejected at env construction; H4 plan
  Decision 2 / spike).
- Frame skip.
- Vectorized parallel Godot envs.
- Reward shaping.
- GPU dependency at acceptance.
- New game mechanics.

Permanent non-goals (charter-pinned, unchanged):

- No offerwalls, no Freecash, no live commercial games, no
  bot-detection evasion, no account farming, no online multiplayer,
  no platforms where automation is prohibited.

## 3. Repo state at packet construction

Substantive H4 implementation landed across the commits in section 1.
The final pre-packet substantive commit is:

- `0567fec` fix(rl): isolate godot tcp port and absolute godot log
  path for h4 live train. Two production bugs fixed inside the H4
  live-smoke boundary: factory now allocates a kernel-assigned
  loopback TCP port per Godot env (eliminating train/eval 8765
  collision when an in-train eval env is constructed alongside the
  train env); `godot_env._launch_godot` now passes an absolute
  `SIGHT_GODOT_LOG_PATH` so Godot's File API does not resolve relative
  to its `--path` working directory. H3 did not hit these because H3
  ran train and out-of-band eval sequentially and pytest's `tmp_path`
  is absolute.

All commits pushed to `origin/main` at
`https://github.com/trzz333/sight.git`. Tree clean at the start of
this acceptance round.

(Substantive HEAD hash for the packet-landing commit: patched into
this section when this packet's commit lands. Placeholder `0e96bfe`.)

## 4. Default test gate

Default command:

```
python -m pytest tests/rl --tb=short
```

Result on this round's HEAD, run on StrongerJr at packet draft time:

- **228 passed, 0 failed, 2 deselected**. 14.74s.
- 230 items collected; the 2 deselected items are the live opt-in
  tests excluded by `pyproject.toml`
  (`addopts = "-ra -m 'not live_mss and not live_godot'"`):
  `test_h3_godot_smoke::test_live_godot_reset_and_100_step_smoke`
  and
  `test_h4_godot_pixel_smoke::test_live_godot_pixel_same_seed_step_by_step_trajectory_equality`.
- Net `+107` tests vs H3-closure baseline (121 -> 228) from the H4
  implementation slices: protocol, env construction, transport,
  pixel CNN policy construction, callbacks, factories, godot config
  plumbing, and the live-smoke regression guards from `0567fec`.

Telemetry posture unchanged: no TensorBoard, no W&B, no MLflow, no
Comet, no network imports under `src/sight_agent/rl/`. H4 still writes
only to loopback TCP and local NDJSON. Charter-pinned.

## 5. Live H4 pixel trajectory equality

Acceptance gate for criterion 6 in `docs/sight-h4-plan.md` section
10: same-seed plus same scripted action sequence produces matching
pixel observations at every post-mode-lock step across two runs
(step-by-step, not merely first-pixel).

Test:
`tests/rl/test_h4_godot_pixel_smoke.py::test_live_godot_pixel_same_seed_step_by_step_trajectory_equality`

Command:

```
python -m pytest tests/rl/test_h4_godot_pixel_smoke.py::test_live_godot_pixel_same_seed_step_by_step_trajectory_equality \
  -m live_godot -v --tb=short -s \
  --basetemp=C:\Projects\Sight\runs\eval\h4_acceptance
```

Result: **1 passed in 4.86s** on StrongerJr.

Rollout: 1 reset + 10 scripted steps with the action sequence
`[1, 0, 2, 1, 0, 2, 1, 0, 2, 1]` (mix of stay/left/right so the test
exercises actual scene change rather than freezing position).

Per-observation invariants asserted (and passed) on every reset and
every step return: `shape == (1, 84, 84)`, `dtype == uint8`, value
range `[0, 255]`, `env.observation_space.contains(obs)`.

Cross-run equality asserted (and passed): `np.array_equal(obs1[i],
obs2[i])` for every `i` from 0 (post-reset) through the last returned
step. Length of both observation lists matches.

Artifact dirs:

- `C:\Projects\Sight\runs\eval\h4_acceptance\test_live_godot_pixel_same_see0\run1\`
- `C:\Projects\Sight\runs\eval\h4_acceptance\test_live_godot_pixel_same_see0\run2\`

Each contains `python.ndjson`, `godot.ndjson`, `godot-stdout.log`,
`godot-stderr.log`. Stdout logs both 172 bytes (engine version
banner). Stderr logs both 0 bytes. `runs/` is gitignored.

### 5.1 Why `-s` was required

Without `-s`, pytest captures stdin and substitutes a handle that
Windows `subprocess._make_inheritable` cannot duplicate; the test
errored at `OSError: [WinError 6] The handle is invalid` during
`_default_process_factory`. With `-s`, stdin inheritance is direct
and Popen succeeds. This is the same family as the H3 closure caveat
documented in `docs/grok-h3-phase-gate-packet.md` section 3 (Popen +
`subprocess.PIPE` deadlock on Windows under a non-console parent),
and is operational; no production code change was authorized in this
round. Tracked as caveat in section 9 below.

## 6. Live H4 acceptance training runs

H4 plan section 10 criteria 8, 9, 10: SB3 PPO `CnnPolicy` constructs
and a short CNN smoke run completes without crash while writing
local artifacts.

Two consecutive same-seed 128-step CPU PPO `CnnPolicy` training runs
against `configs/rl/signal_dodge_ppo_h4_pixel.yaml` on StrongerJr,
both with `SIGHT_GODOT_EXE` set inline in the parent shell to the
console build at the WinGet path.

Command per run:

```
python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h4_pixel.yaml
```

Run dirs:

- `runs\rl\signal_dodge_ppo_h4_pixel\20260511T014058Z_signal_dodge_ppo_h4_pixel_seed0_0e96bfe\`
- `runs\rl\signal_dodge_ppo_h4_pixel\20260511T022255Z_signal_dodge_ppo_h4_pixel_seed0_0e96bfe\`

### 6.1 Exit and PPO metrics

| field | run1 | run2 |
|-------|------|------|
| exit code | 0 | 0 |
| total_timesteps | 128 | 128 |
| iterations | 2 | 2 |
| time_elapsed (PPO logger) | 107 s | 107 s |
| elapsed_seconds (NDJSON run_end) | 105.98 s | 105.76 s |
| fps | 1 | 1 |
| step-128 approx_kl | 8.7335706e-05 | 8.7335706e-05 |
| step-128 clip_fraction | 0 | 0 |
| step-128 entropy_loss | -1.0985275506973267 | -1.0985275506973267 |
| step-128 value_loss | 165.138671875 | 165.138671875 |
| step-128 policy_gradient_loss | 0.001090841367840767 | 0.001090841367840767 |
| step-128 explained_variance | -0.001621842384338379 | -0.001621842384338379 |
| step-128 loss | 81.49404907226562 | 81.49404907226562 |
| step-128 n_updates | 1 | 1 |
| eval mean_reward @64 | 1800.0 | 1800.0 |
| eval mean_reward @128 | 1800.0 | 1800.0 |
| status | ok | ok |

Eval mean_reward 1800.0 corresponds to a 1800-step survival rollout
under the deterministic eval policy at this initialization. Per the
H4 plan and bootstrap guidance, **this is not a learning claim**;
1800.0 reflects scripted-eval-rollout survival under a freshly
initialized network plus Signal Dodge's hazard density at step 0,
not policy quality.

### 6.2 Artifact checklist

Per H4 plan section 10 criterion 10 and the bootstrap "Acceptance
definition" enumeration.

| artifact | run1 | run2 |
|----------|------|------|
| `summary.json` | 1849 B, present | 1849 B, present |
| `events.ndjson` | 4935 B, present | 4935 B, present |
| `config_effective.yaml` | 660 B, present | 660 B, present |
| `model.zip` | 20,237,552 B, present | 20,237,552 B, present |
| `godot-train/godot.ndjson` | 24,873 B, present | 26,818 B, present |
| `godot-train/python.ndjson` | 28,572 B, present | 28,559 B, present |
| `godot-eval/godot.ndjson` | 681,103 B, present | 680,947 B, present |
| `godot-eval/python.ndjson` | 797,041 B, present | 797,056 B, present |

All eight required artifact entries per run are present.

### 6.3 SHA-256 hashes (non-binding telemetry)

| artifact | run1 | run2 |
|----------|------|------|
| `summary.json` | 5941f2f7...c57e8 | edd17132...60e4 |
| `events.ndjson` | 687b8ba9...3a4ca | 8fdccd9d...4a30b1 |
| `config_effective.yaml` | cea7867a...d347 | cea7867a...d347 |
| `model.zip` | cb00c5e3...f104db | 30065e9d...92a715 |
| `godot-train/godot.ndjson` | a357a4e3...2e01c | 46d67d34...a27a49 |
| `godot-train/python.ndjson` | 8dcefb5e...85665e | ff088aa4...4bc0045 |
| `godot-eval/godot.ndjson` | b76e8793...4bc920 | 3bfd2655...796d73 |
| `godot-eval/python.ndjson` | 90f5a9cc...8745f6 | 840a505a...7f058f |

`config_effective.yaml` is byte-identical across runs (same hash).
All other artifacts contain at least one timestamp or per-run id
(`run_id`, `ts_utc`, `ts_unix`, `godot_pid`, `tcp_port`, episode-id),
so byte equality is not expected. Per bootstrap guidance, the gate
does NOT fail on `events.ndjson` or `model.zip` not being byte-
identical.

### 6.4 Config invariants preserved

Each `config_effective.yaml` (same hash) and each `run_start`
`config` block preserves:

| key | required | observed |
|-----|----------|----------|
| `env.observation_mode` | `pixel` | `pixel` |
| `env.headless` | `false` | `false` |
| `env.pixel_channels` | 1 | 1 |
| `env.pixel_height` | 84 | 84 |
| `env.pixel_width` | 84 | 84 |
| `run.seed` | 0 | 0 |
| `train.total_timesteps` | 128 | 128 |
| `algo.name` | `PPO` | `PPO` |
| `algo.policy` | `CnnPolicy` | `CnnPolicy` |
| `algo.device` | `cpu` | `cpu` |

`run_start` event in each `events.ndjson` carries
`env_smoke.obs_shape = [1, 84, 84]` and `env_smoke.action_n = 3`,
matching the H4 contract.

### 6.5 Required event types

Each `events.ndjson` contains exactly:

`{run_start: 1, eval: 2, train_metrics: 2, run_end: 1}`

All four required event types (`run_start`, `train_metrics`, `eval`,
`run_end`) present in both runs. Event-type multiset matches across
runs.

## 7. Pixel-source metadata

H4 plan Decision 4 mandates per-receive obs metadata so reviewers
can audit the capture path from artifacts and source. Required
literals:

- `pixel_source = "godot_windowed_viewport"`
- `capture_point = "RenderingServer.frame_post_draw"`
- `headless_allowed = false`
- `viewport_width`, `viewport_height`: positive ints

### 7.1 Transport-level validation

`src/sight_agent/rl/godot_transport.py` validates these fields' types
on every pixel-mode receive (lines 637-657): `pixel_source` must be
`str`, `capture_point` must be `str`, `headless_allowed` must be
`bool`, `viewport_width` and `viewport_height` must be positive ints.
A receive that lacks any of these fields or has them malformed raises
`GodotProtocolError` and the rollout aborts. Both 128-step runs
completed without aborting, which is positive evidence that all
required fields were present and well-typed on every reset_ok and
step_result reply across roughly 256 pixel-mode receives per run
(128 train + 1 reset, plus the in-train eval rollouts).

### 7.2 Specific literal values

Source-code audit of the required literals:

- `tcp_controller.gd` (lines 580-596) documents that `obs` is a
  Dictionary carrying `mode/shape/dtype/encoding/data/pixel_source/
  capture_point/headless_allowed/viewport_width/viewport_height` and
  that the caller (`main.gd`) is responsible for awaiting
  `RenderingServer.frame_post_draw` and building the dict.
- `godot_env.py` line 194 comment: pixel/both modes reject
  `headless=True` because `--headless` does not emit
  `RenderingServer.frame_post_draw`.
- The pixel-source default per `docs/sight-h4-plan.md` Decision 4 is
  `"godot_windowed_viewport"`; any other value would require explicit
  Jeff approval.

### 7.3 Caveat: literals not pinned in transport

The transport currently validates **types**, not **values**, for
these metadata fields. A regression in `main.gd` that emitted a
different `pixel_source` literal (e.g. `"synthetic_raster"`) would
not be caught by the transport check; only a Jeff-approval audit
would catch it. Pinning the literals would be a small patch
(string-equality check after the type check). Recommended follow-up
but not blocking for the H4 boundary gate, since the artifact-level
audit and the source-code audit agree on the values that were used.

### 7.4 Caveat: metadata not persisted to NDJSON

The wire-payload pixel-source metadata is not separately persisted
to `python.ndjson` or `godot.ndjson`. NDJSON records the
`observation_mode/pixel_width/pixel_height/pixel_channels` fields at
`controller_reset_received` (Godot) and per-step `frame/reward/
terminated/truncated/terminal_reason` (Python). `screen_width=720,
screen_height=540` is captured at `run_start` in `godot.ndjson`.
Persisting `obs.pixel_source/capture_point/headless_allowed/
viewport_width/viewport_height` once per reset would close the
artifact-only audit path. Recommended follow-up, also not blocking
for the H4 boundary gate.

## 8. Determinism posture

The binding determinism criterion for H4 (plan section 10 criterion
6) is same-seed step-by-step pixel observation equality across two
runs under a scripted action sequence.

**Trajectory equality test passed** (section 5): 11 byte-equal pixel
observations across two sequential windowed-Godot rollouts, including
1 post-reset and 10 post-step observations under the mixed
stay/left/right action sequence. No approximate matching, no
per-pixel tolerance; strict `np.array_equal`.

The 128-step CPU PPO `CnnPolicy` training pair (section 6) shows
**identical post-update training metrics** at step 128 across both
runs to all printed digits (approx_kl, entropy_loss, value_loss,
policy_gradient_loss, explained_variance, loss, n_updates). This is
additional non-binding evidence that the pixel pipeline, the policy
forward/backward pass, and SB3's optimizer are deterministic on this
machine at this seed under this CPU build.

Eval mean_reward is identical (1800.0 in both runs at both eval
checkpoints), which is consistent with deterministic eval rollouts
under the same initialization.

The H3 closure caveat about pre-mode-lock physics-tick variance
carries forward unchanged: the trajectory equality assertion applies
only to post-mode-lock observations returned through
`env.reset()` / `env.step()`. The 1945-byte godot.ndjson size
difference between run1 (24,873 B) and run2 (26,818 B) in
godot-train reflects pre-mode-lock physics-tick variance plus
controller_connected/disconnect noise during the bind window,
the same family seen in the H3 packet section 7 caveat 1.

## 9. Caveats and known nuances

1. **`-s` required for the live trajectory equality test under
   Desktop Commander.** Without `-s`, pytest stdin capture
   substitutes a handle that Windows Popen cannot duplicate; the test
   errors at `OSError: [WinError 6]` in `_default_process_factory`.
   With `-s` (no stdin capture), Popen succeeds and the test passes.
   Direct `python -m sight_agent.rl.train` (not under pytest) is
   unaffected. Same family as the H3 `subprocess.PIPE` deadlock
   already documented and fixed; not patched in this round per the
   bootstrap "No code changes unless required to collect acceptance
   evidence" instruction. The acceptance run with `-s` is the live
   evidence; no production behavior is altered.

2. **Pixel-source metadata literals not pinned in transport.** The
   transport validates type, not value. Recommended follow-up: pin
   `pixel_source == "godot_windowed_viewport"`, `capture_point ==
   "RenderingServer.frame_post_draw"`, `headless_allowed == False`.
   See section 7.3.

3. **Pixel-source metadata not persisted to NDJSON.** Artifact-only
   audit currently has to rely on source-code inspection plus
   transport-validation-survival. Recommended follow-up: log the obs
   metadata dict once per reset to `python.ndjson`. See section 7.4.

4. **`SIGHT_GODOT_EXE` not inherited.** Desktop Commander's parent
   shell does not inherit User-scope env vars. The acceptance runs
   set it inline to the WinGet console build path. Operational; same
   as H3 acceptance.

5. **`runs/` is gitignored.** The acceptance artifacts described in
   sections 5 and 6 are NOT committed. They live on disk under
   `C:\Projects\Sight\runs\eval\h4_acceptance\` and
   `C:\Projects\Sight\runs\rl\signal_dodge_ppo_h4_pixel\` on
   StrongerJr for the duration of this packet review. The audit
   script at `runs/eval/h4_acceptance/h4_audit.py` and its successor
   `h4_audit2.py` produce the `_report.json` files that section 6
   draws from; both can be re-run on any clone with the artifacts
   regenerated.

6. **Eval mean_reward of 1800.0 is not a learning signal.** A 1800-
   step eval rollout under a freshly initialized CnnPolicy completed
   without the player being hit by a hazard during the deterministic
   eval rollout. This is consistent with Signal Dodge's first-second
   hazard density and the eval policy's chosen action, not with the
   policy having learned anything in 128 train timesteps. Learning
   quality is H5's gate.

7. **n_updates = 1.** Per the H4 config (`n_epochs: 1`), one
   optimizer update per rollout. Two iterations per run land
   n_updates at the printed 1 (per-iteration counter) per the SB3
   logger's reporting convention. CnnPolicy did execute one forward
   and one backward pass per iteration; the test goal of "constructs
   and runs without crashing while writing artifacts" is met. No
   learning-quality inference is drawn.

## 10. Ethics and non-goal check

All four charter invariants hold for this round:

- No network telemetry. H4 writes only to loopback TCP and local
  NDJSON. No TensorBoard, no W&B, no MLflow, no Comet, no upload of
  any kind.
- No commercial-game surface. H4 operates only on `games/signal-
  dodge`, a custom microgame owned by this repo and MIT-licensed.
- No platform automation. No external target environments. No
  account interaction.
- No bot-detection evasion. Signal Dodge has no bot detection;
  nothing in H4 inspects, evades, or interacts with any anti-bot
  surface. No Freecash, no offerwalls, no account farming, no live
  multiplayer.

## 11. Recommended verdict scope for Grok

Grok is asked to evaluate H4 against the eleven criteria from
`docs/sight-h4-plan.md` section 10:

1. H3 default and live gates pass unchanged: yes / no. (228 passed +
   2 deselected on `tests/rl`; live H3 path runs through the same
   `GodotSignalDodgeEnv` in state mode, unchanged.)
2. `GodotSignalDodgeEnv` accepts `observation_mode = "pixel"` and
   the H4 config sets it: yes / no. (Config and live runs confirm.)
3. Pixel observation space is `Box(0, 255, (1, 84, 84), uint8)`:
   yes / no. (Per-receive transport validation plus
   `env.observation_space.contains(obs)` per step.)
4. `reset(seed=0)` returns valid pixel obs: yes / no. (Trajectory
   equality test asserts shape, dtype, range, contains on every
   reset.)
5. `step(action)` returns valid pixel obs in the Gymnasium 5-tuple:
   yes / no. (Same test asserts on every step.)
6. Same seed plus same scripted action sequence produces matching
   pixel observations at every post-mode-lock step across two runs
   (step-by-step, not first-pixel only): confirmed / not.
   (Trajectory equality test passes; 1 reset + 10 steps byte-equal.)
7. Pixel source matches Decision 2 options 1/2/3, documented:
   confirmed / not. (Option 2, windowed Godot viewport API at
   `RenderingServer.frame_post_draw`. Transport validates fields on
   every receive. Caveats 7.3 and 7.4 noted but not blocking.)
8. SB3 PPO `CnnPolicy` constructs from
   `configs/rl/signal_dodge_ppo_h4_pixel.yaml`: confirmed / not.
   (`policy_class = ActorCriticCnnPolicy` in both summary.json
   files; PPO construction succeeded both runs.)
9. Short CNN smoke run completes without crash and writes local
   artifacts: confirmed / not. (Two 128-step runs, exit 0, all eight
   artifact entries present each run.)
10. Acceptance run artifacts include Python NDJSON, Godot NDJSON,
    config or config hash, summary, and pixel-source metadata
    sufficient to audit shape, dtype, source path, and capture
    point: confirmed / not, with caveats. (All present; metadata
    audit relies on transport-validation-survival plus source-code
    audit per caveat 7.4. The shape/dtype/source/capture_point are
    documented in `tcp_controller.gd` and the H4 spike doc, and the
    config hash is identical across runs.)
11. No new network telemetry, no commercial-game surface, no
    platform automation, no bot-evasion surface, no product framing:
    confirmed. (Section 10.)

Required closure checks per plan section 10:

a. Charter invariants explicitly checked: section 10.
b. H4 phase-gate packet written using the H3 packet pattern: this
   document.
c. `docs/sight-handoff.md` updated with phase, last commit, current
   task, next action, blockers, no more than five notes: yes (this
   round's handoff refresh commit).

If criteria 1-11 read clean and the closure checks (a-c) are
satisfied, the recommended verdict is GREEN and H4 is closed.

If Grok flags YELLOW caveats, the most likely YELLOW areas are
caveats 7.3 (pin pixel-source literals in transport) and 7.4 (log
metadata to NDJSON). Both are small patches and could ride a YELLOW
closure doc.

## 12. Pointers

- Charter: `docs/sight-charter.md`
- H4 plan: `docs/sight-h4-plan.md`
- H4 spike: `docs/sight-h4-spike.md`
- H1 GREEN closure: `docs/grok-h1-final-green.md`
- H1 packet: `docs/grok-h1-phase-gate-packet.md`
- H2 packet: `docs/grok-h2-phase-gate-packet.md`
- H3 packet: `docs/grok-h3-phase-gate-packet.md`
- Handoff: `docs/sight-handoff.md`
- H4 substantive commits: `b7a3e72`, `dbf248d`, `1ad79e1`,
  `fcbcf7e`, `0567fec`, plus this round's packet-landing commit
  (hash patched into section 3 on commit landing).
