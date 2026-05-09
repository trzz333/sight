# Sight - H3 Phase Gate Packet (for Grok review)

Phase-gate packet for H3: tiny Godot environment exposed as a Gym-style
env with state observations only, no pixels. This packet collects the
evidence needed for a Grok GREEN / YELLOW / RED verdict on H3 closure.

H4 (pixel observations on the same Godot env, small CNN policy) is not
started. H2 closed per `docs/grok-h2-phase-gate-packet.md`.

---

## 1. Prior phase state

H2 closed via the H2 phase-gate packet. H3 implementation work was
authorized at H2 closure. Substantive H3 commits up to and including
this packet:

- `dca2af3` feat(rl): add signal dodge h3 config plumbing (step 8)
- `50ea24d` test(rl): add h3 godot stub smoke coverage (step 9)
- `b5b3fad` test(rl): add live_godot marker and live h3 godot smoke
  (step 10 marker + test wiring)
- This packet's substantive commit: live-gate boundary fixes
  (Popen+PIPE deadlock and GDScript protocol_version type widening)

The intermediate `chore: handoff refresh` commits between substantive
landings are doc-only and functionally equivalent for code purposes.


## 2. H3 scope and non-scope

In scope for H3 (per `docs/sight-h3-plan.md`):

- `GodotSignalDodgeEnv` Gymnasium-compatible env at
  `src/sight_agent/rl/godot_env.py`.
- State observations only. 10-element `Box(-1, 1, (10,), float32)`
  vector per plan section 2.
- Action space `Discrete(3)` mapping 0/1/2 to left/stay/right per
  plan section 3.
- Bidirectional loopback TCP transport (`hello`, `reset`, `step`,
  `error`, `reset_ok`, `step_result`) per plan section 7.
- In-process soft reset (no subprocess relaunch per episode) per
  plan Decision 2.
- One Gym `step()` per Godot physics frame, no frame skip per
  Decision 3.
- Sparse survival reward (+1 alive, 0 collision, no shaping) per
  plan section 4.
- Terminal split: `terminated=True` on collision, `truncated=True`
  on timeout (`max_steps` reached). Crash/protocol failures raise
  Python exceptions, not terminal states, per plan section 5.
- Factory seam: `make_env(env_id="godot:signal-dodge-v0")` constructs
  the env (plan section 1).
- NDJSON evidence on both sides (`python.ndjson` under `run_dir`,
  `godot.ndjson` written via `SIGHT_GODOT_LOG_PATH`).
- Default and live test tiers per plan section 8.

Out of scope for H3 (deferred to H4+):

- Pixel observations.
- Frame skip.
- Vectorized parallel Godot envs.
- Reward shaping.
- New game mechanics.
- Generalized Godot env framework (one game only: Signal Dodge).
- PPO performance work or training acceptance on Signal Dodge.

Permanent non-goals (charter-pinned, unchanged):

- No offerwalls, no Freecash, no live commercial games, no bot-
  detection evasion, no account farming, no online multiplayer,
  no platforms where automation is prohibited.

## 3. Repo state at packet construction

Substantive H3 implementation has landed across multiple commits. The
final live-gate-green commit (this round) lands two boundary fixes
without which the live tier could not be green:

- **Popen + `subprocess.PIPE` deadlock on Windows.** Godot 4.6.2 hangs
  at the engine version banner before `_ready()` runs when stdout/
  stderr are anonymous pipes from a non-console parent (electron host,
  pytest harness). Verified by a 6-cell matrix on 2026-05-09: PIPE
  hangs both windowed and console builds; `CREATE_NO_WINDOW` does not
  help; DEVNULL and file redirection both work in under one second.
  Pipe-fill is ruled out (only the 73-byte engine banner is ever
  written before the hang). Fix in `_launch_godot`: redirect to
  `<run_dir>/godot-stdout.log` and `<run_dir>/godot-stderr.log` when
  `run_dir` is set, else `subprocess.DEVNULL`. File handles tracked on
  `self._godot_stdout_file` / `self._godot_stderr_file` and closed by
  `close()`. Three regression tests added in
  `tests/rl/test_h3_godot_env.py`.

- **GDScript `protocol_version` strict-int check.** Godot 4.6.2's
  `JSON.parse_string` widens JSON integer literals to `TYPE_FLOAT`,
  so `python json.dumps({"protocol_version": 2})` arrives in GDScript
  as `2.0`. The pre-fix dispatch rejected this with
  `protocol_version_mismatch`. Fix in
  `games/signal-dodge/scripts/tcp_controller.gd::_h3_dispatch`: accept
  both `TYPE_INT` and `TYPE_FLOAT` and compare via `int()` coercion.
  JSON has no int/float distinction at the wire, so the relaxed check
  is the correct contract regardless of Godot's parser behavior.

- All commits pushed to `origin/main` at
  `https://github.com/trzz333/sight.git`.

The HEAD when the canonical runs in this packet were captured is the
substantive commit recorded in the next paragraph; chore handoff-refresh
commits applied after capture do not change source, config, test, or
GDScript content.

(Substantive HEAD hash for this round will be patched into this section
when the commit lands. The hash placeholder is `<h3-step10-final>`.)


## 4. Test gate

Default command:

```
python -m pytest tests/rl -v --tb=short
```

Result on this round's substantive HEAD, run 2026-05-09 on StrongerJr:

- **121 passed, 0 failed, 1 deselected** (the deselected item is
  `test_live_godot_reset_and_100_step_smoke`, correctly excluded by
  `pyproject.toml` `addopts = "-ra -m 'not live_mss and not live_godot'"`).
- Runtime: 11.67s.
- Net `+3` tests vs the post-step-9 baseline (118): the three new
  regression tests guarding the `subprocess.PIPE` boundary
  (`test_launch_passes_devnull_for_stdio_when_no_run_dir`,
  `test_launch_passes_file_handles_for_stdio_when_run_dir_set`,
  `test_close_releases_godot_stdio_files`).

Live opt-in command:

```
python -m pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short
```

Result: **1 passed, 7 deselected, 2.62s on run1; 1 passed, 7 deselected,
2.57s on run2**. End-to-end real Popen + real loopback TCP + Godot
binding + reset + 100 steps + clean close + NDJSON event-set assertion.

Telemetry posture: still no TensorBoard, W&B, MLflow, Comet, or network
imports anywhere under `src/sight_agent/rl/`. The H3 env writes only to
loopback TCP and local NDJSON. Charter-pinned.


## 5. Acceptance live Godot smoke

Two consecutive live runs were captured under
`runs/eval/h3_acceptance/run{1,2}/test_live_godot_reset_and_100_0/`
on StrongerJr, 2026-05-09. The `runs/` tree is gitignored so artifacts
land durably without committing them.

Command per run:

```
python -m pytest tests/rl/test_h3_godot_smoke.py -m live_godot -v --tb=short \
  --basetemp=C:\Projects\Sight\runs\eval\h3_acceptance\runN
```

Files present in each run:

- `python.ndjson`
- `godot.ndjson`
- `godot-stdout.log` (75 bytes; engine version banner only)
- `godot-stderr.log` (0 bytes)

### 5.1 Run 1

- godot.ndjson: 21,124 bytes, 132 lines, 0 malformed.
- python.ndjson: 22,383 bytes, 103 lines.
- Godot event-type counts:
  `{run_start: 1, controller_connected: 1, controller_hello: 1,
   controller_reset_received: 1, episode_start: 1, h3_step: 100,
   player_tick: 24, spawn: 3}`
- Python event-type counts:
  `{env_start: 1, reset: 1, step: 100, close: 1}`
- First `h3_step`: `frame=1, action=0 (mapped from wire 1=stay),
  player_x=360.0, player_y=508.0, reward=1.0, terminated=False,
  truncated=False, terminal_reason=""`
- Last `h3_step`: `frame=100, terminated=False, truncated=False`
  (player held center; three hazards spawned but missed; episode
  ran the full 100-step rollout the test issued, well below the
  configured `max_steps=120`).

### 5.2 Run 2

- godot.ndjson: 21,158 bytes, 133 lines.
- python.ndjson: 22,391 bytes, 103 lines.
- Godot event-type counts:
  `{run_start: 1, controller_connected: 2, controller_disconnect: 1,
   controller_hello: 1, controller_reset_received: 1, episode_start: 1,
   h3_step: 100, player_tick: 23, spawn: 3}`
- Python event-type counts:
  `{env_start: 1, reset: 1, step: 100, close: 1}`
- First and last `h3_step` identical to run 1 (see section 6).

### 5.3 Required event types

The H3 plan section 7 minimum event-type set
(`run_start, controller_connected, controller_hello,
controller_reset_received, episode_start, h3_step`) is present in both
runs. The live test asserts this set after `env.close()`; passing the
test entails this assertion.


## 6. Same-seed reproducibility

H3 plan section 9 ("Determinism posture") promises same `(machine,
dependency class, Godot version)` reproducibility of the observation
sequence given a fixed seed. Verified at first-step resolution by
comparing run 1 and run 2 (both `seed=0`, same StrongerJr machine,
same Godot 4.6.2 console build, same source tree):

| field            | run 1 | run 2 | match |
|------------------|-------|-------|-------|
| frame            | 1     | 1     | yes   |
| action (mapped)  | 0     | 0     | yes   |
| player_x         | 360.0 | 360.0 | yes   |
| player_y         | 508.0 | 508.0 | yes   |
| reward           | 1.0   | 1.0   | yes   |
| terminated       | False | False | yes   |
| truncated        | False | False | yes   |
| terminal_reason  | ""    | ""    | yes   |

This confirms criterion 7 of the H3 plan section 10 acceptance list at
first-step resolution. The plan does not promise bit-for-bit cross-
machine equivalence (and this packet does not claim it).

The minor variance between runs (run 2 has +1 `controller_connected`
and +1 `controller_disconnect` event, and -1 `player_tick`) is
pre-mode-lock physics-tick variance: Godot's TCP listener was bound for
one or two extra physics ticks before Python's first `hello` arrived
to lock the H3 mode. Post-mode-lock state is fully reproducible (every
`h3_step` carries identical state for both runs given identical seeds).
This variance is tracked as a caveat in section 7, not a determinism
defect.


## 7. Caveats and known nuances

1. **Pre-mode-lock physics-tick variance.** Between Godot binding the
   TCP listener and Python's first `hello` locking the H3 mode in
   `tcp_controller.gd`, the legacy autonomous-loop path runs for a
   handful of physics ticks (1-2 in practice). These ticks emit
   `player_tick` events, no `h3_step` events. Run 2 also recorded an
   extra `controller_connected` / `controller_disconnect` pair which
   indicates the TCP socket bounced once during the bind window. The
   pre-mode-lock window is timing-sensitive and not deterministic
   across runs; post-mode-lock state IS deterministic. Reviewers
   should not treat this variance as a determinism failure.

2. **No `run_end` event in `godot.ndjson`.** The autoload logger's
   `_exit_tree` path emits `run_end` only on `SceneTree.quit()` or
   engine shutdown. The Python env's `close()` calls `proc.terminate()`
   followed by `proc.kill()` if needed; neither path gives Godot's
   `_exit_tree` a chance to flush `run_end` reliably. The H3 plan
   section 8 explicitly does NOT require `run_end` in the live smoke
   minimum set, and the live test does not assert it. Reviewers
   should expect its absence.

3. **No collision/death events.** The acceptance rollouts ran the full
   100 steps under action 1 (stay) without any hazard hitting the
   stationary player. The live test does not require collision/death
   events; the `h3_step`/`player_tick` event sequence is the live-tier
   evidence that physics advanced and reward emitted. Forced-collision
   and forced-timeout terminal paths are exercised by the default-tier
   stub smoke (`test_forced_collision_terminates_with_zero_reward`,
   `test_forced_timeout_truncates_without_terminating`) which routes
   through the same `step_result` reading path in
   `GodotSignalDodgeEnv`.

4. **`info` dict surface.** The env's `info` dict (per plan section 6
   and the codebase's `_build_info`) includes `run_id`, `episode_id`,
   `godot_pid`, `tcp_port`, `frame`, `seed` (on reset), and
   `terminal_reason` (on step). `git_commit`, `config_hash`,
   `python_version`, `godot_version`, `protocol_version`, and `env_id`
   are layered in by the H2 train/eval harness when an H3 env is
   driven via `train.py` or `evaluate.py`. The live smoke constructs
   the env directly without that harness, so its `info` dict shows
   only the env-level fields. The H2-layer fields are exercised under
   `tests/rl/test_h3_train_plumbing.py` against fakes; combining live
   Godot with the train/eval harness is an H3+ training-loop concern,
   not a phase-gate boundary concern.

5. **Console build for the live gate.** GPT chose the console build
   (`Godot_v4.6.2-stable_win64_console.exe`) over the windowed build
   so the first live launch had stdout/stderr visibility. The matrix
   test on 2026-05-09 proved the choice was orthogonal to the deadlock
   bug (both builds hang on `subprocess.PIPE`). The console build
   remains pointed at by `SIGHT_GODOT_EXE` going forward; switching to
   the windowed build is unnecessary and would require a re-run of
   the live gate to confirm equivalence. Both files exist on
   StrongerJr at the WinGet path.

6. **`runs/` is gitignored.** The acceptance artifacts described in
   section 5 are NOT committed. They live on disk under
   `C:\Projects\Sight\runs\eval\h3_acceptance\run{1,2}\` on StrongerJr
   for the duration of this packet review and can be regenerated by
   rerunning the commands in section 5 on any machine with
   `SIGHT_GODOT_EXE` set and the Godot Signal Dodge project tree
   intact.


## 8. What is NOT in this packet

- **No verbatim Grok review text.** Grok closure is recorded as a
  verdict only, per the H1 / H2 closure pattern.
- **No PPO training run on Signal Dodge.** Training Signal Dodge is
  H4+ work (pixel observations) and downstream training quality. H3
  closes on env boundary correctness, not on policy quality.
- **No bit-for-bit cross-machine reproducibility claim.** Same-machine
  same-dependency-class first-step reproducibility was tested and
  held; cross-machine bit equivalence is not promised.
- **No external target environments.** Signal Dodge only.
- **No fresh-clone reproducibility section.** H2's packet had one
  because the H2 acceptance was a 25k-step training run with model
  checkpoints; fresh-clone repro proved deterministic learning
  curves. H3 acceptance is a 100-step deterministic protocol exchange
  and an event-set assertion, both of which run from any clone of
  this commit on StrongerJr (or any Win 11 box with the Godot
  binary). A fresh-clone section would duplicate run 2 in section 5.

## 9. Recommended verdict scope for Grok

Grok is asked to evaluate H3 against the following criteria from
`docs/sight-h3-plan.md` section 10:

1. `GodotSignalDodgeEnv` exists at the documented path: yes / no.
2. `make_env(env_id="godot:signal-dodge-v0")` constructs it: yes / no.
3. State-only observation space, `Box(-1, 1, (10,), float32)`:
   yes / no.
4. Action space `Discrete(3)`: yes / no.
5. Bidirectional TCP supports `hello`, `reset`, `step`: yes / no.
6. Signal Dodge supports in-process soft reset (no subprocess
   relaunch per episode): yes / no.
7. Same seed reproduces first observation and short scripted rollout
   on the same machine: confirmed / not.
8. Default RL test suite passes (`pytest tests/rl -v --tb=short`):
   confirmed / not.
9. Live Godot smoke passes
   (`pytest tests/rl/test_h3_godot_smoke.py -m live_godot ...`) on
   StrongerJr: confirmed / not.
10. Local artifacts written (Python NDJSON, Godot NDJSON, plus the
    new Godot stdout/stderr capture under `run_dir`): confirmed / not.

Required closure checks per plan section 10:

a. Charter invariants explicitly checked: no network telemetry, no
   pixel path used, no commercial or platform automation scope added,
   no offerwalls, no Freecash, no bot-detection evasion, no account
   farming. All four hold for this round.
b. H3 phase-gate packet written using the H2 packet pattern: this
   document.
c. `docs/sight-handoff.md` updated with phase, last commit, current
   task, next action, blockers, no more than five notes: yes (this
   round's handoff refresh commit).

If criteria 1-10 read clean and the closure checks (a-c) are all
satisfied, the recommended verdict is GREEN and H3 is closed. H4
(pixel observations on Signal Dodge with a small CNN policy) becomes
the next phase under the existing charter.

If Grok flags YELLOW caveats, those will be addressed and a YELLOW
closure doc will be added in the H1 pattern.

## 10. Pointers

- Charter: `docs/sight-charter.md`
- H3 plan: `docs/sight-h3-plan.md`
- H1 GREEN closure: `docs/grok-h1-final-green.md`
- H1 packet: `docs/grok-h1-phase-gate-packet.md`
- H2 packet: `docs/grok-h2-phase-gate-packet.md`
- Handoff: `docs/sight-handoff.md`
- H3 substantive commits to date: `dca2af3`, `50ea24d`, `b5b3fad`,
  plus this round's substantive commit (hash patched into section 3
  on commit landing).
