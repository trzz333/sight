# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H3 closed via Grok GREEN on 2026-05-09 (closure artifact `docs/grok-h3-final-green.md`). H4 Steps 1 through 6 are landed, tested, and pushed. Default test gate at 228 passed, 2 deselected on `tests/rl` (up by 6 from Step 5 across `test_rl_config.py` and `test_h3_train_plumbing.py`; live_godot pool unchanged at 2). Live smoke gate at 1 passed (H4 pixel same-seed step-by-step trajectory equality from Step 4). Next slice is H4 Step 7/8: live opt-in 128-step CPU PPO `CnnPolicy` smoke on StrongerJr using `configs/rl/signal_dodge_ppo_h4_pixel.yaml`. This is the first H4 step that actually launches Godot under the new plumbing.

**Last commit:** `b7a3e72` feat(rl): add h4 pixel cnn config plumbing

**Current task:** H4 Step 6 complete on main. Added `configs/rl/signal_dodge_ppo_h4_pixel.yaml` plus the supporting plumbing so the config is truthful end-to-end: `godot_config.resolve_godot_kwargs` now threads optional env-construction kwargs (`max_steps`, `headless`, `observation_mode`, `pixel_width`, `pixel_height`, `pixel_channels`) when present in YAML; `factories.make_env` and `_make_godot_signal_dodge_v0` accept and filter them so the env constructor never receives a literal `None` that would override its own default; `train._godot_smoke_obs_metadata` reports `(1, 84, 84)` for pixel mode and `(10,)` for state mode in the NDJSON `run_start` event without launching Godot. `evaluate.py` picks up the new kwargs automatically through `resolve_godot_kwargs`. H3 production paths are env-level byte-shape unchanged: H3 YAML always carried `max_steps: 1800`, which is also the env default, so the resolver now passes it explicitly but the env behavior is unchanged.

**Next action:** H4 Step 7/8 (= plan implementation sequence step 8, "Optional `--total-timesteps 128` smoke run on StrongerJr"). Live opt-in. Set `SIGHT_GODOT_EXE` to the local Godot binary, then run `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h4_pixel.yaml`. Expected: one windowed Godot launch, two rollouts at `n_steps=64` for `total_timesteps=128`, one in-train eval at step 64, NDJSON evidence under `runs/rl/<run_id>/{events.ndjson, godot-train/, godot-eval/, summary.json, model.zip, config_effective.yaml}`. Acceptance is "constructs and runs while writing artifacts," not "learns." After this lands, H4 Step 9 = acceptance runs with same-seed reproducibility evidence per `docs/sight-h4-plan.md` section 10.

**Blockers:**

- Live H4 train smoke requires `SIGHT_GODOT_EXE` set in the shell environment that runs train.py. Operational, not a code blocker.
- H4 windowed-mode capture pops a real OS window during live runs. Acceptable for StrongerJr local sessions; would need a virtual display for unattended CI, out of current scope.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- **H4 Step 6 plumbing rule.** The resolver does not invent defaults: it includes a key only when YAML has it. The factory does not override the env's defaults: it drops `None` values before calling the env constructor. Net effect: a YAML that explicitly sets `headless: null` cannot mask the env default with `None`, and an H3 YAML that omits `headless` keeps the env's default of `True` exactly. Lock-in tests in `tests/rl/test_rl_config.py` (`test_resolve_godot_kwargs_h3_omits_pixel_fields`) and `tests/rl/test_h3_train_plumbing.py` (the H4 plumbing test).
- **H4 pixel config shape.** `configs/rl/signal_dodge_ppo_h4_pixel.yaml` carries `observation_mode: pixel`, `headless: false`, `(1, 84, 84)` uint8 dims, `max_steps: 1800`, smoke-cheap PPO `n_steps=64, batch_size=32, n_epochs=1`, `total_timesteps=128`, `eval_freq=64`, `n_eval_episodes=1`, `policy: CnnPolicy`, `device: cpu`. MlpPolicy would silently flatten the (1,84,84) tensor; CnnPolicy is the load-bearing choice.
- **H4 Step 5 invariants intact.** `tests/rl/test_h4_cnn_policy_construct.py` still green; the stub-env CnnPolicy construction surface did not need to change for the new plumbing because it does not use `GodotSignalDodgeEnv`.
- **H4 Step 4 invariants intact.** Same-seed plus same scripted-action pixel trajectory equality (`np.array_equal` on every observation) remains the binding determinism criterion. Live smoke verifies this at 1 reset + 10 steps; H4 Step 9 acceptance run will scale this up.
- **H4 windowed-mode launch is canonical.** Pixel mode rejects `headless=True` at env construction (the H4 spike proved Godot 4.6.2 `--headless` does not emit `frame_post_draw`). The new YAML explicitly sets `headless: false` so the env constructor receives the correct value and the run does not depend on env defaults.
