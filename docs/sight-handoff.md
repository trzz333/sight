# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H4 implementation in progress. H4 Steps 1 through 7/8 are landed. Default test gate at 228 passed, 2 deselected on `tests/rl`. Live H4 128-step pixel training smoke green on StrongerJr: train exit 0, all eight expected artifact files present under `C:\Projects\Sight\runs\rl\signal_dodge_ppo_h4_pixel\20260511T011310Z_signal_dodge_ppo_h4_pixel_seed0_4cf8175` (gitignored), no stray `games/signal-dodge/runs/` leak. Next slice is H4 Step 9 acceptance: repeated same-seed reproducibility evidence per `docs/sight-h4-plan.md` section 10, then phase-gate packet prep.

**Last commit:** `0567fec` fix(rl): isolate godot tcp port and absolute godot log path for h4 live train

**Current task:** H4 Step 7/8 complete on main. Live 128-step CPU PPO `CnnPolicy` smoke against the H4 pixel config exits 0. Run dir `C:\Projects\Sight\runs\rl\signal_dodge_ppo_h4_pixel\20260511T011310Z_signal_dodge_ppo_h4_pixel_seed0_4cf8175` contains `summary.json` (status ok, phase H4, total_timesteps 128), `events.ndjson` (run_start with `env_smoke.obs_shape=[1,84,84]`, two `eval` at step 64/128, two `train_metrics`, `run_end`), `config_effective.yaml` preserving `observation_mode: pixel` and `headless: false`, `model.zip` (~20 MB), and `godot-train/` plus `godot-eval/` each containing `python.ndjson` + `godot.ndjson` + `godot-stdout.log` + `godot-stderr.log`. Two latent production bugs surfaced and were fixed inside the H4 live-smoke boundary: factory now allocates a kernel-assigned loopback TCP port per Godot env (eliminating train/eval 8765 collision), and `godot_env._launch_godot` now passes an absolute `SIGHT_GODOT_LOG_PATH` so Godot's File API does not resolve relative to its `--path` working directory.

**Next action:** H4 Step 9 (= `docs/sight-h4-plan.md` section 10). Run the H4 acceptance pair: two same-seed live 128-step pixel training runs on StrongerJr using `configs/rl/signal_dodge_ppo_h4_pixel.yaml` with `SIGHT_GODOT_EXE` set. Acceptance gate: same-seed step-by-step pixel observation equality across the two runs at the env-observation granularity (per the H4 plan section 9 / 10 criterion 6 contract already proven for the standalone env in `tests/rl/test_h4_godot_pixel_smoke.py`). After both runs land green, draft `docs/grok-h4-phase-gate-packet.md` per the H3 phase-gate packet pattern.

**Blockers:**

- Live H4 train smoke requires `SIGHT_GODOT_EXE` set in the shell. The User-scope value is `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe`; Desktop Commander's parent shell does not inherit User-scope env vars, so the live-smoke batch sets it inline. Operational.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Sight runs on the standalone DC remote MCP, unaffected.

**Notes:**

- **H4 Step 7/8 surfaced two production bugs.** First attempt failed at in-train eval with `recv timed out after 5.0s` because train and eval Godot envs both tried to bind `127.0.0.1:8765` (Godot's `tcp_controller.gd` returned WSAEINVAL 22 on `listen()`). H3 acceptance never hit this because H3 ran train and out-of-band eval sequentially. Fix in `factories._make_godot_signal_dodge_v0`: per-env `_allocate_isolated_tcp_port()`. Second attempt produced no `godot.ndjson` because Godot 4.6.2's File API resolves relative paths against the `--path` project working dir, dropping the file at `games\signal-dodge\runs\rl\...\godot-train\godot.ndjson`. Fix in `godot_env._launch_godot`: `str(log_path.resolve())`. Both bugs are pre-existing; H3 missed them because pytest's `tmp_path` is absolute and H3 acceptance ran train/eval sequentially.
- **Live smoke timing.** ~116s wall clock for 128 timesteps on StrongerJr's hybrid GPU under windowed Godot. FPS 1 per train_metrics; this is the windowed-render cost, not a learning signal. Step 9 acceptance runs will share the same per-run budget.
- **H3 acceptance contracts untouched.** Tests that explicitly construct `GodotSignalDodgeEnv(tcp_port=8765)` still get exactly 8765 (the factory-level allocator only fires when the env is built through the factory). The `SIGHT_GODOT_LOG_PATH.endswith("godot.ndjson")` unit assertion still passes against the new absolute path.
- **H4 reset and capture invariants intact.** Same-seed step-by-step pixel trajectory equality test (`test_h4_godot_pixel_smoke.test_live_godot_pixel_same_seed_step_by_step_trajectory_equality`) still green; the new TCP port and log-path fixes touch only env construction and child-process env vars, not the pixel wire path.
- **H4 windowed-mode launch remains mandatory.** Pixel mode rejects `headless=True` at env construction. The H4 config explicitly sets `headless: false`.
