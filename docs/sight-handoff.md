# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Godot eval-of-record (discount-first port). Replica recipe search complete; gamma-0.99 cleared the replica gate. Porting to real Godot Signal Dodge, bar 930.27.

**Last commit:** `02a6b0d` sd-godot: build + smoke-validate discount-first Godot g99 trainer, launch 1M probe

**Current task:** Godot port infra built and SMOKE-VALIDATED; first real run in flight. New durable tool `tools\sd_godot_ppo_g99.py` ports the g99 recipe (m21 + VecNormalize @ gamma 0.99, reward none, 8 envs, MlpPolicy [64,64]) to the real Godot env, NO curriculum (not injectable without GDScript+protocol work). Smoke (2 envs, 2000 steps, 3 eval seeds) exercised multi-process Godot construction with distinct TCP ports, VecNormalize wrap+save, and greedy held-out-seed eval: mean_len 323.0, diverse actions, explained_variance 0.2891, beats_bar false (expected at 2000 steps). Infra correctness HIGH; clearing behavior at real budget UNKNOWN on Godot. First run `g99_godot_1M_s0` (single seed, gamma 0.99, 1M steps, 8 envs headless) launched detached, pid 34116, ~119 steps/s, ETA ~2.3h, log `runs\sd_godot\g99_godot_1M_s0.log`, summary will be `runs\sd_godot\g99_godot_1M_s0_summary.json`.

**Next action:** When `g99_godot_1M_s0_summary.json` lands, read its eval.mean_len and judge against the controlled contrast M2.1 (gamma 0.999 / 1M / Godot -> IQM 418) and the bar 930.27. If mean_len lifts clearly above ~418 toward/past 930.27, the discount transfers: launch the 5M single seed, then on a clear 5M clear launch the 5-seed reliable arm (`g99_godot_5M_s{0..4}`) and run rliable IQM+CI (adapt `tools\sd_fast_reliability.py` or inline bootstrap on the 5 Godot summaries; port gate = 5/5 AND IQM CI lower bound > 930.27). If mean_len is flat near 418, the discount alone does not transfer within budget: switch method to the curriculum injection path (GDScript pre-spawn N hazards at reset + protocol option + `godot_env.reset(options=)`), do not rerun 1M harder. Relaunch pattern for more seeds: `set SIGHT_GODOT_EXE=<winget Godot_v4.6.2 path>` then `.venv-c1\Scripts\python.exe tools\sd_godot_ppo_g99.py --steps <N> --n-envs 8 --seed <s> --run-id <id>` (detached via Popen CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP, DEVNULL).

**Blockers:** None requiring Jeff now. One conditional Jeff scope call, only if BOTH the discount-only port and the curriculum-injection port fail to clear on Godot: whether to accept imitation (BC 1737.3, PPO-ft 1710.5) as the standing solution or keep pursuing from-scratch reliability.

**Notes:**

- Godot binary `Godot_v4.6.2-stable_win64.exe` at `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\`. Not on PATH; set `SIGHT_GODOT_EXE`. `where godot` empty.
- VecNormalize is NOT in `train.py`/`factories.py` (grep zero hits); that is why the port uses a dedicated trainer, not `train.py`. `rl.evaluate` is also unfaithful (no VecNormalize, single global seed not held-out 5000-5029), so eval is wired inside `sd_godot_ppo_g99.py`.
- `make_env` blocks n_envs>1 for Godot; the trainer builds 8 GodotSignalDodgeEnv directly, each with its own kernel-allocated TCP port. DummyVecEnv steps serially so aggregate throughput ~= single-env rate (~119/s headless, 8 envs).
- Replica gate (prior, committed `0970da8`): g99 cleared 5/5, IQM 1792.9, 95% CI [1707.0, 1800.0] vs bar 930.27. The discount cut (horizon ~1000->~100) fixed the critic-variance wall. Full record in `docs\sd-fast-curriculum-findings.md`.
- `runs\` gitignored. Session throwaways deleted (`_chk.py`, `_launch_g99_godot_1M.py`). Pre-existing `tools\_probe_shaped_collapse.py` / `_spawn_shaped_sweep.py` remain (older). Status server pid ~22984 may linger; harmless.
