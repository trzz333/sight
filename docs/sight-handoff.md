# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H3 implementation. Step 7 (factory routing for `godot:signal-dodge-v0` in `src/sight_agent/rl/factories.py::make_env`) closed and validated. Step 8 (config `configs/rl/signal_dodge_ppo_h3.yaml`) is next.

**Last commit:** `060d49c` feat(rl): route signal dodge godot env through factory

**Current task:** H3 step 7 closed. `pytest tests/rl -v --tb=short` 102 passed in 32.33s (100 prior + 2 net new; the H2-era `godot:` prefix rejection pin was flipped into a positive-route test rather than duplicated). Ready for step 8.

**Next action:** Step 8: add `configs/rl/signal_dodge_ppo_h3.yaml` per H3 plan section 1. Minimum posture: `env.id: godot:signal-dodge-v0`, `env.n_envs: 1`, SB3 PPO, MlpPolicy, CPU-safe defaults, smoke-oriented training values only if training is included. The factory now expects `godot_executable` and `project_path` as keyword-only args at call time; step 8 should resolve those from the YAML config (with `SIGHT_GODOT_EXE` / `SIGHT_GODOT_PROJECT` env-var fallback inside the config-loading layer, not inside the factory) and thread them into `make_env(...)` along with `run_dir`. Do not touch the factory in step 8 unless config shape demands it.

**Blockers:**

- Live Godot smoke (plan steps 10/12) still gated on the `live_godot` pytest marker, which is not introduced yet. Step 7 produced no live coverage; the positive route test exercises construction only, not `reset()`. Godot 4.6.2 binary on StrongerJr is already installed and parse-validated, so step 10 itself is unblocked when the time comes.
- Claude Desktop GPU/driver crash on Jeff's primary box. Tracked in `C:\Projects\ops\claude-desktop-crash-ledger.md`. Operational only, not a Sight evidence blocker. Sight sessions run on standalone DC remote MCP (deviceId `64416a67-1bdb-42fc-bf1a-48f988e6901d`).

**Notes:**

- Factory dispatch: `make_env` now branches on `env_id == "godot:signal-dodge-v0"` (module-level `GODOT_SIGNAL_DODGE_V0` constant) before the existing Gymnasium check. Other `godot:` prefixed ids fall through to the Gymnasium loose check, which excludes the `godot:` prefix, then to the unsupported-env_id ValueError. Only the exact lowercase id is routed; unknown Godot games cannot quietly slip in.
- Path resolution precedence inside the Godot branch: explicit kwarg > env var (`SIGHT_GODOT_EXE`, `SIGHT_GODOT_PROJECT`) > (project_path only) repo-root-relative `games/signal-dodge` via `_default_signal_dodge_project_path()`. `godot_executable` has no default; missing kwarg + missing env var raises `ValueError` naming both. Env vars never override explicit kwargs. The factory does not parse YAML; that is step 8.
- Vectorization gate: `n_envs != 1` raises with the exact wording `env_id='godot:signal-dodge-v0' supports n_envs=1 only in H3; vectorized parallel Godot envs are explicitly out of scope. Got n_envs={n_envs}.` No silent clamp. The generic `n_envs >= 1` check still fires first when `n_envs < 1`, matching the existing message contract.
- Lazy imports: `from .godot_env import GodotSignalDodgeEnv` and `from stable_baselines3.common.vec_env import DummyVecEnv` only execute inside the Godot branch. Gymnasium-only training paths (CartPole) do not import the Godot transport. The env's own subprocess-launch path is also lazy (first `reset()`), so wrapping in `DummyVecEnv` does not spawn Godot; the positive-route test asserts `underlying.godot_pid is None` after construction. `run_dir` is exact pass-through to the env constructor; the factory adds no train/eval suffixes and no logging of its own (the env owns `python.ndjson` and the Godot `SIGHT_GODOT_LOG_PATH`).
- Eval seeding: `effective_seed = seed if mode=="train" else seed + 10_000`. Threaded into `GodotSignalDodgeEnv(seed=...)` (consumed by first `reset()`) and through `DummyVecEnv.seed(effective_seed)` to mirror `make_vec_env` behaviour. The vec_env.seed call is wrapped in a defensive try/except so a future VecEnv subclass that raises on seed cannot block construction; this path is not load-bearing.
- HEAD progression: `ab23e36` (H3 step 6 hardening) -> `a801d58` (handoff refresh post step 6 hardening) -> `060d49c` (H3 step 7 factory routing) -> handoff refresh (this commit). Default RL gate: 102 passed in 32.33s on StrongerJr, no live Godot.
