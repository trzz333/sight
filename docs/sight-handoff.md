# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Signal Dodge (930.27 bar), from-scratch (Jeff-ruled). Replica recipe search COMPLETE: gamma-0.99 clears the port gate. Next phase is the Godot eval-of-record.

**Last commit:** `0970da8` sd-fast: gamma-0.99 arm clears the port gate 5/5, port to Godot next

**Current task:** g99 arm JUDGED, PORT verdict. On the fast replica, `sd_fast_m21curr_g99_s{0..4}_5M` (m21 + start-curriculum + gamma 0.99) clears 5/5: held-out per-seed [1671.1, 1800.0, 1778.6, 1800.0, 1800.0], rliable IQM 1792.9, 95% CI [1707.0, 1800.0], CI lower bound 1707 >> bar 930.27. vs none +1059.3 (P(IQM)=1.000); vs curr +398.3 (P(IQM)=0.995). The discount cut (gamma 0.999->0.99, horizon ~1000->~100) is what converted the curriculum arm from not-port-reliable to saturating; first reliable from-scratch clear in project history. Full result + self-audit in `docs\sd-fast-curriculum-findings.md`.

**Next action:** Port DISCOUNT-FIRST to a Godot 5M eval-of-record. The curriculum is NOT injectable into real Godot without GDScript+protocol work (`godot_env.reset()` passes only a seed; no hazard-injection seam), so port gamma-0.99 alone with no curriculum first: it needs zero Godot-side changes and isolates whether the discount fix transfers. Adapt `configs\rl\signal_dodge_ppo_h3.yaml` into `signal_dodge_ppo_g99.yaml` (n_envs 8, total_timesteps 5_000_000, hyperparams gamma 0.99 / n_steps 512 / batch_size 512 / n_epochs 10 / ent_coef 0.01 / clip_range 0.2 / lr 3e-4 / gae_lambda 0.95 / net_arch [64,64]); run `python -m sight_agent.rl.train --config ...`. THREE cheap checks BEFORE the 5M launch (all unverified): (1) does `train.py._build_train_env` (~L157-204) + `factories.py` wrap VecNormalize with configurable gamma? recipe normalizes obs+reward at gamma 0.99; if not applied the port is unfaithful. (2) Godot throughput via the h3 smoke (1024 steps) to size the run (sd_fast did 6551 steps/s; Godot subprocess is far slower). (3) eval-of-record: greedy held-out seeds 5000-5029 vs 930.27 exists for Godot (`h5_baseline_cli` / `rl.evaluate`) or wire it. If discount-only clears reliably, done. If short, THEN build curriculum injection (GDScript pre-spawn + protocol option + `reset(options=)`) and port the full recipe. Do NOT fire an unvalidated config into a multi-hour detached run.

**Blockers:** None requiring Jeff. Godot is the mission target env (not a new-target Jeff call). Scope (from-scratch) resolved.

**Notes:**

- Godot training is config-driven: `sight_agent.rl.train --config <yaml>` -> `factories.make_algo`; recipe knobs live in the YAML `algo.hyperparams`, not CLI. Env id `godot:signal-dodge-v0`, project `games/signal-dodge`, binary via YAML `godot_executable` or `SIGHT_GODOT_EXE`. DEFAULT_MAX_STEPS 1800 (the 930.27/1800 geometry lives here).
- Replica recipe search DONE and archived in `docs\sd-fast-curriculum-findings.md`. rliable harness `tools\sd_fast_reliability.py` now carries none/shaped/curr + the g99 arm (all 5 g99 models present, so it evaluates every run). Reload-eval cache `runs\sd_fast\reliability_eval_cache.json` (gitignored).
- Root-cause chain (HIGH): from-scratch failure was critic-variance under the near-undiscounted (gamma 0.999) return target on an 1800-step survival task, NOT entropy collapse (probe refuted) and NOT policy capacity. Cutting the discount fixed it.
- Twice-failed levers, do NOT retry: reward geometry (K5.5 shaping + PBRS), CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet, ent_coef bump (refuted). Imitation clears reliably (BC IQM 1800, PPO-ft 1738.5) but scope is from-scratch.
- `runs\` gitignored. g99 chain done (CHAIN_G99_DONE); throwaways `tools\_curr_g99_chain.py` and `tools\_g99_watch.py` deleted this session. Pre-existing `tools\_probe_shaped_collapse.py` / `tools\_spawn_shaped_sweep.py` remain (older, harmless). Status server pid ~22984 may linger; harmless.
