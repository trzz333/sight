# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (Phase C 10K slice complete; NOT closure-grade; "more timesteps" falsified)

**Last commit:** `89e5d8f` docs(h5): trained-policy phase C 10k evidence slice

**Current task:** Phase C 10000-timestep PPO CnnPolicy slice (156 updates, 1431s wall) on the existing H4 pixel Signal Dodge profile produced bit-identical per-seed eval results to Phase B (2048 ts, 32 updates). Trained_cnn aggregate (10 seeds 1000-1009): 688.8 reward, 689.7 length, 0.9 collision rate, 0.383 length_ratio. All three H5 section 6 GREEN bars FAIL at the same magnitude as Phase B (+13.9% reward, +13.8% length, 10pp collision-rate reduction). Saturation gate passes. Full 4-policy eval skipped per Phase C decision rule. Findings written to `docs/h5-trained-policy-phase-c-10k-evidence.md`.

**Next action:** GPT decides hyperparameter recipe for the next H5 slice. The "more timesteps" hypothesis is falsified; Phase C is empirically equivalent to Phase B despite 5x training. Primary blocker is the smoke-cheap YAML defaults, specifically `ent_coef=0.0` driving premature entropy collapse (entropy_loss hit -0.07 by iter 36 then -0.003 by iter ~100 and stayed; from iter ~80 `approx_kl=0`, `clip_fraction=0`, `policy_gradient_loss ~1e-7`). Recommended candidates: raise `ent_coef` to 1e-2 or 1e-3, optionally enlarge `n_steps`/`n_epochs`, keep timestep budget similar.

**Blockers:** none operational. Open: hyperparameter recipe call for next slice belongs to GPT.

**Notes:**

- Exact commands: training `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --total-timesteps 10000 --run-id h5_train_phase_c_10k`; eval `python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --run-id h5_eval_phase_c_10k_trained_only --seeds 1000-1009 --mode full --policies trained_cnn --train-run-dir runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_c_10k`. Artifacts under `runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_c_10k/` and `runs/rl/signal_dodge_ppo_h4_pixel/h5_eval_phase_c_10k_trained_only/evaluation/trained_cnn/`.
- Phase C trained_cnn per-seed results are bit-identical to Phase B trained_cnn per-seed results. Seed 1001 timeouts at length 1800 in both. Seed 1006 collides at length 183 in both. Seeds 1000, 1002-1005, 1007-1009 produce identical collision lengths. Negative controls were not re-run; they are seed-deterministic and the Phase B aggregates remain authoritative.
- Training wall 1431s (~24 min) per SB3 `time_elapsed`. Eval wall ~191s aggregate per-seed elapsed (~3 min). No code changes this session. No targeted tests run.
- The committed `configs/rl/signal_dodge_ppo_h4_pixel.yaml` is explicitly tagged "Smoke-cheap PPO values" sized for H4 acceptance ("constructs and runs while writing artifacts"), not learning. `ent_coef` not present → SB3 default of 0.0. `n_steps=64`, `batch_size=32`, `n_epochs=1`. These are the candidates for the next slice's hyperparameter change.
- `SIGHT_GODOT_EXE` was set inline in the live cmd.exe session to `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`, same as Phase B. User-scope env var still points at the `_console.exe` variant.
