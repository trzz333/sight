# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 (first trained-CnnPolicy slice produced; NOT closure-grade)

**Last commit:** `f80489b` docs(h5): trained-policy phase B 2048-timestep evidence slice

**Current task:** First H5 trained-CnnPolicy evidence slice landed. 2048-timestep PPO CnnPolicy training on the existing H4 pixel Signal Dodge profile produced a model that beats negative controls by ~14% on reward and length and ~10pp on collision rate. This is BELOW the H5 section 6 GREEN bar (25% / 25% / 20pp). Profile saturation gate passes cleanly. Findings written to `docs/h5-trained-policy-phase-b-evidence.md`.

**Next action:** GPT decides training budget for the next H5 slice. Primary blocker is insufficient training (32 PPO updates is far below typical pixel-CNN budgets). Recommended: 10K-50K timestep training run on the same H4 pixel profile, same 10-seed full-mode eval, then re-check section 6 GREEN bars. Step 2B (profile hardening) remains unjustified; profile headroom is ample (best negative-control length_ratio = 0.337).

**Blockers:** none operational. Open: training budget call for next slice belongs to GPT.

**Notes:**

- Exact commands used: training `python -m sight_agent.rl.train --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --total-timesteps 2048 --run-id h5_train_phase_b_2048`; eval `python -m sight_agent.rl.h5_baseline_cli --config configs/rl/signal_dodge_ppo_h4_pixel.yaml --run-id h5_eval_phase_b_10seed --seeds 1000-1009 --mode full --train-run-dir runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_b_2048`. Artifacts under `runs/rl/signal_dodge_ppo_h4_pixel/h5_train_phase_b_2048/` and `runs/rl/signal_dodge_ppo_h4_pixel/h5_eval_phase_b_10seed/evaluation/`.
- 10-seed aggregate means (seeds 1000-1009): stay_only=606.0 length / 1.0 collision, seeded_random=414.3 / 1.0, untrained_cnn=606.0 / 1.0, trained_cnn=689.7 / 0.9. Trained vs best negative control: +13.8% length, +13.9% reward, 10pp collision-rate reduction. Saturation gate passes (all negative-control length_ratio <= 0.337, max threshold 0.80).
- `untrained_cnn` produces byte-identical per-seed trajectories to `stay_only` across all 10 seeds. The deterministic argmax of a randomly-initialized SB3 PPO CnnPolicy collapses to action 1 (stay). Practically the H5 negative-control suite delivers 2 independent baselines on this profile, not 3. Already anticipated by `docs/sight-h5-plan.md` section 1; worth flagging for GPT when sizing the next slice.
- No code changes this session. Existing `sight_agent.rl.train` (produces `model.zip`) and `sight_agent.rl.h5_baseline_cli --mode full --train-run-dir <path>` (evaluates all 4 policies including trained_cnn) cover the full slice end-to-end. No targeted tests required; default `tests/rl` and H3/H4 same-seed determinism re-check intentionally deferred since no code paths were touched and live-Godot test budget was spent on training and eval.
- `SIGHT_GODOT_EXE` was set inline in the live cmd.exe session to `C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64.exe`. The User-scope env var currently points at the `_console.exe` variant, which is fine for CLI but the inline setting matches the path used by the H4 pixel-determinism live tests.
