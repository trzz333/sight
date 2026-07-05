# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Godot eval-of-record, curriculum-injection port. The start-state curriculum injection seam is BUILT and validated end to end on real Godot; the full proven recipe (curriculum + gamma 0.99) can now run on the eval-of-record. Bar 930.27.

**Last commit:** `95c57df` sd-godot: build + validate start-state curriculum injection seam (GDScript+protocol+env+trainer)

**Current task:** Seam complete across all four layers and validated. (1) GDScript: `tcp_controller` carries optional `curriculum_n_init` on the H3 reset wire (validated non-negative int, stamped onto the parked request, omitted-when-0 so the clean/eval path is byte-identical); `main._h3_perform_soft_reset` pre-spawns N hazards above the player after `seed()`, mirroring the replica `CurriculumSDF` (x~U(12,708), y~U(-24, player_y-100); 100px headroom > collision threshold 28, so no reset collision and no insta-death). N=0 path unchanged. (2) Protocol/env: `godot_transport.reset` gains `curriculum_n_init` (wire field only when >0); `GodotSignalDodgeEnv` exposes a public `curriculum_n_init` attr (AnnealCurriculum set_attr target) plus an options override, forwarded on reset. (3) Trainer: `tools\sd_godot_ppo_g99.py` gains `--curriculum` (AnnealCurriculum 6->0 over anneal_frac 0.7, pre-seeded via set_attr so the first reset already injects); summary records the curriculum block. (4) Validation this session, all HIGH: single-env smoke SMOKE_OK (clean-start obs byte-identical incl. explicit options=0; N=6 injects exactly 6 above-player hazards with no reset/insta collision; live count mutation to 3 at reset; anneal-terminal 0 returns to clean); a short `--curriculum` train+clean-eval completed (recipe `g99-verbatim-godot+start-curriculum`, curriculum.enabled true, ~88 steps/s); Godot scripts compile+run headless; `pytest` 115 pass (fakes updated + 6 new curriculum tests: wire omit/include/reject-negative, env default/attr-forward/options-override). Prior context (established, committed): discount-only Godot port did NOT transfer (`g99_godot_1M_s0` IQM 476 vs bar 930.27, EV -2.03); seed-selection ruled out (0/50 seeds have early above-player hazards).

**Next action:** Launch the Godot curriculum arm. First the 1M controlled contrast: `tools\sd_godot_ppo_g99.py --curriculum --n-init-max 6 --anneal-frac 0.7 --steps 1000000 --n-envs 8 --run-id g99curr_godot_1M_s0` (SIGHT_GODOT_EXE + PYTHONPATH=src set), detached via subprocess.Popen (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP), monitored by disk polls. Compare its held-out greedy IQM (seeds 5000-5029 vs 930.27 on the clean env) against the discount-only 476 and the bar. If it clears or clearly moves toward the bar, run the 5M + 5-seed rliable arm and apply the gate: 5/5 seeds clear AND IQM 95% CI lower bound > 930.27. If the 1M contrast lands well below the bar with a flat/negative critic like the discount-only arm, do not launch 5M; surface the imitation-vs-from-scratch scope call (Blockers) with the evidence.

**Blockers:** None requiring Jeff now. One conditional Jeff scope call, only if the curriculum-injection port ALSO fails to clear on Godot: whether to accept imitation (BC 1737.3, PPO-ft 1710.5) as the standing solution or keep pursuing from-scratch reliability.

**Notes:**

- Seam anchors: GDScript in `games\signal-dodge\scripts\{tcp_controller,main}.gd`; Python in `src\sight_agent\rl\{godot_transport,godot_env}.py`; trainer flag in `tools\sd_godot_ppo_g99.py`. All in substantive commit `95c57df`.
- Curriculum recipe mirrors the replica `tools\sd_fast_ppo_curriculum.py` (INIT_HEADROOM 100, n_init_max 6, anneal_frac 0.7, AnnealCurriculum ported verbatim). Eval never sends `curriculum_n_init`, so held-out eval stays clean.
- Throwaway smoke script deleted; smoke run artifacts under `runs\sd_godot\_smoke_curr_run` and `runs\sd_godot\smoke_curr_e2e*` are gitignored scratch (safe to delete).
- Godot binary via `SIGHT_GODOT_EXE` = winget `Godot_v4.6.2-stable_win64.exe`. Headless throughput ~85-90 steps/s aggregate on the 2-env smoke.
- `runs\` is gitignored. Reliability stats via scipy trim_mean + bootstrap (rliable package still uninstallable on this machine, arch dependency).
