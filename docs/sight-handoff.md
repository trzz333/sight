# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Godot eval-of-record, curriculum-injection port. Discount-only port did NOT transfer; building the start-state curriculum injection seam so the full proven recipe runs on real Godot. Bar 930.27.

**Last commit:** `02a6b0d` sd-godot: build + smoke-validate discount-first Godot g99 trainer, launch 1M probe

**Current task:** Discount-only Godot port result is in and NEGATIVE. `g99_godot_1M_s0` (gamma 0.99, no curriculum, 1M steps, 8 envs headless) held-out greedy eval (seeds 5000-5029): mean_len 491.5, IQM 476.0 vs the M2.1 controlled contrast (gamma 0.999 / 1M / Godot -> IQM 418) and bar 930.27; beats_bar false; explained_variance -2.03 (critic worse than predicting the mean). Discount alone lifts the IQM ~14% (418 -> 476) but does not approach the bar and the critic never learns a usable value function. The recipe that cleared the replica 5/5 was curriculum + gamma 0.99; the Godot port dropped the curriculum (no injection seam) and kept only the discount. Two Godot from-scratch-no-curriculum attempts now both fail (0.999 -> 418, 0.99 -> 476), so per the contract the 5M discount-only run is NOT launched (it retries a twice-failed, curriculum-omitting recipe with 5x budget). The Python-only shortcut (seed-selection curriculum) was tested and RULED OUT: probe shows `active_hazard_count_above_player` = 0 for all 50 seeds across the first 10 frames, so clean Godot resets have no early above-player density to select for. `godot_env.reset(options=)` exists at the gym layer but is not plumbed to the transport, and Godot has no injection handler. All numbers confirmed against `runs\sd_godot\g99_godot_1M_s0_summary.json` this session (scipy trim_mean for IQM). Confidence HIGH.

**Next action:** Build the Godot start-state curriculum injection seam so the full proven recipe (curriculum + gamma 0.99) runs on Godot. (1) GDScript in `games\signal-dodge`: at reset, read a curriculum hazard count from the reset message and pre-spawn N hazards above the player (headroom ~100px, no reset collision, no insta-death), mirroring the replica `CurriculumSDF`; keep the N=0 clean-start path byte-identical so eval is untouched. (2) Protocol: carry `curriculum_n_init` through the H3 reset wire message. (3) Python: plumb `godot_env.reset(options={...})` -> `transport.reset` -> wire, add an `AnnealCurriculum` callback (n_init_max 6 -> 0 over anneal_frac 0.7). (4) Smoke-validate like the replica: clean-start obs byte-identical, N=6 injects 6 hazards with no reset collision, live count mutation reflected at reset, short train+eval end to end. THEN launch the Godot curriculum arm (1M contrast first, then 5M + 5-seed rliable, gate 5/5 AND IQM CI lower bound > 930.27). Eval stays greedy held-out 5000-5029 vs 930.27 on the clean env.

**Blockers:** None requiring Jeff now. One conditional Jeff scope call, only if the curriculum-injection port ALSO fails to clear on Godot: whether to accept imitation (BC 1737.3, PPO-ft 1710.5) as the standing solution or keep pursuing from-scratch reliability.

**Notes:**

- Result anchor: `runs\sd_godot\g99_godot_1M_s0_summary.json` (mean 491.5, IQM 476.0, EV -2.03, beats_bar false). `runs\` gitignored. Full record and reasoning appended to `docs\sd-fast-curriculum-findings.md`.
- Seed-curriculum feasibility probe `runs\sd_godot\_probe_seedcurr.py`: 0/50 seeds show any above-player hazard in the first 10 frames. HIGH that the GDScript injection seam is required; seed-selection cannot supply the 2-6 density the replica curriculum injected.
- Godot env: `reset(options=)` accepted at the gym layer (`godot_env.py` ~L328) but NOT plumbed to the transport; no Godot-side injection handler. GDScript + protocol work stands. Godot binary via `SIGHT_GODOT_EXE` (winget Godot_v4.6.2 path).
- `tools\sd_godot_ppo_g99.py` is the durable discount-only Godot trainer (keep). The curriculum arm needs a curriculum-aware variant or a `--curriculum` flag added to it plus the `AnnealCurriculum` callback.
- Monitor left running this session (`http://127.0.0.1:8791/monitor.html`, server pid 36524, launcher `runs\sd_godot\_serve.py`); harmless, gitignored, kill when done.
