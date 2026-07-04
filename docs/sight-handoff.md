# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Mission env (Signal Dodge, 930.27 bar) open. Budget-at-speed isolation run on the fast replica; MinAtar adoption banked (3/3).

**Last commit:** `f9c7a37` sd-fast M2.1-recipe budget isolation: 1-of-2 seeds clears (seed variance, not budget)

**Current task:** The clean budget isolation ran: `tools\sd_fast_ppo.py` extended to M2.1's exact recipe (gamma 0.999, VecNormalize norm_obs+norm_reward clip 10/10, n_steps 512, batch 512, n_epochs 10, lr 3e-4, ent_coef 0.01, gae 0.95, 8 envs, MlpPolicy [64,64]), reward none, 5M steps, greedy eval on held-out replica seeds 5000-5029 with obs normalized through saved VecNormalize stats. Seed 0 CLEARS: mean 1119.4, median 1059, IQM 1148.7, diverse actions L/S/R 0.37/0.22/0.41, 30% of episodes at the 1800 cap. All three central metrics beat 930.27; first from-scratch clear of Signal Dodge in project history (replica). Seed 1 FAILS: mean 598, near constant-left (0.86 L), below best-constant 746, critic healthy (EV 0.885). Seed 2 in flight at handoff (~0.5M/5M, chain PID 9532). Verdict MEDIUM: 5M budget makes a from-scratch clear reachable but NOT reliable; the wall is exploration/basin-luck, not critic capacity (EV healthy in both clear and failure). Separately this session, fixed CREATE_NO_WINDOW on every child subprocess spawn (commit `0334d67`) so detached runs stop flashing terminal windows.

**Next action:** Collect the seed-2 summary (`runs\sd_fast\sd_fast_m21_s2_5M_summary.json`; sentinel `runs\sd_fast\m21_confirm.sentinel`) and record the 3-seed IQM spread in `docs\sd-fast-replica-budget-findings.md`. Then move the lever off budget to exploration pressure against the constant-left basin: found-art first (RND Burda 2018, NoisyNets Fortunato 2017; check in-repo `src\sight_agent\rl\noisy_qrdqn.py` and `dyn_qrdqn.py` before building), start with a tightened diversity gate plus restart-on-collapse or scheduled/higher ent_coef. Do NOT launch the Godot 5M eval-of-record until the replica clears reproducibly across seeds.

**Blockers:** None blocking the next action. One Jeff-owned scope call is open (not urgent): whether to keep pursuing from-scratch reliability or accept imitation as the standing mission solution, since BC (1737) and PPO-finetune (1710) already clear reliably and 5x budget did not make from-scratch reliable.

**Notes:**

- Window fix `0334d67`: CREATE_NO_WINDOW on the Godot process factory, the ndjson_logger git stamp, and the git/tasklist helper spawns; detached runs should no longer flash consoles. Confirm on the next detached run.
- Eval correctness: `sd_fast_ppo.py` eval now normalizes held-out obs through the saved VecNormalize stats (training=False). The prior eval predicted on raw obs, which misreads any VecNormalize-trained policy.
- Critic is healthy in BOTH the clear and the failure (EV 0.85-0.89), so the wall is exploration/basin choice, not credit assignment. The diversity gate max(frac)<0.97 is too loose: it passed seed 1's 0.86-left near-constant policy.
- Imitation remains the reliable solution (BC 1737.3, PPO-finetune 1710.5); from-scratch reliability is the open problem, unchanged by 5x budget.
- Confirmation chain launched PID 9532 (seeds 1 then 2, sequential to avoid CPU contention). runs\ is gitignored; summaries live on disk only.
