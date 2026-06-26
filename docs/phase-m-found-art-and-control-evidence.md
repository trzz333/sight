# Phase M - Found-Art Pivot + Harness Control (Signal Dodge from-scratch RL, take 2)

Verdict, one line: offline value-RL is closed for theory reasons (Kumar et al. 2022), and the SB3 harness is now proven known-good on this machine (CartPole-v1 PPO = 500.0), so the from-scratch wall is localized to the Signal Dodge env/reward wiring, not the algorithm or the install. The new approach is on-policy PPO from scratch with a benchmarked reference recipe and pure survival reward.

## Why the pivot (found-art, verdicts)

The offline-RL thread (Phase L) is not reopened. Kumar, Hong, Singh, Levine, "When Should We Prefer Offline RL Over Behavioral Cloning?" (arXiv 2204.05618, VERIFIED this turn) establishes that offline RL beats BC only with suboptimal data that has stitching structure, sparse rewards, or reward-disambiguated critical points; on near-expert data with no stitching, no algorithm beats BC in the worst case. Signal Dodge has dense +1/step reward and a near-expert good slice, so K7's "CQL loses to filtered-BC" was the theoretically expected outcome, not an execution miss. FOUND-ART on "make offline RL win here": do not build, the literature says it will not.

The from-scratch wall is the real open problem. Prior state-obs PPO (K5.5, on-disk `docs\k5-5-state-observation-control-evidence.md`) collapsed to constant actions on all three seeds, but with four specific deficiencies its own routing named: shaped reward (alpha 0.30 threat_weighted_clearance, satisfiable by a constant action), a 10k-step budget, a single env, and no reference control. K5.5's own conclusion routed the blocker to "reward geometry over budget."

FOUND-ART on "fix the from-scratch wall": MIXED.
- ADOPT a benchmarked reference recipe rather than hand-tuning: SB3 RL-Zoo tuned CartPole-v1 PPO config (DLR-RM/rl-baselines3-zoo `hyperparams/ppo.yml`, VERIFIED), CleanRL (vwxyzjn/cleanrl, JMLR 2022, VERIFIED) as the implementation-details reference.
- ADAPT the recipe to the Godot env (SB3 2.8.0 already installed and already wired to `GodotSignalDodgeEnv` from the K5.6 finetune).
- Do NOT BUILD a new algorithm.

The different approach (not a retry of K5.5): on-policy PPO from scratch with all four K5.5 deficiencies fixed at once. On-policy sidesteps both failure families that have repeatedly bitten this project (the offline can't-beat-BC theorem, and off-policy replay-buffer exploration collapse) and carries an explicit entropy term as a direct anti-collapse lever. Key reward insight: under pure survival reward "none" (+1/surviving step), the best constant action provably caps at 845.7 (`constant_left`, K5.2), which is below the 930.27 bar, so the reward geometry itself forces dodging to clear the bar. K5.5's shaped reward did not have that property.

## M1: harness control (the infra-loop audit)

Before spending any compute on Signal Dodge, ran the exact RL-Zoo CartPole-v1 PPO config on SB3 on StrongerJr to confirm the harness reproduces a published benchmark.

Config (RL-Zoo tuned CartPole-v1): MlpPolicy, n_envs 8, n_timesteps 1e5, n_steps 32, batch_size 256, gae_lambda 0.8, gamma 0.98, n_epochs 20, ent_coef 0.0, learning_rate lin_0.001, clip_range lin_0.2. Tool: `tools\m1_cartpole_ppo_control.py`.

Result (HIGH, `runs\phase_m\m1_cartpole_control_report.json`, exit 0): **mean reward 500.0, std 0.0** over 20 deterministic eval episodes vs the 475.0 solved threshold. **PASS**, a perfect policy. Stack: SB3 2.8.0, gymnasium 1.2.3, Python 3.14.4.

Implication: the SB3 + gymnasium + StrongerJr stack trains PPO to optimum on a CartPole-tier task. Every prior Sight from-scratch RL collapse is therefore NOT a broken install or machine; the fault is in the Signal Dodge env wiring or its reward/budget configuration. This is the controlled anchor that K5.5 lacked.

## Next action (Phase M, M2)

Point the same SB3 PPO trainer at state-obs `GodotSignalDodgeEnv` with reward_shaping "none", a reference-grade budget with vectorized envs, a horizon-appropriate gamma (episodes run to 1800, so gamma 0.98 from CartPole is likely too short; raise toward 0.99+), and a nonzero entropy coefficient as the anti-collapse lever. Requires `SIGHT_GODOT_EXE` set. Multi-seed, eval greedy on held-out seeds 1000-1009 vs bar 930.27. PASS condition mirrors the project standard: mean >= 930.27 AND non-degenerate three-way action use (frac_L >= 0.03, frac_R >= 0.03, max(frac) < 0.97). If Signal Dodge collapses where CartPole solved, the bug is definitively in the env, and the env/reward becomes the diagnosable target.

## Self-audit anchors (this turn)

- Kumar 2022 condition (offline RL vs BC): web_search VERIFIED, arXiv 2204.05618, BAIR blog 2022-04-25, AIhub 2022-05-17.
- RL-Zoo CartPole-v1 PPO config: web_search VERIFIED, DLR-RM/rl-baselines3-zoo `hyperparams/ppo.yml`.
- K5.5 collapse + reward-geometry routing: `docs\k5-5-state-observation-control-evidence.md` (read this turn).
- M1 PASS 500.0: `runs\phase_m\m1_cartpole_control_report.json` + `runs\phase_m\m1_control.log` (run this turn, exit 0).
- Git HEAD before this commit: `3ef1158`.

## found-art

MIXED (above). Searches run this turn: "offline RL fails to beat behavior cloning when", "CleanRL single-file reference PPO DQN reproducible benchmark", "rl-baselines3-zoo tuned hyperparameters ppo.yml CartPole". Reference recipes adopted, no new algorithm built.
