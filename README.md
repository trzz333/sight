# Sight

Sight is a local-first reinforcement-learning lab: a hobby and portfolio project for learning modern RL by training small policies, from scratch, on a single older gaming laptop. It is not a product, a startup, or a QA tool. Success is defined by learning progress and reproducible local training, not by users, revenue, or downstream commercial intent.

The environments are **Signal Dodge**, a custom Godot micro-game owned by this repo and exposed as a Gym-style environment (policies are scored against a constant-action baseline of mean episode length 930.27; because that baseline is itself a fixed action, anything that clears it has to actually dodge), and **ViZDoom** `defend_the_center`, an approved second target for pixel-based RL. Gymnasium classic-control environments are used alongside them for formal grounding.

**Non-goals, stated up front and permanently.** Sight does not target live commercial games. It does not touch Freecash, offerwalls, or any paid-engagement platform. It does not develop bot-detection evasion, account farming, or identity spoofing. It does not produce assets or tooling that could drop into a live-service cheat pipeline. These are not current priorities or performance constraints; they are permanent project boundaries. PRs that cross them are closed without review. See `docs/ethics.md` and `CONTRIBUTING.md`.

## Results

| Environment | Method | Score | Verdict |
|---|---|---|---|
| ViZDoom defend_the_center (pixels) | PPO CnnPolicy, gamma 0.99, 2.25M steps, RTX 4080 Laptop | mean 12.2 / IQM 12.8 kills per episode, 30 deterministic eps (untrained ~0) | strong; 30s clip at `runs/vzd/ppo_defend/gameplay.mp4` |
| Signal Dodge (state) | Behavioral cloning from oracle demos | mean episode length 1737.3 vs bar 930.27 | clears reliably |
| Signal Dodge (state) | PPO fine-tune from BC init | mean 1710.5 vs bar 930.27 (10 seeds) | clears reliably |
| Signal Dodge fast replica | From-scratch PPO, gamma 0.99 + start-state curriculum, 5M steps | IQM 1800 (episode cap) on 5/5 seeds | clears, after two structural fixes |
| Signal Dodge (real Godot) | Same from-scratch recipe, 1M / 5M steps | mean 635 / 574 vs bar 930.27 | does not clear; 5x compute did not help |

**The load-bearing finding: discount factor, not exploration.** Every from-scratch failure on Signal Dodge traced to one cause. At gamma 0.999 (effective horizon ~1000 steps) the PPO critic's return targets were too high-variance to fit: explained variance sat near zero and no amount of exploration tuning or capacity helped. Cutting gamma to 0.99 (horizon ~100) fixed the critic outright (explained variance 0.94+), and a start-state curriculum (inject hazards above the player at reset, annealed to zero over 70% of training) was the second required ingredient. The same gamma-0.99 lesson transferred directly to ViZDoom, where the critic was healthy from the first checkpoint.

**The honest negative result.** More compute is not the lever. A 5M-step probe on real Godot (5x the 1M budget) *reshuffled which evaluation start-states the policy survives* rather than accumulating capability: per-eval-seed correlation between the 1M and 5M checkpoints was near zero (Spearman -0.19), the median episode length fell 573 to 393, and the mean fell 635 to 574. Imitation learning clears the bar reliably where from-scratch does not, and is reported as the standing solution for the original env. Results are reported as across-seed distributions with interquartile means (Agarwal et al. 2021), never single lucky seeds.

**Methods exercised.** Value-based RL (DQN), distributional RL (QR-DQN), NoisyNet exploration, imitation learning (BC), policy-gradient fine-tuning (PPO), evolutionary strategies (CMA-ES, CMA-MAE), offline RL, from-scratch PPO from pixels (CNN), and a self-supervised next-state-prediction track. Logging is structured NDJSON with deterministic seeds; evaluation tracks reward, episode length, action distribution, and a failure taxonomy. Full run stories live in `docs/` (start with `docs/vzd-ppo-teacher-findings.md` and `docs/sd-fast-curriculum-findings.md`).

The project is solo and single-voice: one author makes the engineering calls, with an evidence-anchored self-audit (verify every load-bearing claim against on-disk artifacts and re-run the evals) standing in for external review. See `docs/sight-charter.md` for scope, phase gates, success criteria, and decision authority, and `docs/sight-handoff.md` for live status.
