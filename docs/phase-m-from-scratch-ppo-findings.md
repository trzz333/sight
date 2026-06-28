# Phase M findings: from-scratch on-policy PPO on Signal Dodge

Status: M2 and M2.1 both FINAL NEGATIVE. From-scratch on-policy PPO does not
clear the constant-action baseline (mean episode length 930.27) on Signal
Dodge, with or without return normalization. Imitation clears it reliably
(BC 1737.3, PPO-finetune-from-BC 1710.5). The unsolved problem remains
from-scratch RL reliability, not imitation.

## Setup (both experiments)
- Env: Godot Signal Dodge, state observations, reward "none" (pure survival;
  best constant-action argmax 845.7 < 930.27 bar, so the reward argmax is not
  a constant action).
- Algo: SB3 PPO, MlpPolicy, gamma 0.999, ent_coef 0.01, n_envs 8
  (SubprocVecEnv), 1e6 steps/seed, 3 seeds (0/1/2).
- Eval gate: mean episode length over held-out seeds 1000-1009 >= 930.27, plus
  action-diversity gate (frac_L >= 0.03, frac_R >= 0.03, max(frac) < 0.97).
- Control: M1 SB3 PPO on CartPole-v1 (RL-Zoo config) scored mean 500.0 std 0.0,
  localizing all prior collapses to the Signal Dodge env/reward, not the algo.

## M2: vanilla PPO (no normalization)
- Result: 0/3, eval means 472 / 663 / 579, all below bar. FINAL NEGATIVE.
- Named defect: explained_variance ~= 0, value_loss ~311. The critic could not
  fit the large-magnitude high-gamma dense returns. A value-fitting failure.

## M2.1: PPO + VecNormalize(norm_obs, norm_reward, gamma=0.999)
- found-art verdict: ADOPT. VecNormalize is the SB3-standard return-scaling fix
  for explained_variance ~= 0 under high-gamma large-magnitude dense rewards.
- Critic defect fixed, confirmed in full 1e6-step runs:
  explained_variance 0.933 / 0.939 / 0.854 (was ~0), value_loss ~0.02 (was ~311).
- Result: 0/3, eval means (held-out seeds 1000-1009): 417.3 / 696.8 / 778.1, all
  below bar. FINAL NEGATIVE. Runs clean (learn_error null on all three).
- Action distributions diverse, not collapsed (e.g. s2 L 0.35 / stay 0.34 /
  R 0.31). The policy actively chooses; it does not pin to a constant action.
- Per-seed eval lengths show the policy CAN survive full episodes occasionally
  (s2: three of ten eval seeds reached the 1800-step cap) but not reliably; the
  per-seed mean stays below bar on every train seed.

## Conclusion
Fixing the diagnosed value-fitting defect (M2 -> M2.1, explained_variance 0 ->
~0.92) did NOT lift the policy above the constant-action baseline. The failure
mode moved from "critic cannot fit returns" to "policy converges to a
high-variance sub-baseline behavior." From-scratch on-policy PPO has now failed
twice on this env in two distinct ways. Per the operating contract the method is
not retried harder. Honest portfolio finding: from-scratch on-policy RL does not
clear the Signal Dodge baseline, while imitation (BC, PPO-finetune from BC)
clears it reliably and by a wide margin.

## Artifacts
- Eval summary: runs\phase_m\m2_1_eval3\m2_eval_summary.json (gitignored, local).
- Train reports: runs\phase_m\m2_1_s{0,1,2}\m2_train_report.json (gitignored).
- Tools: tools\m2_state_ppo_train.py, tools\m2_state_ppo_eval.py,
  tools\run_m2_1_multiseed.bat (commit b19cf06 carries NLDC migration paths).

## Aggregate statistic: IQM + 95% stratified-bootstrap CI (Agarwal et al. 2021)

Finalizes the M2.1 record independent of the direction call. Computed over the
30 held-out eval episode lengths (3 train seeds x 10 eval seeds 1000-1009).

- found-art verdict: ADAPT. rliable (Agarwal et al. 2021) is the canonical impl
  but is not installed on the Py3.14 global interp and pins older numpy; rather
  than perturb the load-bearing M-phase env, the two definitions were reproduced
  exactly: IQM = scipy.stats.trim_mean(x, 0.25) (identical to
  rliable.metrics.aggregate_iqm) and a stratified-by-seed percentile bootstrap
  (resample episodes with replacement within each seed, pool, recompute IQM;
  50000 reps, rng seed 0). IQM primitive unit-checked (trim_mean(1..100,0.25)==50.5).

- Survival bar: 930.27
- Per-seed mean lengths: 417.3 / 696.8 / 778.1
- Grand mean (30 eps): 630.73
- IQM: 418.25
- 95% CI (stratified bootstrap, percentile): [314.44, 670.50]
- P(bootstrap IQM >= bar): 0.0003

Read: the IQM and its entire 95% CI fall below the 930.27 bar; the upper bound
(670.50) does not reach it, and essentially no bootstrap resample clears it
(0.03%). IQM (418) sits well below the grand mean (631) because the right tail of
1800-step cap survivals inflates the mean; trimming those tails is exactly the
robustness IQM provides, and the gap quantifies the "high-variance sub-baseline"
failure mode. M2.1 is FINAL NEGATIVE at the aggregate level, not only at the
single-seed point estimates. Confidence HIGH (hand-verified IQM, unit-checked
primitive, reproducible committed script; result JSON local).

- Tool: tools\m2_1_iqm_ci.py (committed; reproduces these numbers from the eval JSON).
- Result: runs\phase_m\m2_1_eval3\iqm_ci_result.json (gitignored, local; numbers above are the record).
