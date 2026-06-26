# Phase L - Offline RL Findings (Signal Dodge)

Verdict, one line: on the full mixed-quality Signal Dodge dataset, value-based offline RL (DiscreteCQL) collapses to the single modal action and fails the baseline (mean 606.0), while imitation on filtered good data (filtered-BC) clears it (mean 1299.7); the working lever is data quality, not the CQL conservative coefficient. Phase L value-RL thread is CLOSED.

This doc is the claimable synthesis of the offline-RL pivot. Every number below is pulled from an on-disk artifact named at the point of use and re-read during the session that wrote this doc.

## The bar and the baselines

- Above-baseline bar: **930.27** (mean episode length of the constant-action baseline; a policy that clears it has to actually dodge, not stand still). Source: every report's `bar` field and `README.md`.
- Best fixed-action policy: **845.7** (`constant_left`, K5.2). Source: every report's `best_constant` field. A method that scores below 845.7 is beaten by the best single hard-coded action.
- Episode-length cap (timeout): 1800 steps. All evals are greedy/argmax in-env on held-out seeds 1000-1009, disjoint from any training seeds.

## Results across every method tried (anchored)

| Method | Family | Score (mean ep len) | vs bar 930.27 | Verdict | Source artifact |
|--------|--------|--------------------:|--------------:|---------|-----------------|
| DiscreteCQL (K7 / K7.1) | offline value-RL | 606.0 | -324.27 | FAIL | `runs\phase_k\k7_offline\real\k7_eval_report.json` |
| filtered-BC (K7) | offline imitation | 1299.7 | +369.43 | PASS | `runs\phase_k\k7_offline\real\k7_eval_report.json` |
| NoisyNet QR-DQN (K5.8, N=10 reliability) | from-scratch distributional RL | IQM 729.2 | below | FAIL (reliability) | `runs\phase_k\k5_8_reliability_report.json` |
| Self-supervised aux head OFF (K6, N=5) | from-scratch + world-model probe | IQM 660.4 | below | FAIL | `runs\phase_k\k6_off_reliability_report.json` |
| Self-supervised aux head ON (K6, N=5) | from-scratch + world-model probe | IQM 626.1 | below | FAIL | `runs\phase_k\k6_on_reliability_report.json` |
| BC (K5.6) | imitation | 1737.3 | +807.03 | PASS | `docs\k5-6-bc-evidence.md` |
| PPO-finetune from BC (K5.6) | imitation warm-start + RL | 1710.5 | +780.23 | PASS | `docs\k5-6-ppo-finetune-evidence.md` |

Only imitation-derived policies clear the bar. Every value-based-RL approach (from-scratch or offline) fails it.

## The offline-RL result in detail (K7 / K7.1)

Dataset: 80,470 transitions, 98 episodes, mixed-quality with a genuine competence gradient (random + 8 QR-DQN checkpoint stages + BC), returns spanning roughly 182 to 1800. The two offline learners were trained on the **same** npz (no recollection between runs) for 100k steps each, then evaluated greedily in-env on seeds 1000-1009.

### DiscreteCQL: stay-only collapse (FAIL)

Mean episode length **606.0**, collision on all 10 seeds (collision_rate 1.0, timeout_rate 0.0), pooled action fractions **[0.0, 1.0, 0.0]** (Left/Stay/Right). The policy picks Stay on every step of every seed. 606.0 is below the bar (930.27) and below even the best fixed action (845.7): a value-RL method that collapsed to one action scores worse than the best hand-picked single action. Source: `k7_eval_report.json` `cql` block, per-seed `action_counts_LSR` all `[0, n, 0]`.

Per-seed lengths (steps, all collisions): 333, 273, 843, 963, 1203, 1263, 543, 183, 183, 273. Sum 6060 / 10 = 606.0.

### filtered-BC: real three-way dodging (PASS)

Mean episode length **1299.7**, collision_rate 0.5, timeout_rate 0.5 (5/10 reach the 1800 cap), pooled action fractions **[0.212, 0.307, 0.481]** - a genuinely moving policy. Trained on the top-quality slice of the same dataset. Source: `k7_eval_report.json` `filtered_bc` block.

Per-seed lengths: 1800, 465, 1800, 573, 753, 1800, 1800, 583, 1623, 1800. Sum 12997 / 10 = 1299.7.

### The retry that closed the thread (K7.1)

The K7.1 retry changed the method rather than retrying it harder: DiscreteCQL `n_critics` 1 -> 3 and conservative `alpha` 1.0 -> 0.5, on the same dataset. Result: identical 606.0 stay-only collapse to the K7 default run (606.0 to the decimal). The found-art hypothesis (that the conservative coefficient drives the collapse) was therefore **falsified**: lowering alpha moved the collapse by zero. Stop condition met after two value-RL attempts with the same failure mode; no third retry. Cross-run equality of the two CQL runs is documented in `docs\sight-handoff.md` (commit `7f1d21d`); the current committed `k7_eval_report.json` carries the 606.0 result. Confidence: HIGH on the current 606.0 (re-read from file); MEDIUM on the exact-decimal equality of the prior K7 default run (handoff-attested, prior report since overwritten).

## Why DiscreteCQL collapsed (mechanism, not tuning artifact)

The collapse is structural. Discrete-CQL's conservative penalty pushes Q toward a behavior-cloning NLL on the dataset's action distribution; on a Stay-dominant mixed dataset, that target is Stay. The penalty and the modal-action bias point the same way, so the greedy policy degenerates to the modal action regardless of the conservative coefficient in the tested range. That is why alpha tuning produced zero movement. The lever that works is upstream of the algorithm: filter the dataset to high-return trajectories so the imitation target is competent play, which is exactly what filtered-BC does and why it passes on the same raw data. Confidence: MEDIUM (consistent with both CQL runs landing identically and with filtered-BC passing on the same npz; not independently ablated beyond the two alpha settings).

## From-scratch and imitation history (context for the claim)

From-scratch value/distributional RL never reached reliable above-bar play:

- NoisyNet QR-DQN (K5.8) was the first from-scratch policy to clear the bar **on a single seed** (best seed 980.7, PASS), but across N=10 seeds at a 200k-step budget the reliability is IQM 729.2 (CI95 [607.9, 835.8]), with only 1/10 seeds above the bar and half the seeds degenerate. So: a lucky seed can dodge, but from-scratch is not dependable. Source: `k5_8_reliability_report.json`.
- Adding a self-supervised next-state prediction head (K6) did not lift reliability: OFF arm IQM 660.4, ON arm IQM 626.1, both below bar, the auxiliary head if anything slightly worse. FINAL NEGATIVE. Source: `k6_off_reliability_report.json`, `k6_on_reliability_report.json`.

Imitation cleared the bar reliably from the start: BC 1737.3, PPO-finetune-from-BC 1710.5. Competence in this project came from behavioral cloning of a hand-built oracle, not from value-based RL learning to dodge on its own.

## The claimable finding

On a CartPole-tier env where a competent policy provably exists (BC reaches 1737.3 through the same env path), value-based reinforcement learning - whether online from-scratch (QR-DQN, NoisyNet, self-supervised aux head) or offline conservative (DiscreteCQL) - consistently fails to produce reliable above-baseline play, collapsing toward a single action. Imitation learning on competent demonstrations clears the baseline every time. The decisive variable across the whole sweep is the quality of the action signal the learner is fed (good demonstrations, or filtered good trajectories), not the RL algorithm's own machinery or its conservative/exploration knobs. This is a documented, reproducible negative result on value-based RL plus a positive result on imitation, on a fully owned env with deterministic seeds.

## Threats to validity / honest caveats

- The bar (930.27) and best_constant (845.7) differ; 930.27 is the pass threshold used uniformly across all reports, 845.7 is the best measured single fixed action. Both are reported as-is; the gap is not re-derived here.
- The CQL mechanism explanation is supported by two alpha settings landing identically and by filtered-BC passing on the same data; it is not a full ablation across `n_critics`, network size, or dataset-balance interventions. Stated at MEDIUM.
- filtered-BC (1299.7) is imitation on the top-25% slice; it is below pure BC's 1737.3 and is not a value-RL win. The "offline RL works" reading would be wrong: what works offline here is imitation, same as online.
- All evals are greedy/argmax, 10 fixed held-out seeds. Single-seed peaks (e.g. K5.8's 980.7) are reported as such and never as the headline; the headline is always the across-seed distribution.
- runs/ is gitignored per project pattern; the cited run artifacts live on disk and were re-read this session. Tracked evidence is this doc plus the K5.6 docs.

## Self-audit anchors (read this session)

- K7/K7.1 CQL 606.0 + filtered-BC 1299.7, per-seed tables and action fractions: `runs\phase_k\k7_offline\real\k7_eval_report.json`. Arithmetic re-derived (6060/10, 12997/10).
- K5.8 reliability IQM 729.2, best seed 980.7, frac_above_bar 0.1: `runs\phase_k\k5_8_reliability_report.json`.
- K6 OFF IQM 660.4, ON IQM 626.1: `runs\phase_k\k6_off_reliability_report.json`, `runs\phase_k\k6_on_reliability_report.json`.
- BC 1737.3 / PPO-ft 1710.5: `docs\k5-6-bc-evidence.md`, `docs\k5-6-ppo-finetune-evidence.md`.
- Git HEAD at writing: `fb61710` (clean, in sync with origin/main).

## found-art

ADOPT. Problem generalized: document a negative ML result plus a positive comparator. Adopted the project's existing `-evidence.md` structure (claim / bar / results table / mechanism / threats / self-audit anchors) used by `k5-6-bc-evidence.md` and `k5-6-ppo-finetune-evidence.md`; no new format built. Reliability methodology (IQM + bootstrap CI + performance profile) is itself an ADOPT of Agarwal et al. 2021 (rliable), reimplemented on numpy/scipy as recorded in the report `method` fields.
