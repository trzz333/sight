# Phase N / C2 (CMA-MAE) — Screen Findings: NEGATIVE

Paradigm 2 of the Phase N from-scratch screen. Method: CMA-MAE quality-diversity
(`ribs` 0.11.0) over the same 5,059-param SB3 MlpPolicy actor (10->64->64->3).
Objective = raw mean episode length (archive supplies diversity, so no shaping).
Measures = 2D action-fraction simplex (frac_left, frac_right). 20x20 GridArchive,
learning_rate 0.01, threshold_min 0.0, ranker "imp" (canonical CMA-MAE,
arXiv 2303.00191). Seeds-per-gen 2, rotated per generation
(`fresh_training_seeds(gen=gen, k)`), 100 generations, 3 seeds. Held-out gate
unchanged from M2/C1: mean episode length >= 930.27 AND frac_left >= 0.03 AND
frac_right >= 0.03 AND max(action_fraction) < 0.97, on eval seeds 1000-1009.

## Verdict: C2 CMA-MAE screen NEGATIVE (all 3 seeds sub-bar, both vectors)

Held-out gen-100 results, each anchored to `c1_eval_summary.json` on disk under
`runs\phase_n\c2_screen_s*\eval_{actor,mean}`. best-actor = `archive.best_elite`
solution; CMA-mean = `emitter._opt.mean` (population center). Confidence HIGH.

| seed | best-actor (elite)      | CMA-mean (emitter center) | verdict |
|------|-------------------------|---------------------------|---------|
| 0    | 523.0  (fracR 0.000)    | 845.7  (fracR 0.007)      | FAIL    |
| 1    | 689.7  (fracR 0.964)    | 586.6  (diverse)          | FAIL    |
| 2    | 845.7  (fracR 0.012)    | 704.7  (diverse)          | FAIL    |

All 6 C2 held-out evaluations are FAIL. Best single number across the whole C2
screen is 845.7 (seed-0 mean and seed-2 actor, tied), missing the 930.27 bar by
~9%. No seed clears the bar; no vector both clears the bar and passes diversity.

The failure splits into the same two modes seen in C1. The best-elite vectors
tend to collapse a lateral action on held-out (seed-0 fracR 0.000, seed-1 fracR
0.964 i.e. near right-collapse with fracL 0.023, seed-2 fracR 0.012): sharp on
the few train seeds they were selected under, degenerate off-distribution. The
CMA-mean vectors mostly keep a diverse action mix (seed-1/seed-2 pass all
diversity sub-gates) but are mediocre in raw competence (586.6, 704.7, and
seed-0's 845.7 which is left-heavy). Diverse but not competent, or competent-
looking on train but collapsed on held-out. Neither reaches the bar.

## Why this closes C2

CMA-MAE was pre-registered (foolproof-design Axis B) specifically to beat the
C1 diagnosis: keep an archive of behaviorally-diverse elites so search cannot
prematurely collapse to a narrow action mix. On TRAIN seeds it did exactly that
(archives grew to 166-202 cells with diverse fracs, train archive_best 1741.5).
The diversity intervention worked as designed and still did not transfer: the
archived diversity does not survive the jump to held-out seeds, and the best
held-out number (845.7) is actually BELOW C1's best (906.4). Adding QD machinery
did not move the held-out wall; it slightly lowered the ceiling. Per the
method-fails-twice rule we do not retry QD harder.

## The real wall (cross-paradigm, C1 + C2)

Two structurally distinct black-box weight searches over this actor (C1 single-
point CMA-ES, C2 archive-based CMA-MAE) fail identically on held-out, and the
train-seed protocol rules out the obvious culprit: seeds ROTATE per generation
(`fresh_training_seeds(gen=gen)`), so this is not fixed-seed overfitting. The
gap survives rotating training seeds. What both paradigms share is the decisive
mechanism: they select a "best" candidate by return measured on only ~2 seeds
per generation, in an environment with high between-seed variance (held-out
per-seed lengths span 183 to 1800 for a single fixed policy). Return-only
selection on a tiny noisy sample rewards seed-luck, not seed-robust competence,
and seed-luck does not generalize.

This is consistent with the single most reproducible finding in the whole
project (phase-l-offline-rl-findings, anchored to k5_8_reliability_report.json):
the decisive variable is the quality of the ACTION SIGNAL the learner is fed
(good demonstrations, or filtered good trajectories), not the optimizer or its
exploration knobs. Every method that gets a good action signal (BC 1737.3,
PPO-finetune-from-BC 1710.5) clears the bar reliably; every from-scratch
return-only method (PPO Phase M, QR-DQN/NoisyNet K5.8 at IQM 729.2 with 1/10
seeds above bar, offline value-RL Phase L, C1 CMA-ES, C2 CMA-MAE) does not.
NoisyNet's one lucky seed at 980.7 is the exception that proves it: from-scratch
return-only can occasionally stumble into competence, never dependably.

## Next paradigm: C3 = self-imitation from filtered self-play (from-scratch)

Decision (mine; technical/tactics call): run one C3 shot, then FINAL NEGATIVE
if it fails. One-line reason: the project's only reproducible positive
(imitation / filtered-good-signal clears; return-only from-scratch does not) has
never been run as a from-scratch SELF-imitation paradigm, so it is the highest-
EV, structurally-distinct third shot, and it is the direct attack on the
diagnosed wall (action-signal quality / generalization) rather than a third
weight-search.

found-art verdict ADAPT. Search named: "self-imitation learning Oh 2018
arXiv 1806.05635", "filtered behavior cloning self-generated trajectories",
"best-of-N self-distillation RL". Mechanism: the agent rolls out across many
rotated seeds, KEEPS only its own top-return trajectories (a growing self-
generated good-trajectory buffer, no external/human/scripted demos), and
imitates them, iterating. The competence is discovered from reward (return
filtering) and amplified by imitation, so it is from-scratch RL (SIL is a
published RL method), NOT BC-on-external-demos. This is why a C3 clear would
count as mission success where BC did not: no known-good controller is ever
handed to the agent.

Why ADAPT not BUILD: the repo already has a working BC trainer, the Godot
rollout/worker pool, and the unchanged held-out gate. C3 reuses all three and
adds only a self-filtering outer loop (roll -> filter top-k by return ->
imitate -> repeat). Structurally distinct from C1/C2 (not weight search) and
from Phase M/K5.8/Phase L (not vanilla policy-gradient or value-RL); it is the
one untried family the evidence actually points at.

Why C3 could still fail (honest, confidence LOW-to-MEDIUM it clears): SIL needs
occasional good episodes to bootstrap from. The env does produce lucky long
episodes (1800s appear even under failed policies), so there is signal to
filter, but if early self-play never clears the noise floor there is nothing to
imitate and C3 collapses to the same wall. One honest shot, verify hard against
the reliability gate before any PASS claim.

## Paradigm accounting

Phase N stopping rule unchanged: 3 structurally distinct paradigms, one honest
shot each, then FINAL NEGATIVE if none clear. C1 (CMA-ES) spent NEGATIVE. C2
(CMA-MAE) spent NEGATIVE. One shot remains: C3 = self-imitation from filtered
self-play. If C3 fails its reliability gate, Phase N closes FINAL NEGATIVE with
a clean, documented result: on this fully-owned env, from-scratch return-only
learning (weight-search or value/policy-gradient) does not reliably clear the
constant-action baseline, while imitation of any good action signal does.
