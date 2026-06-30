# Phase N / C1 (CMA-ES) — Screen Findings: NEGATIVE

Paradigm 1 of the Phase N from-scratch screen. Method: separable CMA-ES
(`pycma` 4.4.4) over the 5,059-param SB3 MlpPolicy actor (10->64->64->3),
shaped diversity fitness, seeds-per-gen 4, 100 generations, held-out gate
on eval seeds 1000-1009. Gate: mean episode length >= 930.27 AND frac_left
>= 0.03 AND frac_right >= 0.03 AND max(action_fraction) < 0.97.

## Verdict: C1 ES screen NEGATIVE (all 3 seeds sub-bar, both vectors)

Held-out gen-100 results, anchored to `c1_eval_summary.json` on disk
(`runs\phase_n\c1_screen_s*\eval_*`):

| seed | CMA-mean (xfavorite) | best-actor (sampled) | verdict |
|------|----------------------|----------------------|---------|
| 0    | 906.4                | 845.4                | FAIL    |
| 1    | 591.0                | 707.7                | FAIL    |
| 2    | 564.0                | 663.0                | FAIL    |

Every one of the 12 C1 eval summaries (3 seeds x {mean, actor} x checkpoints)
is C1-FAIL. Best single number across the whole screen is seed 0's gen-100
CMA-mean at 906.4, a near-miss that did NOT reproduce: seed 1 and seed 2 means
landed at 591.0 and 564.0. High between-seed variance, no seed clears 930.27.

Seed 0 within-run trajectory was non-monotonic (mean-vec g5 845.7, g18 519.0,
g33 729.9, g100 906.4): the objective is flat/noisy enough that CMA wanders
rather than climbing, consistent with the diagnosed premature behavioral
convergence (early gens collapsed toward a narrow action mix).

Note: seed 2's gen-100 vectors did NOT collapse (fracs L .65 / S .17 / R .18
for the mean; L .79 / S .04 / R .18 for the actor, all diversity sub-gates
passed). The FAIL is purely the mean being sub-bar, not single-action collapse.
Diversity shaping worked; raw competence did not clear the bar.

## Why this closes C1

Separable CMA-ES as a pure-optimization shot over the actor params cannot
reliably clear the survival bar on Signal Dodge. The decisive lever in this
project has repeatedly been exploration, not optimizer tuning (NoisyNet was
the only from-scratch method to ever clear the bar; PPO and offline value-RL
both closed NEGATIVE). Retrying CMA-ES harder is the same method; per the
method-fails-twice rule we change the method.

## Next paradigm: C2 = pyribs CMA-MAE (quality-diversity)

found-art verdict ADAPT (search: "pyribs CMA-MAE quality-diversity",
arXiv 2303.00191, PyPI `ribs`, icaros-usc/pyribs; pre-registered in
`docs\phase-n-foolproof-design.md` Axis B). CMA-MAE keeps an archive of
behaviorally-diverse elites and anneals exploration->exploitation, built to
be robust to flat objectives, i.e. it attacks the diagnosed failure directly.
It reuses the same pycma ask/tell interface, so the rollout/worker/eval/gate
infra is unchanged; the only new design choice is a 1-2 dim behavior
descriptor (action-fraction simplex or trajectory mean x).

Paradigm accounting: CMA-MAE is counted as C2, a structurally distinct
paradigm (archive-based illumination, not single-point optimization), not a
re-run of C1. ADOPT/ADAPT (reuse the library and infra) and "counts as a
distinct paradigm shot" are separate axes; both hold here. Phase N stopping
rule unchanged: 3 distinct paradigms, one honest shot each, then FINAL
NEGATIVE if none clear. C1 spent. Two shots remain (C2 = CMA-MAE, C3 TBD).

## Infra finding (carried, not part of the verdict)

Windowless durable overnight execution is PROVEN this session. Seed 2 ran
gen 8 -> 100 (rc=0, sentinel EXIT 0) over ~8 h with no visible window and
survived active shell churn. Pattern: supervisor launched DETACHED_PROCESS
(pool-less, immune to console Ctrl/CLOSE), trainer spawned CREATE_NO_WINDOW
(hidden console gives the 8-Godot pool valid handles), kill_godot + status
poll also CREATE_NO_WINDOW, plus FOR_DISABLE_CONSOLE_CTRL_HANDLER=1. This
reconciles `phase-n-foolproof-design.md` Axis A: DETACHED_PROCESS is harmful
only for the pool-bearing trainer, not the pool-less supervisor. NSSM is now
optional (reboot-survival only) rather than required for the no-reboot case.
The prior crash loop was console-control kills (exit 0xC000013A =
STATUS_CONTROL_C_EXIT), not reboots.
