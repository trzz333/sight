# MinAtar ADOPT spike - first from-scratch clear (SB3 PPO, Breakout)

Date: 2026-07-03. Phase: post-Phase-N, first spike on the Jeff-approved next
target environment (MinAtar). Single-voice execution.

## Verdict

ADOPT mainline `minatar` 1.0.15 (PyPI). SB3 PPO with a small MinAtar CNN learns
MinAtar/Breakout-v1 from scratch and clears the published-scale baseline (~9.4)
on held-out seeds. This is the first from-scratch clear in the project's history.

## found-art

Search named (found-art doc `next-target-env-found-art.md`, prior session):
MinAtar compute-limited from-scratch baselines; SB3-compatible MinAtar fork.
- Fork `rlai-lab/MinAtar-Faster` was named but NOT needed: mainline `minatar`
  1.0.15 already targets gymnasium (modern reset/step, registers
  `MinAtar/<Game>-v0` full action set and `-v1` minimal action set). ADOPT
  mainline, fewer moving parts; the fork's only edge is raw speed, irrelevant
  to a first yes/no.
- Net architecture: ADAPT, not BUILD. Young & Tian (2019) small conv net (one
  3x3x16 conv + ReLU + FC128), reproduced in qlan3/gym-games, wrapped as an SB3
  `BaseFeaturesExtractor`. Needed because obs is a 10x10xC grid, below SB3
  NatureCNN's 36px assertion.

## Setup (all on-disk, reproducible)

- Env: `MinAtar/Breakout-v1` (minimal action set, `Discrete(3)`), obs (10,10,4)
  bool -> wrapped to (4,10,10) float32 channel-first.
- Algo: SB3 2.8.0 PPO, MlpPolicy + custom MinAtarCNN extractor (features_dim
  128, pi/vf heads empty), n_envs=8, n_steps=128, batch 256, n_epochs 4,
  gamma .99, gae_lambda .95, clip .2, ent_coef .01, vf_coef .5, lr 2.5e-4.
- Steps: 5,000,000. CPU torch 2.11.0, Python 3.14.6, `.venv-c1`.
- Code: `src/sight_agent/rl/minatar.py` (env layer + extractor),
  `tools/minatar_ppo_spike.py` (trainer + held-out eval),
  `tools/minatar_sanity.py` (random-floor probe).
- Curves/models/summaries under `runs/minatar/` (gitignored).

## Results (evidence: `runs/minatar/*_summary.json`)

- Random-policy floor: return mean 0.333 (30 eps, seed-varied). The honest floor.
- Held-out eval = deterministic policy over seeds 1000-1029, disjoint from the
  seed-0..2 training envs.

| seed | held-out mean | std | clears 9.4 | train s | steps/s |
|------|---------------|-----|-----------|---------|---------|
| 0    | 11.5          | 4.08 | yes      | 863.7   | 5789    |
| 1    | 14.7          | 4.49 | yes      | 828.3   | ~6000   |
| 2    | in flight     | -    | -        | -       | -       |

2/2 completed seeds clear. Seed 2 running at write time; collect next session.
Throughput ~5.8-6.1k steps/s on CPU, 8 envs; full 5M run ~14 min. Overnight
detached infra not needed for MinAtar.

## Confidence

- Infra learns a published-benchmark game from scratch, reproducibly, on
  held-out seeds, above a published-scale bar: HIGH (two independent seeds, disk
  summaries, disjoint eval seeds).
- Byte-exact reproduction of the Young & Tian protocol: LOW. Breakout-v1 uses
  the minimal 3-action set and PPO is a different algorithm family than the
  paper's AC/Q. 9.4 is a reference bar, not an identical-protocol target. The
  clear is real; the "matches the paper exactly" claim is not made.

## Why this matters for the mission (Signal Dodge)

Same machine, same SB3, same CPU torch that went 0-for-5 from scratch on Signal
Dodge (PPO+VecNormalize, offline DiscreteCQL, CMA-ES, CMA-MAE, elite-BC; best
held-out 906.4 vs 930.27 bar) now clears a real benchmark from scratch. That
relocates the Phase N wall: it was almost certainly Signal Dodge's own design,
not an infra or method incapacity. A constant action already survives ~930 steps
in Signal Dodge, so the from-scratch learnable signal is thin. MinAtar Breakout
gives dense, immediately-credited reward (random 0.33 -> learned ~11-15), which
the same stack exploits fine.

Lateral read: the structurally different move that worked was changing BOTH
levers at once (environment: Signal Dodge -> MinAtar; algorithm family:
ES/BC -> on-policy policy-gradient). The next diagnostic step ports the working
recipe's lesson (dense credited reward) back to Signal Dodge's reward/obs design,
rather than a sixth from-scratch method on the unchanged env.

## Next

1. Collect seed 2; record 3-seed spread.
2. Decide Signal Dodge reward/observation redesign informed by the MinAtar
   contrast (dense per-step credit), OR establish MinAtar Freeway as a second
   confirmation game. Technical call, mine to make next session.
