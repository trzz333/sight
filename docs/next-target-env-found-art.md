# Next Target Environment - found-art: ADOPT (MinAtar + benchmarked PPO/DQN)

Context: Phase N closed FINAL NEGATIVE (five from-scratch methods, no clear of the
930.27 constant-action baseline on the custom Signal Dodge env). Recommendation on
record was to change the environment, not the algorithm. Jeff asked for a different
open-source game with a proven, reproducible open-source approach. This is that
found-art. Verdict: ADOPT both sides, no build needed.

Searches run: "MinAtar compute-limited from-scratch baselines"; "CleanRL
reproducible single-file benchmark openrlbenchmark"; "slimevolleygym hardmaru
self-play CMA-ES PPO from scratch".

## Recommendation (Claude's technical call): MinAtar, Breakout or Freeway first, trained with a benchmarked PPO or DQN reference

One-line reason: MinAtar is the field-standard compute-limited testbed for
from-scratch deep RL with independently reproduced published baselines, so it turns
"can our infra learn a game from scratch" into a clean cheap reproducible yes/no
against a real bar instead of a hand-rolled constant-action one.

Anchored facts (external sources, this session):
- MinAtar (Young & Tian 2019): five miniaturized Atari games (Breakout, Asterix,
  Freeway, Seaquest, Space Invaders) on a 10x10xN binary grid, built for reduced
  computation to enable thorough algorithmic comparison. Per-game action counts
  documented (Breakout 3, Asterix 5, Freeway 3, Seaquest 6, Space Invaders 4).
  Published from-scratch baselines with tight variance: Freeway ~52.8, Space
  Invaders ~45.4, Seaquest ~16.1, Asterix ~12.5, Breakout ~9.4; DQN peaks within
  the standard 5M steps. SB3-compatible fork exists (MinAtar ~10x faster than
  original Breakout, patched for Stable-Baselines3); rlai-lab MinAtar-Faster ships
  optimized code plus standard-algorithm benchmarks. [HIGH]
- CleanRL: single-file implementations benchmarked on par with reputable sources,
  backed by Open RL Benchmark storing exact command, frozen deps, and seed per run,
  plus a reproduce utility; JMLR 2022. [HIGH]
- Sight already runs SB3 2.8.0 on CPU torch, so lowest-friction proven route is SB3
  PPO/DQN on the MinAtar fork, cross-checked against CleanRL / Open RL Benchmark
  curves. [HIGH]
- CPU wall-clock per seed: single-digit hours on the tiny CNN, inside the existing
  overnight detached-run infra. [MEDIUM: anchored "peak within 5M frames" and
  "reduced compute by design", not a clean CPU-only timing.]

## Runner-up (held as follow-on, not the lead): Slime Volleyball + CMA-ES self-play

hardmaru/slimevolleygym is a real game where the exact CMA-ES/ES family that failed
on Signal Dodge is proven from scratch: self-play with GA, PPO, and cooperative
CMA-ES produce agents that beat the built-in baseline. Dependency-light (gym +
numpy only) and cheap on Sight's hardware profile (~12.5K timesteps/sec on a 2015
core-i7 CPU for state observations). It would directly confirm the Phase N wall was
the environment, not the method. Held second because it keeps pulling the same ES
lever and rides legacy tooling (pre-trained PPO on stable-baselines v2.10, Gym
0.19; later Gym has API-breaking changes), so the PPO path needs a compat shim
though the numpy+cma path dodges it. [HIGH on proven/cheap; legacy-gym friction is
why it is not the lead.]

## Lateral audit

Leading with PPO/DQN on MinAtar changes both levers at once (environment and
algorithm family). Phase N was three evolutionary/BC variants in a row; the
structurally different move is a value or policy-gradient method on a proven env,
not a fourth evolutionary method on a friendlier game. That is also why
Slime-Volley-CMA-ES is the confirmation experiment, not the headline.

## Ethics

Both are open-source research environments. No live commercial game, no ToS or
bot-detection surface. Clean per docs\ethics.md.

## Status

Gated on a Jeff-owned scope call: approve MinAtar as Sight's next target
environment. On approval, the deterministic spike is: install the MinAtar SB3 fork,
run one seed of SB3 PPO on Breakout, check against the published baseline, point
the existing held-out-eval harness at MinAtar episode-return, report the first
reproducible-or-not from-scratch curve. If Jeff redirects to Slime Volley, flip the
order; the CMA-ES-on-SlimeVolley path reuses Sight's existing ES plumbing most
directly.
