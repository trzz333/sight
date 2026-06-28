# Phase N - from-scratch RL via structurally-distinct paradigms

Opened after Jeff's direction call (C): keep attempting from-scratch RL with
structurally different methods, bounded by the stopping rule below. Phase M
(from-scratch on-policy PPO) is closed FINAL NEGATIVE (M2 critic-broken; M2.1
critic-fixed-but-sub-baseline; aggregate IQM 418.25, 95% CI [314.44, 670.50],
entire interval below the 930.27 bar).

## Success criterion (reliability, not a lucky seed)
From-scratch RL has already cleared the bar ONCE (K5.8 NoisyNet seed, 980.7), so
feasibility is not the open question; reliability is. C-method succeeds only when
it clears the bar reliably: bootstrap CI lower bound >= 930.27 over a seed sweep
(Agarwal et al. 2021 framework, already adopted), not a single-seed peak. Eval on
held-out seeds 1000-1009; gate also requires action diversity (max(frac)<0.97).

## Stopping rule (Claude-owned governance; cap is Jeff's lever)
Pre-registered slate of THREE structurally-distinct paradigms. Each gets ONE
honest shot at the gate (a cheap screen first; full sweep only if the screen
clears). A method that fails its screen is recorded NEGATIVE and NOT retried
harder (operating contract). If all three distinct paradigms fail, from-scratch-
RL-reliability is recorded FINAL NEGATIVE for the project and Phase N closes.
Three paradigm failures past an already-imitation-solved bar is a result, not a
reason to continue. Raising the cap past 3 is the one Jeff-owned lever here.

Slate (order may change as evidence comes in):
- C1: Evolution Strategies (black-box, no critic, episode-consistent perturbation).
- C2: intrinsic-motivation / exploration-bonus value-RL (RND-style), IF C1 fails.
      NOTE: distinct from K5.8 NoisyNet (already N=10 negative) and from PPO
      (already 2x negative); must be genuinely new, not a reskin.
- C3: one more distinct paradigm TBD by C1/C2 evidence (e.g. population-based).

## found-art verdict (this turn, searches named)
FOUND-ART: MIXED. ADOPT an ES optimizer library; ADAPT the env binding; two
candidate methods REJECTED after verified checks.

- gSDE (Raffin/Kober/Stulp 2021, "Smooth Exploration", PMLR v164) - VERIFIED,
  REJECTED. Continuous-control only: it replaces the Gaussian action distribution
  for SAC/PPO. Signal Dodge is Discrete(3). `use_sde=True` does not apply.
- EvoTorch 0.6.1 (NNAISENSE) - VERIFIED, REJECTED on this machine. Hard-depends
  on ray>=1.0; ray has no Python 3.14 distribution (pip: "no matching
  distributions available for your environment: ray"). Not installable on the
  C:\Python314 global interp. Do not retry.
- Evolution Strategies (Salimans et al. 2017, arXiv 1703.03864) - VERIFIED, the
  paradigm for C1. Optimizes policy-net weights by black-box perturbation,
  discrete actions supported, no critic (sidesteps the entire M2/M2.1 critic-
  fitting saga), each candidate evaluated over full episodes = perturbation is
  consistent within an episode (the K5.8 exploration-structure lever).
- pycma / cma 4.4.4 (Hansen) - VERIFIED installable (pure-Python, numpy-only dep;
  pip dry-run resolved clean on Py3.14). ADOPT for the ES optimizer.

## C1 design (locked enough to build)
- Optimizer: separable CMA-ES via pycma (CMA_diagonal). Full-covariance CMA is
  infeasible: actor net 10->64->64->3 ~= 5059 params; O(d^2) covariance and
  O(d^3) eigendecomp at d~5k. Separable/diagonal is O(d). Fallbacks if needed:
  OpenAI-ES (Salimans antithetic + rank-normalized, ~40 lines), or a linear /
  single-hidden-layer policy + ARS (Mania et al. 2018) which converges in far
  fewer generations on simple tasks.
- Policy class: reuse SB3 MlpPolicy [64,64] actor (10->64->64->3) for apples-to-
  apples with the PPO/BC comparison. Flatten actor params to a vector; load each
  candidate vector in; argmax over 3 logits for the action.
- Fitness: mean episode length over a few FRESH training seeds per candidate
  (NOT the held-out 1000-1009 eval seeds; those stay sealed for the final gate).
- Env: existing GodotSignalDodgeEnv (src\sight_agent\rl\godot_env.py), state mode
  Box(-1,1,(10,)), reward "none" (+1/step). Reuse the m2 env builder; wrap a
  single-episode rollout as the fitness function (this is the ADAPT).

## Open risks to resolve at build time (honest, unverified)
1. THROUGHPUT is the likely killer, not correctness. ES is sample-inefficient
   (Salimans needed millions of episodes) and every rollout is a TCP round-trip
   to a Godot subprocess. SMOKE-TEST first: measure episodes/sec across N parallel
   Godot subprocs, estimate generations-to-signal, and only launch the long run
   if the wall-clock is sane. If not, drop to linear-policy + ARS (fewer gens).
2. System-wide install. New packages must land in C:\Python314\Lib\site-packages,
   NOT --user (WMI-detached training has no APPDATA and cannot see user-site;
   it crashed every seed at import numpy last migration). The dry-run shell wrote
   to user-site and reported "normal site-packages is not writeable" -> the
   system-wide install of `cma` may need an elevated shell. Handle before launch.
3. pycma diagonal-mode ergonomics; if awkward, OpenAI-ES is the clean fallback.

## Next action
Build C1: install `cma` system-wide; write the ES trainer (pycma sep-CMA-ES,
actor-vector <-> MlpPolicy, rollout-as-fitness reusing the m2 env builder);
SMOKE-TEST throughput (episodes/sec, gens-to-signal estimate) BEFORE any long
run; if sane, run a 3-seed screen against the gate; only then the full sweep.
Do not launch a multi-hour run until the throughput smoke-test says it is sane.
