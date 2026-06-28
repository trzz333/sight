# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N (from-scratch RL via structurally-distinct paradigms). Opened on Jeff's direction call (C). Phase M closed FINAL NEGATIVE (M2 critic-broken, M2.1 critic-fixed-but-sub-baseline, aggregate IQM 418.25 CI [314.44, 670.50] below bar). On NLDC (hostname MSI).

**Last commit:** `ab9c5e1` Phase N opened (Jeff chose C): from-scratch RL via structurally-distinct paradigms; C1 = ES, plan in docs\phase-n-plan.md.

**Current task:** Phase N opened (Jeff chose C). found-art done this turn (searches named in `docs\phase-n-plan.md`). Verdict MIXED: ADOPT pycma (cma 4.4.4, installs clean on Py3.14) for a separable ES optimizer; ADAPT the rollout-as-fitness binding to the existing Godot env. TWO dead ends pruned with VERIFIED checks so no future session re-treads them: gSDE (Raffin 2021) is continuous-control only, inapplicable to Discrete(3); EvoTorch 0.6.1 hard-depends on ray>=1.0 which has NO Python 3.14 wheel, uninstallable here. C1 = Evolution Strategies (Salimans 2017): no critic (sidesteps the M2/M2.1 critic saga), episode-consistent perturbation (the K5.8 exploration lever), discrete actions supported. C1 fully scoped in the plan doc; no code written or run yet (stated plainly, not faked).

**Next action:** Build C1 per `docs\phase-n-plan.md`. (1) install `cma` SYSTEM-WIDE into C:\Python314\Lib\site-packages (likely needs an elevated shell; --user is invisible to WMI-detached training). (2) write the ES trainer: pycma separable CMA-ES, flatten<->load the SB3 MlpPolicy [64,64] actor (10->64->64->3, ~5059 params), rollout-as-fitness reusing the m2 env builder, fitness = mean length over fresh training seeds (held-out 1000-1009 stay sealed). (3) THROUGHPUT smoke-test FIRST (episodes/sec across parallel Godot subprocs, gens-to-signal estimate); do NOT launch any multi-hour run until throughput is sane. If throughput/sample-efficiency is the wall, drop to linear-policy + ARS (Mania 2018). (4) 3-seed screen vs the gate, full sweep only if it clears.

**Blockers:** None blocking. Jeff resolved the direction call: C (attempt structurally-different from-scratch methods). The one remaining Jeff-owned lever is the slate cap: Phase N defaults to THREE distinct paradigms, each one honest shot at the reliability gate, then FINAL NEGATIVE and close if all fail. Raising that cap is Jeff's; the default proceeds without asking. Everything else (method choice, tactics, governance) is Claude's.

**Notes:**

- Migration StrongerJr -> NLDC (hostname MSI, user maste) complete. Python now at `C:\Python314`; packages installed SYSTEM-WIDE into `C:\Python314\Lib\site-packages` (a --user-only install is invisible to WMI-detached training, no APPDATA in that minimal env, crashed every seed at `import numpy`; system-wide is the durable fix). Godot 4.6.2-stable at the winget path (matches scripts' DEFAULT_EXE). Bat path fixes in `b19cf06`.
- M2.1 critic fix worked but bought no competence: explained_variance 0 -> ~0.92 across seeds, yet eval still 0/3 below the 930.27 bar. Fixing the diagnosed defect did not lift the policy. Do not retry PPO harder.
- Portfolio context for the direction call: imitation clears the bar reliably and wide (BC 1737.3, PPO-finetune-from-BC 1710.5). The unsolved problem is from-scratch RL reliability, not imitation. Eval gate rejects single-action survival via `max(frac)<0.97` (proven). Reward "none" best-constant 845.7 < bar forces dodging; do not add reward shaping for M2.x.
- M-phase from-scratch PPO runs the GLOBAL interp (SB3 2.8.0, gymnasium 1.2.3, torch 2.11.0, numpy 2.4.4). `.venv-d3rlpy` offline stack is off the active path; never import d3rlpy globally if revisited. `requirements-lock.txt` reproduces the env (`pip install -r`, strip the self-referential `sight_agent` git line).
- noautoreboot: LEAVE set (Jeff, this session). The Dell/StrongerJr keeps it for legal-corpus access; the old "revert before reboot" instruction is retired. Not a Sight concern post-migration.

---
