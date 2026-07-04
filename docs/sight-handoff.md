# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. MinAtar adoption spike (Jeff-approved next target environment). Signal Dodge mission still open.

**Last commit:** `9144198` MinAtar ADOPT spike: first from-scratch clear (SB3 PPO Breakout)

**Current task:** First from-scratch clear in project history is banked. SB3 2.8.0 PPO with the Young & Tian small CNN (custom SB3 extractor) learns MinAtar/Breakout-v1 from scratch over 5M steps. Held-out eval is the deterministic policy over seeds 1000-1009 range (1000-1029, disjoint from the seed-0..2 training envs): seed 0 mean 11.5 (std 4.08), seed 1 mean 14.7 (std 4.49), both clearing the ~9.4 published-scale bar; seed 2 is still training in the background at handoff time. Random-policy floor is 0.333. Throughput ~5.8-6.1k steps/s on CPU (8 envs), full 5M run ~14 min, so no overnight detached infra is needed. All numbers anchored to `runs\minatar\*_summary.json` and rev-parse this session. Findings written to `docs\minatar-adopt-spike-findings.md`.

**Next action:** Collect the seed-2 summary (`runs\minatar\ppo_Breakout-v1_s2_full_summary.json`) and record the 3-seed spread in `docs\minatar-adopt-spike-findings.md`. Then port the MinAtar lesson back to the mission: redesign Signal Dodge's reward toward dense per-step credit (the diagnosed Phase N wall, since a constant action already survives ~930 steps) and run one from-scratch PPO seed on the redesigned env. Freeway as a second MinAtar confirmation game is the fallback only if the Signal Dodge redesign stalls. Technical call, Claude's to make, no Jeff gate.

**Blockers:** None requiring Jeff.

**Notes:**

- Anchors: seed summaries in `runs\minatar\` (gitignored). seed0 11.5, seed1 14.7 held-out both clear 9.4; seed2 in flight at handoff. Random floor 0.333. HEAD `9144198`.
- Cross-env read: the same stack that went 0/5 from scratch on Signal Dodge (best held-out 906.4 vs 930.27 bar) clears MinAtar Breakout from scratch. Wall relocated to Signal Dodge's own design (thin learnable signal), not an infra or method incapacity.
- Reusable: `src\sight_agent\rl\minatar.py` (gymnasium env layer + Young & Tian CNN extractor), `tools\minatar_ppo_spike.py` (trainer + held-out eval), `tools\minatar_sanity.py` (random floor).
- found-art: mainline `minatar` 1.0.15 is gymnasium-native, so the rlai-lab MinAtar-Faster fork was not needed. ADOPT mainline; CNN is ADAPT (Young & Tian 2019 / qlan3 gym-games).
- Honesty: Breakout-v1 uses the minimal 3-action set and PPO is not the paper's AC/Q, so 9.4 is a reference bar. The from-scratch clear is real (HIGH); byte-exact reproduction of the paper protocol is not claimed (LOW).
