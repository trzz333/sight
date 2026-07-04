# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Mission env (Signal Dodge, 930.27 bar) open. On the fast replica: budget isolation closed (1/3 seeds clear at 5M); next lever is reward geometry (potential-based shaping), sweep in flight.

**Last commit:** `f8e137d` sd-fast: potential-based reward shaping + matched-seed sweep; 3-seed budget spread recorded

**Current task:** Matched-seed none-vs-shaped 5M sweep RUNNING DETACHED on the replica. Launcher PID 16080 (`tools\sd_fast_shaped_sweep.py`, spawned windowless). Chain of 7 runs sequential (~12 min each, ~85 min total): none seeds 3,4 then shaped seeds 0-4. Writes `runs\sd_fast\shaped_sweep.sentinel` = CHAIN_DONE at the end. Purpose: does potential-based shaping raise the from-scratch CLEAR-RATE above the reward=none baseline of 1/5 (existing none seeds 0,1,2 = 1/3; this extends none to n=5 on seeds 0-4 for a matched comparison).

**Next action:** When `runs\sd_fast\shaped_sweep.sentinel` reads CHAIN_DONE, compute the clear-rate for both arms with an IQM spread (adapt `tools\sd_fast_iqm_spread.py`: none arm = `sd_fast_m21_s{0..4}_5M`, shaped arm = `sd_fast_m21sh_s{0..4}_5M`), and record the none-vs-shaped clear-rate + IQM spread in `docs\sd-fast-replica-budget-findings.md`. Decision rule: if shaped clears more seeds than none, tune the coefficient (try coef 1.0) or port the shaped recipe to a Godot 5M run for the eval of record (build CheckpointCallback+resume first, Phase M saw Godot worker crashes). If shaped does NOT beat none, potential-based shaping is refuted at 5M and the open Jeff-owned scope call goes live (below). Do NOT launch the Godot 5M eval-of-record until the replica clears reproducibly across seeds.

**Blockers:** None blocking the next action (sweep is self-completing). One Jeff-owned scope call is open and becomes live only if the shaped sweep does not lift the clear-rate: whether to keep pursuing from-scratch reliability or accept imitation (BC 1737, PPO-finetune 1710, both clear reliably) as the standing mission solution. Not urgent; do not raise unless the shaped result forces it.

**Notes:**

- 3-seed budget spread (reload-eval, `tools\sd_fast_iqm_spread.py`): s0 IQM 1148.7 CLEAR, s1 IQM 482.1 constant-left collapse, s2 IQM 750.1 diverse-but-sub-bar. 1/3 clears 930.27. Two distinct failure modes: basin collapse (s1) and mediocre-but-diverse dodging (s2). Budget lifts ceiling, not reliability.
- Corrected vs prior handoff (disk beats memory): final EV does NOT track clearing (clear s0 EV 0.18; sub-bar s1/s2 EV 0.885/0.907). Basin collapse is a MINORITY mode: 2/3 seeds are action-diverse.
- Reward shaping: `SignalDodgeFast(reward_mode="shaped", shape_coef=0.5, shape_gamma=0.999)`. Potential-based (Ng 1999), Phi = imminence-weighted horizontal clearance to nearest hazard above player. `reward_mode="none"` byte-identical (dynamics verified identical 20 seeds, `tools\sd_fast_shaped_check.py`). Eval is reward-agnostic (counts survival steps), so shaping only affects training.
- found-art on this lever is recorded in the findings doc: NoisyNet/exploration already refuted (K5.8 1/10, Phase L); existing Godot threat_weighted_clearance shaping falsified only at 10k with alpha too small (K4.1). PBS at 5M with coef above the margin floor is a genuinely new test, not a Phase G rerun.
- Imitation remains the reliable solution (BC 1737.3, PPO-finetune 1710.5); from-scratch reliability is the open problem.
- runs\ is gitignored; all summaries/logs/models live on disk only. Detached launch pattern used: CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP via `tools\_spawn_shaped_sweep.py`.
