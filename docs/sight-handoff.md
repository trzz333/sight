# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Signal Dodge (930.27 bar) open. Fast-replica reward-geometry lever CLOSED NEGATIVE: potential-based shaping shows no reliability lift at 5M.

**Last commit:** `0802918` sd-fast: none-vs-shaped 5M eval-of-record, PBRS refuted as reliability fix

**Current task:** The matched-seed none-vs-shaped 5M sweep is complete (`shaped_sweep.sentinel` = CHAIN_DONE, all 10 models on disk) and evaluated. Eval of record via `tools\sd_fast_reliability.py` (rliable / Agarwal 2021 methodology ported to numpy: IQM + percentile seed-bootstrap 95% CI + probability of improvement), greedy reload-eval on held-out seeds 5000-5029, obs through each run's saved VecNormalize stats. Result: reward=none IQM 733.6 CI [613.0, 1042.2] clears 1/5; shaped IQM 741.0 CI [669.9, 1451.1] clears 1/5. IQM diff +7.4 (~1%), P(IQM_shaped>IQM_none) 0.607, rliable POI 0.569. Verdict HIGH: PBRS does NOT lift reliability at 5M. Clear-rate tied 1/5, 95% CIs overlap almost completely, POI is a hair off a coin flip. The harness's built-in binary printed "SHAPED >= NONE" on a mechanical diff>0 test; that is too permissive and is overridden on the evidence. Diagnostic detail: shaped s0 and s1 have byte-identical held-out length distributions despite different greedy action sequences (weak policies, seed-locked death timing, `tools\_probe_shaped_collapse.py`); shaped's higher pool mean 925.6 is carried by one lucky seed (s4 1735.1) that IQM trims out. Reward-geometry engineering for reliability has now failed twice on this env (K5.5-era dense shaping, PBRS here); method changes, no retry harder. Full record in `docs\sd-fast-replica-budget-findings.md`.

**Next action:** Inventory the imitation artifacts (BC 1737.3, PPO-finetune 1710.5): locate the on-disk checkpoints, confirm their training env, and determine whether `tools\sd_fast_reliability.py` can reload-eval them on held-out seeds 5000-5029 for an apples-to-apples IQM / CI / POI versus the from-scratch arms. This readies the imitation reliability number for whichever way the scope call lands. Do NOT open a new from-scratch lever and do NOT launch the Godot 5M eval-of-record until Jeff rules on the scope call.

**Blockers:** One Jeff-owned scope call is LIVE. From-scratch reliability has now failed across every pre-registered lever: Phase N (CMA-ES, CMA-MAE, elite-BC, all FINAL NEGATIVE), budget isolation (1/5 clear at 5M), exploration (K5.8 NoisyNet 1/10), and reward geometry (this result, no lift). Imitation clears reliably (BC 1737.3, PPO-finetune 1710.5). Decision: keep pursuing from-scratch reliability on a new lever, or accept imitation as the standing mission solution. Claude's recommendation, earned not a menu: accept imitation, because from-scratch reliability has failed across every lever tried while imitation clears every time. Direction/scope, reserved to Jeff.

**Notes:**

- Eval of record is rliable (Agarwal 2021) ported to numpy (package will not build here, `arch` needs MSVC). `tools\sd_fast_reliability.py`; cache `runs\sd_fast\reliability_eval_cache.json` (gitignored, re-runs instant).
- Reward shaping is refuted for reliability at 5M. Do NOT retry as PBRS coef 1.0 or a Godot port of the shaped recipe; both are retrying a twice-failed method.
- Status dashboard reused, not rebuilt: `tools\sd_fast_status_server.py` + `tools\launch_sd_fast_status_detached.py`, windowless on localhost:8767, reads sweep logs/summaries/sentinel. Adapted from the `sight_status_server.py` stoplight shell.
- Do NOT launch the Godot 5M eval-of-record: no from-scratch recipe clears the replica reproducibly, so nothing reliability-worthy exists to port.
- `runs\` is gitignored; summaries, logs, models, and the eval cache live on disk only.
