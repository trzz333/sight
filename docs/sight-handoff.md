# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Post-Phase-N. Signal Dodge (930.27 bar) open. From-scratch (Jeff-ruled). Curriculum lever judged better-than-none but not port-reliable; a gamma-0.99 variance-reduction arm is training to try to make it reliable.

**Last commit:** `fdb9d0a` sd-fast: refute entropy-collapse, diagnose value variance, launch gamma-0.99 arm

**Current task:** The gamma-0.99 5-seed arm is IN FLIGHT. Prior result (committed b7f81f9): the start-state curriculum arm `sd_fast_m21curr_s{0..4}_5M` is a large real improvement over the m21 none arm (rliable IQM 1394.5 vs 733.6, +661, P(IQM)=0.970, clears 3/5 vs 1/5) but NOT port-reliable: seeds s2 (887.7) and s3 (669.9) collapse below the bar and the curr IQM 95% CI [742.5, 1730.1] straddles it. This session diagnosed WHY, evidence-anchored. First refuted the entropy-collapse guess by probing the trained policies: the failing seeds are not low-entropy (s3, worst at 670, has the HIGHEST policy entropy 0.756 nats; winners s0 0.552, s4 0.669), so an ent_coef bump targets the wrong mechanism. found-art (arXiv 2301.05104 / 2311.02129 / 2111.04504) reframed it as value-estimation variance under gamma 0.999 (effective horizon ~1000) on an 1800-step survival task, the recipe forces the critic to regress near-undiscounted survival ~1000 steps out. Decision: cut the discount, not the exploration. Launched `sd_fast_m21curr_g99_s{0..4}_5M` (gamma 0.99, effective horizon ~100, else m21 + curriculum verbatim) detached this session, chain log `runs\sd_fast\curr_g99_chain.log`, seed 0 training at handoff. Root-cause detail and the refutation are in `docs\sd-fast-curriculum-findings.md`.

**Next action:** Collect the gamma-0.99 arm as it lands (`runs\sd_fast\sd_fast_m21curr_g99_s{0..4}_5M_summary.json`, ~12.5 min/seed, ~60 min total). Once all 5 are on disk, run `.venv-c1\Scripts\python.exe tools\sd_fast_reliability.py`; the g99 arm is already wired in (guarded on all 5 models present) and prints g99-vs-none, g99-vs-curr, and a verdict. Port gate: clears 5/5 AND IQM CI lower bound > 930.27. If g99 clears reliably, port the recipe to a Godot 5M eval-of-record (bar 930.27). If it lifts but is still short, next un-tried knob is `anneal_frac` 0.7 -> 0.9 or higher `n_init_max` (hold the curriculum scaffold longer for slow seeds), not ent_coef (refuted this session).

**Blockers:** None requiring Jeff. Scope call resolved (keep from-scratch). Do not open a structurally new lever until the gamma-0.99 arm is judged.

**Notes:**

- Curriculum trainer: `tools\sd_fast_ppo_curriculum.py` (CLI: --seed --steps --gamma --n-init-max --anneal-frac --ent-coef --run-id; m21 defaults verbatim). gamma flows into both PPO and VecNormalize return-norm. g99 arm uses --gamma 0.99, everything else default.
- rliable harness `tools\sd_fast_reliability.py` now carries none / shaped / curr (required) plus a guarded g99 arm (evaluated only when all 5 g99 models exist). Reload-evals greedily on held-out seeds 5000-5029; cache `runs\sd_fast\reliability_eval_cache.json` (gitignored). Verified this session it still runs on the 15 present models and skips the absent g99 arm.
- Root-cause finding (HIGH): from-scratch failure is a shared constant-collapse attractor (byte-identical held-out length arrays across none-s3/curr-s3/shaped-s0/shaped-s1, mean 669.9) driven by value-estimation variance, NOT entropy collapse (probe refuted that). Full writeup in `docs\sd-fast-curriculum-findings.md`.
- Twice-failed levers, do NOT retry: reward geometry (K5.5 shaping + PBRS), CMA-ES, CMA-MAE, elite-BC, budget 5M, NoisyNet. The discount is none of these. Imitation still clears reliably (BC IQM 1800, PPO-ft 1738.5) but scope is from-scratch.
- `runs\` gitignored. Untracked throwaway `tools\_curr_g99_chain.py` is the live chain launcher, delete next session once CHAIN_G99_DONE. Leftover `-i` REPLs and the status server (pid ~22984) may linger; harmless.
