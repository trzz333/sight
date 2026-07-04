# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N CLOSED FINAL NEGATIVE. Three pre-registered structurally-distinct from-scratch paradigms, one honest shot each at the held-out reliability gate, all spent: C1 (separable CMA-ES) NEGATIVE best held-out 906.4, C2 (CMA-MAE QD) NEGATIVE 845.7, C3 (reward-ranked elite-BC self-imitation) NEGATIVE 606.0. Closing finding written to `docs\phase-n-c3-findings.md`. Across five total from-scratch methods (PPO+VecNormalize, offline value-RL/DiscreteCQL, CMA-ES, CMA-MAE, elite-BC) none clears the 930.27 constant-action baseline on Signal Dodge; the same 5059-param actor clears it only with demonstrations (BC 1737.3, PPO-finetune-from-BC 1710.5), which the charter does not count as from-scratch success.

**Last commit:** (this session) Phase N C3 screen CLOSED NEGATIVE; Phase N FINAL NEGATIVE closing finding. Substantive hash filled by handoff commit.

**Current task:** None active. Phase N is closed. No run is live: seed-0 completed 60 iters (report `runs\phase_n\c3_screen_s0\c3_report.json`, elapsed 6936s), the orchestrator gated both vectors on held-out 1000-1009, wrote `c3_screen_verdict.json` (screen_result C3-NEGATIVE-seed0-clear-miss) and `c3_screen_all.sentinel` (EXIT 0), and correctly did NOT stage seeds 1+2 (606.0 < 880 near-miss floor). Status server on :8766 may still be up (harmless, windowless); kill if a clean box is wanted.

**Next action:** Jeff-owned direction/scope call. From-scratch RL on Signal Dodge is an answered negative: five structurally distinct methods, wall did not move, held-out trend across the last three paradigms actually declined (906.4 -> 845.7 -> 606.0). The only structurally different lever left is the target environment, not a sixth algorithm on the same game (that is a "new target environment" call, which is Jeff's). Options are his to weigh: (a) open a new phase against a different, more credit-assignment-tractable environment; (b) redefine success to allow demonstration-seeded RL; (c) record the from-scratch-on-Signal-Dodge arc as concluded. Claude's one-line recommendation: if the mission continues, change the environment, not the algorithm; do not run a sixth from-scratch method on Signal Dodge. No technical work is queued until Jeff picks a direction.

**Blockers:** Direction/scope decision above (Jeff-owned). No technical blockers.

**Notes:**

- Phase N verdict is fully disk-anchored, not memory: `c3_screen_verdict.json` (actor vec 606.0 fracs .281/.438/.281 gate FAIL; final vec 443.2 fracs .328/.344/.328 gate FAIL; staged_seeds_1_2 false), `c3_report.json` (60 iters, dev-best 790.1 at iter 17), sentinel `EXIT 0 C3-NEGATIVE-seed0-clear-miss`. C1/C2 verdicts re-confirmed off-disk in their findings docs before asserting FINAL NEGATIVE.
- C3 failure mode differs from C1: no action-collapse (max frac .438, balanced/stay-heavy), so C3 failed on raw competence, not single-action degeneracy. The elite buffer stayed diverse; dense BC on it plateaued near ~600 held-out steps.
- Interpretation: not a capacity wall (identical actor clears the bar with demos). It is a from-scratch credit-assignment/exploration wall at hobby-lab compute (seeds-per-gen 2, ~60 iters, high between-seed variance).
- Eval gate UNCHANGED and reusable for any future environment port: held-out band, PASS = mean>=bar AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97. `tools\c1_es_eval.py` is the gate. `runs\` is gitignored: histories/eval summaries/verdict live only on disk, folded into `docs\phase-n-c3-findings.md`.
- Infra that survives Phase N (reusable next environment): windowless detached execution pattern (DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP, stdin DEVNULL, CREATE_NO_WINDOW for Godot pool); autonomous sequential-seed screen orchestrator; stdlib status server on :8766; 8-worker headless Godot pool. All proven this phase.
- Prior-session temp files in `runs\` (gitignored, harmless): `_inspect_c3.py`, `_verify_prior_findings.py`, and any `_c2_commit_msg.txt`. Delete opportunistically.
