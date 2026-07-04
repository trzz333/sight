# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase N CLOSED FINAL NEGATIVE. Three pre-registered structurally-distinct from-scratch paradigms, one honest shot each at the held-out reliability gate, all spent: C1 (separable CMA-ES) 906.4, C2 (CMA-MAE QD) 845.7, C3 (reward-ranked elite-BC self-imitation) 606.0, all below the 930.27 bar. Across five total from-scratch methods (PPO+VecNormalize, offline value-RL/DiscreteCQL, CMA-ES, CMA-MAE, elite-BC) none clears; the same 5059-param actor clears only with demonstrations (BC 1737.3, PPO-finetune 1710.5), which the charter does not count. Next-target-environment found-art done this session; recommendation is MinAtar. Mission continuation is a Jeff-owned scope call (new target environment).

**Last commit:** `c987de6` found-art: next target environment ADOPT MinAtar + benchmarked PPO/DQN

**Current task:** Phase N is closed; no run is live. Next-target-env found-art complete and written to `docs\next-target-env-found-art.md`. Verdict ADOPT both sides, no build. Recommendation (Claude's technical call): MinAtar (Breakout or Freeway first) trained with a benchmarked PPO or DQN reference, run via SB3 (already installed, 2.8.0) on the SB3-compatible MinAtar fork, cross-checked against CleanRL Open RL Benchmark curves. MinAtar is the field-standard compute-limited from-scratch testbed with published reproducible baselines (Freeway ~52.8, Space Invaders ~45.4, Seaquest ~16.1, Asterix ~12.5, Breakout ~9.4; DQN peaks within 5M steps), so it gives a real published bar and a clean cheap yes/no on whether the infra can learn a game from scratch. Runner-up held as follow-on: Slime Volleyball (hardmaru/slimevolleygym) + CMA-ES self-play, which proves the exact ES family Signal Dodge defeated and reuses Sight's ES plumbing, but rides legacy gym 0.19 / SB v2.10 so it is the confirmation experiment, not the lead. All facts web-anchored with citations this session.

**Next action:** On Jeff approving MinAtar as the target environment, stand up the deterministic ADOPT spike: install the SB3-compatible MinAtar fork into `.venv-c1`, run one seed of SB3 PPO on MinAtar Breakout, check the return against the published baseline, point the existing held-out-eval harness at MinAtar episode-return, and report the first reproducible-or-not from-scratch curve. Do not commit to the new environment before approval (it is a Jeff-owned scope call). If Jeff redirects to Slime Volley, flip the order and run CMA-ES self-play on slimevolleygym first.

**Blockers:** One, Jeff-owned: approve the next target environment (MinAtar recommended; Slime Volley is the named alternative). A new game is a scope/direction call per the charter. No technical blockers; the spike is deterministic once the environment is approved.

**Notes:**

- Phase N verdict fully disk-anchored: `c3_screen_verdict.json` (actor 606.0 gate FAIL, final 443.2 gate FAIL, staged_seeds_1_2 false), `c3_report.json` (60 iters, dev-best 790.1 iter 17), sentinel `EXIT 0 C3-NEGATIVE-seed0-clear-miss`. C1/C2 verdicts re-confirmed off their findings docs before asserting FINAL NEGATIVE. Closing finding in `docs\phase-n-c3-findings.md`.
- Cross-paradigm read: held-out declined across paradigms (906.4 -> 845.7 -> 606.0) as machinery grew. Not a capacity wall (identical actor clears with demos); a from-scratch credit-assignment/exploration wall at hobby-lab compute. C3 failed on competence, not action-collapse (max frac .438).
- Reusable infra surviving Phase N (ports to any new env): held-out eval gate (`tools\c1_es_eval.py`, PASS = mean>=bar AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97), windowless detached-execution pattern, autonomous sequential-seed screen orchestrator, stdlib status server on :8766, 8-worker headless Godot pool.
- found-art recommendation detail and citations in `docs\next-target-env-found-art.md`. Lateral audit: leading with PPO/DQN on MinAtar changes both env and algorithm-family levers; Slime-Volley-CMA-ES keeps the ES lever so it is second.
- `runs\` is gitignored: screen histories/eval summaries/verdict live only on disk, folded into findings docs. Prior temp files cleaned this session.
