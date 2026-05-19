# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** Phase K (K5.1 high-force clearance reward alpha=0.30 produced a constant-action regime shift, not a behavior break; trained policy collapses to constant-STAY with player_x exactly 360.0 on every reached eval step; FAIL on GPT "another constant-action basin" clause; routing to K5.2-grok env-dynamics sanity check)

**Last commit:** `1db19a9` Phase K K5.1 alpha=0.30 high-force clearance reward: constant-action regime shift, FAIL

**Current task:** K5.1 evidence on disk in `docs/k5-1-clearance-alpha030-visible-behavior-evidence.md` (271 lines) and pushed. Sibling config `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml` at reward_shaping_alpha=0.30 / lookahead 270 / safe_lateral 180. Smoke parser `tools/h5_smoke_parse.py` made alpha-parametric via --alpha CLI flag, default 0.05 preserved for backward compat. Smoke aggregate frac_active_threat_saturated_norm = 0.416 < 0.50 across the representative sample; the single per-episode FAIL on ep-000005 (0.524) is byte-identical to Phase G smoke because the saturation predicate clearance_norm >= 0.98 is alpha-invariant by construction. Training: 10k steps seed=0 with --reward-scale-divisor 30 to match the K3.5c regime that K4.1 measured the margin floor on; 388 s wall time, exit 0. Trained-only eval seeds 1000-1009 mode=full: collision_rate 1.00, timeout_rate 0.00, mean episode length 606. Action distribution on 6060 reached eval steps: action=0 (stay) on 100%, action_wire=1 on 100%, action=-1 (left) on 0%. Player_x on every reached step across all 10 seeds: exactly 360.0 (screen center, zero lateral motion). K3.5c baseline for comparison: constant-LEFT, player_x <= 16 on 92.55% of 8214 steps. K4.1-style logit-margin probe deferred (behavioral evidence is unambiguous binary FAIL without it).

**Next action:** Execute K5.2-grok env-dynamics sanity check per GPT K5.1 scope item 7. Grok scope: action timing per Godot physics tick, hazard kinematics, observation freshness across H3 transport boundary, frame-stack contract, and player kinematics. K5.2-grok now has two anchored facts to explain rather than one: K3.5c-divisor30 seed 0 with reward_shaping=none -> constant-LEFT at left wall; same regime with threat_weighted_clearance alpha=0.30 -> constant-STAY at center. Both deterministic-argmax fixed points at 10k steps, ~40 PPO updates. Whatever env-layer property forces single-action collapse must explain both attractor positions, not just one. GPT directive forbids coefficient sweeping in between: no alpha=0.50, alpha=1.0, safe-lateral tweaks, entropy tuning, or another reward formulation before K5.2-grok runs.

**Blockers:** None requiring Jeff. Routing follows GPT K5.1 scope item 7.

**Notes:**

- Tactical-divisor call recorded in K5.1 evidence section 2: K5.1 used --reward-scale-divisor 30 to match the K3.5c regime K4.1 probed. GPT specified seed=0 only, without divisor; Claude filled the unspecified parameter so the run would land in the K4.1 reference regime rather than an untested intermediate.
- Smoke-gate revision applied at K5.1 under charter "Claude revises GPT's decisions on evidence": Phase G smoke disposition carried over because the saturation predicate is alpha-invariant. Trained-policy aggregate frac_sat_active = 0.263 over 4972 active-threat steps, well below the Phase G 0.50 monitoring bar.
- K5.1 trained-policy mean reward 716.66 across 6060 reached steps decomposes to base 6050 + clearance ~1116. Shaped bonus is 18.4% of total reward; base dominates 5.4-to-1, but per-step shaping was enough to bias PPO advantage estimates toward the constant-stay fixed point that maximizes minimum hazard clearance.
- The Phase E entropy recipe at ent_coef=0.01 did not prevent collapse to constant-action; eval-time entropy is moot (deterministic argmax) but the training-time entropy term moved (entropy_loss -0.858 -> -0.697 over 40 iterations), so the issue is not "no entropy regularization."
- New tracked artifacts this session: `configs/rl/signal_dodge_ppo_h5_pixel_entropy_shaped_alpha030.yaml`, `tools/h5_smoke_parse.py` (modified), `docs/k5-1-clearance-alpha030-visible-behavior-evidence.md`. Smoke + train + eval driver bats under `runs/smoke/` and `runs/rl/` remain gitignored.
