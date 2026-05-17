# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** H5 amended; Phase K K0 training-time entropy-collapse probe complete at both 2048 and 10000 timesteps. K0-2048 verdict K-C (pilot), K0-10k verdict K-A (late). K-B detector tightened to AND semantics before the 10k rerun.

**Last commit:** `<PENDING>` Phase K K0-10k evidence + K-B AND patch (see commit message body).

**Current task:** K0-10k ran one instrumented 10000-timestep training session (train_seed=2, `configs/rl/signal_dodge_ppo_h5_pixel_entropy.yaml`, 40 PPO updates, 312.0 s wall) via the patched `tools/h5_training_entropy_probe.py`. Collapse threshold crosses at update 25 (ts=6400): `H_post=0.1776` (entropy < 0.20), `rollout_action_stats.top_action_fraction=0.973` (>= 0.95). Raw margin never crosses 4.0 (peak 3.76 at upd 26 pre). Evidence at `docs/h5-phase-k-training-entropy-probe-evidence.md` sections 11-17. NDJSON sha256 `5a8d7a4d…`, summary sha256 `bd420e0f…` (both gitignored under `runs/phase_k/`).

**Next action:** Hold for Jeff's go signal. K-C clause is discharged. Per Phase J option ladder, K1 architecture probe (`policy_kwargs.net_arch = dict(pi=[64], vf=[64])`, same seed, same 10000 timesteps, same entropy YAML otherwise) is the structurally correct next slice. Before K1 executes, GPT should resolve one open question raised by K0-10k finding 1: Phase H's basin definitions were measured on `rollout_action_stats.top_action` (sampled), not `policy_state.top_argmax_action` (deterministic). The deterministic argmax locks at upd 9 (ts=2304) and never reverses through upd 40, which is much earlier and much firmer than Phase H's narrative suggested. Phase H Class A/B/C labels may need re-derivation against the deterministic stat before K1 has a basin to compare against. K2 train-seed asymmetry probe remains held behind K1.

**Blockers:** None. Open question for GPT noted under Next action.

**Notes:**

- K0-10k finding 1 falsifies the K0-2048 handoff's "K0 rollout argmax oscillates left/stay across 8 updates" framing. That stat was `rollout_action_stats.top_action` (sampled actions actually drawn), not `policy_state.top_argmax_action` (deterministic argmax over the same 256 rollout obs). The deterministic argmax was at 1.000 every one of the 40 updates and flipped only twice: left → stay at upd 2, stay → left at upd 9. After upd 9 it never reversed. The "Class B = train_seed=2 picks left" identity locks at ts=2304, inside the 2048-10000 window the K0-2048 evidence predicted. The probe records both stats by design; future evidence docs must report both and never conflate them.
- Collapse is steady drift, not a phase change. From upd 9 to upd 25, entropy slides 1.03 → 0.18, sampled top-action fraction rises 0.41 → 0.97, raw margin grows 0.43 → 3.62. No single-update step change exceeds ~0.3 entropy or ~0.1 EV. Mechanism is standard PPO commitment once EV crosses ~0.10 (upd 9, ev=0.134); the value head produces structured advantage, PPO commits to argmax basin, logit margin sharpens, softmax sharpens, entropy collapses, feedback loop closes by upd 25.
- K-B AND patch behaved correctly: adv_std stayed in [0.68, 18.45] (never below 0.05), so K-B did not fire despite EV being below 0.10 on updates 1-8. Under the pre-patch OR rule, K-B would have falsely tripped on upd 3 (ev=-0.008) and outranked the genuine K-A late signal. The patch lands the K0-2048 evidence section 6 recommendation.
- Raw margin 4.0 threshold is too strict for this config: K0-10k margin peaked at 3.76 (upd 26 pre), so K-D wedge-only would never have triggered even with sampled-fraction collapse present. Consider relaxing the margin threshold or dropping it from K-D classification in future probes. Not blocking K1.
- Launch pattern for ~5-min probes: bat-with-sentinel wrapper at `C:\Users\maste\AppData\Local\Temp\run_k0_10k.bat` with `SIGHT_GODOT_EXE` set inline, stdout/stderr redirected to a log, done-sentinel written on exit. Started with `start "k0_10k" /MIN cmd /c <bat>` from an interactive shell, then polled the sentinel via PowerShell `Test-Path` every 60-120 s. Durable across the 4-min MCP timeout. Inline invocation under `interact_with_process` died on first try because `SIGHT_GODOT_EXE` was user-scope only; setting it inline in the wrapper bat avoided the issue entirely.
