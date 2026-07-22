# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 COMPLETE, portfolio packaging in progress. Project purpose per Jeff 2026-07-22: resume enrichment. Machine: MSI Raider 18 HX (hostname MSI, RTX 4080 Laptop 12GB).

**Last commit:** 61cb08b style: de-AI-tic pass on resume-facing content

**Current task:** Results write-up is done and pushed. docs\vzd-deadly-corridor-findings.md has section 9 (three-seed replication: pilot s3 2279.43 / s5 2279.67, seed-1 s3 2276.58 / s5 2280.44, seed-2 s3 2277.95 / s5 2279.69, all IQM over 30 deterministic episodes vs bars 683.9 and 93.6; every s5 probe FIGHT at 5.77-5.9 kills/ep, 28-29/30 survived; all numbers re-verified from summary.json plus independent trim_mean recompute this session) and a three-seed ledger table. README has the deadly_corridor rows, the curriculum-transfer paragraph, and a 5.7 MB demo GIF above the fold (docs\media\corridor_s5_demo.gif, one full skill-5 clear). Portfolio clip recorded: runs\vzd\demos\corridor_s5_ft_seed1.mp4 (40s, 47.5 MB, probe-identical episodes, gitignored). A de-AI-tic pass (WikiProject AI Cleanup checklist: negation pivots, kicker aphorisms, emphasis bolding) ran over README and findings sections 6/9; dated log sections 1-8 kept their original voice deliberately.

**Next action:** Execute the Jeff-approved portfolio sequence, in order: (1) add a "Reproduce this result" section to README (exact commands, versions vizdoom 1.3.0 / sb3 2.8.0 / Python 3.14, hardware, expected numbers); (2) upload the three ViZDoom models (ppo_defend, corridor s3_shaped, corridor s5_ft_seed1) to Hugging Face Hub via huggingface_sb3 package_to_hub for auto model cards + replay videos, Jeff authenticates the HF account when prompted; (3) draft the technical blog post from findings sections 1-9 (entropy collapse story, curriculum fix, replication), target GitHub Pages on the sight repo; (4) draft resume and LinkedIn bullets from the results table, delivered inline in copy-ready fenced blocks. Session runs in Claude Cowork with desktop control. Apply the de-tic checklist to everything written for external eyes.

**Blockers:** One Jeff touchpoint mid-sequence: Hugging Face login/account at step 2 (identity-owned). Not blocking steps 1, 3 draft, or 4.

**Notes:**

- **VZD-3 three-seed ledger:** six passing evals in 2276.58-2280.44 vs bars 683.9 (s3) and 93.6 (s5). Seed-1 s3 leg has no standalone combat probe (chain fixed mid-seed-1); covered by pilot s3 and seed-2 s3 probes. SHOTS_FIRED/accuracy still unreliable, omit everywhere; KILLCOUNT/HITCOUNT/DAMAGE_TAKEN clean.
- **De-tic discipline (Jeff directive):** run the AI-tic checklist (negation pivot "X, not Y" without information, staccato triplets, kicker aphorisms, emphasis bolding, "highlighting/underscoring" tack-ons) on all future resume-facing prose, including the blog post and bullets. Keep contrasts that carry real content.
- **Doom hider daemon UNKNOWN x2 carried:** silent owner deaths (no Python traceback, suspect external kill) and simultaneous-launch interpreter wedge. Read runs\vzd\doom_hider.log then window_watch.log before theorizing on any popup report. Scheduled tasks: Sight-Monitor, Sight-DoomHider only.
- **Process-kill rule:** before killing any duplicate in a port-owning family, check Get-NetTCPConnection ownership; verify AFTER mutating. Monitor serving verification is HTTP-vs-disk byte equality, never status 200. Monitor URL goes to Jeff at every launch, first thing.
- **Standing training caveats:** --ent-coef reapplied post PPO.load; resume from numbered _steps.zip + step-matched VecNormalize pkl, never model.zip; eval is RAW and cross-run comparable; DC waits under ~50s, long jobs detached via pythonw launch_hidden, verified by output files; runs\ gitignored.
