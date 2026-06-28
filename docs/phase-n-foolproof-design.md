# Phase N — Foolproof-Design found-art pass (2026-06-28)

Triggered by Jeff: "/found-art for an even-more-foolproof design; I'm not
averse to new OS environments; see if GitHub has better model incentive
treats." Two problem classes searched: durable long-run supervision (infra)
and better exploration tooling (algorithm). Verdict-first below.

---

## AXIS A — Durable long-run supervision on Windows

**FOUND-ART: MIXED** — adopt NSSM for long runs, reject DETACHED_PROCESS
(proven harmful this turn), keep the console+env-var path for short runs.

Generalized problem: run an 8h+ training process that survives session
disconnects / console-close events and auto-recovers from crashes, while
its multiprocessing + Godot worker pool keeps valid I/O.

Prior art (closest first):
- **NSSM (Non-Sucking Service Manager)** — nssm.cc, GitHub kirillkovalenko/nssm.
  [VERIFIED this turn] Runs an arbitrary exe/batch as a Windows service:
  session-independent (survives logout/disconnect — kills the window-CLOSE
  failure class at the root), proper AppStdout/AppStderr file redirection
  (valid child handles), per-exit-code actions (`AppExit 0 Exit` = stop on
  clean finish, default `Restart` = auto-restart on crash), AppThrottle /
  AppRestartDelay to avoid crash loops. Crash -> NSSM restarts -> our
  `--resume` reloads `es_state.pkl` = unattended recovery. Last stable build
  2014 but still standard; widely used for exactly this.
- **DETACHED_PROCESS (WMI CreateFlags=8)** — [VERIFIED harmful this turn].
  Tried it to drop the console entirely. Silently killed the worker pool:
  multiprocessing spawn + Godot subprocs get no valid std handles with no
  console, so the run died before logging anything (c1_smoke_detach: START
  only, no gens, no sentinel, no surviving process). Rejected.
- **WSL2 / Linux port** — [REMEMBERED, unchecked]. No forrtl/console model;
  systemd or nohup/tmux. Viable since Godot exports a Linux headless build,
  but it's a migration (port project, Linux Godot, rebuild venv). Held as a
  heavier option; Jeff has unlocked OS installs so it stays on the table if
  NSSM disappoints.

Gap NSSM does NOT auto-solve: service account. NSSM defaults to LOCALSYSTEM
in session 0 (no desktop — fine, trainer already runs Godot `headless=True`,
verified line 174). Open check: LOCALSYSTEM read access to the Godot exe under
`C:\Users\maste\AppData\Local\...WinGet`; if ACL-blocked, run the service as
`maste` (needs that password -> Jeff-owned credential step, not mine to enter).

Recommendation: adopt NSSM for the full 100-gen screen post-gaming. Setup is
the "skilled env work" Jeff unlocked; do it carefully (not rushed before a
game session). For short/bounded runs the WMI console pattern +
`FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` + `--resume` + `--max-wall-s` is enough.

---

## AXIS B — Better "incentive treats" (exploration)

**FOUND-ART: ADAPT** — pyribs CMA-MAE, deploy if the ES screen plateaus
sub-bar. Replaces the planned ARS fallback (stronger, targets the diagnosed
failure directly).

Generalized problem: ES collapses to a narrow behavior (gen-5 mean vec: actions
L .91 / R .09 / S 0) — premature behavioral convergence on a flat-ish objective,
exactly the project's recurring "exploration is the decisive lever" finding
(NoisyNet was the only from-scratch method to ever clear the bar).

Prior art (closest first):
- **pyribs (icaros-usc)** — pyribs.org, PyPI `ribs`, GitHub icaros-usc/pyribs,
  arXiv 2303.00191. [VERIFIED this turn] Official impl of CMA-ME, CMA-MAE,
  CMA-MEGA. Quality-Diversity: keeps an ARCHIVE of behaviorally-diverse elites
  instead of one mean, so it resists collapse. **CMA-MAE** anneals
  exploration->exploitation via a learning rate and is explicitly built to be
  robust to flat objectives — our failure mode. Crucially uses the SAME pycma
  ask-tell interface, so it ADAPTS into the existing rollout/eval infra: swap
  the optimizer, define a behavior descriptor (action fractions or mean
  x-position), keep workers/eval/gate unchanged.
- **ARS (Mania 2018)** — the prior planned C1 second attempt. Weaker, less
  targeted: linear policy + random search changes the policy class, not the
  exploration mechanism that we've repeatedly found decisive. Demote it.

Gap: need to choose a behavior descriptor for the archive (1-2 cheap dims:
action-fraction simplex, or trajectory mean x). Small design step, not a build.

---

## Adopted sequence

1. (running now) bounded ES screen, seed 0, 90-min wall, resumable -> learning
   stat now + a checkpoint to continue from. Respects the gaming window.
2. (next, post-gaming) stand up NSSM for seed 0; resume to full 100 gens.
   Auto-restart + `--resume` = unattended-foolproof. Run the account/ACL check.
3. (only if 100-gen ES plateaus sub-bar) ADAPT to pyribs CMA-MAE on the same
   infra — NOT ARS. QD attacks the diagnosed collapse head-on.

## Infra changes shipped this turn (tools\c1_es_train.py + batches)
- `--resume`: atomic per-gen pickle of full CMA state (`es_state.pkl`); a crash
  resumes instead of restarting.
- `--max-wall-s`: clean, checkpointed stop after N seconds (CMA maxiter stays
  `--gens`), so bounded runs finish + resume.
- `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` in batches: neutralises the Intel-runtime
  forrtl-200 window-CLOSE abort that killed the gen-6 screen.
- Launcher reverted to the proven WMI console pattern; DETACHED_PROCESS rejected.
