# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-1 (ViZDoom defend_the_center: RL teacher run + validated BC-from-demos pipeline; Godot Signal Dodge closed with imitation as the standing solution)

**Last commit:** PENDING

**Current task:** PPO CnnPolicy (gray 60x80, skip 4, stack 4, gamma 0.99) training on VizdoomDefendCenter-v1, resumed from the 750k checkpoint after the first run was killed at 772k, running detached PID 15376, log runs\vzd\ppo_defend\train_log2.txt. At 758k steps ep_rew_mean 9.7 (about ten kills per episode vs about zero untrained), entropy 0.31, explained variance 0.90. ETA roughly 105 min from 13:15 to finish 1.5M, then it writes model.zip, summary.json (30-ep deterministic eval, mean + IQM), and a DONE sentinel. BC pipeline (vzd_record_demo, vzd_extract_dataset, vzd_bc_train, vzd_bc_eval) validated end to end on stand-in data at ce0b3ad; desktop icon "Record Doom Demos" deployed for optional human demos.

**Next action:** Check runs\vzd\ppo_defend for DONE and a fresh summary.json. If mean/IQM is strong, write a watch/eval tool for the SB3 model (vzd_bc_eval covers BC nets only), record a 30s gameplay clip, then execute the approved resume packaging: README rewrite with results table, the gamma/critic-collapse story, and a short writeup. If the run died again, diagnose train_log2.txt; a second unexplained death means change the launch method (schtasks or supervising wrapper), not relaunch harder.

**Blockers:** None requiring Jeff.

**Notes:**

- Run-death postmortem: first 1.5M run killed at 772k by Claude's own kill_process fired at a stale PID mid-triage; the ESRCH reply was misleading, the tree died anyway. Rule: never kill_process near a live training run; finish assessment first. Salvaged via 250k-interval checkpoints plus a new --resume flag.
- CUDA is on: torch 2.13.0+cu126, RTX 4080 Laptop GPU. Throughput 120 fps with 8 SubprocVecEnv workers.
- Terminal popups root cause: cmd /c wrappers spawn console hosts, hundreds per session. PowerShell-native commands only from now on (also stored in cross-session memory). Jeff to confirm the flashes stopped.
- Resume packaging approved by Jeff: gameplay video is the LinkedIn-legible artifact; README results table and honest negative-results writeup are the hiring-manager-legible ones (found-art search this session).
- Stale-sentinel footgun: smoke runs wrote DONE/summary.json into the real out dir; deleted before relaunch. Future smokes take --out to a scratch dir.
