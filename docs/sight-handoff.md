# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** VZD-3 in flight (deadly_corridor PPO teacher training; VZD-2 distillation and all prior results stand)

**Last commit:** 715028f vzd: generalize PPO trainer/watcher to --env-id and --doom-skill for VZD-3

**Current task:** VZD-3 baseline run LIVE, launched 2026-07-12 ~10:40 CDT: vzd_ppo_train.py --env-id VizdoomDeadlyCorridor-v1 --steps 1500000, skill 5 (cfg default, the honest baseline), out runs\vzd\ppo_deadly_corridor, log runs\vzd\ppo_deadly_corridor_train.log. At 57k steps it ran ~113 fps, ETA ~3.5-4h. Both smokes passed pre-launch: corridor skill-3 smoke (reward ~767 scale sane, doom_skill recorded in summary) and defend default smoke (dirs and schema unchanged), scratch dirs runs\vzd\_smoke_*. Generalization committed at 715028f.

**Next action:** When DONE sentinel appears in runs\vzd\ppo_deadly_corridor: read summary.json (30-ep mean/IQM), judge against training curve (no pre-registered numeric bar for this scenario; smoke-untrained was ~767 at skill 3), record clip with vzd_ppo_watch.py --env-id VizdoomDeadlyCorridor-v1 --record, update README results table. If the run stalled, pre-registered levers in priority order: (1) --doom-skill curriculum 1->3->5, (2) scenario reward config (living_reward/death_penalty), (3) reward normalization/scaling (VecNormalize or wrapper) - added this session because early train stats showed approx_kl ~0.39 and clip_fraction ~0.55 at clip_range 0.1 with value_loss ~7e3, consistent with the ~1000x reward scale vs defend. NOT more steps.

**Blockers:** None requiring Jeff.

**Notes:**

- Monitor: http://127.0.0.1:8791/monitor.html now serves runs\vzd ROOT (new runs\_launch_monitor_vzd_root.py, server pid 33916, old ppo_defend-rooted server killed). runs\vzd\monitor.html points at the corridor log/summary and decodes UTF-16 PowerShell logs (BOM sniff + null-byte heuristic in JS).
- DC relay failure mode seen today: relay can return "No approval received"/"No devices available" while the device still EXECUTES the call (verified in ~/.claude-server-commander/tool-history.jsonl). If relay errors return, treat DC as write-only: issue commands with file redirects, read results via Filesystem MCP. Registry last_seen is unreliable.
- Pairing rule stands: rollout-trained students eval with --obs rgb2; human-demo students native.
- deadly_corridor ground truth: env id VizdoomDeadlyCorridor-v1, cfg skill 5, death_penalty 100, no living_reward, reward = WAD distance shaping toward armor, 7 buttons -> Discrete(8), timeout 2100 tics.
- DC transport still dies on blocking calls >= ~4 min; launch with *> file logging, poll with instant reads only. PowerShell *> logs are UTF-16.
