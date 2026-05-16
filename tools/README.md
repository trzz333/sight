# Sight tools

Repo-local helper scripts. Run from `C:\Projects\Sight`.

## handoff_update.py

Atomic Sight handoff: edit `docs\sight-handoff.md` per-field, commit, push, refresh hash, commit, push. See header docstring for input JSON schema and exit codes. Canonical workflow lives in the user skill at `tools\skills\sight-handoff\SKILL.md`.

## h5_smoke_parse.py

Parser for the H5 amendment pre-training smoke. Reads `python.ndjson` from a shaped and a default (`reward_shaping: none`) seeded_random rollout and evaluates the hard pass criteria from `docs\h5-reward-amendment-proposal.md` section 10 plus the tightening recorded in `docs\h5-reward-amendment-smoke-evidence.md`. Volatile per-run identity fields (`ts_unix`, `run_id`, `godot_pid`, `tcp_port`, `episode_id`) are normalized out before the default-path schema check. Exit codes: 0 all-pass, 1 criteria-fail, 2 missing input.

## skills/

Authored copies of [Claude.ai](http://Claude.ai) user skills that target this repo. The mounted `/mnt/skills/user/<name>/SKILL.md` versions inside the agent container are read-only; deploy changes by uploading the files here via [Claude.ai](http://Claude.ai) Settings -&gt; Skills.

Current skills tracked:

- `sight-handoff/SKILL.md`
