# Sight tools

Repo-local helper scripts. Run from `C:\Projects\Sight`.

## handoff_update.py

Atomic Sight handoff: edit `docs\sight-handoff.md` per-field, commit, push, refresh hash, commit, push. See header docstring for input JSON schema and exit codes. Canonical workflow lives in the user skill at `tools\skills\sight-handoff\SKILL.md`.

## skills/

Authored copies of [Claude.ai](http://Claude.ai) user skills that target this repo. The mounted `/mnt/skills/user/<name>/SKILL.md` versions inside the agent container are read-only; deploy changes by uploading the files here via [Claude.ai](http://Claude.ai) Settings -&gt; Skills.

Current skills tracked:

- `sight-handoff/SKILL.md`
