# Sight Skills

Project-specific Claude skills for Sight. Version-controlled in this repo so they can be reinstalled from any machine.

## Skills

- `sight-handoff/` - end-of-session handoff. Updates `docs/sight-handoff.md` in the locked schema, commits and pushes, emits two bootstrap messages (Claude, GPT).

## Install

Skills are installed through the Anthropic skills mechanism (upload via Claude settings, or whatever per-host process is in use). The canonical copy lives here; installed copies on any given machine should be synced from this directory.

Trigger phrase for the handoff skill: `/sight-handoff`, `/handoff`, or any natural wrap-up language like "push and handoff" or "session done."
