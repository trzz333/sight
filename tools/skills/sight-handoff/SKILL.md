---

## name: sight-handoff description: End-of-session handoff for the Sight project. Trigger when the user says "/sight-handoff", "/handoff", "handoff", "wrap up", "push and handoff", "session done", or similar, or proactively when context is heavy (30+ tool calls, long conversation, a significant phase just completed). This skill updates docs/sight-handoff.md in the canonical terse schema (phase, last commit, current task, next action, blockers, up to 5 notes), commits and pushes to trzz333/sight on GitHub so GPT can see it, and ALWAYS outputs two separate bootstrap messages (one for Claude, one for GPT) each in its own copy/paste markdown code block. Both messages every time. Never combine them. Never skip GPT because the current session was Claude-facing. Trigger proactively when context is heavy, do not wait to be asked.

# Sight Handoff

Every invocation produces:

1. Updated `docs\sight-handoff.md` matching the locked schema
2. Clean push to `origin/main`
3. A Claude bootstrap message in its own fenced code block
4. A GPT bootstrap message in its own fenced code block

Items 3 and 4 are two SEPARATE blocks. Always. If tempted to produce only one because the session felt Claude-only, or to combine to save space, do neither. Two blocks, every time, labeled outside the fence.

Style rules carry over: terse, tool-first, no em dashes, minimal colons outside the schema, Windows absolute paths.

## 1. Summarize this session

One to six bullets. Factual, terse. Commits, decisions, work completed, new limitations surfaced. No padding.

## 2. Run the handoff script

Canonical: `C:\Projects\Sight\tools\handoff_update.py`

The script does the deterministic mechanical work in one call: rewrites only the named schema fields via per-field `re.subn` (no whole-file rewrite, no transcription drift), commits everything dirty under a substantive message, pushes, refreshes the `**Last commit:**` line with the new short hash, commits the chore, pushes again. Returns JSON with both hashes plus the substantive subject.

### Invocation

Write the new field values to a JSON file via `python -i` REPL. Inline `python -c "..."` and `git commit -m "..."` are unreliable through Desktop Commander (cmd.exe word-splits the quoted args). Tempfile + `git commit -F` is the only reliable path.

```python
# In python -i
import json
payload = {
    "phase":          "<phase + short descriptor>",
    "current_task":   "<2-4 sentences>",
    "next_action":    "<one concrete move>",
    "blockers":       "<none, or specifics>",
    "notes":          ["note 1", "note 2", "note 3"],
    "commit_subject": "handoff: <72-char subject>",
    "commit_body":    "<optional 1-3 sentence body>",
}
open(r"C:\Projects\Sight\handoff_update.json", "w", encoding="utf-8").write(json.dumps(payload))
```

Then in cmd.exe:

```
cd C:\Projects\Sight
C:\Users\maste\AppData\Local\Python\bin\python.exe tools\handoff_update.py --input handoff_update.json
del handoff_update.json
```

The script prints JSON like:

```
{"substantive_hash": "787b9f0", "refresh_hash": "67fa151", "subject": "handoff: ..."}
```

Capture `substantive_hash`. Both bootstrap messages reference it (NOT the refresh hash).

### Schema (locked)

The script preserves this schema exactly. Do not add new top-level sections or restructure.

```
# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** <phase id and short descriptor>

**Last commit:** <short hash> <one-line summary>

**Current task:** <2-4 sentences describing where work actually is>

**Next action:** <single clearest next move. No menus.>

**Blockers:** <none, or specific items that need Jeff's attention>

**Notes:**

- <note 1>
- ...up to 5 total
```

Field rules: short hash is 7 chars; current task is present-tense; next action is one concrete move (no "A or B, Jeff picks"); blockers say "None" when clean; max 5 notes.

### Pre-script checks

Before invoking the script:

- `git status --short` to confirm what will be picked up by the substantive commit. If unrelated work is dirty, commit it separately first.
- The script does NOT enforce a clean tree; it bundles everything dirty into the substantive commit. Bundling unrelated work into the handoff commit violates the project pattern.

### Manual fallback

If the script fails (anchor not found, JSON malformed, push rejected, etc.):

1. `python -i` REPL with `re.subn` per field on `docs\sight-handoff.md`. Anchors are the bold `**Field:**` markers. Each field gets exactly one match. Atomic via `os.replace`. Never whole-file rewrite.
2. Write commit message to `COMMIT_MSG.txt` via the same REPL.
3. `git add -A`, `git commit -F COMMIT_MSG.txt`, `git push origin main`. Capture the short hash.
4. Second `re.subn` to update `**Last commit:**` with the real hash. `chore: refresh handoff hash` commit. Push.

## 3. Output BOTH bootstrap messages

Non-negotiable. Two separate fenced code blocks. Plain fences, no language hint. Labeled outside the fence with `**Claude bootstrap**` / `**GPT bootstrap**`. No combining. No substituting.

### 3a. Claude bootstrap template

```
Sight session resume on <hostname>. Orient and stop.
1. cd C:\Projects\Sight
2. Read docs\sight-charter.md and docs\sight-handoff.md end to end.
3. git log --oneline -5; git status (confirm clean, HEAD = <short hash> or newer, remote in sync)
4. Report back:
   - Current phase
   - Last commit hash and one-line summary
   - Current task
   - Next action
   - Anything in blockers that needs Jeff's attention
<one line: Do NOT start new work plus any current-phase guardrails. Replace with a specific queued execution task only if one is explicitly ready.>

Operating reminders for this project:
- Terse, tool-first output. No narration of routine operations. No em dashes. Minimal colons.
- Windows paths, absolute. Desktop Commander's start_process mangles `git commit -m "..."` and inline `python -c "..."` (cmd.exe word-splits the quoted args). Use `python -i` REPL for code, tempfile + `git commit -F COMMIT_MSG.txt` for commits.
- For handoffs use C:\Projects\Sight\tools\handoff_update.py with a JSON input file. Manual fallback in the skill.
- GPT leads planning; Claude executes, revises, commits, handoffs; Grok on trigger; Jeff decides direction.
- Ethics are hard constraints per docs\ethics.md and docs\sight-charter.md non-goals. No live commercial games, no bot-detection evasion, no Freecash, no account farming, no cheat-pipeline assets. Flag and stop if a request drifts.
- One phase per prompt. Desktop Claude has no continue button; tool-call budget deaths are silent. Chunk accordingly.
- Commit + push at end of any substantive session. Update docs\sight-handoff.md via the script. No "A or B, Jeff picks."
End with "No Jeff action required" unless genuinely blocked.
```

### 3b. GPT bootstrap template

```
Sight session update for GPT.

Read these first in the trzz333/sight GitHub repo:
- docs/sight-charter.md
- docs/sight-handoff.md
<any other doc directly relevant to the open decision>

State: <one-sentence state summary>
Latest commit: <short hash> <subject>
Phase: <phase id + descriptor>

Open question for you to think through before the next execution round:
<the architecture, sequencing, or parameter-choice question Jeff will eventually decide but where GPT's framing helps. One paragraph max.>

Role split stands: you plan and research, Claude executes, Grok is pulled only for charter-defined triggers, Jeff decides direction.
```

### 3c. Why two messages

Both messages are orient signals, not briefings. The canonical handoff in the repo holds the detail. The messages exist so Jeff can paste into fresh Claude and GPT sessions without rewriting context. Even if this session was 100% Claude work, GPT still needs orienting because GPT is the planner and the next session may route through GPT first.

## 4. End with next steps

Two lines max:

- Next best step for Claude: usually nothing until the next conversation
- Next best step for Jeff: paste one or both bootstrap messages, or run any verification flagged

If a real blocker is flagged, surface it once more in one sentence. Otherwise end with `No Jeff action required.`

## Do not

- Produce only one bootstrap message. Both, every invocation, separate blocks.
- Combine the Claude and GPT messages into one block.
- Skip the GPT message because the session felt Claude-only.
- Whole-file rewrite the handoff doc. The script applies per-field `re.subn`; the manual fallback also uses `re.subn`. Whole-file rewrites cause silent transcription drift on unchanged fields.
- Use `git commit -m "..."` through Desktop Commander. Quote-mangling silently splits the message into pathspecs. Always `git commit -F COMMIT_MSG.txt`.
- Use inline `python -c "..."` for non-trivial code through Desktop Commander. Quote-mangling breaks it. Use `python -i` REPL or run a script file.
- Widen scope during a handoff. Record what happened, do not also refactor or add features.
- Bundle unrelated work into the handoff commit. Commit unrelated changes separately first.
- Exceed the schema. Five notes max. No new top-level sections.
- Invent state. If something is unclear, surface it as a blocker.
- Use em dashes anywhere.
- Produce an "A or B, Jeff picks" Next action. State Jeff's decision under Blockers; Next action is the deterministic prep.
- Generate bootstrap messages before the push. Push first so messages truthfully reference the pushed hash.
- Touch the charter (`docs\sight-charter.md`) as part of a handoff unless a charter amendment is the explicit work of the session.

## Success criteria

- TWO separate fenced code blocks: one Claude, one GPT. Non-negotiable.
- Both reference the substantive (non-chore) short hash returned by the script.
- `docs\sight-handoff.md` matches the locked schema and reflects real state.
- `git status` clean, `origin/main` in sync.
- Next steps explicit and minimal.

## Updating this skill

This file lives at `C:\Projects\Sight\tools\skills\sight-handoff\SKILL.md`. The Claude.ai user-skill copy at `/mnt/skills/user/sight-handoff/SKILL.md` is read-only inside the agent container; the canonical source is here in the repo. To deploy a change so future Claude sessions pick it up, upload this file via Claude.ai Settings -> Skills -> sight-handoff -> replace.
