---
name: sight-handoff
description: End-of-session handoff for the Sight project. Trigger when the user says "/sight-handoff", "/handoff", "handoff", "wrap up", "push and handoff", "session done", or similar, or proactively when context is heavy (30+ tool calls, long conversation, a significant phase just completed). This skill updates docs/sight-handoff.md in the canonical terse schema (phase, last commit, current task, next action, blockers, up to 5 notes), commits and pushes to trzz333/sight on GitHub so GPT can see it, and ALWAYS outputs two separate bootstrap messages (one for Claude, one for GPT) each in its own copy/paste markdown code block. Both messages every time. Never combine them. Never skip GPT because the current session was Claude-facing. Trigger proactively when context is heavy, do not wait to be asked.
---

# Sight Handoff

Every invocation of this skill produces the SAME four deliverables:

1. An updated `docs\sight-handoff.md` matching the locked schema.
2. A clean git push to `origin/main`.
3. A Claude bootstrap message in its own fenced code block.
4. A GPT bootstrap message in its own fenced code block.

Three and four are two SEPARATE blocks. Always. If you are tempted to produce only one because the session felt Claude-only, or to combine them to save space, do neither. Two blocks, every time, labeled outside the fence so Jeff can copy cleanly.

Style rules for the project carry into the handoff output: terse, tool-first, no em dashes, minimal colons outside the schema, Windows absolute paths. Do not narrate routine operations.

## 1. Summarize this session

List what was accomplished. Factual, terse. Commits made with short hashes, decisions recorded, work completed, new limitations surfaced. No narrative padding. One to six bullets.

## 2. Update and push the canonical handoff

### Schema (locked)

The canonical handoff lives at:

```
C:\Projects\Sight\docs\sight-handoff.md
```

Preserve this schema exactly. Do not add new top-level sections. Do not restructure.

```
# Sight - Session Handoff

Canonical handoff document. Updated at the end of every session by whoever executed. Target cold-start resume time: 10 minutes.

---

**Phase:** <phase id and short descriptor, e.g. "P2 in progress (agent loop)">

**Last commit:** <short hash> <one-line summary>

**Current task:** <2-4 sentences describing where work actually is>

**Next action:** <single clearest next move, whether for Jeff, GPT, or Claude. No menus.>

**Blockers:** <none, or specific items that need Jeff's attention>

**Notes:**

- <note 1>
- <note 2>
- <...up to 5 total>
```

### Field rules

- **Phase** matches charter phase gates (P0/P1/P2/...). Short parenthetical only if it helps disambiguate.
- **Last commit** uses the short hash (first 7 chars) and the first line of the commit message. Update AFTER the push in step 2b.
- **Current task** is present-tense, what-is-actually-true. Not a plan.
- **Next action** is a single concrete move. No "A or B, Jeff picks." If the decision is genuinely Jeff's, state the decision point in Blockers and put the deterministic prep step in Next action.
- **Blockers**: write "none" when clean. Only real blockers.
- **Notes**: 5 max. Prune. If something belongs in the charter or a spec, put it there.

### 2a. Update the file

Read the current handoff first. Rewrite only the fields above. Do not add new sections.

Files touched this session that have not been committed must be committed in the handoff commit (if part of the same unit of work) or separately first. Do not bundle unrelated work.

### 2b. Commit and push

Use Desktop Commander. Do not describe commands, execute them.

```
cd C:\Projects\Sight
git status --short
git add -A
$msg = @"
<concise subject, <=72 chars>

<optional 1-3 sentence body. No bullet lists.>
"@
# Do NOT use Out-File -Encoding utf8 on Windows PowerShell 5.1: it writes a BOM
# which leaks as an invisible character at the start of the commit subject.
# Use WriteAllText (no BOM) or PowerShell 7's utf8NoBOM encoding.
[System.IO.File]::WriteAllText("$(Get-Location)\COMMIT_MSG.txt", $msg)
git commit -F COMMIT_MSG.txt
Remove-Item COMMIT_MSG.txt
git push origin main
git rev-parse --short HEAD
git log --oneline -3
```

Capture the short hash. BOTH bootstrap messages in step 3 reference it.

Then update the `**Last commit:**` line of the handoff doc with the new hash. Either amend or add a trailing `chore: refresh handoff hash` commit, whichever keeps the repo clean.

If `git status` is not clean at the end, stop and fix before producing bootstrap messages.

## 3. Output BOTH bootstrap messages

Non-negotiable. Every invocation produces TWO separate markdown fenced code blocks, in this order:

1. Message for next Claude session
2. Message for GPT

Each block is standalone and copy-pasteable. Each is labeled OUTSIDE the block with a plain heading like `**Claude bootstrap**` or `**GPT bootstrap**`, not inside. No preamble inside blocks. No em dashes anywhere. No combining. No substituting one for the other.

### Shared structure (both messages)

Both bootstraps follow the same shape:

1. One-line session frame ("Sight execution round for Claude on <host>." / "Sight planning round for GPT.").
2. Numbered `Next steps for you to ...` block leading the message. Concrete, prioritized, tied to current state. Not a menu.
3. `Handoff subtext:` block at the bottom with state, latest commit, phase, role split, and operating reminders.

Reason: the next model that reads this message needs the WHAT-TO-DO in its first scan. Handoff detail is orienting background, not the headline.

### 3a. Claude bootstrap template

Fill in the placeholders. ~22-32 lines. Plain fenced block, no language hint.

```
Sight execution round for Claude on <hostname>.

Next steps for you to execute when Jeff kicks this session:

1. Orient first. cd C:\Projects\Sight. Read docs\sight-charter.md and docs\sight-handoff.md end to end. Run git log --oneline -5 and git status. Confirm HEAD is <short hash> or newer and origin/main is in sync. Report phase, last commit, current task, next action, anything in blockers that needs Jeff.

2. <Specific execution task for this session, derived from Next action in the handoff. Be concrete about commands, paths, and the verification criterion. If the task is conditional on something Jeff needs to do first, say so.>

3. <Fallback path if the step-2 precondition is not met. Usually: do NOT scaffold ahead of phase, surface the blocker to Jeff, and stop.>

4. <Optional: handling for parallel work from GPT or other prompts. Reinforce the one-phase-per-prompt chunking rule if relevant.>

5. End of session: run /sight-handoff. Produce both bootstrap messages (Claude and GPT), every time, subtext style.

Handoff subtext:
- State: <one-to-two-sentence state summary>
- Latest commit: <short hash> <subject>
- Phase: <phase id + descriptor>
- Role split: GPT plans; Claude executes, revises, commits, handoffs; Grok on charter-defined triggers; Jeff decides direction.
- Operating reminders: terse, tool-first output, no narration of routine operations, no em dashes, minimal colons, Windows absolute paths, one phase per prompt, ethics hard constraints per docs\ethics.md, flag and stop if a request drifts.
- End every response with "No Jeff action required" unless genuinely blocked.
```

### 3b. GPT bootstrap template

Fill in the placeholders. ~18-26 lines. Plain fenced block. No implementation instructions. GPT plans and researches.

```
Sight planning round for GPT.

Next steps for you to work through before Jeff kicks the next execution round:

1. <First planning or architecture question. One short paragraph. Name tradeoffs, ask for a pick.>

2. <Second. Typically evaluator, metrics, or data-schema design.>

3. <Third. Typically module boundaries, file layout, or scaffold sequencing.>

4. <Fourth, optional. Post-run parameter review or sequencing decisions that depend on the next live run.>

Handoff subtext:
- Read docs/sight-charter.md and docs/sight-handoff.md in the trzz333/sight GitHub repo.
- State: <one-sentence state summary>
- Latest commit: <short hash> <subject>
- Phase: <phase id + descriptor>
- Role split stands: you plan and research, Claude executes, Grok on charter-defined triggers, Jeff decides direction.
```

### 3c. Purpose

Both messages are orient signals, not briefings. The canonical handoff in the repo holds the detail. The messages exist so Jeff can paste into fresh Claude and GPT sessions without rewriting context. Even if this session was 100% Claude work, GPT still needs orienting because GPT is the planner and the next session may route through GPT first.

## 4. End with next steps

Two lines at most:

- Next best step for Claude: usually nothing until the next conversation.
- Next best step for Jeff: paste one or both bootstrap messages into fresh sessions, or run any verification the handoff flagged.

If the handoff flagged a real blocker, surface it once more in one sentence. Otherwise end with `No Jeff action required.`

## Do not

- Produce only one bootstrap message. Both, every invocation, separate blocks.
- Combine the Claude and GPT messages into one block.
- Skip the GPT message because the session felt Claude-only. GPT is the project planner; it needs orienting every cycle regardless of who actually typed in this session.
- Widen scope during a handoff. Record what happened, do not also refactor or add features.
- Bundle unrelated work into the handoff commit. Commit unrelated changes separately first.
- Exceed the schema. Five notes max. No new top-level sections in the handoff doc.
- Invent state. If something is unclear, surface it as a blocker.
- Use em dashes anywhere in the handoff doc or bootstrap messages.
- Produce an "A or B, Jeff picks" Next action. If Jeff's decision, state it under Blockers. Next action remains the deterministic prep.
- Generate bootstrap messages before the push. Push first, then generate, so messages can truthfully reference the pushed hash.
- Touch the charter (`docs\sight-charter.md`) as part of a handoff unless a charter amendment is the explicit work of the session.

## Success criteria

- TWO separate fenced code blocks exist in the final response: one Claude, one GPT. Non-negotiable. This is the failure mode this skill exists to prevent.
- Both blocks reference the pushed short hash and are internally consistent with the updated handoff doc.
- `docs\sight-handoff.md` matches the locked schema and reflects real current state.
- `git status` clean, `origin/main` in sync, hash captured.
- Next steps explicit and minimal.