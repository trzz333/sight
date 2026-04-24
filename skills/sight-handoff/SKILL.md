---
name: sight-handoff
description: End-of-session handoff for the Sight project. Trigger when the user says "/sight-handoff", "/handoff", "handoff", "wrap up", "push and handoff", "session done", or similar, or proactively when the context is heavy (30+ tool calls, long conversation, a significant phase just completed). This skill updates docs/sight-handoff.md in the canonical terse schema (phase, last commit, current task, next action, blockers, <=5 notes), commits and pushes to trzz333/sight on GitHub so GPT can see it, and outputs two separate bootstrap messages (one for Claude, one for GPT) each in its own copy/paste markdown code block. Use this instead of ad-hoc handoff writing whenever a Sight session is being wrapped. Trigger proactively when context is heavy, do not wait to be asked.
---

# Sight Handoff

End-of-session handoff for the Sight project. Four phases: summarize, update-and-push, produce two bootstrap messages, end with next steps.

Style rules for this project carry into the handoff output: terse, tool-first, no em dashes, minimal colons outside the schema, Windows absolute paths. Do not narrate routine operations.

## 1. Summarize this session

List what was accomplished. Factual, terse. Commits made with short hashes, decisions recorded, work completed, new limitations surfaced. No narrative padding. One to six bullets.

## 2. Update and push the canonical handoff

The canonical handoff is a single file at a fixed path:

```
C:\Projects\Sight\docs\sight-handoff.md
```

It follows a locked schema. Preserve it exactly.

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

Rules for the fields:

- **Phase** matches the charter phase gates (P0/P1/P2/...). Append a short parenthetical only if it helps disambiguate.
- **Last commit** uses the short hash (first 7 chars) and the first line of the commit message. Update this AFTER the push in step 2b, not before.
- **Current task** is present-tense, what-is-actually-true. Not a plan.
- **Next action** is a single concrete move. No "A or B, Jeff picks." If a decision is genuinely Jeff's, state the decision point in Blockers and put the deterministic prep step in Next action.
- **Blockers**: write "none" when clean. Only list real blockers, not aspirations.
- **Notes**: 5 max. Prune. If something belongs in the charter or a spec doc, put it there, not here. Keep primary-host line, any cross-machine state, chunking reminders, and ethics armor status if relevant.

### 2a. Update the file

Read the current handoff first. Rewrite only the fields above. Do not restructure. Do not add new top-level sections.

Any other files touched this session that have not been committed yet must either be committed in this handoff commit (if they are part of the same unit of work) or committed separately first. Do not bundle unrelated work.

### 2b. Commit and push

Use Desktop Commander. Do not describe commands, execute them. PowerShell here-strings are the cleanest way to write commit messages with both a subject and a body.

```
cd C:\Projects\Sight
git status --short
git add -A
$msg = @"
<concise subject, <=72 chars>

<optional 1-3 sentence body; what shipped and why. No bullet lists.>
"@
# Do NOT use Out-File -Encoding utf8 on Windows PowerShell 5.1: it writes a BOM
# which ends up as an invisible character at the start of the commit subject.
# Use WriteAllText (no BOM) or PowerShell 7's utf8NoBOM encoding.
[System.IO.File]::WriteAllText("$(Get-Location)\COMMIT_MSG.txt", $msg)
git commit -F COMMIT_MSG.txt
Remove-Item COMMIT_MSG.txt
git push origin main
git rev-parse --short HEAD
git log --oneline -3
```

Capture the resulting short hash. Both bootstrap messages reference it. Go back and update the **Last commit** line of the handoff doc with the new hash, then amend or add a follow-up commit if the handoff's own Last commit line would otherwise be stale. One common pattern: commit code + handoff together, then update Last commit in a trailing `chore: refresh handoff hash` commit. Use whichever keeps the repo clean.

If `git status` is not clean at the end, stop and fix before producing bootstrap messages.

## 3. Output two bootstrap messages

Produce two separate markdown code blocks in the chat, each standalone and copy-pasteable. Label each clearly outside the block. No preamble inside the blocks. No em dashes.

### Message for the next Claude session (~14-20 lines)

Must include:

- One line: `Read C:\Projects\Sight\docs\sight-charter.md and docs\sight-handoff.md first.`
- The Sight role split in one sentence (GPT plans, Claude executes, Grok on trigger, Jeff decides).
- Ethics armor line: `Ethics hard constraints per docs\ethics.md. Flag and stop if a request drifts.`
- Current phase and what the last commit shipped (short hash).
- The next move. If a Claude task, say what. If waiting on Jeff or GPT, say that clearly.
- Explicit: `Do not start new work without orienting first. Do not propose game concepts or scaffold ahead of phase.` (Unless a specific execution task is queued, in which case state it.)
- End with: `End with "No Jeff action required" unless genuinely blocked.`

### Message for GPT (~10-14 lines)

Must include:

- One line: `Read docs/sight-charter.md and docs/sight-handoff.md in the trzz333/sight GitHub repo.`
- One-sentence state summary.
- Latest short hash and what it shipped.
- Current phase.
- The open question or decision GPT should think about next. Architecture, sequencing, parameter choices, or evaluator design, per the role split.
- No implementation instructions. GPT plans and researches; Claude executes.

Both messages are orient signals, not briefings. The canonical handoff in the repo holds the detail.

## 4. End with next steps

Two lines at most:

- Next best step for Claude: usually nothing until the next conversation.
- Next best step for Jeff: paste one or both bootstrap messages into fresh sessions, or run whatever verification the handoff flagged.

If the handoff flagged a real blocker, surface it one more time here in one sentence. Otherwise end with `No Jeff action required.`

## Do not

- Widen scope during a handoff. Record what happened, do not also refactor or add features.
- Bundle unrelated work into the handoff commit. Commit unrelated changes separately first.
- Exceed the schema. Five notes maximum. No new top-level sections in the handoff doc.
- Invent state. If something is unclear, surface it as a blocker, not a guess.
- Use em dashes anywhere in the handoff doc or bootstrap messages.
- Produce an "A or B, Jeff picks" Next action. If the decision is Jeff's, it goes under Blockers as a decision point and Next action is the deterministic prep.
- Generate bootstrap messages before the push. Push first, then generate, so the messages can truthfully reference the pushed hash.
- Touch the charter (`docs\sight-charter.md`) as part of a handoff unless a charter amendment is the explicit work of the session.

## Success criteria

- `docs\sight-handoff.md` matches the locked schema and reflects real current state.
- `git status` clean, `origin/main` in sync, hash captured.
- Two copy-paste code blocks rendered in chat, one for Claude, one for GPT, each self-contained and labeled.
- Next steps explicit and minimal.
