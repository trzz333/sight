# Sight - Project Charter

## Mission

Build a local-first AI game-agent lab that perceives game state, decides actions, executes them, and improves through measurement. Ship the framework as a product and portfolio piece while maintaining unambiguous ethical and legal hygiene.

## Scope

- Custom Godot micro-games (primary; Jeff owns game and assets)
- Open-source games where automation is permitted (e.g., 0 A.D. single-player)
- Formal RL benchmark environments via Gymnasium (credibility layer)
- Council integration as optional decision layer (phase 2+)

## Non-Goals (explicit and permanent)

- No live commercial game automation
- No Freecash / offerwall / paid-engagement work, ever
- No bot-detection evasion
- No account farming or identity spoofing
- No work on any platform where automation is prohibited by ToS
- No assets or tooling that could drop into a live-service cheat pipeline

## Ethics Armor

- README and `ethics.md` state the non-goals verbatim
- License: MIT
- [CONTRIBUTING.md](http://CONTRIBUTING.md) rejects PRs targeting commercial live games
- Every progression video frames its legitimate target environment explicitly

## Stack

- Languages: Python (agent), GDScript (Godot games)
- Perception: OpenCV + template matching first; small vision model (moondream via llama.cpp CPU) when genuinely needed
- Policy: rule-based first; ML/RL layered in once loop is reliable
- Decision layer (phase 2+): Council integration
- Logging: structured NDJSON (pattern from Workbench)
- Evaluation: win rate, episode length, action distribution, failure taxonomy

## Repo Architecture

```
C:\Projects\Sight\
  docs\       sight-charter.md, sight-handoff.md, ethics.md
  src\
    capture\       screen/state capture
    perception\    CV and optional vision model
    policy\        decision logic
    controller\    action execution
    logger\        event recording
    evaluator\     metrics and charts
    council\       optional deliberation (phase 2+)
  tests\
  games\      Godot micro-games (separate subprojects)
  runs\       episode artifacts
  CONTRIBUTING.md, LICENSE, README.md
```

## Roles

- GPT: plans, researches, drives technical direction
- Claude: executes, revises GPT, vetoes on evidence grounds, owns end-of-round commit + handoff update
- Grok: pulled for (a) unresolved GPT/Claude disagreement after 2 rounds, (b) domains where both are weak (RL internals, Godot specifics, CV tradeoffs on low-spec hardware), (c) phase-gate sanity checks
- Jeff: relays, synthesizes, ground-truth backstop, decides on direction, approvals, and anything touching supervision

## Phase Gates (90 days)

- P1 (days 1-14): Godot game chosen; repo scaffolded with charter and ethics
- P2 (days 15-30): Minimal agent loop (capture -&gt; rules -&gt; controller -&gt; logger)
- P3 (days 31-45): Measurement layer (evaluator, failure taxonomy)
- P4 (days 46-60): Progression video series launched; 2-3 milestones published
- P5 (days 61-75): Dashboard / replay tool (QA-harness shape)
- P6 (days 76-90): Product decision (SaaS, contract pitch, or both)

Each gate: Grok sanity check before proceeding.

## Success Criteria

- Technical: working agent loop on &gt;=1 Godot game with reliable measurement
- Content: 3+ published progression videos with coherent narrative
- Product shape: at least one of {paying customer, signed contract, serious inbound interest}

## Decision Authority

- Jeff-only: direction, first commit approval, first public release, anything touching supervision, money/legal/IP
- GPT+Claude consensus: technical implementation, phase tactics, tooling
- Claude-only within charter: execution, cleanup, routine commits on already-approved work

## Handoff Protocol

- Canonical handoff: `C:\Projects\Sight\docs\sight-handoff.md`
- Structure: phase | last commit | current task | next action | blockers | &lt;=5 notes
- Updated at end of every session by whoever executed
- Target: 10-minute cold-start resume

## Status

- Draft written by Claude (web) 2026-04-23
- Pending: GPT stress-test, convergence, first commit by Sight-project Claude
