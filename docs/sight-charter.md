# Sight - Project Charter

## Mission

Sight is a local-first ethical game-agent lab for learning RL and game-AI techniques and training small policies on a single older gaming laptop with 64 GB RAM and a weaker discrete GPU. Sight is a hobby and research project. Not a product. Not a startup. Not a customer-facing tool. Not a QA platform.

Success is defined by learning progress and reproducible local training, not by buyers, revenue, contracts, pilots, or downstream commercial intent.

## Scope

- Gymnasium classic control and toy-text environments (CartPole first)
- Custom Godot microgames owned by this repo (Signal Dodge and successors)
- Approved open-source single-player games, only after explicit license and automation review

## Non-Goals (explicit and permanent)

- No offerwalls
- No Freecash or any paid-engagement work
- No bot-detection evasion
- No account farming or identity spoofing
- No online multiplayer
- No live-service games
- No live commercial games
- No platforms where automation is prohibited by ToS
- No assets or tooling that could drop into a live-service cheat pipeline
- No product, customer, contract, pilot, buyer, or SaaS framing
- No proprietary commercial games (e.g., Diablo II) until an explicit legal and ToS posture has been verified for that specific game and that specific use

## Ethics Armor

- README and ethics.md state the non-goals verbatim
- License: MIT
- CONTRIBUTING.md rejects PRs targeting commercial live games or any other non-goal
- Every published artifact frames its target environment explicitly

## Hardware Profile

- Target machine: older gaming laptop, 64 GB RAM, weaker discrete GPU
- Optimize for small models, fast feedback loops, deterministic seeds, and reproducibility
- Out of scope: large foundation models, multi-GPU training, frontier-scale RL

## Stack

- Languages: Python (agent and training), GDScript (Godot games)
- RL frameworks: Stable-Baselines3 or CleanRL (whichever lands first in H1)
- Perception: state observations first, pixel observations only after the state-based loop is reliable
- Logging: structured NDJSON, deterministic seeds
- Evaluation: training curves, reward, episode length, action distribution, failure taxonomy

## Repo Architecture

```
C:\Projects\Sight\
  docs\       sight-charter.md, sight-handoff.md, ethics.md
  src\
    capture\       screen/state capture
    perception\    CV and optional small vision model
    policy\        decision logic
    controller\    action execution
    logger\        event recording
    evaluator\     metrics
    rl\            RL training loops, env adapters (added in H1+)
  tests\
  games\      Godot microgames (separate subprojects)
  runs\       episode and training artifacts
  CONTRIBUTING.md, LICENSE, README.md
```

## Roles

- GPT: plans, researches, drives technical direction
- Claude: executes, revises GPT, vetoes on evidence grounds, owns end-of-round commit and handoff update
- Grok: pulled for unresolved GPT/Claude disagreement after 2 rounds, weak-domain questions (RL internals, Godot specifics, low-spec hardware tradeoffs), and phase-gate sanity checks
- Jeff: relays, decides direction, approvals, anything touching supervision, money, legal, IP, or new target environments

## Phase Gates (hobby track)

Phases gate on technical readiness, not calendar days.

- H1: local RL baseline on Gymnasium CartPole using Stable-Baselines3 or CleanRL with NDJSON training-metric logging
- H2: reusable training and eval harness with deterministic seeds, NDJSON logs, reproducible from a config file
- H3: tiny Godot environment exposed as a Gym-style env with state observations only, no pixels
- H4: pixel observations on the same Godot environment, small CNN policy
- H5: evaluate the small CNN policy on Signal Dodge or its successor microgame

Each gate: Grok sanity check before proceeding.

## Disallowed Phase Gates (former product track, now dead)

The original P4-P6 gates are retired. They are listed here so they cannot quietly return:

- Progression video series as a substitute for measurement or buyer evidence
- Dashboard or replay tool framed as a QA harness or product surface
- Product, SaaS, or contract decision
- Buyer discovery, lived-use studies, or any external-validation gate predicated on a paying customer or budget holder

If a future Jeff wants to revive a product track, that is a separate decision under a separate charter. The hobby charter does not reserve that path.

## Success Criteria

- Working RL baseline on at least one Gymnasium environment with reproducible logs
- Working RL training loop on at least one custom Godot environment owned by this repo
- A small policy that learns on Signal Dodge or its successor microgame
- Ethical posture intact across every iteration

## Decision Authority

- Jeff-only: direction, scope changes, anything touching supervision, money, legal, IP, target-environment additions
- GPT and Claude consensus: technical implementation, phase tactics, tooling
- Claude-only within charter: execution, cleanup, routine commits on already-approved work

## Handoff Protocol

- Canonical handoff: `C:\Projects\Sight\docs\sight-handoff.md`
- Structure: phase | last commit | current task | next action | blockers | notes (max 5)
- Updated at end of every session by whoever executed
- Target cold-start resume time: 10 minutes

## Status

- Recharter from product track to hobby and research lab on 2026-04-29
- Pre-pivot product-era WIP (~696 line uncommitted Python harness slice from prior session) preserved on branch `pivot-preserve-p3-wip` at commit `a29beb3`, marked unverified
- Pivot lands on branch `pivot/hobby-rl-lab` for review before any merge to main
