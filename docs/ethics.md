# Sight - Ethics

These are the permanent non-goals of the Sight project. They are reproduced verbatim from `docs/sight-charter.md`. They are not subject to revision based on performance, opportunity, or schedule. A contribution, issue, or direction that crosses any of these lines is out of scope regardless of who proposes it.

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
- CONTRIBUTING.md rejects PRs targeting commercial live games
- Every progression video frames its legitimate target environment explicitly

## Enforcement

The maintainer closes PRs and issues that cross these boundaries without debate. If a legitimate in-scope contribution is mistakenly flagged, open a short issue referencing the specific line above that it does not violate. The burden of establishing scope is on the contributor.

If a proposed direction sits near a line but does not clearly cross it, the default answer is no. Ambiguity resolves toward refusal.

## Approved Target Environments (per-environment review record)

Per the charter, approved open-source single-player games require an explicit license and automation review before adoption. Record of reviews:

### ViZDoom (approved by Jeff, 2026-07-07)

- Platform: ViZDoom 1.3.0, a Farama Foundation research platform built explicitly for RL and learning-from-demonstration research on the 1993 Doom mechanics via the ZDoom engine.
- License: ViZDoom original code MIT. ZDoom engine components carry mixed upstream licenses; usage here is local research, no redistribution of engine binaries beyond the unmodified pip package.
- Assets: bundled research scenario WADs plus freedoom1/freedoom2 (free assets). No commercial Doom WADs are used or required.
- Automation posture: automation is the platform's designed purpose. Offline, local, single-player scenarios only. No multiplayer, no deathmatch against humans, no live services, no ToS to violate.
- Scope within Sight: human demonstrations recorded locally by the project owner, behavioral cloning, and RL finetuning on bundled or repo-owned scenarios.
