# Contributing to Sight

Thanks for the interest. Sight is a small, opinionated research project. Read this before opening an issue or PR.

## Hard rejections

The following are permanent project boundaries. PRs, issues, or discussions that target these areas will be closed without review.

- **Live commercial games.** Sight does not target any commercial, live-service, multiplayer, or competitive game. No exceptions for single-player modes of commercial games where the publisher prohibits automation.
- **Bot-detection evasion.** No anti-anti-cheat work. No input humanization intended to defeat detection systems. No research into detection signatures for the purpose of avoiding them.
- **Paid-engagement platforms.** No Freecash, offerwall, survey-site, ad-view, or any platform that pays users for engagement. Permanently out of scope.
- **Account farming or identity spoofing.** No multi-account tooling, no identity rotation, no CAPTCHA solving for the purpose of automation-at-scale.
- **Cheat pipeline assets.** No code, models, or tooling whose primary plausible use is a live-service cheat. If a perception or control component looks drop-in for that, it does not land here.

See `docs/ethics.md` for the verbatim non-goals list from the project charter.

## What is in scope

- Custom Godot micro-games authored for this project
- Open-source games where automation is explicitly permitted (0 A.D. single-player is the motivating example)
- Formal RL benchmark environments via Gymnasium
- Perception, policy, controller, logging, and evaluation code that targets the above
- Tooling, tests, documentation, and measurement improvements

## PR expectations

- Small, focused diffs
- Explicit statement of which in-scope target the change supports
- Tests or measurements where behavior changes
- No new dependencies without justification
- No scraped assets, no copyrighted game content, no binaries that cannot be rebuilt from source

## Issues

Bug reports and measurement-improvement ideas welcome. Feature requests that cross the non-goals above will be closed with a link to this file.
