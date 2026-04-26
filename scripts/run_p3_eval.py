"""P3 eval runner scaffold.

Refuses to start when SIGHT_TCP_IGNORE_DEATH is set in the inherited environment
(spec invariant from docs/sight-p3-metrics.md). Live Godot execution is NOT
implemented in this slice; this script currently supports only --from-artifacts
mode, which loads existing episode NDJSON files, runs the loader-to-aggregator
plumbing, and writes summary.json. The live mode hook will land in a later slice
once GPT plans it.

Future live-execution layout (not implemented here):

    runs\\eval\\<run_id>\\episodes\\<episode_id>\\godot.ndjson
    runs\\eval\\<run_id>\\episodes\\<episode_id>\\python.ndjson
    runs\\eval\\<run_id>\\summary.json

CLI:

    python scripts/run_p3_eval.py --from-artifacts runs/eval/<run_id> \\
        --actions-budget 300 --out-dir runs/eval/<run_id>

The refusal guard runs unconditionally before any subcommand.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Mapping


# Spec invariant: this env var, set non-empty, blocks any P3 metric run.
IGNORE_DEATH_ENV_VAR: str = "SIGHT_TCP_IGNORE_DEATH"  # spec invariant: refusal env-var name


class IgnoreDeathRefusal(SystemExit):
    """Raised when the refusal guard rejects an inherited environment."""

    EXIT_CODE: int = 2

    def __init__(self, message: str) -> None:
        super().__init__(self.EXIT_CODE)
        self.message = message


def refuse_if_ignore_death(env: Mapping[str, str] | None = None) -> None:
    """Spec invariant: refuse to start if the ignore-death env var is set.

    Reads from ``env`` if provided, otherwise os.environ. Raises
    ``IgnoreDeathRefusal`` (a SystemExit subclass with code 2) on detection.
    """
    if env is None:
        env = os.environ
    raw = env.get(IGNORE_DEATH_ENV_VAR)
    if raw is not None and str(raw).strip() != "":
        msg = (
            f"refusing: {IGNORE_DEATH_ENV_VAR} is set in the environment; "
            "P3 metric runs require death-respecting mode "
            "(see docs/sight-p3-metrics.md spec invariant)."
        )
        # Print before raising so the message reaches stderr regardless of caller.
        print(msg, file=sys.stderr)
        raise IgnoreDeathRefusal(msg)


def _ts_run_id() -> str:
    return "p3-" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_p3_eval",
        description="Sight P3 evaluation runner (scaffold; live mode pending).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes (live mode, future slice).",
    )
    p.add_argument(
        "--actions-budget",
        type=int,
        default=300,
        help="Per-episode action budget; success_budget_reached fires at this count.",
    )
    p.add_argument(
        "--wall-time-budget-sec",
        type=float,
        default=120.0,
        help="Per-episode wall-time safety ceiling in seconds.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory; defaults to runs/eval/<auto-run-id>.",
    )
    p.add_argument(
        "--from-artifacts",
        type=str,
        default=None,
        help=(
            "Load episodes from an existing runs/eval/<run_id> tree instead of "
            "launching Godot. Required in this scaffold slice."
        ),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id; defaults to a timestamped value.",
    )
    return p.parse_args(argv)


def discover_episode_dirs(run_root: Path) -> list[Path]:
    """Return sorted list of <run_root>/episodes/* directories that have artifacts."""
    episodes_dir = run_root / "episodes"
    if not episodes_dir.exists():
        return []
    out: list[Path] = []
    for d in sorted(episodes_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "godot.ndjson").exists() or (d / "python.ndjson").exists():
            out.append(d)
    return out


def from_artifacts_mode(args: argparse.Namespace) -> int:
    """Load episodes from disk, aggregate, write summary.json. No Godot launch."""
    # Local imports keep CLI parse fast and let tests import the module without
    # forcing the package import path before the working directory is set.
    from sight_agent.evaluator.episodes import load_episode
    from sight_agent.evaluator.metrics import aggregate

    run_root = Path(args.from_artifacts)
    if not run_root.exists():
        print(f"--from-artifacts path not found: {run_root}", file=sys.stderr)
        return 3

    episode_dirs = discover_episode_dirs(run_root)
    if not episode_dirs:
        print(
            f"no episode artifacts under {run_root}/episodes/",
            file=sys.stderr,
        )
        return 4

    episodes = []
    for d in episode_dirs:
        ep = load_episode(
            godot_path=d / "godot.ndjson",
            python_path=d / "python.ndjson",
            episode_id=d.name,
            actions_budget=args.actions_budget,
            wall_time_budget_sec=args.wall_time_budget_sec,
        )
        episodes.append(ep)

    summary = aggregate(episodes)
    summary["run_id"] = args.run_id or run_root.name
    summary["mode"] = "from_artifacts"
    summary["episode_ids"] = [ep.episode_id for ep in episodes]
    summary["actions_budget"] = args.actions_budget
    summary["wall_time_budget_sec"] = args.wall_time_budget_sec

    out_dir = Path(args.out_dir) if args.out_dir else run_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(out_path), "episodes": len(episodes)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    refuse_if_ignore_death()
    args = parse_args(argv)

    if args.from_artifacts:
        return from_artifacts_mode(args)

    print(
        "live P3 mode is not implemented in this slice; "
        "use --from-artifacts <run_root>",
        file=sys.stderr,
    )
    return 5


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IgnoreDeathRefusal as exc:
        sys.exit(exc.EXIT_CODE)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
