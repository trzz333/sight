"""ViZDoom human demo recorder (SPECTATOR mode).

Jeff-facing tool: opens a visible Doom window, Jeff plays with keyboard
and mouse, each episode is saved as a ViZDoom .lmp demo file plus a
manifest. LMP files are replayable, so datasets can be re-extracted at
any resolution or frame-skip later without re-recording.

Default scenario: defend_the_center (turn left / turn right / attack,
an aiming task). Episodes end on death or timeout; the next one starts
automatically. Close the window or Ctrl+C in the console to stop early.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_record_demo.py
  .venv-c1\\Scripts\\python.exe tools\\vzd_record_demo.py --scenario basic --episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import vizdoom as vzd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default=None, help="demo dir (default runs\\vzd\\demos\\<scenario>)")
    ap.add_argument("--resolution", default="RES_640X480", help="window size while playing")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else REPO_ROOT / "runs" / "vzd" / "demos" / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, args.scenario + ".cfg"))
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.set_screen_resolution(getattr(vzd.ScreenResolution, args.resolution))
    game.init()

    existing = sorted(out_dir.glob("ep_*.lmp"))
    start_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "scenario": args.scenario, "episodes": []}

    print(f"Recording to {out_dir}")
    print("Play each episode. Close window or Ctrl+C to stop.")
    try:
        for i in range(start_idx, start_idx + args.episodes):
            lmp = out_dir / f"ep_{i:03d}.lmp"
            game.new_episode(str(lmp))
            tics = 0
            while not game.is_episode_finished():
                game.advance_action()
                tics += 1
            total = game.get_total_reward()
            manifest["episodes"].append({
                "file": lmp.name, "tics": tics, "total_reward": total,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"ep_{i:03d}: {tics} tics, reward {total}")
    except (KeyboardInterrupt, vzd.ViZDoomIsNotRunningException,
            vzd.SignalException):
        print("Stopped by user.")
    finally:
        game.close()
    print(f"Done. {len(manifest['episodes'])} episodes in manifest.")


if __name__ == "__main__":
    main()
