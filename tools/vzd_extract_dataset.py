"""Extract a BC dataset from recorded ViZDoom .lmp demos.

Replays every ep_*.lmp in a demo dir headless at GRAY8 160x120,
samples every --frame-skip tics, and writes one npz:
  frames (N,120,160) uint8, labels (N,) int64 combo class,
  episode_ids (N,) int32, combo_map (list of button tuples), buttons.

Combo classes are the unique button vectors actually observed in the
demos, so the BC head only learns actions the human actually used.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_extract_dataset.py --scenario defend_the_center
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import vizdoom as vzd

REPO_ROOT = Path(__file__).resolve().parents[1]


def replay_episode(game: vzd.DoomGame, lmp: Path, frame_skip: int):
    frames, actions = [], []
    game.replay_episode(str(lmp))
    tic = 0
    while not game.is_episode_finished():
        st = game.get_state()
        game.advance_action()
        a = game.get_last_action()
        if st is not None and tic % frame_skip == 0:
            frames.append(st.screen_buffer.copy())
            actions.append(tuple(int(x) for x in a))
        tic += 1
    return frames, actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--demo-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frame-skip", type=int, default=4)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir) if args.demo_dir else \
        REPO_ROOT / "runs" / "vzd" / "demos" / args.scenario
    lmps = sorted(demo_dir.glob("ep_*.lmp"))
    if not lmps:
        raise SystemExit(f"no .lmp files in {demo_dir}")

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, args.scenario + ".cfg"))
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.init()
    buttons = [str(b) for b in game.get_available_buttons()]

    all_frames, all_combos, ep_ids = [], [], []
    for ep_i, lmp in enumerate(lmps):
        fr, ac = replay_episode(game, lmp, args.frame_skip)
        all_frames.extend(fr)
        all_combos.extend(ac)
        ep_ids.extend([ep_i] * len(fr))
        print(f"{lmp.name}: {len(fr)} samples")
    game.close()

    combo_map = sorted(set(all_combos))
    combo_to_cls = {c: i for i, c in enumerate(combo_map)}
    labels = np.array([combo_to_cls[c] for c in all_combos], dtype=np.int64)
    frames = np.stack(all_frames).astype(np.uint8)
    ep_ids = np.array(ep_ids, dtype=np.int32)

    out = Path(args.out) if args.out else \
        REPO_ROOT / "runs" / "vzd" / f"bc_dataset_{args.scenario}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, frames=frames, labels=labels, episode_ids=ep_ids,
        combo_map=np.array(json.dumps([list(c) for c in combo_map])),
        buttons=np.array(json.dumps(buttons)),
        frame_skip=args.frame_skip)
    counts = np.bincount(labels, minlength=len(combo_map))
    print(f"dataset {out}: {len(frames)} samples, {len(combo_map)} combos")
    for c, n in zip(combo_map, counts):
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
