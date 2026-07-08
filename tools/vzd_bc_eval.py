"""Evaluate a ViZDoom BC policy in-env; optional --watch to spectate.

Headless mode: N episodes, greedy argmax, reports per-episode rewards
plus mean and IQM (scipy trim_mean 0.25). Watch mode: visible window,
real-time pace, so a human can watch the policy play.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_bc_eval.py --scenario defend_the_center
  .venv-c1\\Scripts\\python.exe tools\\vzd_bc_eval.py --scenario defend_the_center --watch --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd
from scipy.stats import trim_mean

from vzd_bc_train import VzdBCNet, STACK

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pol_path = Path(args.policy) if args.policy else \
        REPO_ROOT / "runs" / "vzd" / f"bc_{args.scenario}" / "vzd_bc_policy.pt"
    ckpt = torch.load(pol_path, map_location="cpu", weights_only=False)
    net = VzdBCNet(ckpt["n_classes"], ckpt["stack"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    combo_map = [list(map(int, c)) for c in ckpt["combo_map"]]

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, args.scenario + ".cfg"))
    game.set_window_visible(args.watch)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.init()

    rewards, lengths = [], []
    for ep in range(args.episodes):
        game.set_seed(args.seed + ep)  # distinct, reproducible episodes
        game.new_episode()
        stack: deque = deque(maxlen=STACK)
        steps = 0
        while not game.is_episode_finished():
            st = game.get_state()
            if st is None:
                game.advance_action()
                continue
            f = st.screen_buffer.astype(np.float32) / 255.0
            while len(stack) < STACK:
                stack.append(f)
            stack.append(f)
            x = torch.from_numpy(np.stack(stack)[None])
            with torch.no_grad():
                cls = int(net(x).argmax(1))
            game.make_action(combo_map[cls], args.frame_skip)
            steps += 1
            if args.watch:
                time.sleep(args.frame_skip / 35.0)
        rewards.append(game.get_total_reward())
        lengths.append(steps)
        print(f"ep {ep:02d}: reward {rewards[-1]:.1f} steps {steps}")
    game.close()

    r = np.array(rewards, float)
    res = {"scenario": args.scenario, "episodes": args.episodes,
           "mean_reward": float(r.mean()), "iqm_reward": float(trim_mean(r, 0.25)),
           "std_reward": float(r.std()), "mean_len": float(np.mean(lengths)),
           "policy": str(pol_path)}
    print(json.dumps(res, indent=2))
    out = pol_path.parent / "vzd_bc_eval.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
