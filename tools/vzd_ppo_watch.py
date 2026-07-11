"""Watch / record a trained SB3 PPO model on ViZDoom defend_the_center.

Loads a PPO checkpoint (.zip) and runs deterministic episodes with the
exact training preprocessing (GrayStride, SkipFrames, VecFrameStack 4).
Modes:
  --watch          visible game window (render_mode=human)
  --record X.mp4   headless, writes full-res RGB tics to an H.264 mp4
  (neither)        headless eval, prints per-episode rewards + mean/IQM

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_watch.py --model runs\\vzd\\ppo_defend\\model.zip --episodes 5
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_watch.py --model ... --record runs\\vzd\\ppo_defend\\gameplay.mp4 --seconds 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vzd_ppo_train import ENV_ID, GrayStride, SkipFrames  # noqa: E402


class FrameTap(gym.Wrapper):
    """Copies the raw RGB screen of every inner tic into a sink."""

    def __init__(self, env, sink: list):
        super().__init__(env)
        self._sink = sink

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._sink.append(obs["screen"].copy())
        return obs, r, term, trunc, info

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._sink.append(obs["screen"].copy())
        return obs, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--record", default=None, help="output .mp4 path")
    ap.add_argument("--seconds", type=int, default=30,
                    help="max clip length when recording (35 tics/s)")
    ap.add_argument("--scale", type=int, default=2,
                    help="integer upscale factor for the recorded clip")
    args = ap.parse_args()

    import vizdoom.gymnasium_wrapper  # noqa: F401  (registers envs)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    frames: list = []

    def _f():
        kw = {"render_mode": "human"} if args.watch else {}
        env = gym.make(ENV_ID, **kw)
        if args.record:
            env = FrameTap(env, frames)
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env.reset(seed=args.seed)
        return env

    ev = VecFrameStack(DummyVecEnv([_f]), 4)
    model = PPO.load(args.model, device="cuda")

    max_frames = args.seconds * 35 if args.record else None
    rewards, ep_r = [], 0.0
    obs = ev.reset()
    while len(rewards) < args.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = ev.step(act)
        ep_r += float(r[0])
        if done[0]:
            rewards.append(ep_r)
            print(f"episode {len(rewards)}: reward {ep_r:.1f}")
            ep_r = 0.0
        if max_frames and len(frames) >= max_frames:
            break
    ev.close()

    if rewards:
        from scipy.stats import trim_mean
        arr = np.array(rewards, float)
        print(f"episodes {len(arr)}  mean {arr.mean():.2f}  "
              f"iqm {trim_mean(arr, 0.25):.2f}")

    if args.record and frames:
        import imageio.v2 as imageio
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip = frames[: args.seconds * 35]
        if args.scale > 1:
            import cv2
            clip = [cv2.resize(f, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_NEAREST) for f in clip]
        w = imageio.get_writer(str(out), fps=35, codec="libx264",
                               quality=8, pixelformat="yuv420p")
        for f in clip:
            w.append_data(f)
        w.close()
        print(f"wrote {out} ({len(clip)} frames, {len(clip)/35:.1f}s, "
              f"{clip[0].shape[1]}x{clip[0].shape[0]})")


if __name__ == "__main__":
    main()
