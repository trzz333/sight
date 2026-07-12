"""Extract a BC dataset from trained-teacher rollouts (VZD-2 distillation).

Rolls out the SB3 PPO teacher in its exact training env chain
(VizdoomDefendCenter-v1 + GrayStride + SkipFrames + VecFrameStack) and
records, per decision step, the student frame and the button combo the
game actually applied (game.get_last_action, same ground truth as the
LMP extractor). Student frames are mean-gray stride-2 of the 240x320
RGB screen -> (120,160) uint8, so the npz is schema-identical to
vzd_extract_dataset.py output and feeds vzd_bc_train.py unchanged.

Eval note: a student trained on this dataset must be evaluated with
vzd_bc_eval.py --obs rgb2 (matched derivation), not the native GRAY8
render used for human-demo students.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_rollout_dataset.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from vzd_ppo_train import ENV_ID, GrayStride, SkipFrames

REPO_ROOT = Path(__file__).resolve().parents[1]


def derive_student_frame(screen_rgb: np.ndarray) -> np.ndarray:
    """(240,320,3) RGB uint8 -> (120,160) gray uint8, mean-gray stride 2.

    Shared derivation for teacher-rollout datasets and --obs rgb2 eval.
    """
    return screen_rgb[::2, ::2].mean(axis=2).astype(np.uint8)


class RecordScreen(gym.Wrapper):
    """Stash the raw RGB screen and the last applied button combo.

    Sits directly above the base Vizdoom env (below GrayStride) so
    last_action is read before any auto-reset can clear it.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_screen = None
        self.last_combo = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.last_screen = obs["screen"].copy()
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.last_screen = obs["screen"].copy()
        self.last_combo = tuple(
            int(x) for x in self.env.unwrapped.game.get_last_action())
        return obs, r, term, trunc, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(
        REPO_ROOT / "runs" / "vzd" / "ppo_defend" / "model.zip"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20_000)
    ap.add_argument("--stochastic", action="store_true",
                    help="sample teacher actions instead of argmax")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scenario", default="defend_the_center")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    rec_holder = {}

    def make_env():
        import vizdoom.gymnasium_wrapper  # noqa: F401
        env = gym.make(ENV_ID)
        env = RecordScreen(env)
        rec_holder["rec"] = env
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env.reset(seed=args.seed)
        return env

    venv = VecFrameStack(DummyVecEnv([make_env]), 4)
    model = PPO.load(args.model, device="cuda")
    rec = rec_holder["rec"]
    buttons = [str(b) for b in rec.unwrapped.game.get_available_buttons()]

    all_frames, all_combos, ep_ids, ep_rewards = [], [], [], []
    obs = venv.reset()
    ep_i, ep_r = 0, 0.0
    while ep_i < args.episodes:
        frame = derive_student_frame(rec.last_screen)
        act, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, r, done, _info = venv.step(act)
        all_frames.append(frame)
        all_combos.append(rec.last_combo)
        ep_ids.append(ep_i)
        ep_r += float(r[0])
        if done[0]:
            ep_rewards.append(ep_r)
            print(f"ep {ep_i:03d}: reward {ep_r:.1f}", flush=True)
            ep_i, ep_r = ep_i + 1, 0.0
    venv.close()

    combo_map = sorted(set(all_combos))
    combo_to_cls = {c: i for i, c in enumerate(combo_map)}
    labels = np.array([combo_to_cls[c] for c in all_combos], dtype=np.int64)
    frames = np.stack(all_frames).astype(np.uint8)
    ep_arr = np.array(ep_ids, dtype=np.int32)

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "vzd" / \
        f"bc_dataset_{args.scenario}_teacher.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, frames=frames, labels=labels, episode_ids=ep_arr,
        combo_map=np.array(json.dumps([list(c) for c in combo_map])),
        buttons=np.array(json.dumps(buttons)), frame_skip=4)

    r = np.array(ep_rewards, float)
    from scipy.stats import trim_mean
    meta = {
        "source": "teacher_rollout", "model": args.model,
        "episodes": args.episodes, "deterministic": not args.stochastic,
        "seed": args.seed, "n_samples": int(len(frames)),
        "teacher_rollout_mean": float(r.mean()),
        "teacher_rollout_iqm": float(trim_mean(r, 0.25)),
        "rewards": ep_rewards, "obs_deriv": "rgb2 (mean-gray stride-2 of 240x320 RGB)"}
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    counts = np.bincount(labels, minlength=len(combo_map))
    print(f"dataset {out}: {len(frames)} samples, {len(combo_map)} combos", flush=True)
    for c, n in zip(combo_map, counts):
        print(f"  {c}: {n}", flush=True)
    print(f"teacher rollout mean {r.mean():.2f} iqm {trim_mean(r, 0.25):.2f}", flush=True)
    (out.parent / "ROLLOUT_DONE").write_text("ok")


if __name__ == "__main__":
    main()
