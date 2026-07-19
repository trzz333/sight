"""Answer "fight or run past" for a trained deadly_corridor policy.

The raw scenario reward is distance-toward-armor plus a death penalty, so a
high eval score (VZD-3 stage 2: IQM 2279.43) is fully consistent with a policy
that sprints to the vest and never fires. Section 2 of
docs\\vzd-deadly-corridor-findings.md warns that skill-3 success is often
combat-free, and that a run-past policy will not transfer to skill 5 (faster,
more accurate enemies punish ignoring them).

This runs the SAME deterministic eval as vzd_ppo_train.py (raw reward, no
shaping, no VecNormalize, seed 10000, skill from --doom-skill) and additionally
records the engine's own combat counters per episode via
unwrapped.game.get_game_variable(), plus an optional gameplay clip from the
same pass.

Counters are tapped INSIDE the wrapper stack and accumulated as per-episode
maxima. DummyVecEnv auto-resets the instant an episode ends, so reading the
counters after done[0] would report the next episode's zeros; and ViZDoom may
return stale/zero values once an episode is finished. KILLCOUNT/HITCOUNT are
monotone within an episode, so a running max is the honest read.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_probe_combat.py ^
    --model runs\\vzd\\ppo_deadly_corridor_s3_shaped\\model.zip ^
    --env-id VizdoomDeadlyCorridor-v1 --doom-skill 3 --episodes 30 ^
    --out runs\\vzd\\ppo_deadly_corridor_s3_shaped\\combat_probe.json ^
    --record runs\\vzd\\demos\\corridor_s3_shaped.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vzd_ppo_train import DEFAULT_ENV_ID, GrayStride, SkipFrames  # noqa: E402

VARS = ("KILLCOUNT", "HITCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN", "AMMO2", "HEALTH")


class CombatTap(gym.Wrapper):
    """Read engine counters every tic; expose per-episode aggregates via info."""

    def __init__(self, env, frame_sink: list | None = None):
        super().__init__(env)
        self._sink = frame_sink
        self._acc = {}
        self._ammo0 = 0.0
        self._tics = 0

    def _read(self) -> dict:
        from vizdoom import GameVariable as GV
        g = self.env.unwrapped.game
        return {n: float(g.get_game_variable(getattr(GV, n))) for n in VARS}

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        v = self._read()
        # maxima for monotone counters; live value for the rest
        self._acc = {n: v[n] for n in VARS}
        self._ammo0 = v["AMMO2"]
        self._tics = 0
        if self._sink is not None:
            self._sink.append(obs["screen"].copy())
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._tics += 1
        v = self._read()
        for n in ("KILLCOUNT", "HITCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN"):
            self._acc[n] = max(self._acc[n], v[n])
        # AMMO2 falls as it fires; keep the minimum reached and the last health
        self._acc["AMMO2"] = min(self._acc["AMMO2"], v["AMMO2"])
        self._acc["HEALTH"] = v["HEALTH"]
        info["combat"] = dict(self._acc)
        info["combat"]["SHOTS_FIRED"] = self._ammo0 - self._acc["AMMO2"]
        info["combat"]["TICS"] = self._tics
        if self._sink is not None:
            self._sink.append(obs["screen"].copy())
        return obs, r, term, trunc, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env-id", default=DEFAULT_ENV_ID)
    ap.add_argument("--doom-skill", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    import vizdoom.gymnasium_wrapper  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    frames: list = [] if args.record else None
    max_frames = args.seconds * 35 if args.record else None

    def _f():
        kw = {} if args.doom_skill is None else {"doom_skill": args.doom_skill}
        env = gym.make(args.env_id, **kw)
        env = CombatTap(env, frames)
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env.reset(seed=args.seed)
        return env

    ev = VecFrameStack(DummyVecEnv([_f]), 4)
    model = PPO.load(args.model, device="cuda")

    eps, ep_r = [], 0.0
    obs = ev.reset()
    while len(eps) < args.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, infos = ev.step(act)
        ep_r += float(r[0])
        if frames is not None and len(frames) > max_frames:
            frames[:] = frames[:max_frames]  # stop growing, keep evaluating
        if done[0]:
            c = infos[0].get("combat", {})
            rec = {"episode": len(eps) + 1, "raw_reward": round(ep_r, 3),
                   **{k: c.get(k) for k in
                      ("KILLCOUNT", "HITCOUNT", "DAMAGECOUNT", "DAMAGE_TAKEN",
                       "SHOTS_FIRED", "HEALTH", "TICS")}}
            eps.append(rec)
            print(json.dumps(rec), flush=True)
            ep_r = 0.0
    ev.close()

    from scipy.stats import trim_mean
    rw = np.array([e["raw_reward"] for e in eps], float)
    kills = np.array([e["KILLCOUNT"] or 0.0 for e in eps], float)
    hits = np.array([e["HITCOUNT"] or 0.0 for e in eps], float)
    shots = np.array([e["SHOTS_FIRED"] or 0.0 for e in eps], float)
    dmg = np.array([e["DAMAGE_TAKEN"] or 0.0 for e in eps], float)
    hp = np.array([e["HEALTH"] or 0.0 for e in eps], float)
    verdict = ("FIGHT" if kills.mean() >= 2.0 else
               "RUN_PAST" if kills.mean() < 0.5 else "MIXED")
    summary = {
        "model": args.model, "env": args.env_id, "doom_skill": args.doom_skill,
        "episodes": len(eps), "seed": args.seed, "deterministic": True,
        "eval_reward": "raw scenario (unshaped, unnormalized)",
        "mean_reward": float(rw.mean()), "iqm_reward": float(trim_mean(rw, 0.25)),
        "kills_mean": float(kills.mean()), "kills_max": float(kills.max()),
        "episodes_with_a_kill": int((kills > 0).sum()),
        "hits_mean": float(hits.mean()), "shots_mean": float(shots.mean()),
        "accuracy": float(hits.sum() / shots.sum()) if shots.sum() else None,
        "damage_taken_mean": float(dmg.mean()),
        "final_health_mean": float(hp.mean()),
        "episodes_survived": int((hp > 0).sum()),
        "verdict": verdict, "per_episode": eps}
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"},
                     indent=2))

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2))
        print("wrote", p)

    if args.record and frames:
        import imageio.v2 as imageio
        out = Path(args.record)
        out.parent.mkdir(parents=True, exist_ok=True)
        clip = frames[:max_frames]
        if args.scale > 1:
            import cv2
            clip = [cv2.resize(f, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_NEAREST) for f in clip]
        w = imageio.get_writer(str(out), fps=35, codec="libx264", quality=8,
                               pixelformat="yuv420p")
        for f in clip:
            w.append_data(f)
        w.close()
        print(f"wrote {out} ({len(clip)} frames, {len(clip)/35:.1f}s, "
              f"{clip[0].shape[1]}x{clip[0].shape[0]})")


if __name__ == "__main__":
    main()
