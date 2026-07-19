"""PPO from pixels on bundled ViZDoom scenarios (SB3, GPU).

RL teacher for the vzd track: trains a CnnPolicy on a bundled
vizdoom gymnasium env (--env-id, default VizdoomDefendCenter-v1;
VZD-3 uses VizdoomDeadlyCorridor-v1). Gray 60x80 (stride-4 downsample
of 240x320), frame-skip 4, frame-stack 4, gamma 0.99 (the K-phase
lesson). --doom-skill overrides the scenario cfg skill (deadly_corridor
ships at skill 5; skill curriculum is the standard lever there).
Writes progress to
<out>/log.csv, checkpoints, final model.zip + summary.json, and a
DONE sentinel for detached monitoring.

Note on --resume semantics: SB3's reset_num_timesteps=False treats
--steps as ADDITIONAL steps on top of the checkpoint, not a total
target. Resuming a 750k checkpoint with --steps 1500000 trains to
2.25M total.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_train.py --steps 1500000
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_train.py --steps 2000 --smoke
  .venv-c1\\Scripts\\python.exe tools\\vzd_ppo_train.py --env-id VizdoomDeadlyCorridor-v1 --doom-skill 1 --steps 1500000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_ID = "VizdoomDefendCenter-v1"
ENV_ID = DEFAULT_ENV_ID  # compat alias: vzd_rollout_dataset.py and friends import this
STRIDE = 4  # 240x320 RGB -> 60x80 gray


def out_slug(env_id: str) -> str:
    """VizdoomDefendCenter-v1 -> ppo_defend (legacy), else ppo_<snake>."""
    if env_id == DEFAULT_ENV_ID:
        return "ppo_defend"  # keep the existing run dir stable
    import re
    core = re.sub(r"^Vizdoom|-v\d+$", "", env_id)
    return "ppo_" + re.sub(r"(?<!^)(?=[A-Z])", "_", core).lower()


class GrayStride(gym.ObservationWrapper):
    """screen dict RGB (H,W,3) -> gray uint8 (H//s, W//s, 1), numpy only."""

    def __init__(self, env, stride: int = STRIDE):
        super().__init__(env)
        self._s = stride
        h, w, _ = env.observation_space["screen"].shape
        self.observation_space = gym.spaces.Box(
            0, 255, (h // stride, w // stride, 1), np.uint8)

    def observation(self, obs):
        g = obs["screen"][:: self._s, :: self._s].mean(axis=2)
        return g.astype(np.uint8)[..., None]


class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total, terminated, truncated = 0.0, False, False
        obs, info = None, {}
        for _ in range(self._skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            total += r
            if terminated or truncated:
                break
        return obs, total, terminated, truncated, info


class SafeDoom(gym.Wrapper):
    """Rebuild the ViZDoom engine in-process when its binary dies.

    ViZDoom instances die stochastically under multi-instance load and the rate
    scales with instance count (Farama-Foundation/ViZDoom#169, open since 2017,
    no upstream fix; issue #430 confirms the exception just means "the binary
    died", cause unrecoverable from Python). Under SubprocVecEnv an uncaught
    engine fault kills the worker, the parent's pipe EOFs, and the entire run
    dies. Both shaped corridor runs died exactly this way, at ~47k and ~290k
    steps, while the policy itself was healthy.

    Catching the fault at the env boundary drops the cost of an engine death
    from "run over" to "one episode lost". Only ViZDoom's own exception types
    are caught; anything else still surfaces.
    """

    def __init__(self, make_base, max_rebuilds: int = 100):
        super().__init__(make_base())
        self._make_base = make_base
        self._max_rebuilds = max_rebuilds
        self.rebuilds = 0

    @staticmethod
    def _faults():
        import vizdoom as vzd
        return (vzd.ViZDoomErrorException, vzd.ViZDoomUnexpectedExitException,
                vzd.ViZDoomIsNotRunningException, vzd.SharedMemoryException,
                vzd.MessageQueueException)

    def _rebuild(self, where: str, exc: BaseException) -> None:
        self.rebuilds += 1
        print(f"[SafeDoom] engine fault in {where}: {type(exc).__name__}: {exc} "
              f"-> rebuild {self.rebuilds}/{self._max_rebuilds}", flush=True)
        if self.rebuilds > self._max_rebuilds:
            raise RuntimeError("SafeDoom: rebuild budget exhausted") from exc
        try:
            self.env.close()
        except BaseException:
            pass
        for attempt in range(5):
            try:
                self.env = self._make_base()
                return
            except BaseException as e2:
                print(f"[SafeDoom] rebuild attempt {attempt} failed: {e2}", flush=True)
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError("SafeDoom: could not rebuild engine") from exc

    def reset(self, **kw):
        try:
            return self.env.reset(**kw)
        except self._faults() as e:
            self._rebuild("reset", e)
            return self.env.reset(**kw)

    def step(self, action):
        try:
            return self.env.step(action)
        except self._faults() as e:
            self._rebuild("step", e)
            obs, info = self.env.reset()
            info["engine_fault"] = True
            return obs, 0.0, True, False, info


class ShapedCorridorReward(gym.Wrapper):
    """Renotte-style game-variable shaping for deadly_corridor.

    shaped = raw_scenario_reward            (WAD distance shaping toward armor)
             + d(HITCOUNT)     * w_hit      pay for landing shots
             - d(DAMAGE_TAKEN) * w_dmg      punish getting shot
             + d(AMMO2)        * w_ammo     d is negative when firing -> punish spray

    Coefficients default to the nicknochnack/DoomReinforcementLearning notebook
    (200 / 10 / 5). Variables are read via unwrapped.game.get_game_variable(),
    which works for any variable regardless of what the cfg declares available
    (deadly_corridor.cfg declares HEALTH only). AMMO2 is the pistol ammo pool;
    SELECTED_WEAPON_AMMO reads -1 here and is unusable. Verified on vizdoom 1.3.0.
    """

    def __init__(self, env, w_hit: float = 200.0, w_dmg: float = 10.0,
                 w_ammo: float = 5.0):
        super().__init__(env)
        self.w_hit, self.w_dmg, self.w_ammo = w_hit, w_dmg, w_ammo
        self._prev = (0.0, 0.0, 0.0)

    def _vars(self):
        from vizdoom import GameVariable as GV
        g = self.env.unwrapped.game
        return (g.get_game_variable(GV.HITCOUNT),
                g.get_game_variable(GV.DAMAGE_TAKEN),
                g.get_game_variable(GV.AMMO2))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev = self._vars()
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if info.get("engine_fault"):
            # SafeDoom rebuilt the engine under us; game vars restarted at 0.
            # Do not shape across that discontinuity.
            self._prev = (0.0, 0.0, 0.0)
            return obs, r, terminated, truncated, info
        hit, dmg, ammo = self._vars()
        p_hit, p_dmg, p_ammo = self._prev
        shaped = (r
                  + (hit - p_hit) * self.w_hit
                  - (dmg - p_dmg) * self.w_dmg
                  + (ammo - p_ammo) * self.w_ammo)
        self._prev = (hit, dmg, ammo)
        info["raw_reward"] = r
        return obs, shaped, terminated, truncated, info


def _sibling_vecnormalize(ckpt: Path) -> Path | None:
    """ppo_x_250000_steps.zip -> ppo_x_vecnormalize_250000_steps.pkl, if present.

    CheckpointCallback(save_vecnormalize=True) writes the stats beside the model
    with the step count in the name; match on that count so a resume can never
    silently pair a checkpoint with the wrong return statistics.
    """
    import re
    m = re.search(r"^(.*?)_(\d+)_steps\.zip$", ckpt.name)
    if not m:
        return None
    prefix, steps = m.group(1), m.group(2)
    p = ckpt.parent / f"{prefix}_vecnormalize_{steps}_steps.pkl"
    return p if p.exists() else None


def make_env(env_id: str, seed: int, doom_skill: int | None = None,
             shape_reward: bool = False):
    def _f():
        import vizdoom.gymnasium_wrapper  # noqa: F401  (registers envs)
        from stable_baselines3.common.monitor import Monitor
        kw = {} if doom_skill is None else {"doom_skill": doom_skill}
        env = SafeDoom(lambda: gym.make(env_id, **kw))
        if shape_reward:
            env = ShapedCorridorReward(env)
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env = Monitor(env)  # emits ep_rew_mean / ep_len_mean to the log
        env.reset(seed=seed)
        return env
    return _f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--env-id", default=DEFAULT_ENV_ID)
    ap.add_argument("--doom-skill", type=int, default=None,
                    help="override scenario cfg doom_skill (deadly_corridor ships at 5)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shape-reward", action="store_true",
                    help="game-variable shaping (hit/damage/ammo); train-only, eval stays raw")
    ap.add_argument("--norm-reward", action="store_true",
                    help="VecNormalize returns; fixes the corridor value_loss ~5e4 "
                         "that collapses entropy via the shared CNN trunk")
    ap.add_argument("--ent-coef", type=float, default=None,
                    help="entropy bonus; default 0.01 fresh. On --resume this "
                         "OVERRIDES the checkpoint's saved ent_coef (PPO.load "
                         "restores it), which is the lever for re-opening "
                         "exploration on a converged policy")
    ap.add_argument("--ckpt-every", type=int, default=50_000,
                    help="timesteps between checkpoints; this is the worst-case "
                         "work lost when the supervisor restarts a dead leg")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .zip to resume from; --steps is ADDITIONAL steps on top "
                         "of the checkpoint (SB3 reset_num_timesteps=False), not a total target")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
    from stable_baselines3.common.callbacks import CheckpointCallback

    out = Path(args.out) if args.out else REPO_ROOT / "runs" / "vzd" / out_slug(args.env_id)
    out.mkdir(parents=True, exist_ok=True)

    venv = SubprocVecEnv([make_env(args.env_id, args.seed + i, args.doom_skill,
                                   args.shape_reward)
                          for i in range(args.n_envs)])
    venv = VecFrameStack(venv, 4)
    if args.norm_reward:
        from stable_baselines3.common.vec_env import VecNormalize
        # On resume, restore the running return statistics. A fresh VecNormalize
        # would restart its std estimate at 1.0 and re-inflate the returns that
        # collapsed entropy in the first place, so a naive restart would undo
        # the fix it is meant to protect.
        vn_pkl = _sibling_vecnormalize(Path(args.resume)) if args.resume else None
        if vn_pkl is not None:
            venv = VecNormalize.load(str(vn_pkl), venv)
            venv.training, venv.norm_reward = True, True
            print("restored VecNormalize stats from", vn_pkl)
        else:
            venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                                clip_reward=10.0)
            if args.resume:
                print("WARNING: resuming without VecNormalize stats "
                      "(no sibling .pkl); return scale will re-estimate")

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cuda")
        # PPO.load restores the CHECKPOINT's ent_coef, so --ent-coef is silently
        # ignored on resume unless it is reapplied here. Stage 2 ended at
        # entropy_loss -0.08 (near-deterministic), so the skill-5 finetune needs
        # this lever to actually move or the raised value is a lie in summary.json.
        if args.ent_coef is not None:
            print(f"ent_coef {model.ent_coef} -> {args.ent_coef} (resume override)")
            model.ent_coef = args.ent_coef
        args.ent_coef = model.ent_coef  # so summary.json records what ran
        print("resumed from", args.resume, "at", model.num_timesteps, "steps")
    else:
        if args.ent_coef is None:
            args.ent_coef = 0.01
        model = PPO(
            "CnnPolicy", venv, verbose=1, seed=args.seed,
            n_steps=256, batch_size=512, learning_rate=2.5e-4,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1,
            ent_coef=args.ent_coef, vf_coef=0.5, n_epochs=4, device="cuda",
            tensorboard_log=None)
    print("obs space:", venv.observation_space)

    # A checkpoint is the supervisor's restart point, so --ckpt-every is the
    # worst-case work an engine death can cost.
    cb = None if args.smoke else CheckpointCallback(
        save_freq=max(args.ckpt_every // args.n_envs, 1), save_path=str(out),
        name_prefix=out_slug(args.env_id), save_vecnormalize=args.norm_reward)

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False,
                reset_num_timesteps=not args.resume)
    train_s = time.time() - t0
    model.save(out / "model.zip")
    venv.close()

    # deterministic eval, fresh single env.
    # Deliberately RAW: no shaping, no VecNormalize. The eval number must stay
    # comparable to the flat-run baselines (skill-5 IQM 93.6, skill-3 IQM 683.9).
    from stable_baselines3.common.vec_env import DummyVecEnv
    ev = VecFrameStack(DummyVecEnv([make_env(args.env_id, 10_000, args.doom_skill,
                                             shape_reward=False)]), 4)
    rewards = []
    n_eval = 3 if args.smoke else 30
    obs = ev.reset()
    ep_r, done_count = 0.0, 0
    while done_count < n_eval:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = ev.step(act)
        ep_r += float(r[0])
        if done[0]:
            rewards.append(ep_r)
            ep_r = 0.0
            done_count += 1
    ev.close()

    r = np.array(rewards, float)
    from scipy.stats import trim_mean
    summary = {
        "env": args.env_id, "doom_skill": args.doom_skill,
        "steps": int(model.num_timesteps), "steps_this_leg": args.steps,
        "n_envs": args.n_envs,
        "seed": args.seed, "train_seconds": round(train_s, 1),
        "steps_per_sec": round(args.steps / train_s, 1),
        "eval_episodes": n_eval, "mean_reward": float(r.mean()),
        "iqm_reward": float(trim_mean(r, 0.25)), "rewards": rewards,
        "gamma": 0.99, "ent_coef": args.ent_coef,
        "shape_reward": bool(args.shape_reward),
        "norm_reward": bool(args.norm_reward),
        "eval_reward": "raw scenario (unshaped, unnormalized)",
        "recipe": "PPO CnnPolicy gray60x80 skip4 stack4"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text("ok")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
