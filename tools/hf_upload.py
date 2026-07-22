"""Upload the three ViZDoom PPO models to Hugging Face Hub.

Uses huggingface_sb3.package_to_hub: re-evaluates each model (30 deterministic
episodes, raw reward, same protocol as the published numbers), records a replay
video, generates a model card, and pushes to <user>/<repo>.

Requires a logged-in HF account (hf auth login) with a WRITE token.

Usage:
  .venv-c1\\Scripts\\python.exe tools\\hf_upload.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gymnasium as gym
from vzd_ppo_train import GrayStride, SkipFrames, SafeDoom

MODELS = [
    # (run_dir, repo_name, env_id, doom_skill, blurb)
    ("runs/vzd/ppo_defend", "sight-ppo-vizdoom-defend-center",
     "VizdoomDefendCenter-v1", None,
     "PPO CnnPolicy from pixels, 2.25M steps."),
    ("runs/vzd/ppo_deadly_corridor_s3_shaped", "sight-ppo-vizdoom-deadly-corridor-s3",
     "VizdoomDeadlyCorridor-v1", 3,
     "Curriculum stage 1: skill 3, shaped reward + VecNormalize, 1.5M steps."),
    ("runs/vzd/ppo_deadly_corridor_s5_ft_seed1", "sight-ppo-vizdoom-deadly-corridor-s5",
     "VizdoomDeadlyCorridor-v1", 5,
     "Curriculum stage 2: skill-5 finetune of the s3 policy, 3.0M total steps."),
]


def make_eval_env(env_id: str, doom_skill: int | None):
    """Raw eval env, same chain as vzd_ppo_train's eval: gray60x80 skip4 stack4.

    render_mode='rgb_array' so package_to_hub's VecVideoRecorder gets frames.
    No shaping, no VecNormalize: the model card number stays comparable to the
    published raw bars.
    """
    def _f():
        import vizdoom.gymnasium_wrapper  # noqa: F401  (registers envs)
        from stable_baselines3.common.monitor import Monitor
        kw = {"render_mode": "rgb_array"}
        if doom_skill is not None:
            kw["doom_skill"] = doom_skill
        env = SafeDoom(lambda: gym.make(env_id, **kw))
        env = GrayStride(env)
        env = SkipFrames(env, 4)
        env = Monitor(env)
        env.reset(seed=10_000)
        return env

    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    return VecFrameStack(DummyVecEnv([_f]), 4)


def main() -> None:
    from huggingface_hub import whoami
    try:
        user = whoami()["name"]
    except Exception as e:
        sys.exit(f"Not logged in to Hugging Face ({e}). Run: "
                 f".venv-c1\\Scripts\\hf.exe auth login  (write token)")
    print("logged in as", user, flush=True)

    from stable_baselines3 import PPO
    from huggingface_sb3 import package_to_hub

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for run_dir, repo_name, env_id, skill, blurb in MODELS:
        if only and only not in repo_name:
            continue
        model_path = REPO_ROOT / run_dir / "model.zip"
        assert model_path.exists(), f"missing {model_path}"
        print(f"\n=== {repo_name} <- {run_dir} ===", flush=True)
        model = PPO.load(str(model_path), device="cuda")
        eval_env = make_eval_env(env_id, skill)
        package_to_hub(
            model=model, model_name=repo_name, model_architecture="PPO",
            env_id=env_id, eval_env=eval_env,
            repo_id=f"{user}/{repo_name}",
            commit_message=f"{blurb} Trained in trzz333/sight.",
        )
        print(f"pushed {user}/{repo_name}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
