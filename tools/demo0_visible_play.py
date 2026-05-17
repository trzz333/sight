"""Demo-0: visible AI playing Signal Dodge.

Loads a trained CnnPolicy PPO model from a completed H5 train run,
launches Signal Dodge in windowed (non-headless) pixel mode, steps the
env with deterministic argmax actions, and writes a watchable artifact
set under ``runs/demo0/<label>/``:

    frames/frame_NNNNN.png   84x84 grayscale upscaled 4x (336x336)
    demo0.mp4                MP4 of the same frames at --fps fps
    steps.ndjson             per-step record (action, reward, terminals)
    manifest.json            run metadata (model, seed, git, terminals)
    demo0.log                stdout/stderr of this script

This is a DEMO artifact, not H5 acceptance evidence. The trained policy
from Phase E may exhibit constant-action or wedge behavior; the artifact
visualizes whatever the policy does on the current pixel observation
stream, faithfully.

Usage (PowerShell or cmd):
  set SIGHT_GODOT_EXE=<path to Godot>.exe
  python tools\\demo0_visible_play.py ^
    --train-run-dir runs\\rl\\signal_dodge_ppo_h5_pixel_entropy\\h5_train_phase_e_seed2_entropy_10k ^
    --seed 1008 ^
    --out-dir runs\\demo0\\trained_seed2_eval1008 ^
    --episodes 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sight_agent.rl.godot_config import is_godot_env_id, resolve_godot_kwargs  # noqa: E402
from sight_agent.rl.godot_env import GodotSignalDodgeEnv  # noqa: E402

ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}


def _short_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def _load_effective_config(train_run_dir: Path) -> dict[str, Any]:
    cfg_path = train_run_dir / "config_effective.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_effective.yaml not found at {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config_effective.yaml not a mapping: {cfg_path}")
    return cfg


def _resolve_model_path(train_run_dir: Path) -> Path:
    summary_path = train_run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    paths = summary.get("artifact_paths") or {}
    candidate = paths.get("model")
    if isinstance(candidate, str) and candidate:
        p = Path(candidate)
        if not p.is_absolute():
            p = train_run_dir / p
        if p.exists():
            return p
    fallback = train_run_dir / "model.zip"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No model.zip under {train_run_dir}")


def _build_env(train_cfg: dict[str, Any], run_dir: Path, seed: int) -> GodotSignalDodgeEnv:
    env_cfg = train_cfg.get("env", {})
    if not is_godot_env_id(env_cfg.get("id")):
        raise ValueError(f"train run env id is not a godot id: {env_cfg.get('id')!r}")
    extra = resolve_godot_kwargs(train_cfg, repo_root=REPO_ROOT)
    if not extra.get("godot_executable"):
        raise ValueError(
            "godot_executable unresolved. Set SIGHT_GODOT_EXE in this shell "
            "(it cannot rely on user-scope env vars for Godot launches)."
        )
    extra["headless"] = False
    kwargs = {k: v for k, v in extra.items() if v is not None}
    env = GodotSignalDodgeEnv(
        run_dir=str(run_dir / "godot"),
        seed=int(seed),
        connect_timeout_s=30.0,
        step_timeout_s=10.0,
        **kwargs,
    )
    return env


def _obs_to_gray_frame(obs: np.ndarray) -> np.ndarray:
    """Convert a (C,H,W) uint8 pixel obs to (H,W) uint8 grayscale frame."""
    if obs.ndim == 3 and obs.shape[0] == 1:
        return obs[0]
    if obs.ndim == 3:
        return obs.mean(axis=0).astype(np.uint8)
    if obs.ndim == 2:
        return obs.astype(np.uint8)
    raise ValueError(f"Unexpected obs shape: {obs.shape}")


def _write_artifacts(
    *,
    frames: list[np.ndarray],
    step_records: list[dict[str, Any]],
    out_dir: Path,
    fps: int,
    upscale: int,
) -> dict[str, Any]:
    import cv2

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not frames:
        return {"frame_count": 0, "video_path": None, "frames_dir": str(frames_dir)}

    h, w = frames[0].shape
    up_h, up_w = h * upscale, w * upscale

    # Frame PNGs (upscaled, nearest neighbor preserves the AI's coarse view).
    for idx, f in enumerate(frames):
        up = cv2.resize(f, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), up)

    # MP4 via OpenCV VideoWriter. mp4v is broadly available on Windows builds.
    video_path = out_dir / "demo0.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (up_w, up_h), False)
    if not writer.isOpened():
        writer.release()
        return {
            "frame_count": len(frames),
            "video_path": None,
            "frames_dir": str(frames_dir),
            "video_error": "VideoWriter failed to open (mp4v codec unavailable?)",
        }
    for f in frames:
        up = cv2.resize(f, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
        writer.write(up)
    writer.release()

    # Per-step NDJSON sidecar.
    ndjson_path = out_dir / "steps.ndjson"
    with ndjson_path.open("w", encoding="utf-8") as f:
        for rec in step_records:
            f.write(json.dumps(rec, separators=(",", ":")))
            f.write("\n")

    return {
        "frame_count": len(frames),
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "steps_ndjson": str(ndjson_path),
        "frame_dimensions": [up_h, up_w],
        "fps": fps,
        "upscale": upscale,
    }


def _run_episode(env: GodotSignalDodgeEnv, model: PPO, *, episode_index: int,
                 max_steps_override: int | None) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    obs, info_reset = env.reset()
    frames.append(_obs_to_gray_frame(obs))

    total_reward = 0.0
    terminal_reason: str | None = None
    terminated = False
    truncated = False
    step_idx = 0
    t0 = time.monotonic()

    while True:
        action_arr, _state = model.predict(obs, deterministic=True)
        action_int = int(np.asarray(action_arr).reshape(-1)[0])
        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += float(reward)
        frames.append(_obs_to_gray_frame(obs))
        rec = {
            "episode_index": episode_index,
            "step": step_idx,
            "action": action_int,
            "action_name": ACTION_NAMES.get(action_int, str(action_int)),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        if isinstance(info, dict) and "terminal_reason" in info:
            rec["terminal_reason"] = info["terminal_reason"]
            if terminated or truncated:
                terminal_reason = str(info["terminal_reason"])
        records.append(rec)
        step_idx += 1
        if terminated or truncated:
            break
        if max_steps_override is not None and step_idx >= max_steps_override:
            truncated = True
            terminal_reason = terminal_reason or "demo0_max_steps_override"
            break

    elapsed = time.monotonic() - t0
    summary = {
        "episode_index": episode_index,
        "steps": step_idx,
        "total_reward": total_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "terminal_reason": terminal_reason,
        "wall_seconds": elapsed,
    }
    return frames, records, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo0_visible_play")
    parser.add_argument("--train-run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1008)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--max-steps-override", type=int, default=None)
    args = parser.parse_args(argv)

    train_run_dir: Path = args.train_run_dir.resolve()
    if not train_run_dir.is_dir():
        print(f"ERROR: train run dir not found: {train_run_dir}", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = _load_effective_config(train_run_dir)
    model_path = _resolve_model_path(train_run_dir)
    print(f"Demo-0: model={model_path}")
    print(f"Demo-0: train_run_dir={train_run_dir}")
    print(f"Demo-0: out_dir={out_dir}")

    env = _build_env(train_cfg, run_dir=out_dir, seed=int(args.seed))
    try:
        model = PPO.load(str(model_path), device="cpu")
    except Exception:
        env.close()
        raise

    all_frames: list[np.ndarray] = []
    all_records: list[dict[str, Any]] = []
    episode_summaries: list[dict[str, Any]] = []
    status = "ok"
    error_payload: dict[str, Any] | None = None

    try:
        for ep_i in range(int(args.episodes)):
            frames, records, ep_summary = _run_episode(
                env, model,
                episode_index=ep_i,
                max_steps_override=args.max_steps_override,
            )
            all_frames.extend(frames)
            all_records.extend(records)
            episode_summaries.append(ep_summary)
            print(
                f"Demo-0 episode {ep_i}: steps={ep_summary['steps']}, "
                f"reward={ep_summary['total_reward']:.3f}, "
                f"terminal={ep_summary['terminal_reason']!r}"
            )
    except Exception as exc:
        status = "error"
        error_payload = {"error_type": type(exc).__name__, "message": str(exc)}
        print(f"Demo-0 ERROR: {error_payload}", file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception:
            pass


    art_info = _write_artifacts(
        frames=all_frames,
        step_records=all_records,
        out_dir=out_dir,
        fps=int(args.fps),
        upscale=int(args.upscale),
    )

    manifest = {
        "schema_version": 1,
        "kind": "demo0_visible_play",
        "label": out_dir.name,
        "seed": int(args.seed),
        "episodes_requested": int(args.episodes),
        "episodes_completed": len(episode_summaries),
        "fps": int(args.fps),
        "upscale": int(args.upscale),
        "model_path": str(model_path),
        "train_run_dir": str(train_run_dir),
        "config_effective_path": str(train_run_dir / "config_effective.yaml"),
        "out_dir": str(out_dir),
        "git_commit": _short_git_commit(),
        "status": status,
        "error": error_payload,
        "artifact_paths": {
            "frames_dir": art_info.get("frames_dir"),
            "video": art_info.get("video_path"),
            "steps_ndjson": art_info.get("steps_ndjson"),
        },
        "frame_count": art_info.get("frame_count"),
        "video_dimensions": art_info.get("frame_dimensions"),
        "episodes": episode_summaries,
        "caveat": (
            "Demo-0 visible artifact only. The Phase E seed2 entropy-recipe "
            "policy is not H5 acceptance-grade; expected behavior includes "
            "constant-action or wedge tendencies under deterministic argmax. "
            "Artifact visualizes the policy's actual decisions on the live "
            "pixel observation stream."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Demo-0 manifest: {manifest_path}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
