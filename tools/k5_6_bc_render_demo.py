"""K5.6 demo render: BC policy playing Signal Dodge, watchable mp4.

Faithful top-down render of the SAME state-mode rollout the BC policy
was evaluated on (k5_6_bc_eval_inenv). Runs one episode under greedy
argmax, captures the true per-step game geometry from
info.godot_info.reward_state (player_x/y, hazards_above x/y), and draws
each frame with cv2 (player box + hazard boxes on a 720x540 canvas),
writing demo.mp4 + steps.ndjson + manifest.json under the out dir.

Not pixel obs: the policy consumes the 10-dim state vector exactly as in
eval, so on-screen behavior matches the 930.27-clearing verdict. Frames
are drawn from logged geometry, not from the policy's input.

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_6_bc_render_demo.py --ckpt runs\\phase_k\\k5_6_bc\\bc_policy.pt --seed 1009
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from k5_2_env_dynamics_probe import _build_env  # noqa: E402
from k5_6_bc_train import BCPolicyNet  # noqa: E402

SCREEN_W = 720
SCREEN_H = 540
PLAYER_SIZE = 32
HAZARD_SIZE = 24
PLAYER_HALF = PLAYER_SIZE // 2
HAZARD_HALF = HAZARD_SIZE // 2
DEFAULT_PLAYER_Y = 508.0
ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}


def _short_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def load_policy(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = BCPolicyNet(
        in_dim=arch["in_dim"], hidden=arch["hidden"], n_actions=arch["n_actions"]
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mu = np.asarray(ck["feat_mean"], dtype=np.float32)
    sd = np.asarray(ck["feat_std"], dtype=np.float32)
    sd[sd < 1e-6] = 1.0
    return model, mu, sd


def greedy_action(model, mu, sd, obs) -> int:
    x = (np.asarray(obs, dtype=np.float32) - mu) / sd
    with torch.no_grad():
        return int(model(torch.from_numpy(x).unsqueeze(0)).argmax(1).item())


def _draw_frame(player_x: float, player_y: float, hazards: list,
                action: int, step: int) -> np.ndarray:
    import cv2
    img = np.full((SCREEN_H, SCREEN_W, 3), 18, dtype=np.uint8)  # dark bg
    # hazards: red boxes
    for h in hazards:
        try:
            hx, hy = float(h["x"]), float(h["y"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0 = int(hx - HAZARD_HALF), int(hy - HAZARD_HALF)
        cv2.rectangle(img, (x0, y0), (x0 + HAZARD_SIZE, y0 + HAZARD_SIZE),
                      (60, 60, 220), -1)
    # player: cyan box
    px0, py0 = int(player_x - PLAYER_HALF), int(player_y - PLAYER_HALF)
    cv2.rectangle(img, (px0, py0), (px0 + PLAYER_SIZE, py0 + PLAYER_SIZE),
                  (220, 200, 40), -1)
    # HUD
    cv2.putText(img, f"step {step}  act {ACTION_NAMES.get(action, action)}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1,
                cv2.LINE_AA)
    return img


def run(ckpt_path: Path, out_dir: Path, seed: int, *, fps: int = 30,
        max_steps: int = 1800) -> dict:
    import cv2
    model, mu, sd = load_policy(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(
        observation_mode="state", run_dir=out_dir / "godot",
        seed=int(seed), max_steps=max_steps, reward_shaping="none",
    )
    frames: list[np.ndarray] = []
    records: list[dict] = []
    steps = 0
    term = trunc = False
    reason = ""
    action_counts = [0, 0, 0]
    t0 = time.monotonic()
    try:
        obs, info = env.reset(seed=int(seed))
        rs = (info.get("godot_info") or {}).get("reward_state") or {}
        while steps < max_steps:
            a = greedy_action(model, mu, sd, obs)
            action_counts[a] += 1
            px = float(rs.get("player_x", SCREEN_W / 2.0))
            py = float(rs.get("player_y", DEFAULT_PLAYER_Y))
            hz = rs.get("hazards_above", []) or []
            frames.append(_draw_frame(px, py, hz, a, steps))
            obs, r, term, trunc, info = env.step(a)
            rs = (info.get("godot_info") or {}).get("reward_state") or {}
            records.append({"step": steps, "action": a,
                            "action_name": ACTION_NAMES.get(a, a),
                            "player_x": px, "reward": float(r),
                            "terminated": bool(term), "truncated": bool(trunc)})
            steps += 1
            if term or trunc:
                reason = info.get("terminal_reason", "")
                # draw the terminal frame too
                px = float(rs.get("player_x", px))
                hz = rs.get("hazards_above", []) or []
                frames.append(_draw_frame(px, py, hz, a, steps))
                break
    finally:
        env.close()
    wall = time.monotonic() - t0
    print(f"rollout seed={seed} steps={steps} reason={reason} "
          f"acts={action_counts} wall={wall:.1f}s", flush=True)
    return _write(frames, records, out_dir, ckpt_path, seed, steps, term,
                  trunc, reason, action_counts, fps)


def _write(frames, records, out_dir: Path, ckpt_path: Path, seed: int,
           steps: int, term: bool, trunc: bool, reason: str,
           action_counts: list, fps: int) -> dict:
    import cv2
    video_path = out_dir / "demo.mp4"
    video_err = None
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (w, h), True)
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()
        else:
            writer.release()
            video_err = "VideoWriter failed to open (mp4v unavailable?)"
            video_path = None
    else:
        video_path = None
        video_err = "no frames"

    (out_dir / "steps.ndjson").write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "kind": "k5_6_bc_render_demo",
        "ckpt": str(ckpt_path),
        "seed": int(seed),
        "steps": int(steps),
        "terminated": bool(term),
        "truncated": bool(trunc),
        "terminal_reason": reason,
        "action_counts_LSR": action_counts,
        "frame_count": len(frames),
        "fps": int(fps),
        "video": str(video_path) if video_path else None,
        "video_error": video_err,
        "git_commit": _short_git_commit(),
        "screen": [SCREEN_W, SCREEN_H],
        "note": (
            "Faithful top-down render of the checkpoint's state-mode rollout "
            "(same env path as k5_6_bc_eval_inenv, bar 930.27). Player = cyan "
            "box, hazards = red boxes, true game geometry. See manifest 'ckpt'."
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"WROTE {video_path} frames={len(frames)} reason={reason}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_6_bc_render_demo")
    p.add_argument("--ckpt", type=Path,
                   default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "bc_policy.pt")
    p.add_argument("--seed", type=int, default=1009)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=1800)
    args = p.parse_args(argv)
    out = args.out or (REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "demo"
                       / f"seed{args.seed}")
    run(args.ckpt.resolve(), out.resolve(), int(args.seed),
        fps=int(args.fps), max_steps=int(args.max_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
