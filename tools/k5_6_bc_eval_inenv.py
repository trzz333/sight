"""K5.6 in-env greedy eval for the BC policy. THE verdict for the BC pivot.

Loads bc_policy.pt, runs greedy argmax over the 10-dim state obs against
the production GodotSignalDodgeEnv (state mode, headless, reward_shaping
"none", max_steps 1800) across held-out eval seeds 1000-1009 (disjoint
from BC training seeds 2000-2035). Reports per-seed survived steps, mean
episode length, collision/timeout split, vs the 930.27 bar (K5.2 best
constant 845.7 + 10% margin 84.57). BC val accuracy does NOT certify
survival under covariate shift; this in-env mean is the real number.

Mirrors the K5.2 layer-6 _run_one_episode loop and _build_env kwargs so
episode length is directly comparable to the 845.7 / 1762.8 baselines.

Usage (cmd, SIGHT_GODOT_EXE inline):
  python tools\\k5_6_bc_eval_inenv.py --ckpt runs\\phase_k\\k5_6_bc\\bc_policy.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from k5_2_env_dynamics_probe import _build_env  # noqa: E402
from k5_6_bc_train import BCPolicyNet  # noqa: E402

BAR = 930.27
BEST_CONSTANT = 845.7


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


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
        logits = model(torch.from_numpy(x).unsqueeze(0))
        return int(logits.argmax(1).item())


def run(ckpt_path: Path, out_dir: Path, seeds: list[int], max_steps: int = 1800) -> dict:
    model, mu, sd = load_policy(ckpt_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = _build_env(
        observation_mode="state",
        run_dir=out_dir / "godot",
        seed=int(seeds[0]),
        max_steps=max_steps,
        reward_shaping="none",
    )
    episodes: list[dict] = []
    try:
        for s in seeds:
            obs, info = env.reset(seed=int(s))
            steps = 0
            term = trunc = False
            reason = ""
            action_counts = [0, 0, 0]
            while steps < max_steps:
                a = greedy_action(model, mu, sd, obs)
                action_counts[a] += 1
                obs, r, term, trunc, info = env.step(a)
                steps += 1
                if term or trunc:
                    reason = info.get("terminal_reason", "")
                    break
            episodes.append({
                "seed": int(s), "steps": steps,
                "terminated": bool(term), "truncated": bool(trunc),
                "terminal_reason": reason, "action_counts_LSR": action_counts,
            })
            print(
                f"seed {s}: steps={steps} reason={reason} acts={action_counts}",
                flush=True,
            )
    finally:
        env.close()

    lengths = [e["steps"] for e in episodes]
    mean_len = float(np.mean(lengths))
    n = len(episodes)
    coll = sum(1 for e in episodes if e["terminal_reason"] == "collision") / n
    tmo = sum(1 for e in episodes if e["terminal_reason"] == "timeout") / n
    verdict = "PASS" if mean_len >= BAR else "FAIL"
    report = {
        "ckpt": str(ckpt_path),
        "eval_seeds": [int(s) for s in seeds],
        "max_steps": max_steps,
        "per_seed": episodes,
        "mean_episode_length": round(mean_len, 2),
        "min_episode_length": int(min(lengths)),
        "max_episode_length": int(max(lengths)),
        "collision_rate": round(coll, 3),
        "timeout_rate": round(tmo, 3),
        "bar": BAR,
        "best_constant": BEST_CONSTANT,
        "delta_vs_bar": round(mean_len - BAR, 2),
        "delta_vs_best_constant": round(mean_len - BEST_CONSTANT, 2),
        "verdict": verdict,
    }
    out_json = out_dir / "bc_eval_inenv_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"VERDICT {verdict} mean={round(mean_len, 2)} bar={BAR} "
        f"delta_vs_bar={round(mean_len - BAR, 2)} "
        f"coll={round(coll, 3)} timeout={round(tmo, 3)}",
        flush=True,
    )
    print("REPORT", json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="k5_6_bc_eval_inenv")
    p.add_argument(
        "--ckpt", type=Path,
        default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "bc_policy.pt",
    )
    p.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "runs" / "phase_k" / "k5_6_bc" / "eval_inenv",
    )
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--max-steps", type=int, default=1800)
    args = p.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("no seeds parsed")
    run(args.ckpt.resolve(), args.out.resolve(), seeds, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
