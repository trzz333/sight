"""Phase M / M2 - greedy eval of from-scratch PPO policies on Signal Dodge.

Loads one or more M2 train run dirs (each with model.zip), rolls each
policy deterministically on GodotSignalDodgeEnv (state obs, reward "none")
over held-out eval seeds 1000-1009, and applies the M2 PASS gate:

    PASS (per training seed) =
        mean_episode_length >= 930.27
        AND frac_left  >= 0.03
        AND frac_right >= 0.03
        AND max(action_fraction) < 0.97

Under reward "none" (+1 per surviving step) episode length == total
reward, so the survival bar applies to mean length directly. The gate's
action-fraction clauses reject the degenerate stay-pinned / single-action
collapse that K5.1/K5.5 fell into even when survival was incidentally high.

Overall M2 verdict mirrors the project's dependability posture: report
how many training seeds individually clear the full gate. Single-seed
peaks are not a reliability claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

SURVIVAL_BAR = 930.27
MAX_STEPS = 1800
ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}
DEFAULT_EXE = (
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64.exe"
)
DEFAULT_PROJECT = str(_REPO_ROOT / "games" / "signal-dodge")


def _alloc_port() -> int:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _build_eval_env(*, port: int, run_dir: str, exe: str, project: str):
    from sight_agent.rl.godot_env import GodotSignalDodgeEnv
    return GodotSignalDodgeEnv(
        godot_executable=exe, project_path=project, run_dir=run_dir,
        max_steps=MAX_STEPS, headless=True, observation_mode="state",
        reward_shaping="none", tcp_port=port,
    )


def parse_runs(s: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        label, path = tok.split("=", 1)
        out[label.strip()] = Path(path.strip())
    if not out:
        raise ValueError("no --runs entries")
    return out


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


def roll_episode(model, vec, seed: int) -> dict:
    try:
        vec.seed(int(seed))
    except Exception:
        pass
    obs = vec.reset()
    counts = {0: 0, 1: 0, 2: 0}
    length = 0
    while length < MAX_STEPS:
        pred, _ = model.predict(np.asarray(obs), deterministic=True)
        act = int(np.asarray(pred).reshape(-1)[0])
        obs, rewards, dones, infos = vec.step(np.asarray([act]))
        counts[act] += 1
        length += 1
        if bool(np.asarray(dones).any()):
            break
    return {"seed": int(seed), "length": int(length), "action_counts": counts}


def eval_model(label: str, run_dir: Path, vec, seeds: list[int]) -> dict:
    model = PPO.load(str(run_dir / "model.zip"), env=None, device="cpu")
    eps = [roll_episode(model, vec, s) for s in seeds]
    lengths = [e["length"] for e in eps]
    tot = {0: 0, 1: 0, 2: 0}
    for e in eps:
        for a in (0, 1, 2):
            tot[a] += e["action_counts"][a]
    total = sum(tot.values()) or 1
    frac = {ACTION_NAMES[a]: tot[a] / total for a in (0, 1, 2)}
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    max_frac = max(frac.values())
    passed = (
        mean_len >= SURVIVAL_BAR
        and frac["left"] >= 0.03
        and frac["right"] >= 0.03
        and max_frac < 0.97
    )
    return {
        "label": label,
        "run_dir": str(run_dir),
        "mean_episode_length": mean_len,
        "min_episode_length": float(min(lengths)) if lengths else 0.0,
        "max_episode_length": float(max(lengths)) if lengths else 0.0,
        "n_seeds": len(seeds),
        "action_fractions": frac,
        "max_action_fraction": max_frac,
        "per_seed_length": {str(e["seed"]): e["length"] for e in eps},
        "gate_pass": bool(passed),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m2_state_ppo_eval")
    p.add_argument("--runs", required=True,
                   help="label=run_dir,label2=run_dir2 (each holds model.zip)")
    p.add_argument("--seeds", default="1000-1009")
    p.add_argument("--exe", default=DEFAULT_EXE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    runs = parse_runs(args.runs)
    seeds = parse_seeds(args.seeds)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eval_run_dir = out / "godot_eval"
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    port = _alloc_port()
    vec = DummyVecEnv([
        partial(_build_eval_env, port=port, run_dir=str(eval_run_dir),
                exe=args.exe, project=args.project)
    ])

    t0 = time.time()
    results = []
    try:
        for label, run_dir in runs.items():
            if not (run_dir / "model.zip").is_file():
                results.append({"label": label, "run_dir": str(run_dir),
                                "error": "model.zip missing", "gate_pass": False})
                continue
            results.append(eval_model(label, run_dir, vec, seeds))
    finally:
        try:
            vec.close()
        except Exception:
            pass

    n_pass = sum(1 for r in results if r.get("gate_pass"))
    verdict = "M2-PASS" if n_pass >= 1 else "M2-FAIL"
    summary = {
        "phase": "M2-eval",
        "survival_bar": SURVIVAL_BAR,
        "eval_seeds": seeds,
        "gate": "mean>=930.27 AND frac_L>=0.03 AND frac_R>=0.03 AND max(frac)<0.97",
        "n_train_seeds_pass": n_pass,
        "n_train_seeds_total": len(runs),
        "verdict": verdict,
        "per_model": results,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with (out / "m2_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
