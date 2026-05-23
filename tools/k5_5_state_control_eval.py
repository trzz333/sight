"""K5.5 state-observation PPO control eval.

Loads the three K5.5 state-observation PPO checkpoints (train seeds
0/1/2), rolls each deterministically on the Godot Signal Dodge env over
eval seeds 1000-1009, records per-step action distributions, and
classifies the pooled result into one of the five K5.5 buckets.

K5.5 isolates whether K5.1's stay-pinned argmax is caused by the
single-frame pixel representation or by PPO/objective/budget failure
even when production state geometry is handed directly to the policy.

No training. No env code edit. Loads only committed train artifacts.
Reuses tools/h5_logit_compare.py helpers (load_and_fingerprint,
get_action_logits, obs_hash); PPO.load is not CnnPolicy-specific so
those helpers apply unchanged to the MlpPolicy state checkpoints.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
_TOOLS = _REPO_ROOT / "tools"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from sight_agent.rl.config import load_config  # noqa: E402
from sight_agent.rl.factories import make_env  # noqa: E402
from sight_agent.rl.godot_config import resolve_godot_kwargs  # noqa: E402

import h5_logit_compare as h5lc  # noqa: E402

ACTION_NAMES = {0: "left", 1: "stay", 2: "right"}
MATERIAL_SURVIVAL_BAR = 930.27


def parse_models_arg(models_str: str) -> dict[str, Path]:
    """Parse ``label1=dir1,label2=dir2`` into an ordered label->path map.

    Accepts one or more entries (K5.5 supplies three train-seed dirs).
    """
    out: dict[str, Path] = {}
    for tok in models_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"--models token missing '=': {tok!r}")
        label, path = tok.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--models token has empty label or path: {tok!r}")
        if label in out:
            raise ValueError(f"--models label repeated: {label!r}")
        out[label] = Path(path)
    if not out:
        raise ValueError("at least one --models entry required")
    return out


def parse_seed_range(spec: str) -> list[int]:
    """Parse ``1000-1009`` or comma-separated ints/ranges into a seed list."""
    out: list[int] = []
    for raw in spec.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"hi<lo in seed range: {tok!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("no eval seeds parsed")
    return out


def _stats(vals: list[float]) -> dict[str, float | int]:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def run_one_episode(
    model: Any, env: Any, label: str, eval_seed: int, max_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Roll one deterministic episode; return (step_rows, episode_summary)."""
    try:
        env.seed(int(eval_seed))
    except (AttributeError, TypeError):
        pass
    obs = env.reset()
    rows: list[dict[str, Any]] = []
    action_counts = {0: 0, 1: 0, 2: 0}
    total_reward = 0.0
    player_x_vals: list[float] = []
    step = 0
    collision = False
    timeout = False
    t0 = time.time()
    while step < int(max_steps):
        obs_arr = np.asarray(obs)
        pred, _ = model.predict(obs_arr, deterministic=True)
        act = int(np.asarray(pred).reshape(-1)[0])
        _logits, probs, entropy_v, argmax = h5lc.get_action_logits(model, obs_arr)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        obs, rewards, dones, infos = env.step(np.asarray([act]))
        reward = float(np.asarray(rewards).reshape(-1)[0])
        total_reward += reward
        action_counts[act] += 1
        info = infos[0] if isinstance(infos, (list, tuple)) and infos else {}
        godot_info = info.get("godot_info") if isinstance(info, dict) else None
        reward_state = (
            godot_info.get("reward_state") if isinstance(godot_info, dict) else None
        )
        player_x = None
        if isinstance(reward_state, dict) and reward_state.get("player_x") is not None:
            player_x = float(reward_state["player_x"])
            player_x_vals.append(player_x)
        done = bool(np.asarray(dones).any())
        truncated = bool(info.get("TimeLimit.truncated", False)) if done else False
        rows.append({
            "kind": "step", "label": label, "eval_seed": int(eval_seed),
            "step": int(step), "obs_hash": h5lc.obs_hash(obs_arr),
            "action": int(act), "p_left": float(probs[0]),
            "p_stay": float(probs[1]), "p_right": float(probs[2]),
            "argmax": int(argmax), "predict_argmax_match": bool(act == int(argmax)),
            "entropy": float(entropy_v), "reward": reward, "player_x": player_x,
            "done": done, "truncated": truncated if done else None,
        })
        step += 1
        if done:
            collision = not truncated
            timeout = truncated
            break
    else:
        timeout = True
    ep = {
        "kind": "episode", "label": label, "eval_seed": int(eval_seed),
        "episode_length": int(step), "total_reward": float(total_reward),
        "collision": bool(collision), "timeout": bool(timeout),
        "action_counts": {ACTION_NAMES[a]: int(action_counts[a]) for a in (0, 1, 2)},
        "player_x": _stats(player_x_vals),
        "elapsed_seconds": float(time.time() - t0),
    }
    return rows, ep


def _action_fractions(action_counts: dict[int, int]) -> dict[str, float]:
    total = sum(action_counts.values())
    if total == 0:
        return {"left": 0.0, "stay": 0.0, "right": 0.0, "nonstay": 0.0}
    f = {ACTION_NAMES[a]: action_counts[a] / total for a in (0, 1, 2)}
    f["nonstay"] = f["left"] + f["right"]
    return f


def summarize_model(
    label: str, fp: dict[str, Any], episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-model summary across that model's eval episodes."""
    lengths = [e["episode_length"] for e in episodes]
    rewards = [e["total_reward"] for e in episodes]
    n = len(episodes)
    collisions = sum(1 for e in episodes if e["collision"])
    timeouts = sum(1 for e in episodes if e["timeout"])
    counts = {0: 0, 1: 0, 2: 0}
    for e in episodes:
        for a in (0, 1, 2):
            counts[a] += int(e["action_counts"][ACTION_NAMES[a]])
    px_all: list[float] = []
    for e in episodes:
        s = e["player_x"]
        if s.get("n", 0) > 0:
            px_all.extend([s["min"], s["max"], s["mean"]])
    return {
        "label": label,
        "model_fingerprint": {
            "sha256": fp.get("sha256"),
            "param_count": fp.get("param_count"),
            "state_dict_blake2b16": fp.get("state_dict_blake2b16"),
        },
        "n_eval_seeds": int(n),
        "episode_length": _stats([float(x) for x in lengths]),
        "mean_episode_length": float(statistics.fmean(lengths)) if lengths else 0.0,
        "collision_rate": (collisions / n) if n else 0.0,
        "timeout_rate": (timeouts / n) if n else 0.0,
        "reward_mean": float(statistics.fmean(rewards)) if rewards else 0.0,
        "action_counts": {ACTION_NAMES[a]: int(counts[a]) for a in (0, 1, 2)},
        "action_fractions": _action_fractions(counts),
        "player_x": _stats(px_all),
    }


def classify_k5_5(
    model_summaries: list[dict[str, Any]],
    pooled: dict[str, Any],
    mechanical_ok: bool,
) -> dict[str, Any]:
    """Five-bucket K5.5 classification. First match wins, packet order."""
    success_seed_count = sum(
        1 for m in model_summaries
        if m["mean_episode_length"] >= MATERIAL_SURVIVAL_BAR
    )
    pcr = pooled["collision_rate"]
    pf = pooled["action_fractions"]
    inputs = {
        "material_survival_bar": MATERIAL_SURVIVAL_BAR,
        "success_seed_count": int(success_seed_count),
        "pooled_collision_rate": pcr,
        "pooled_stay_action_fraction": pf["stay"],
        "pooled_nonstay_action_fraction": pf["nonstay"],
        "pooled_left_action_fraction": pf["left"],
        "pooled_right_action_fraction": pf["right"],
        "per_seed_mean_episode_length": {
            m["label"]: m["mean_episode_length"] for m in model_summaries
        },
    }
    if not mechanical_ok:
        return {"bucket": "STATE-CONTROL-UNKNOWN",
                "reason": "mechanical failure during eval", "inputs": inputs}
    if (success_seed_count >= 2 and pcr <= 0.80 and pf["nonstay"] >= 0.20
            and pf["left"] >= 0.03 and pf["right"] >= 0.03):
        return {"bucket": "STATE-CONTROL-PASS",
                "reason": ">=2 seeds clear survival bar; pooled collision<=0.80; "
                          "bidirectional non-stay motion", "inputs": inputs}
    if success_seed_count == 1:
        return {"bucket": "STATE-CONTROL-SEED-SENSITIVE",
                "reason": "exactly 1 of 3 seeds clears survival bar", "inputs": inputs}
    if success_seed_count == 0 and pf["stay"] >= 0.90 and pcr >= 0.80:
        return {"bucket": "STATE-CONTROL-FAIL-STAY",
                "reason": "0 seeds clear bar; pooled stay>=0.90; collision>=0.80",
                "inputs": inputs}
    if success_seed_count == 0 and pf["nonstay"] >= 0.20 and pcr >= 0.80:
        return {"bucket": "STATE-CONTROL-FAIL-ACTIVE-BAD",
                "reason": "0 seeds clear bar; pooled non-stay>=0.20; collision>=0.80",
                "inputs": inputs}
    return {"bucket": "STATE-CONTROL-UNKNOWN",
            "reason": "no bucket rule matched (mixed signal)", "inputs": inputs}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="k5_5_state_control_eval",
        description="K5.5 state-observation PPO control deterministic eval.",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--models", required=True,
                   help="Comma-separated label=train_run_dir pairs.")
    p.add_argument("--seeds", required=True, help="e.g. 1000-1009")
    p.add_argument("--max-steps", type=int, default=1800)
    p.add_argument("--out-dir", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    models_map = parse_models_arg(args.models)
    seeds = parse_seed_range(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = out_dir / "k5_5_state_control.ndjson"
    summary_path = out_dir / "k5_5_state_control.summary.json"

    loaded: list[tuple[str, Any, dict[str, Any]]] = []
    seen_sha: dict[str, str] = {}
    for label, run_dir in models_map.items():
        model, fp = h5lc.load_and_fingerprint(Path(run_dir))
        if fp["sha256"] in seen_sha:
            print(f"WARN: {label} shares sha256 with {seen_sha[fp['sha256']]}")
        seen_sha[fp["sha256"]] = label
        loaded.append((label, model, fp))
    distinct_models = len(seen_sha) == len(loaded)

    env_id = cfg["env"]["id"]
    base_seed = int(cfg.get("run", {}).get("seed", 0))
    godot_extra = resolve_godot_kwargs(cfg)
    eval_run_dir = out_dir / "godot_eval"
    eval_run_dir.mkdir(parents=True, exist_ok=True)
    if godot_extra:
        env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval",
                       run_dir=str(eval_run_dir), **godot_extra)
    else:
        env = make_env(env_id, n_envs=1, seed=base_seed, mode="eval")

    header = {
        "_header": True, "kind": "header",
        "tool": "tools/k5_5_state_control_eval.py",
        "config": str(args.config),
        "models": {l: fp for l, _, fp in loaded},
        "eval_seeds": seeds, "max_steps": int(args.max_steps),
        "distinct_models": bool(distinct_models),
        "ran_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    started = time.time()
    episodes_by_label: dict[str, list[dict[str, Any]]] = {l: [] for l, _, _ in loaded}
    mechanical_ok = True
    fh = ndjson_path.open("w", encoding="utf-8", newline="")
    try:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for label, model, _fp in loaded:
            for eval_seed in seeds:
                try:
                    rows, ep = run_one_episode(
                        model, env, label, int(eval_seed), int(args.max_steps))
                except Exception as exc:  # noqa: BLE001
                    mechanical_ok = False
                    print(f"ERROR {label} seed={eval_seed}: {type(exc).__name__} {exc}")
                    continue
                for r in rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                fh.write(json.dumps(ep, separators=(",", ":")) + "\n")
                episodes_by_label[label].append(ep)
                print(f"  {label} seed={eval_seed} len={ep['episode_length']} "
                      f"coll={ep['collision']} to={ep['timeout']}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        fh.close()
    return _finalize(loaded, episodes_by_label, header, summary_path,
                     mechanical_ok, time.time() - started)


def _finalize(
    loaded: list[tuple[str, Any, dict[str, Any]]],
    episodes_by_label: dict[str, list[dict[str, Any]]],
    header: dict[str, Any],
    summary_path: Path,
    mechanical_ok: bool,
    elapsed: float,
) -> int:
    """Build per-model + pooled summary, classify, write summary.json."""
    model_summaries: list[dict[str, Any]] = []
    for label, _model, fp in loaded:
        eps = episodes_by_label[label]
        if not eps:
            mechanical_ok = False
        model_summaries.append(summarize_model(label, fp, eps))

    all_eps = [e for eps in episodes_by_label.values() for e in eps]
    pooled_counts = {0: 0, 1: 0, 2: 0}
    for e in all_eps:
        for a in (0, 1, 2):
            pooled_counts[a] += int(e["action_counts"][ACTION_NAMES[a]])
    n_all = len(all_eps)
    pooled = {
        "n_episodes": int(n_all),
        "episode_length": _stats([float(e["episode_length"]) for e in all_eps]),
        "collision_rate": (
            sum(1 for e in all_eps if e["collision"]) / n_all) if n_all else 0.0,
        "timeout_rate": (
            sum(1 for e in all_eps if e["timeout"]) / n_all) if n_all else 0.0,
        "reward_mean": (
            float(statistics.fmean([e["total_reward"] for e in all_eps]))
            if all_eps else 0.0),
        "action_counts": {ACTION_NAMES[a]: int(pooled_counts[a]) for a in (0, 1, 2)},
        "action_fractions": _action_fractions(pooled_counts),
    }
    classification = classify_k5_5(model_summaries, pooled, mechanical_ok)
    summary = {
        "_header": header,
        "elapsed_seconds": float(elapsed),
        "mechanical_ok": bool(mechanical_ok),
        "per_model": model_summaries,
        "pooled": pooled,
        "k5_5_classification": classification,
        "comparison_anchors": {
            "k5_1_pixel_alpha030_deterministic": {
                "mean_episode_length": 606.0, "collision_rate": 1.00},
            "k5_2_best_constant": 845.7,
            "material_survival_bar": MATERIAL_SURVIVAL_BAR,
            "k5_2_k5_4_oracle": 1762.8,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"k5_5 done: episodes={n_all} elapsed={elapsed:.1f}s "
          f"verdict={classification['bucket']} "
          f"pooled_len_mean={pooled['episode_length'].get('mean')} "
          f"pooled_collision={pooled['collision_rate']} "
          f"out={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
