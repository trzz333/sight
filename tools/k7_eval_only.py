"""K7 in-env eval of already-trained TorchScript policies. GLOBAL interp.

Reuses collect_offline_dataset.eval_torchscript + _build_env (found-art
ADOPT: no eval logic rewritten). Evals cql_policy.pt and fbc_policy.pt on
held-out seeds vs the 930.27 bar and writes k7_eval_report.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(r"C:\Projects\Sight")
for p in (REPO / "src", REPO / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from collect_offline_dataset import _build_env, eval_torchscript, _parse_seeds, BAR  # noqa: E402


def main() -> int:
    out = REPO / "runs" / "phase_k" / "k7_offline" / "real"
    seeds = _parse_seeds("1000-1009")
    env = _build_env(observation_mode="state", run_dir=out / "eval_godot",
                     seed=seeds[0], max_steps=1800, reward_shaping="none")
    rep: dict = {}
    try:
        for name, pt in (("cql", out / "cql_policy.pt"),
                         ("filtered_bc", out / "fbc_policy.pt")):
            if pt.exists():
                print(f"[eval] {name}: {pt.name}", flush=True)
                rep[name] = eval_torchscript(env, pt, seeds, 1800)
            else:
                rep[name] = {"error": f"missing {pt}"}
    finally:
        env.close()
    (out / "k7_eval_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    line = []
    for name in ("cql", "filtered_bc"):
        r = rep.get(name, {})
        if "mean_episode_length" in r:
            line.append(f"{name} mean={r['mean_episode_length']} "
                        f"d_bar={r['delta_vs_bar']} {r['verdict']}")
        else:
            line.append(f"{name} {r.get('error', 'n/a')}")
    print("VERDICT " + " | ".join(line) + f" | bar={BAR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
