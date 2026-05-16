"""H5 amendment pre-training smoke parser.

Reads python.ndjson from the shaped and default seeded_random smoke
rollouts and evaluates the hard pass criteria recorded in
docs/h5-reward-amendment-proposal.md section 10 plus GPT's tightening
recorded in docs/h5-reward-amendment-smoke-evidence.md. Exit code is 0
if every criterion passes, 1 if any criterion fails, 2 on missing input.

Volatile per-run identity fields (ts_unix, run_id, godot_pid, tcp_port,
episode_id) are normalized out before the default-path schema check.
The criterion is normalized key-set equality plus reward semantics, not
literal byte equality.

Usage:

    python tools/h5_smoke_parse.py \
      --shaped  runs/rl/<run_name>/<shaped_run_id>/godot-eval-seeded_random/python.ndjson \
      --default runs/rl/<run_name>/<default_run_id>/godot-eval-seeded_random/python.ndjson

See docs/h5-reward-amendment-smoke-evidence.md for the reproduction
recipe (temp YAML configs under runs/smoke/, run_smoke.bat driver,
seeded_random policy, seeds 1000-1002, max_steps=600, windowed pixel
mode, SIGHT_GODOT_EXE set inline).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# Field set the default (reward_shaping=none) step path is contractually
# allowed to emit, AFTER normalizing out volatile per-run identity fields
# (ts_unix from writer; run_id, godot_pid, tcp_port from _log_event).
# The contract is normalized schema equality, not literal byte equality,
# per GPT's tightening of section 8 of the H5 amendment proposal.
VOLATILE_FIELDS = {"ts_unix", "run_id", "godot_pid", "tcp_port", "episode_id"}

DEFAULT_STEP_FIELDS_EXPECTED = {
    "type",
    "frame",
    "reward",
    "terminated",
    "truncated",
    "terminal_reason",
}

# Field set the shaped path adds on top of the default set.
SHAPED_ONLY_FIELDS = {
    "base_reward",
    "clearance_bonus",
    "threat_weight_sum",
    "active_hazard_count_above_player",
}

ALPHA = 0.05
FLOAT_TOL = 1e-9
BONUS_UPPER = ALPHA + FLOAT_TOL
SATURATION_THRESHOLD = ALPHA - 0.001  # 0.049


def load_steps(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"python.ndjson not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "step":
                rows.append(rec)
    return rows


def group_by_episode(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in steps:
        eid = str(s.get("episode_id", "?"))
        out[eid].append(s)
    # Sort each episode's steps by frame to be safe.
    for eid in out:
        out[eid].sort(key=lambda r: r.get("frame", 0))
    return out


def fmt_check(label: str, ok: bool, detail: str = "") -> str:
    tag = "PASS" if ok else "FAIL"
    if detail:
        return f"  [{tag}] {label} | {detail}"
    return f"  [{tag}] {label}"


def evaluate_shaped(ep_steps: list[dict[str, Any]], eid: str) -> dict[str, Any]:
    """Per-episode hard checks on the shaped run."""
    results: dict[str, Any] = {"episode_id": eid, "checks": [], "all_pass": True}
    n = len(ep_steps)
    results["n_steps"] = n
    if n == 0:
        results["all_pass"] = False
        results["checks"].append(("empty_episode", False, "0 step rows"))
        return results

    # Required-fields presence on every step.
    missing_field_steps = 0
    for s in ep_steps:
        for k in ("base_reward", "clearance_bonus", "threat_weight_sum",
                  "active_hazard_count_above_player"):
            if k not in s:
                missing_field_steps += 1
                break
    ok = missing_field_steps == 0
    results["checks"].append((
        "shaped_required_fields_present",
        ok,
        f"steps_missing_a_field={missing_field_steps}",
    ))
    results["all_pass"] &= ok

    # Identify terminal-collision step (terminated=True): last step if so.
    terminal_collision = ep_steps[-1].get("terminated") is True
    truncation = ep_steps[-1].get("truncated") is True and not terminal_collision
    results["terminal"] = (
        "collision" if terminal_collision else ("truncation" if truncation else "open")
    )

    # Bonus bounds across all non-terminal steps.
    out_of_bounds = []
    for s in ep_steps[:-1] if terminal_collision else ep_steps:
        b = s.get("clearance_bonus")
        if not isinstance(b, (int, float)):
            out_of_bounds.append(s.get("frame"))
            continue
        if b < -FLOAT_TOL or b > BONUS_UPPER:
            out_of_bounds.append(s.get("frame"))
    ok = len(out_of_bounds) == 0
    results["checks"].append((
        "clearance_bonus_in_[0,alpha]",
        ok,
        f"out_of_bounds_frames={out_of_bounds[:5]}{'...' if len(out_of_bounds)>5 else ''}",
    ))
    results["all_pass"] &= ok

    # Reward decomposition consistency.
    decomp_mismatch = []
    for s in ep_steps:
        r = s.get("reward")
        br = s.get("base_reward")
        cb = s.get("clearance_bonus")
        if not all(isinstance(x, (int, float)) for x in (r, br, cb)):
            continue
        if abs(float(r) - (float(br) + float(cb))) > 1e-6:
            decomp_mismatch.append(s.get("frame"))
    ok = len(decomp_mismatch) == 0
    results["checks"].append((
        "reward == base_reward + clearance_bonus (all steps)",
        ok,
        f"mismatch_frames={decomp_mismatch[:5]}{'...' if len(decomp_mismatch)>5 else ''}",
    ))
    results["all_pass"] &= ok

    # Base reward is only 0.0 or 1.0.
    bad_base = []
    for s in ep_steps:
        br = s.get("base_reward")
        if isinstance(br, (int, float)) and float(br) not in (0.0, 1.0):
            bad_base.append((s.get("frame"), br))
    ok = len(bad_base) == 0
    results["checks"].append((
        "base_reward in {0.0, 1.0}",
        ok,
        f"violations={bad_base[:5]}",
    ))
    results["all_pass"] &= ok

    # Terminal-collision step constraints.
    if terminal_collision:
        last = ep_steps[-1]
        cond = (
            abs(float(last.get("base_reward", 1.0))) < 1e-9
            and abs(float(last.get("clearance_bonus", 1.0))) < 1e-9
            and abs(float(last.get("reward", 1.0))) < 1e-9
        )
        results["checks"].append((
            "collision step base=0, bonus=0, reward=0",
            cond,
            f"base={last.get('base_reward')}, bonus={last.get('clearance_bonus')}, "
            f"reward={last.get('reward')}",
        ))
        results["all_pass"] &= cond
    else:
        results["checks"].append((
            "collision step base=0, bonus=0, reward=0",
            True,
            "no-collision-this-episode",
        ))

    # Step-level activity metrics.
    nonterm = ep_steps[:-1] if terminal_collision else ep_steps
    n_nt = len(nonterm)
    if n_nt == 0:
        results["checks"].append((
            "frac_nonterm_with_bonus>=0.20",
            False,
            "no non-terminal steps",
        ))
        results["all_pass"] = False
        return results
    bonus_vals = [float(s.get("clearance_bonus", 0.0)) for s in nonterm]
    tws_vals = [float(s.get("threat_weight_sum", 0.0)) for s in nonterm]
    frac_bonus = sum(1 for b in bonus_vals if b > 0.0) / n_nt
    frac_tws = sum(1 for t in tws_vals if t > 0.0) / n_nt
    mean_bonus = sum(bonus_vals) / n_nt
    active_idx = [i for i, t in enumerate(tws_vals) if t > 0.0]
    mean_bonus_active = (
        sum(bonus_vals[i] for i in active_idx) / len(active_idx)
        if active_idx else 0.0
    )
    sat_count = sum(1 for i in active_idx if bonus_vals[i] >= SATURATION_THRESHOLD)
    frac_sat = sat_count / len(active_idx) if active_idx else 0.0

    results["metrics"] = {
        "n_nonterm": n_nt,
        "frac_nonterm_with_bonus": frac_bonus,
        "frac_nonterm_with_active_threat": frac_tws,
        "mean_bonus_all_nonterm": mean_bonus,
        "mean_bonus_active_threat": mean_bonus_active,
        "frac_active_threat_saturated": frac_sat,
        "max_bonus_observed": max(bonus_vals) if bonus_vals else 0.0,
    }

    ok = frac_bonus >= 0.20
    results["checks"].append((
        "frac_nonterm_with_bonus >= 0.20",
        ok,
        f"={frac_bonus:.3f}",
    ))
    results["all_pass"] &= ok

    ok = frac_tws >= 0.20
    results["checks"].append((
        "frac_nonterm_with_active_threat >= 0.20",
        ok,
        f"={frac_tws:.3f}",
    ))
    results["all_pass"] &= ok

    ok = 0.005 <= mean_bonus <= 0.045
    results["checks"].append((
        "mean_bonus_all_nonterm in [0.005, 0.045]",
        ok,
        f"={mean_bonus:.5f}",
    ))
    results["all_pass"] &= ok

    if active_idx:
        ok = 0.01 <= mean_bonus_active <= 0.045
        results["checks"].append((
            "mean_bonus_active_threat in [0.01, 0.045]",
            ok,
            f"={mean_bonus_active:.5f}",
        ))
        results["all_pass"] &= ok

        ok = frac_sat < 0.50
        results["checks"].append((
            "frac_active_threat_saturated < 0.50",
            ok,
            f"={frac_sat:.3f}",
        ))
        results["all_pass"] &= ok
    else:
        results["checks"].append((
            "active-threat statistics",
            False,
            "no active-threat steps in episode",
        ))
        results["all_pass"] = False

    return results


def evaluate_default(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Default-path checks: schema match + reward semantics."""
    results: dict[str, Any] = {"checks": [], "all_pass": True, "n_steps": len(steps)}
    if not steps:
        results["all_pass"] = False
        results["checks"].append(("empty_default_run", False, "0 step rows"))
        return results

    # Normalized key-set comparison: each row's key set must equal
    # DEFAULT_STEP_FIELDS_EXPECTED. No shaped-only field is permitted.
    bad_rows = []
    shaped_field_hits = 0
    for s in steps:
        keys = set(s.keys()) - VOLATILE_FIELDS
        if keys != DEFAULT_STEP_FIELDS_EXPECTED:
            bad_rows.append((s.get("frame"), sorted(keys ^ DEFAULT_STEP_FIELDS_EXPECTED)))
        if keys & SHAPED_ONLY_FIELDS:
            shaped_field_hits += 1
    ok_schema = len(bad_rows) == 0 and shaped_field_hits == 0
    results["checks"].append((
        "default schema == pre-amendment field set",
        ok_schema,
        f"row_mismatches={bad_rows[:3]}, shaped_field_hits={shaped_field_hits}",
    ))
    results["all_pass"] &= ok_schema

    # Reward semantics: only 0.0 or 1.0; 0.0 only on terminated=True.
    bad_reward = []
    for s in steps:
        r = s.get("reward")
        term = s.get("terminated")
        if not isinstance(r, (int, float)):
            bad_reward.append((s.get("frame"), "non-numeric"))
            continue
        if float(r) not in (0.0, 1.0):
            bad_reward.append((s.get("frame"), r))
            continue
        if float(r) == 0.0 and term is not True:
            bad_reward.append((s.get("frame"), "zero_reward_without_terminated"))
    ok = len(bad_reward) == 0
    results["checks"].append((
        "default reward in {0.0,1.0}; 0.0 only on terminated",
        ok,
        f"violations={bad_reward[:5]}",
    ))
    results["all_pass"] &= ok

    return results


def cross_check_default_vs_shaped(
    default_eps: dict[str, list[dict[str, Any]]],
    shaped_eps: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Same seeded_random seed should produce identical trajectory length
    and terminal state under shaped vs default because reward shaping is
    Python-side only and must not affect Godot physics or action stream.

    Episode IDs differ between runs (Godot uses per-reset UUIDs), so the
    pairing is positional: nth episode_id in shaped is paired with nth
    episode_id in default, by reset order. With the same seed list and
    one Godot launch per policy this is deterministic.
    """
    results: dict[str, Any] = {"checks": [], "all_pass": True}
    s_eids = sorted(shaped_eps.keys(), key=lambda e: shaped_eps[e][0].get("frame", 0))
    d_eids = sorted(default_eps.keys(), key=lambda e: default_eps[e][0].get("frame", 0))
    # Better: sort by ts_unix of first step.
    s_eids = sorted(shaped_eps.keys(), key=lambda e: shaped_eps[e][0].get("ts_unix", 0.0))
    d_eids = sorted(default_eps.keys(), key=lambda e: default_eps[e][0].get("ts_unix", 0.0))

    if len(s_eids) != len(d_eids):
        results["checks"].append((
            "episode_count(shaped) == episode_count(default)",
            False,
            f"shaped={len(s_eids)}, default={len(d_eids)}",
        ))
        results["all_pass"] = False
        return results
    results["checks"].append((
        "episode_count(shaped) == episode_count(default)",
        True,
        f"={len(s_eids)}",
    ))

    mismatches = []
    for i, (se, de) in enumerate(zip(s_eids, d_eids)):
        sh = shaped_eps[se]
        dh = default_eps[de]
        s_len = len(sh)
        d_len = len(dh)
        s_term = sh[-1].get("terminated")
        d_term = dh[-1].get("terminated")
        s_trunc = sh[-1].get("truncated")
        d_trunc = dh[-1].get("truncated")
        if s_len != d_len or s_term != d_term or s_trunc != d_trunc:
            mismatches.append({
                "pair_index": i,
                "shaped_eid": se,
                "default_eid": de,
                "shaped_len": s_len,
                "default_len": d_len,
                "shaped_term_trunc": (s_term, s_trunc),
                "default_term_trunc": (d_term, d_trunc),
            })
    ok = not mismatches
    results["checks"].append((
        "per-seed trajectory parity (length + terminal flags)",
        ok,
        f"mismatches={mismatches}",
    ))
    results["all_pass"] &= ok
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shaped", required=True, type=Path)
    p.add_argument("--default", required=True, type=Path,
                   dest="default_path")
    args = p.parse_args()

    try:
        shaped_steps = load_steps(args.shaped)
        default_steps = load_steps(args.default_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    shaped_eps = group_by_episode(shaped_steps)
    default_eps = group_by_episode(default_steps)

    all_pass = True

    print("=" * 78)
    print("H5 amendment pre-training smoke parse")
    print("=" * 78)
    print(f"shaped python.ndjson:  {args.shaped}")
    print(f"  step rows: {len(shaped_steps)}, episodes: {len(shaped_eps)}")
    print(f"default python.ndjson: {args.default_path}")
    print(f"  step rows: {len(default_steps)}, episodes: {len(default_eps)}")
    print()

    print("--- SHAPED RUN ----------------------------------------------------")
    for eid in sorted(shaped_eps.keys(),
                      key=lambda e: shaped_eps[e][0].get("ts_unix", 0.0)):
        ep = shaped_eps[eid]
        res = evaluate_shaped(ep, eid)
        m = res.get("metrics", {})
        print(f"episode {eid} | n_steps={res['n_steps']} | terminal={res['terminal']}")
        if m:
            print(f"  frac_bonus={m['frac_nonterm_with_bonus']:.3f} "
                  f"frac_active={m['frac_nonterm_with_active_threat']:.3f} "
                  f"mean_bonus={m['mean_bonus_all_nonterm']:.5f} "
                  f"mean_bonus_active={m['mean_bonus_active_threat']:.5f} "
                  f"frac_sat={m['frac_active_threat_saturated']:.3f} "
                  f"max_bonus={m['max_bonus_observed']:.5f}")
        for label, ok, detail in res["checks"]:
            print(fmt_check(label, ok, detail))
        if not res["all_pass"]:
            all_pass = False
    print()

    print("--- DEFAULT RUN ---------------------------------------------------")
    def_res = evaluate_default(default_steps)
    print(f"default: n_steps={def_res['n_steps']}")
    for label, ok, detail in def_res["checks"]:
        print(fmt_check(label, ok, detail))
    if not def_res["all_pass"]:
        all_pass = False
    print()

    print("--- CROSS-CHECK DEFAULT vs SHAPED ---------------------------------")
    cross = cross_check_default_vs_shaped(default_eps, shaped_eps)
    for label, ok, detail in cross["checks"]:
        print(fmt_check(label, ok, detail))
    if not cross["all_pass"]:
        all_pass = False
    print()

    print("=" * 78)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
