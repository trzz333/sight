"""K3.5 evidence extractor: parse K3.5 NDJSONs and compute gate metrics.

Reads:
  runs/phase_k/k3_5_reward_scale_div30_2048.ndjson
  runs/phase_k/k3_5_reward_scale_div100_2048.ndjson
  runs/phase_k/k3_3_baseline_vfcoef_0_5_2048.ndjson  (reference)
  runs/phase_k/k3_4_value_bias_first_rollout_mean_2048.ndjson  (reference)

Emits:
  runs/phase_k/k3_5_reward_scale_table.csv  (per-update extraction)
  Prints gate decision summary to stdout.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PHASE_K = Path(r"C:\Projects\Sight\runs\phase_k")


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("_header"):
                header = obj
            else:
                records.append(obj)
    assert header is not None
    return header, records


def extract_per_update(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in records:
        post = r["post_update"]
        fp = post.get("fixed_panel_policy_state")
        lvf = fp["feature_chain_diversity"]["latent_vf"] if fp else None
        vp = fp["feature_chain_diversity"]["value_predictions"] if fp else None
        adv = r["adv_ret_val_stats"]
        vf = r["value_fit"]
        out.append({
            "update_idx": int(r["update_idx"]),
            "lvf_post": int(lvf["n_dims_above_eps"]) if lvf else 0,
            "vp_std_post": float(vp["std"]) if vp else 0.0,
            "vp_mean_post": float(vp["mean"]) if vp else 0.0,
            "ret_mean": float(adv["returns"]["mean"]),
            "ret_std": float(adv["returns"]["std"]),
            "val_mean_rollout": float(adv["values"]["mean"]),
            "post_val_mean_rollout": float(vf["post_rollout"]["value_pred_mean"]),
            "pre_ratio": vf["pre_rollout"]["model_to_constant_mse_ratio"],
            "post_ratio": vf["post_rollout"]["model_to_constant_mse_ratio"],
            "value_loss_mean": float(r["losses"]["value_loss_mean"]),
            "policy_grad_loss_mean": float(r["losses"]["policy_gradient_loss_mean"]),
            "entropy_mean_post": float(post["policy_state"]["entropy_mean"]),
            "gn_total_mean": float(r["grad_norms_preclip"]["total_mean"]),
            "gn_mlp_vf_w_mean": float(r["grad_norms_preclip"]["mlp_value_net_weights_mean"]),
            "gn_scalar_vh_mean": float(r["grad_norms_preclip"]["scalar_value_head_mean"]),
            "ev": float(r["explained_variance"]),
        })
    return out


def deltas(values: list[float]) -> list[float]:
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def run_summary(label: str, divisor: float, records: list[dict[str, Any]]) -> None:
    rows = extract_per_update(records)
    print(f"\n=== {label} (divisor={divisor}) ===")
    print(f"{'u':>2} {'lvf_post':>9} {'vp_std_post':>12} {'post_ratio':>10} "
          f"{'ret_mean_scaled':>15} {'ret_mean_raw':>13} {'v_loss':>10} {'|g|':>8}")
    for row in rows:
        raw_ret = row["ret_mean"] * divisor
        pr = row["post_ratio"]
        pr_str = f"{pr:.4f}" if pr is not None else "  None"
        print(f"{row['update_idx']:>2} {row['lvf_post']:>9} "
              f"{row['vp_std_post']:>12.4e} {pr_str:>10} "
              f"{row['ret_mean']:>15.4f} {raw_ret:>13.4f} "
              f"{row['value_loss_mean']:>10.4f} {row['gn_total_mean']:>8.4f}")

    # Drift metrics
    scaled_ret = [r["ret_mean"] for r in rows]
    raw_ret = [r * divisor for r in scaled_ret]
    if len(scaled_ret) >= 2:
        scaled_first_to_last = scaled_ret[-1] / scaled_ret[0] if scaled_ret[0] != 0 else float("nan")
        raw_first_to_last = raw_ret[-1] / raw_ret[0] if raw_ret[0] != 0 else float("nan")
        scaled_deltas = deltas(scaled_ret)
        raw_deltas = deltas(raw_ret)
        print(f"\n  ret_mean drift summary:")
        print(f"    scaled first-to-last ratio: {scaled_first_to_last:.4f}x")
        print(f"    raw    first-to-last ratio: {raw_first_to_last:.4f}x")
        print(f"    scaled per-update deltas: {[f'{d:+.4f}' for d in scaled_deltas]}")
        print(f"    raw    per-update deltas: {[f'{d:+.4f}' for d in raw_deltas]}")

    # Gates
    if len(rows) >= 3:
        u3 = rows[2]
        print(f"\n  PRIMARY GATE (update 3):")
        print(f"    lvf_post = {u3['lvf_post']} (need >= 16): "
              f"{'PASS' if u3['lvf_post'] >= 16 else 'FAIL'}")
        print(f"    vp_std_post = {u3['vp_std_post']:.4e} (need > 1e-6): "
              f"{'PASS' if u3['vp_std_post'] > 1e-6 else 'FAIL'}")
        pr3 = u3["post_ratio"]
        if pr3 is not None:
            print(f"    post_ratio = {pr3:.4f} (need < 22.84 K3.3 baseline): "
                  f"{'PASS' if pr3 < 22.84 else 'FAIL'}")
        else:
            print(f"    post_ratio = None")
    if len(rows) >= 8:
        u8 = rows[7]
        print(f"  STRONG GATE (update 8):")
        print(f"    lvf_post = {u8['lvf_post']} (need >= 16): "
              f"{'PASS' if u8['lvf_post'] >= 16 else 'FAIL'}")
        print(f"    vp_std_post = {u8['vp_std_post']:.4e} (need > 1e-6): "
              f"{'PASS' if u8['vp_std_post'] > 1e-6 else 'FAIL'}")
        pr8 = u8["post_ratio"]
        if pr8 is not None:
            print(f"    post_ratio = {pr8:.4f} (need < 6.18 K3.3 baseline): "
                  f"{'PASS' if pr8 < 6.18 else 'FAIL'}")
        else:
            print(f"    post_ratio = None")
    return rows


def write_csv(rows_30: list[dict[str, Any]], rows_100: list[dict[str, Any]]) -> None:
    out = PHASE_K / "k3_5_reward_scale_table.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "divisor", "update_idx", "lvf_post", "vp_std_post", "vp_mean_post",
            "ret_mean_scaled", "ret_mean_raw", "ret_std_scaled",
            "val_mean_rollout_scaled", "post_val_mean_rollout_scaled",
            "pre_ratio", "post_ratio",
            "value_loss_mean", "policy_grad_loss_mean", "entropy_mean_post",
            "gn_total_mean", "gn_mlp_vf_w_mean", "gn_scalar_vh_mean", "ev",
        ])
        for rows, div in [(rows_30, 30.0), (rows_100, 100.0)]:
            for r in rows:
                w.writerow([
                    div, r["update_idx"], r["lvf_post"], r["vp_std_post"], r["vp_mean_post"],
                    r["ret_mean"], r["ret_mean"] * div, r["ret_std"],
                    r["val_mean_rollout"], r["post_val_mean_rollout"],
                    r["pre_ratio"], r["post_ratio"],
                    r["value_loss_mean"], r["policy_grad_loss_mean"], r["entropy_mean_post"],
                    r["gn_total_mean"], r["gn_mlp_vf_w_mean"], r["gn_scalar_vh_mean"], r["ev"],
                ])
    print(f"\nwrote {out}")


def main() -> int:
    _, recs_30 = load_run(PHASE_K / "k3_5_reward_scale_div30_2048.ndjson")
    _, recs_100 = load_run(PHASE_K / "k3_5_reward_scale_div100_2048.ndjson")
    rows_30 = run_summary("K3.5 /30", 30.0, recs_30)
    rows_100 = run_summary("K3.5 /100", 100.0, recs_100)
    write_csv(rows_30, rows_100)

    # K3.4 reference for comparison
    print("\n=== REFERENCE: K3.4 bias-init (divisor=1, no scaling) ===")
    _, recs_k34 = load_run(PHASE_K / "k3_4_value_bias_first_rollout_mean_2048.ndjson")
    rows_k34 = extract_per_update(recs_k34)
    print(f"{'u':>2} {'lvf_post':>9} {'vp_std_post':>12} {'post_ratio':>10} "
          f"{'ret_mean':>10} {'v_loss':>10} {'|g|':>8}")
    for row in rows_k34:
        pr = row["post_ratio"]
        pr_str = f"{pr:.4f}" if pr is not None else "  None"
        print(f"{row['update_idx']:>2} {row['lvf_post']:>9} "
              f"{row['vp_std_post']:>12.4e} {pr_str:>10} "
              f"{row['ret_mean']:>10.4f} {row['value_loss_mean']:>10.4f} "
              f"{row['gn_total_mean']:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
