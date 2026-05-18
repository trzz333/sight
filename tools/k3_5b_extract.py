"""K3.5b 10k confirmation extractor.

Reads runs/phase_k/k3_5_confirm_reward_scale_div30_10000.ndjson and
emits per-update gate evaluation across all 40 updates, a CSV, and a
summary block for the evidence doc.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


PHASE_K = Path(r"C:\Projects\Sight\runs\phase_k")
NDJSON = PHASE_K / "k3_5_confirm_reward_scale_div30_10000.ndjson"
CSV_OUT = PHASE_K / "k3_5_confirm_reward_scale_div30_10000_table.csv"


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


def extract(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in records:
        post = r["post_update"]
        fp = post.get("fixed_panel_policy_state")
        lvf = fp["feature_chain_diversity"]["latent_vf"] if fp else None
        vp = fp["feature_chain_diversity"]["value_predictions"] if fp else None
        adv = r["adv_ret_val_stats"]
        vf = r["value_fit"]
        rs = r["rollout_action_stats"]
        out.append({
            "update_idx": int(r["update_idx"]),
            "lvf_post": int(lvf["n_dims_above_eps"]) if lvf else 0,
            "vp_std_post": float(vp["std"]) if vp else 0.0,
            "vp_mean_post": float(vp["mean"]) if vp else 0.0,
            "panel_top_argmax_frac": float(
                fp["top_argmax_fraction"]
            ) if fp else None,
            "panel_num_det_actions": int(
                fp["num_det_actions"]
            ) if fp else None,
            "ret_mean_scaled": float(adv["returns"]["mean"]),
            "ret_mean_raw": float(adv["returns"]["mean"]) * 30.0,
            "val_mean_rollout": float(adv["values"]["mean"]),
            "post_val_mean_rollout": float(vf["post_rollout"]["value_pred_mean"]),
            "pre_ratio": vf["pre_rollout"]["model_to_constant_mse_ratio"],
            "post_ratio": vf["post_rollout"]["model_to_constant_mse_ratio"],
            "value_loss_mean": float(r["losses"]["value_loss_mean"]),
            "policy_grad_loss_mean": float(r["losses"]["policy_gradient_loss_mean"]),
            "entropy_mean_post": float(post["policy_state"]["entropy_mean"]),
            "rollout_top_action": rs["top_action"],
            "rollout_top_action_fraction": float(rs["top_action_fraction"]),
            "gn_total_mean": float(r["grad_norms_preclip"]["total_mean"]),
            "gn_mlp_vf_w_mean": float(r["grad_norms_preclip"]["mlp_value_net_weights_mean"]),
            "gn_scalar_vh_mean": float(r["grad_norms_preclip"]["scalar_value_head_mean"]),
            "ev": float(r["explained_variance"]),
        })
    return out


def main() -> int:
    header, records = load_run(NDJSON)
    rows = extract(records)
    print(f"=== K3.5b 10k confirmation: divisor 30, seed 3 ===")
    print(f"header.reward_scale_divisor={header['reward_scale_divisor']}")
    print(f"header.reward_scale_applied={header['reward_scale_applied']}")
    print(f"header.value_bias_init_mode={header['value_bias_init_mode']}")
    print(f"header.value_bias_init_applied={header['value_bias_init_applied']}")
    print(f"header.vf_coef={header['vf_coef']}")
    print(f"header.policy_kwargs={header['policy_kwargs']}")
    print(f"header.elapsed_seconds={header['elapsed_seconds']:.1f}")
    print(f"n_updates={len(records)}")
    print()
    print(f"{'u':>3} {'lvf':>4} {'vp_std':>10} {'post_ratio':>10} "
          f"{'ret_raw':>9} {'top_act':>8} {'topfrac':>7} "
          f"{'panel_frac':>10} {'detN':>4} {'v_loss':>9} {'|g|':>6} {'ev':>7}")
    for r in rows:
        pr = r["post_ratio"]
        pr_s = f"{pr:.3f}" if pr is not None else "  None"
        pf = r["panel_top_argmax_frac"]
        pf_s = f"{pf:.3f}" if pf is not None else "  None"
        print(f"{r['update_idx']:>3} {r['lvf_post']:>4} "
              f"{r['vp_std_post']:>10.3e} {pr_s:>10} "
              f"{r['ret_mean_raw']:>9.3f} "
              f"{r['rollout_top_action']:>8} "
              f"{r['rollout_top_action_fraction']:>7.3f} "
              f"{pf_s:>10} {r['panel_num_det_actions']:>4} "
              f"{r['value_loss_mean']:>9.4f} "
              f"{r['gn_total_mean']:>6.3f} "
              f"{r['ev']:>7.3f}")

    # Sustained-life gates across the full probe.
    n = len(rows)
    lvf_min = min(r["lvf_post"] for r in rows)
    lvf_min_at = [r["update_idx"] for r in rows if r["lvf_post"] == lvf_min][0]
    vp_std_min = min(r["vp_std_post"] for r in rows)
    vp_std_min_at = [r["update_idx"] for r in rows if r["vp_std_post"] == vp_std_min][0]
    post_ratio_max = max(
        r["post_ratio"] for r in rows if r["post_ratio"] is not None
    )
    post_ratio_max_at = [
        r["update_idx"] for r in rows if r["post_ratio"] == post_ratio_max
    ][0]
    top_frac_max = max(r["rollout_top_action_fraction"] for r in rows)
    top_frac_max_at = [
        r["update_idx"] for r in rows if r["rollout_top_action_fraction"] == top_frac_max
    ][0]
    panel_frac_max = max(
        r["panel_top_argmax_frac"] for r in rows if r["panel_top_argmax_frac"] is not None
    )
    panel_frac_max_at = [
        r["update_idx"] for r in rows
        if r["panel_top_argmax_frac"] == panel_frac_max
    ][0]
    panel_det_min = min(
        r["panel_num_det_actions"] for r in rows if r["panel_num_det_actions"] is not None
    )
    panel_det_min_at = [
        r["update_idx"] for r in rows
        if r["panel_num_det_actions"] == panel_det_min
    ][0]

    print()
    print(f"=== Sustained-life gates across {n} updates ===")
    print(f"lvf_post min: {lvf_min} at update {lvf_min_at} (need >= 16): "
          f"{'PASS' if lvf_min >= 16 else 'FAIL'}")
    print(f"vp_std_post min: {vp_std_min:.3e} at update {vp_std_min_at} (need > 1e-6): "
          f"{'PASS' if vp_std_min > 1e-6 else 'FAIL'}")
    print(f"post_ratio max: {post_ratio_max:.3f} at update {post_ratio_max_at} "
          f"(K3.3 strong baseline 6.18)")
    print(f"rollout top_action_fraction max: {top_frac_max:.3f} at update "
          f"{top_frac_max_at} (wedge threshold 0.95): "
          f"{'PASS' if top_frac_max < 0.95 else 'FAIL'}")
    print(f"panel top_argmax_fraction max: {panel_frac_max:.3f} at update "
          f"{panel_frac_max_at} (constant-action threshold 0.95): "
          f"{'PASS' if panel_frac_max < 0.95 else 'FAIL'}")
    print(f"panel num_det_actions min: {panel_det_min} at update "
          f"{panel_det_min_at} (need >= 2): "
          f"{'PASS' if panel_det_min >= 2 else 'FAIL'}")

    # Drift
    first_ret = rows[0]["ret_mean_raw"]
    last_ret = rows[-1]["ret_mean_raw"]
    print()
    print(f"=== Reward drift (raw, divisor=30 scaled by *30) ===")
    print(f"first ret_mean_raw u1 = {first_ret:.3f}")
    print(f"last  ret_mean_raw u{n} = {last_ret:.3f}")
    print(f"first-to-last ratio: {last_ret / first_ret:.3f}x")

    # CSV
    with CSV_OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow(list(r.values()))
    print(f"\nwrote {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
