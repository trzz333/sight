"""Phase N / C2 - CMA-MAE (quality-diversity) on Signal Dodge.

Paradigm 2 of the Phase N from-scratch screen. Structurally distinct from
C1 (separable CMA-ES): instead of driving a single distribution toward one
optimum, CMA-MAE maintains an ARCHIVE of behaviorally-diverse elites and
anneals exploration -> exploitation via an archive learning rate. Built to
be robust to flat/noisy objectives, which is exactly the diagnosed C1
failure (premature behavioral convergence on a flat objective).

found-art verdict ADAPT (search: "pyribs CMA-MAE quality-diversity",
arXiv 2303.00191, PyPI `ribs` 0.11.0, icaros-usc/pyribs). This reuses the
ENTIRE C1 infra unchanged: the 5059-param SB3 actor, the Godot worker pool,
the packed lockstep rollout (evaluate_generation), the flatten/load helpers,
and the held-out eval gate (c1_es_eval.py runs against C2 output unchanged).
The only swap is the optimizer: cma.CMAEvolutionStrategy (single-point) ->
ribs Scheduler(GridArchive + EvolutionStrategyEmitter) (archive illumination).

Objective: RAW mean episode length (NOT the C1 shaped fitness). CMA-MAE
sources diversity from the archive + behavior measures, so the objective is
pure competence; the C1 collapse-penalty shaping is redundant here and would
double-count. This is the structural point of the switch.

Behavior descriptor (measures): 2D action-fraction simplex
(frac_left, frac_right), each in [0,1]. The diagnosed failure is behavioral
(action-mix collapse), so illuminating the L/R fraction plane targets it
directly, and both fractions are already computed by the rollout for free.
frac_stay = 1 - L - R, so 2D captures the full 3-way simplex.

CMA-MAE knobs (canonical, arXiv 2303.00191):
  - GridArchive learning_rate (alpha) = 0.01
  - GridArchive threshold_min = 0.0 (episode length is non-negative)
  - EvolutionStrategyEmitter ranker = "imp" (improvement ranking)
  - batch_size = n_workers (one ask() == one packed wave)

Two-vector held-out eval (mirrors C1 CMA-mean vs best-actor):
  - best_actor_vec.npy  = archive.best_elite["solution"] (best behavior)
  - best_mean_vec.npy   = emitter._opt.mean (CMA distribution mean, robust)

Windowless durable pattern (PROVEN in C1): supervisor DETACHED_PROCESS +
trainer CREATE_NO_WINDOW. Do NOT relaunch with CREATE_NEW_CONSOLE.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tools"))

# Detached-launch env fix (same rationale as c1_es_train): a WMI-detached
# process runs with a stripped env, so inject user-site onto sys.path before
# importing third-party packages. Foreground imports already work; this only
# adds the path when missing.
_USER_SITE = r"C:\Users\maste\AppData\Roaming\Python\Python314\site-packages"
if _USER_SITE not in sys.path and Path(_USER_SITE).is_dir():
    sys.path.insert(0, _USER_SITE)

import numpy as np  # noqa: E402

# Reuse the C1 infra wholesale. These are the exact objects the C1 screen and
# the unchanged eval gate use, so C2 is apples-to-apples with C1 by construction.
from c1_es_train import (  # noqa: E402
    ACTION_NAMES,
    DEFAULT_EXE,
    DEFAULT_PROJECT,
    SURVIVAL_BAR,
    build_policies,
    build_worker_pool,
    evaluate_generation,
    flatten_actor,
    fresh_training_seeds,
    load_actor,
)

# CMA-MAE hyperparameters (canonical, arXiv 2303.00191).
ARCHIVE_DIMS = (20, 20)          # 400 cells over the (frac_L, frac_R) plane
ARCHIVE_RANGES = [(0.0, 1.0), (0.0, 1.0)]
ARCHIVE_LR = 0.01                # archive learning rate alpha
THRESHOLD_MIN = 0.0              # episode length is non-negative
RANKER = "imp"                   # improvement ranking == CMA-MAE


def measures_from_result(r: dict) -> np.ndarray:
    """2D behavior descriptor (frac_left, frac_right) from a rollout result.

    evaluate_generation already accumulates action_fractions per candidate;
    the archive illuminates the L/R simplex plane (frac_stay = 1 - L - R).
    """
    fr = r["action_fractions"]
    return np.array([fr["left"], fr["right"]], dtype=np.float64)


def build_scheduler(x0: np.ndarray, sigma0: float, batch_size: int, seed: int):
    """CMA-MAE object graph: GridArchive + one EvolutionStrategyEmitter.

    learning_rate + threshold_min on the archive plus ranker='imp' on the
    emitter is the canonical CMA-MAE configuration. result_archive defaults
    to a passthrough of the main archive (no separate high-res grid needed
    at 400 cells). Returns (scheduler, archive, emitter).
    """
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler

    archive = GridArchive(
        solution_dim=int(x0.size),
        dims=list(ARCHIVE_DIMS),
        ranges=ARCHIVE_RANGES,
        learning_rate=ARCHIVE_LR,
        threshold_min=THRESHOLD_MIN,
        seed=seed,
    )
    emitter = EvolutionStrategyEmitter(
        archive,
        x0=np.asarray(x0, dtype=np.float64),
        sigma0=float(sigma0),
        ranker=RANKER,
        es="cma_es",
        selection_rule="mu",
        restart_rule="basic",
        batch_size=int(batch_size),
        seed=seed,
    )
    sched = Scheduler(archive, [emitter])
    return sched, archive, emitter


def run_smoke(args, vec_env, policies, actor_keys, numel) -> dict:
    """Throughput smoke-test: time ONE realistic CMA-MAE generation.

    One ask() -> batch_size candidates, scored packed over seeds_per_gen
    seeds via the shared evaluate_generation, one tell(). Reports eps/sec and
    gens/hour, confirms the archive fills and is picklable, then exits without
    a long run. Mandatory gate before any multi-hour launch (mirrors C1).
    """
    seeds = fresh_training_seeds(gen=0, k=args.seeds_per_gen)
    x0 = flatten_actor(policies[0], actor_keys)
    sched, archive, emitter = build_scheduler(
        x0, args.sigma0, args.batch_size, args.seed)
    t0 = time.time()
    sols = sched.ask()
    res = evaluate_generation(policies, actor_keys, vec_env, args.n_workers,
                              [np.asarray(s) for s in sols], seeds)
    objectives = np.array([r["mean_length"] for r in res], dtype=np.float64)
    measures = np.array([measures_from_result(r) for r in res], dtype=np.float64)
    sched.tell(objectives, measures)
    elapsed = time.time() - t0
    # picklability check: a crash must be able to --resume, not restart.
    pickle_ok = True
    try:
        pickle.dumps({"sched": sched, "archive": archive, "emitter": emitter})
    except Exception as e:  # noqa: BLE001
        pickle_ok = False
        pickle_err = repr(e)
    total_eps = len(sols) * len(seeds)
    eps_per_sec = total_eps / max(elapsed, 1e-9)
    out = {
        "mode": "smoke",
        "optimizer": "CMA-MAE (pyribs)",
        "n_workers": args.n_workers,
        "batch_size": args.batch_size,
        "seeds_per_gen": args.seeds_per_gen,
        "episodes_this_gen": total_eps,
        "elapsed_seconds": round(elapsed, 2),
        "episodes_per_sec": round(eps_per_sec, 3),
        "secs_per_generation": round(elapsed, 1),
        "generations_per_hour": round(3600.0 / max(elapsed, 1e-9), 2),
        "archive_entries_after_1_gen": int(len(archive)),
        "random_policy_mean_length": round(float(objectives.mean()), 1),
        "random_policy_best_length": round(float(objectives.max()), 1),
        "state_picklable": pickle_ok,
    }
    if not pickle_ok:
        out["pickle_error"] = pickle_err
    return out


def run_mae(args, vec_env, policies, actor_keys, numel, out: Path) -> dict:
    """CMA-MAE optimization loop. Saves best-actor + archive-mean vectors,
    per-gen history, and an atomic full-state checkpoint for --resume."""
    ckpt = out / "mae_state.pkl"
    x0 = flatten_actor(policies[0], actor_keys)  # SB3 default init as archive x0

    if args.resume and ckpt.exists():
        with ckpt.open("rb") as f:
            st = pickle.load(f)
        sched = st["sched"]
        archive = st["archive"]
        emitter = st["emitter"]
        history = st["history"]
        best_obj = st["best_obj"]
        best_vec = st["best_vec"]
        gen = st["gen"]
        elapsed_offset = st["elapsed"]
        print(json.dumps({"resumed_from_gen": gen,
                          "elapsed_offset_s": round(elapsed_offset, 1)}))
    else:
        sched, archive, emitter = build_scheduler(
            x0, args.sigma0, args.batch_size, args.seed)
        history = []
        best_obj = -1.0
        best_vec = x0.copy()
        gen = 0
        elapsed_offset = 0.0

    t0 = time.time() - elapsed_offset
    launch_t0 = time.time()
    while gen < args.gens:
        if args.max_wall_s > 0 and (time.time() - launch_t0) >= args.max_wall_s:
            print(json.dumps({"wall_budget_stop_s": round(args.max_wall_s, 1),
                              "stopped_before_gen": gen}))
            break
        seeds = fresh_training_seeds(gen=gen, k=args.seeds_per_gen)
        sols = sched.ask()
        res = evaluate_generation(policies, actor_keys, vec_env,
                                  args.n_workers, [np.asarray(s) for s in sols],
                                  seeds)
        objectives = np.array([r["mean_length"] for r in res], dtype=np.float64)
        measures = np.array([measures_from_result(r) for r in res],
                            dtype=np.float64)
        sched.tell(objectives, measures)

        gi = int(np.argmax(objectives))
        if objectives[gi] > best_obj:
            best_obj = float(objectives[gi])
            best_vec = np.asarray(sols[gi]).copy()

        be = archive.best_elite
        arch_best_obj = float(be["objective"]) if be is not None else -1.0
        rec = {
            "gen": gen,
            "gen_best_mean_length": round(float(objectives.max()), 1),
            "gen_mean_mean_length": round(float(objectives.mean()), 1),
            "gen_best_action_fractions": res[gi]["action_fractions"],
            "archive_size": int(len(archive)),
            "archive_best_objective": round(arch_best_obj, 1),
            "running_best_mean_length": round(best_obj, 1),
            "elapsed_seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec))

        # Persist incrementally: best-actor = highest-objective archive elite;
        # archive-mean = CMA distribution mean (noise-robust). Both feed the
        # unchanged c1_es_eval gate.
        actor_vec = (np.asarray(be["solution"], dtype=np.float64)
                     if be is not None else best_vec)
        np.save(str(out / "best_actor_vec.npy"), actor_vec)
        np.save(str(out / "best_mean_vec.npy"),
                np.asarray(emitter._opt.mean, dtype=np.float64))
        with (out / "mae_history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        tmp = out / "mae_state.pkl.tmp"
        with tmp.open("wb") as f:
            pickle.dump({"sched": sched, "archive": archive, "emitter": emitter,
                         "history": history, "best_obj": best_obj,
                         "best_vec": best_vec, "gen": gen + 1,
                         "elapsed": time.time() - t0}, f)
        tmp.replace(ckpt)
        gen += 1

    be = archive.best_elite
    actor_vec = (np.asarray(be["solution"], dtype=np.float64)
                 if be is not None else best_vec)
    np.save(str(out / "best_actor_vec.npy"), actor_vec)
    np.save(str(out / "best_mean_vec.npy"),
            np.asarray(emitter._opt.mean, dtype=np.float64))
    return {
        "mode": "mae",
        "optimizer": "CMA-MAE (pyribs)",
        "generations_run": gen,
        "batch_size": args.batch_size,
        "seeds_per_gen": args.seeds_per_gen,
        "best_mean_length_train_seeds": round(best_obj, 1),
        "archive_final_size": int(len(archive)),
        "archive_best_objective": round(
            float(be["objective"]) if be is not None else -1.0, 1),
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="c2_mae_train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8,
                   help="parallel Godot subprocs = candidates scored per wave")
    p.add_argument("--vec", choices=("dummy", "subproc"), default="subproc")
    p.add_argument("--sigma0", type=float, default=0.1,
                   help="CMA initial step size in weight space")
    p.add_argument("--batch-size", type=int, default=8,
                   help="emitter batch (== one packed wave). Keep == n-workers "
                        "so one ask() fills exactly one lockstep wave.")
    p.add_argument("--gens", type=int, default=100)
    p.add_argument("--seeds-per-gen", type=int, default=1,
                   help="seeds each candidate is averaged over per generation "
                        "(common random numbers within a gen).")
    p.add_argument("--smoke", action="store_true",
                   help="throughput smoke-test only; times one packed gen")
    p.add_argument("--resume", action="store_true",
                   help="resume full CMA-MAE state from out/mae_state.pkl if "
                        "present, else start fresh")
    p.add_argument("--max-wall-s", type=float, default=0.0,
                   help="if >0, stop THIS launch cleanly after N seconds and "
                        "checkpoint (resumable).")
    p.add_argument("--exe", default=DEFAULT_EXE)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool_root = out / "worker_envs"
    pool_root.mkdir(parents=True, exist_ok=True)

    policies, actor_keys, numel = build_policies(args.n_workers)
    assert numel == 5059, f"actor param count drift: {numel}"

    vec_env = build_worker_pool(
        n_workers=args.n_workers, base_seed=args.seed, run_root=pool_root,
        exe=args.exe, project=args.project, vec=args.vec,
    )
    try:
        if args.smoke:
            report = run_smoke(args, vec_env, policies, actor_keys, numel)
        else:
            report = run_mae(args, vec_env, policies, actor_keys, numel, out)
    finally:
        try:
            vec_env.close()
        except Exception:
            pass

    import ribs
    report.update({
        "actor_param_count": numel,
        "survival_bar": SURVIVAL_BAR,
        "seed": args.seed,
        "archive_dims": list(ARCHIVE_DIMS),
        "archive_learning_rate": ARCHIVE_LR,
        "ranker": RANKER,
        "python": sys.version.split()[0],
        "ribs": ribs.__version__,
        "sb3": __import__("stable_baselines3").__version__,
    })
    with (out / "c2_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
