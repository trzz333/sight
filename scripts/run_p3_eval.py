"""P3 eval runner.

Death-respecting batch runner for Sight P3 metric evaluation. Refuses to start
when SIGHT_TCP_IGNORE_DEATH is set (spec invariant from
docs/sight-p3-metrics.md). Two operating modes:

- Live mode (default): launch one Godot process per episode, drive the Phase B
  hello/action wire contract, write per-episode artifacts under
  runs/eval/<run_id>/episodes/<ep_id>/, then aggregate to summary.json.
- From-artifacts mode (--from-artifacts <run_root>): replay a previously
  captured run through loader -> aggregate without launching Godot.

Per-episode artifacts:

    runs/eval/<run_id>/
      summary.json
      episodes/
        ep_001/
          godot.ndjson
          python.ndjson
          meta.json
          godot_stdout.log
          godot_stderr.log

meta.json carries harness-side metadata that the loader uses to override
terminal classification (harness_abort) when the harness itself failed. One
Godot process per episode is intentional. There is no reset wire frame. The
action budget is enforced harness-side; success_budget_reached remains the
loader's decision.

Exit codes:
- 0 success
- 1 unhandled error
- 2 ignore-death refusal (spec invariant; preserved across slices)
- 3 missing --from-artifacts path
- 4 no episode artifacts found
- 5 live mode preflight failed (was: live mode not implemented)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


# Spec invariant: this env var, set non-empty, blocks any P3 metric run.
IGNORE_DEATH_ENV_VAR: str = "SIGHT_TCP_IGNORE_DEATH"  # spec invariant: refusal env-var name

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Phase B fallback paths. Do not introduce new Jeff-specific absolute paths
# beyond what already exists in scripts/run_phase_b_live.py.
_FALLBACK_GODOT_EXE: str = (
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64_console.exe"
)
_FALLBACK_GODOT_RUNS_DIR_TAIL: str = r"Godot\app_userdata\Signal Dodge\runs"

META_SCHEMA_VERSION: int = 1
GAME_ID: str = "signal_dodge"
SUCCESS_TERMINAL: str = "success_budget_reached"


class IgnoreDeathRefusal(SystemExit):
    """Raised when the refusal guard rejects an inherited environment."""

    EXIT_CODE: int = 2

    def __init__(self, message: str) -> None:
        super().__init__(self.EXIT_CODE)
        self.message = message


def refuse_if_ignore_death(env: Mapping[str, str] | None = None) -> None:
    """Spec invariant: refuse to start if the ignore-death env var is set.

    Reads from ``env`` if provided, otherwise os.environ. Raises
    ``IgnoreDeathRefusal`` (a SystemExit subclass with code 2) on detection.
    """
    if env is None:
        env = os.environ
    raw = env.get(IGNORE_DEATH_ENV_VAR)
    if raw is not None and str(raw).strip() != "":
        msg = (
            f"refusing: {IGNORE_DEATH_ENV_VAR} is set in the environment; "
            "P3 metric runs require death-respecting mode "
            "(see docs/sight-p3-metrics.md spec invariant)."
        )
        print(msg, file=sys.stderr)
        raise IgnoreDeathRefusal(msg)


def build_child_env(parent_env: Mapping[str, str], port: int) -> dict[str, str]:
    """Compose the Godot child environment.

    Sets SIGHT_TCP_MODE=1 and SIGHT_TCP_PORT=<port>. Defensively removes the
    ignore-death env var even though the parent guard already refuses on it
    (defense in depth: a future caller could bypass the parent guard).
    """
    env = dict(parent_env)
    env.pop(IGNORE_DEATH_ENV_VAR, None)  # spec invariant: child never inherits this
    env["SIGHT_TCP_MODE"] = "1"
    env["SIGHT_TCP_PORT"] = str(port)
    return env


def episode_id_for_index(index: int) -> str:
    """Zero-padded, 1-based episode id."""
    if index < 1:
        raise ValueError(f"episode index is 1-based; got {index}")
    return f"ep_{index:03d}"


def wire_run_id(batch_run_id: str, episode_id: str) -> str:
    return f"{batch_run_id}-{episode_id}"


def episode_dir(out_dir: Path, episode_id: str) -> Path:
    return out_dir / "episodes" / episode_id


def meta_json_path(ep_dir: Path) -> Path:
    return ep_dir / "meta.json"


def write_meta_json(ep_dir: Path, payload: Mapping[str, Any]) -> Path:
    ep_dir.mkdir(parents=True, exist_ok=True)
    path = meta_json_path(ep_dir)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def read_meta_json(ep_dir: Path) -> dict | None:
    path = meta_json_path(ep_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ts_run_id() -> str:
    return "p3-" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _default_godot_exe(parent_env: Mapping[str, str]) -> str:
    val = parent_env.get("GODOT_EXE")
    if val:
        return val
    return _FALLBACK_GODOT_EXE


def _default_project_path() -> Path:
    return REPO_ROOT / "games" / "signal-dodge"


def _default_godot_runs_dir(parent_env: Mapping[str, str]) -> str:
    val = parent_env.get("SIGHT_GODOT_RUNS_DIR")
    if val:
        return val
    appdata = parent_env.get("APPDATA")
    if appdata:
        return os.path.join(appdata, _FALLBACK_GODOT_RUNS_DIR_TAIL)
    return os.path.join(os.path.expanduser("~"), _FALLBACK_GODOT_RUNS_DIR_TAIL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_p3_eval",
        description="Sight P3 evaluation runner (live + from-artifacts).",
    )
    # Existing flags
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--actions-budget", type=int, default=300)
    p.add_argument("--wall-time-budget-sec", type=float, default=120.0)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--from-artifacts", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None)
    # Live-mode flags
    p.add_argument(
        "--godot-exe",
        type=str,
        default=None,
        help="Path to Godot executable. Falls back to $GODOT_EXE then a Phase B default.",
    )
    p.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to the Godot project. Defaults to <repo>/games/signal-dodge.",
    )
    p.add_argument(
        "--godot-runs-dir",
        type=str,
        default=None,
        help=(
            "Where Godot writes its NDJSON runs. Falls back to "
            "$SIGHT_GODOT_RUNS_DIR then $APPDATA\\Godot\\app_userdata\\Signal Dodge\\runs."
        ),
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--connect-timeout-sec", type=float, default=25.0)
    p.add_argument("--interval-sec", type=float, default=0.033)
    p.add_argument("--agent", type=str, default="p3-stub")
    p.add_argument(
        "--apply-grace-sec",
        type=float,
        default=2.5,
        help="Grace period after the action loop for Godot to flush apply events.",
    )
    return p.parse_args(argv)


def discover_episode_dirs(run_root: Path) -> list[Path]:
    """Sorted list of <run_root>/episodes/* dirs that hold ndjson artifacts."""
    episodes_dir = run_root / "episodes"
    if not episodes_dir.exists():
        return []
    out: list[Path] = []
    for d in sorted(episodes_dir.iterdir()):
        if not d.is_dir():
            continue
        if (d / "godot.ndjson").exists() or (d / "python.ndjson").exists():
            out.append(d)
    return out


def from_artifacts_mode(args: argparse.Namespace) -> int:
    """Replay episodes from disk through loader -> aggregate -> summary.json."""
    from sight_agent.evaluator.episodes import load_episode
    from sight_agent.evaluator.metrics import aggregate

    run_root = Path(args.from_artifacts)
    if not run_root.exists():
        print(f"--from-artifacts path not found: {run_root}", file=sys.stderr)
        return 3

    episode_dirs = discover_episode_dirs(run_root)
    if not episode_dirs:
        print(f"no episode artifacts under {run_root}/episodes/", file=sys.stderr)
        return 4

    episodes = []
    for d in episode_dirs:
        meta = read_meta_json(d) or {}
        harness_status = meta.get("harness_status")
        ep = load_episode(
            godot_path=d / "godot.ndjson",
            python_path=d / "python.ndjson",
            episode_id=d.name,
            actions_budget=args.actions_budget,
            wall_time_budget_sec=args.wall_time_budget_sec,
            harness_status=harness_status,
        )
        episodes.append(ep)

    summary = aggregate(episodes)
    summary["run_id"] = args.run_id or run_root.name
    summary["mode"] = "from_artifacts"
    summary["episode_ids"] = [ep.episode_id for ep in episodes]
    summary["actions_budget"] = args.actions_budget
    summary["wall_time_budget_sec"] = args.wall_time_budget_sec

    out_dir = Path(args.out_dir) if args.out_dir else run_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(out_path), "episodes": len(episodes)}))
    return 0


# --- live mode ------------------------------------------------------------


def _snapshot_run_files(godot_runs_dir: str) -> set[str]:
    pattern = os.path.join(godot_runs_dir, "*.ndjson")
    return set(glob.glob(pattern))


def _find_new_run_file(godot_runs_dir: str, before: set[str]) -> str | None:
    pattern = os.path.join(godot_runs_dir, "*.ndjson")
    after = set(glob.glob(pattern))
    new_files = after - before
    if not new_files:
        return None
    return max(new_files, key=os.path.getmtime)


def _stop_godot(godot: subprocess.Popen | None) -> None:
    if godot is None:
        return
    try:
        if godot.poll() is None:
            godot.terminate()
            try:
                godot.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                godot.kill()
                godot.wait(timeout=5.0)
    except Exception:
        pass


def _finalize_meta(
    *,
    meta: dict[str, Any],
    ep_dir: Path,
    godot: subprocess.Popen | None,
    stdout_f,
    stderr_f,
    godot_runs_dir: str,
    snap_before: set[str],
    grace_sec: float,
    sock=None,
) -> dict:
    """Stop subprocess, copy artifacts, write meta.json. Returns meta dict."""
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass

    if grace_sec > 0:
        time.sleep(grace_sec)

    _stop_godot(godot)
    if godot is not None:
        meta["godot_exit_code"] = godot.returncode

    for f in (stdout_f, stderr_f):
        try:
            f.close()
        except Exception:
            pass

    time.sleep(0.3)

    new_path = _find_new_run_file(godot_runs_dir, snap_before)
    if new_path is not None:
        meta["godot_source_ndjson"] = new_path
        try:
            shutil.copy2(new_path, ep_dir / "godot.ndjson")
            meta["godot_ndjson_copied"] = True
        except OSError as e:
            meta["godot_ndjson_copied"] = False
            if meta["harness_status"] == "ok":
                meta["harness_status"] = "harness_abort"
                meta["harness_reason"] = f"godot_ndjson_copy_failed: {e}"
    else:
        if meta["harness_status"] == "ok":
            meta["harness_status"] = "harness_abort"
            meta["harness_reason"] = "no_godot_ndjson_found"

    godot_ndjson_path = ep_dir / "godot.ndjson"
    if godot_ndjson_path.exists():
        try:
            from sight_agent.evaluator.episodes import (
                load_ndjson,
                run_metadata_from_events,
            )
            events = load_ndjson(godot_ndjson_path)
            md = run_metadata_from_events(events)
            meta["ignore_death_active_resolved"] = bool(md["ignore_death_active"])
        except Exception:
            pass

    meta["end_ts_unix_ns"] = time.time_ns()
    write_meta_json(ep_dir, meta)
    return meta


def run_live_episode(
    *,
    args: argparse.Namespace,
    parent_env: Mapping[str, str],
    batch_run_id: str,
    episode_id: str,
    out_dir: Path,
    godot_exe: str,
    project_path: Path,
    godot_runs_dir: str,
) -> dict:
    """Drive a single live Godot episode end to end. Returns meta payload."""
    from sight_agent.harness.tcp_client import (
        build_action,
        build_decision,
        build_hello,
        connect_with_retry,
        send_json_line,
        wait_for_port_bind,
    )

    ep_dir = episode_dir(out_dir, episode_id)
    ep_dir.mkdir(parents=True, exist_ok=True)
    wire_id = wire_run_id(batch_run_id, episode_id)

    meta: dict[str, Any] = {
        "schema_version": META_SCHEMA_VERSION,
        "game_id": GAME_ID,
        "success_terminal": SUCCESS_TERMINAL,
        "batch_run_id": batch_run_id,
        "episode_id": episode_id,
        "wire_run_id": wire_id,
        "actions_budget": args.actions_budget,
        "wall_time_budget_sec": args.wall_time_budget_sec,
        "host": args.host,
        "port": args.port,
        "configured_godot_exe": godot_exe,
        "configured_project_path": str(project_path),
        "configured_godot_runs_dir": godot_runs_dir,
        "start_ts_unix_ns": time.time_ns(),
        "harness_status": "ok",
        "harness_reason": None,
        "godot_exit_code": None,
        "godot_source_ndjson": None,
        "godot_ndjson_copied": False,
        "python_ndjson_path": str(ep_dir / "python.ndjson"),
        "actions_sent": 0,
        "ignore_death_active_resolved": None,
    }

    snap_before = _snapshot_run_files(godot_runs_dir)
    child_env = build_child_env(parent_env, args.port)

    stdout_path = ep_dir / "godot_stdout.log"
    stderr_path = ep_dir / "godot_stderr.log"
    stdout_f = stdout_path.open("wb")
    stderr_f = stderr_path.open("wb")

    godot: subprocess.Popen | None = None
    sock = None
    try:
        try:
            godot = subprocess.Popen(
                [godot_exe, "--headless", "--path", str(project_path)],
                env=child_env,
                stdout=stdout_f,
                stderr=stderr_f,
            )
        except (FileNotFoundError, OSError) as e:
            meta["harness_status"] = "harness_abort"
            meta["harness_reason"] = f"godot_launch_failed: {e}"
            return _finalize_meta(
                meta=meta, ep_dir=ep_dir, godot=godot,
                stdout_f=stdout_f, stderr_f=stderr_f,
                godot_runs_dir=godot_runs_dir, snap_before=snap_before,
                grace_sec=0.0,
            )

        if not wait_for_port_bind(args.host, args.port, args.connect_timeout_sec):
            meta["harness_status"] = "harness_abort"
            meta["harness_reason"] = "godot_did_not_bind_port"
            return _finalize_meta(
                meta=meta, ep_dir=ep_dir, godot=godot,
                stdout_f=stdout_f, stderr_f=stderr_f,
                godot_runs_dir=godot_runs_dir, snap_before=snap_before,
                grace_sec=0.0,
            )

        try:
            sock = connect_with_retry(args.host, args.port, args.connect_timeout_sec)
        except (ConnectionError, OSError) as e:
            meta["harness_status"] = "harness_abort"
            meta["harness_reason"] = f"tcp_connect_failed: {e}"
            return _finalize_meta(
                meta=meta, ep_dir=ep_dir, godot=godot,
                stdout_f=stdout_f, stderr_f=stderr_f,
                godot_runs_dir=godot_runs_dir, snap_before=snap_before,
                grace_sec=0.0,
            )

        try:
            send_json_line(sock, build_hello(wire_id, args.agent))
        except OSError as e:
            meta["harness_status"] = "harness_abort"
            meta["harness_reason"] = f"hello_send_failed: {e}"
            return _finalize_meta(
                meta=meta, ep_dir=ep_dir, godot=godot,
                stdout_f=stdout_f, stderr_f=stderr_f,
                godot_runs_dir=godot_runs_dir, snap_before=snap_before,
                grace_sec=0.0, sock=sock,
            )

        py_path = ep_dir / "python.ndjson"
        deadline = time.monotonic() + args.wall_time_budget_sec
        actions_sent = 0
        with py_path.open("w", encoding="utf-8") as f:
            for seq in range(args.actions_budget):
                if godot.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    break
                ts_ns = time.time_ns()
                try:
                    sent_ts = send_json_line(sock, build_action(seq, ts_ns))
                except OSError:
                    break
                f.write(
                    json.dumps(
                        build_decision(wire_id, seq, ts_ns, sent_ts),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                f.flush()
                actions_sent += 1
                if args.interval_sec > 0:
                    time.sleep(args.interval_sec)

        meta["actions_sent"] = actions_sent

        return _finalize_meta(
            meta=meta, ep_dir=ep_dir, godot=godot,
            stdout_f=stdout_f, stderr_f=stderr_f,
            godot_runs_dir=godot_runs_dir, snap_before=snap_before,
            grace_sec=args.apply_grace_sec, sock=sock,
        )
    except Exception as e:
        meta["harness_status"] = "harness_abort"
        meta["harness_reason"] = f"unexpected_error: {e}"
        return _finalize_meta(
            meta=meta, ep_dir=ep_dir, godot=godot,
            stdout_f=stdout_f, stderr_f=stderr_f,
            godot_runs_dir=godot_runs_dir, snap_before=snap_before,
            grace_sec=0.0, sock=sock,
        )


def live_mode(args: argparse.Namespace, parent_env: Mapping[str, str]) -> int:
    """Run --episodes Godot episodes, write artifacts, aggregate to summary.json."""
    from sight_agent.evaluator.episodes import load_episode
    from sight_agent.evaluator.metrics import aggregate

    batch_run_id = args.run_id or _ts_run_id()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (REPO_ROOT / "runs" / "eval" / batch_run_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    godot_exe = args.godot_exe or _default_godot_exe(parent_env)
    project_path = (
        Path(args.project_path) if args.project_path else _default_project_path()
    )
    godot_runs_dir = args.godot_runs_dir or _default_godot_runs_dir(parent_env)

    if not Path(godot_exe).exists():
        print(f"godot exe not found: {godot_exe}", file=sys.stderr)
        return 5
    if not project_path.exists():
        print(f"godot project path not found: {project_path}", file=sys.stderr)
        return 5
    if not Path(godot_runs_dir).exists():
        # Not fatal: Godot may create the directory on first run.
        print(
            f"warning: godot runs dir does not exist yet: {godot_runs_dir}",
            file=sys.stderr,
        )

    episode_ids: list[str] = []
    for i in range(1, args.episodes + 1):
        ep_id = episode_id_for_index(i)
        episode_ids.append(ep_id)
        run_live_episode(
            args=args,
            parent_env=parent_env,
            batch_run_id=batch_run_id,
            episode_id=ep_id,
            out_dir=out_dir,
            godot_exe=godot_exe,
            project_path=project_path,
            godot_runs_dir=godot_runs_dir,
        )

    episodes = []
    for ep_id in episode_ids:
        d = episode_dir(out_dir, ep_id)
        meta = read_meta_json(d) or {}
        harness_status = meta.get("harness_status")
        ep = load_episode(
            godot_path=d / "godot.ndjson",
            python_path=d / "python.ndjson",
            episode_id=ep_id,
            actions_budget=args.actions_budget,
            wall_time_budget_sec=args.wall_time_budget_sec,
            harness_status=harness_status,
        )
        episodes.append(ep)

    summary = aggregate(episodes)
    summary["run_id"] = batch_run_id
    summary["mode"] = "live"
    summary["episode_ids"] = episode_ids
    summary["actions_budget"] = args.actions_budget
    summary["wall_time_budget_sec"] = args.wall_time_budget_sec
    summary["game_id"] = GAME_ID

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(summary_path), "episodes": len(episodes)}))
    return 0


# --- entrypoint -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    refuse_if_ignore_death()
    args = parse_args(argv)

    if args.from_artifacts:
        return from_artifacts_mode(args)

    return live_mode(args, dict(os.environ))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IgnoreDeathRefusal as exc:
        sys.exit(exc.EXIT_CODE)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
