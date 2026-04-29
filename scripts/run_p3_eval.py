"""P3 eval runner.

Two modes (mutually exclusive, both off by default):

  --live              Launch Godot headless, drive scripts/run_phase_b.py over
                      TCP, collect per-episode artifacts, aggregate.
  --from-artifacts    Load existing runs/eval/<run_id>/episodes/* and aggregate.

Spec invariants:

  - refuse_if_ignore_death() runs first in main(). If SIGHT_TCP_IGNORE_DEATH is
    set non-empty in the inherited environment, the runner refuses (exit 2).
  - Live-mode child env never carries SIGHT_TCP_IGNORE_DEATH. The exclusion
    invariant from docs/sight-p3-metrics.md is preserved at the harness layer.
  - Godot terminal evidence wins over Python client failure: a nonzero
    run_phase_b.py exit is not automatically harness_abort if godot.ndjson
    already contains death/collision or success_budget_reached.

GDScript surface relied on (current main):

  - SIGHT_GODOT_LOG_PATH         logger.gd writes only to this absolute path
  - SIGHT_TCP_MODE / SIGHT_TCP_PORT  TCP server mode and port
  - SIGHT_P3_ACTIONS_BUDGET      main.gd emits success_budget_reached at threshold
  - SIGHT_EPISODE_ID             echoed into success_budget_reached payload

Eval artifact layout:

    runs\\eval\\<batch_run_id>\\episodes\\<episode_id>\\godot.ndjson
    runs\\eval\\<batch_run_id>\\episodes\\<episode_id>\\python.ndjson
    runs\\eval\\<batch_run_id>\\episodes\\<episode_id>\\meta.json
    runs\\eval\\<batch_run_id>\\summary.json

Diagnostics layout:

    runs\\diagnostics\\p3-live-<batch_run_id>\\<episode_id>\\godot_stdout.log
    runs\\diagnostics\\p3-live-<batch_run_id>\\<episode_id>\\godot_stderr.log
    runs\\diagnostics\\p3-live-<batch_run_id>\\<episode_id>\\python_stdout.log
    runs\\diagnostics\\p3-live-<batch_run_id>\\<episode_id>\\python_stderr.log
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping, Optional


# Spec invariant: this env var, set non-empty, blocks any P3 metric run.
IGNORE_DEATH_ENV_VAR: str = "SIGHT_TCP_IGNORE_DEATH"  # spec invariant: refusal env-var name

DEFAULT_PORT: int = 8765
DEFAULT_PROJECT_REL: str = r"games\signal-dodge"
GODOT_EXE_ENV_VAR: str = "SIGHT_GODOT_EXE"
DEFAULT_GODOT_EXE_FALLBACK: str = (
    r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages"
    r"\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\Godot_v4.6.2-stable_win64_console.exe"
)

# Wait window for Godot to bind the TCP port after launch.
PORT_BIND_TIMEOUT_SEC: float = 25.0
PORT_BIND_POLL_SEC: float = 0.25
DEFAULT_APPLY_GRACE_SEC: float = 5.0

# Authoritative Godot terminal types. Used to decide whether a nonzero Python
# client exit should escalate to harness_abort. Mirrors the loader contract.
_GODOT_TERMINAL_TYPES: frozenset[str] = frozenset(
    {"death", "collision", "success_budget_reached"}
)


class IgnoreDeathRefusal(SystemExit):
    """Raised when the refusal guard rejects an inherited environment."""

    EXIT_CODE: int = 2

    def __init__(self, message: str) -> None:
        super().__init__(self.EXIT_CODE)
        self.message = message


def refuse_if_ignore_death(env: Mapping[str, str] | None = None) -> None:
    """Spec invariant: refuse to start if the ignore-death env var is set."""
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


def _ts_run_id() -> str:
    return "p3-" + _dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


# --- argparse ---------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_p3_eval",
        description=(
            "Sight P3 evaluation runner. Exactly one mode required: "
            "--live or --from-artifacts."
        ),
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--live",
        action="store_true",
        help=(
            "Launch Godot headless and drive scripts/run_phase_b.py over TCP. "
            "Never the default; must be passed explicitly."
        ),
    )
    mode.add_argument(
        "--from-artifacts",
        type=str,
        default=None,
        help="Load episodes from runs/eval/<run_id> instead of launching Godot.",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of live episodes (live mode only).",
    )
    p.add_argument(
        "--actions-budget",
        type=int,
        default=300,
        help=(
            "Per-episode action budget; success_budget_reached fires when "
            "applied_count >= this in TCP mode."
        ),
    )
    p.add_argument(
        "--wall-time-budget-sec",
        type=float,
        default=120.0,
        help="Per-episode wall-time safety ceiling in seconds.",
    )
    p.add_argument(
        "--apply-grace-sec",
        type=float,
        default=DEFAULT_APPLY_GRACE_SEC,
        help=(
            "Grace period above wall-time-budget before forced kill, and grace "
            "for Godot to flush its terminal event after the client exits."
        ),
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional batch run id; defaults to a timestamped value.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Eval root; defaults to runs/eval/<run_id>.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Godot TCP port.",
    )
    p.add_argument(
        "--godot-exe",
        type=str,
        default=None,
        help=(
            f"Godot executable; defaults to ${GODOT_EXE_ENV_VAR} env var or the "
            "known local Winget install if present."
        ),
    )
    p.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help=f"Godot project dir; defaults to <repo>/{DEFAULT_PROJECT_REL}.",
    )
    return p.parse_args(argv)


# --- from-artifacts mode ---------------------------------------------------


def discover_episode_dirs(run_root: Path) -> list[Path]:
    """Return sorted list of <run_root>/episodes/* directories that have artifacts."""
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
    """Load episodes from disk, aggregate, write summary.json. No Godot launch."""
    from sight_agent.evaluator.episodes import load_episode
    from sight_agent.evaluator.metrics import aggregate

    run_root = Path(args.from_artifacts)
    if not run_root.exists():
        print(f"--from-artifacts path not found: {run_root}", file=sys.stderr)
        return 3

    episode_dirs = discover_episode_dirs(run_root)
    if not episode_dirs:
        print(
            f"no episode artifacts under {run_root}/episodes/",
            file=sys.stderr,
        )
        return 4

    episodes = []
    for d in episode_dirs:
        ep = load_episode(
            godot_path=d / "godot.ndjson",
            python_path=d / "python.ndjson",
            episode_id=d.name,
            actions_budget=args.actions_budget,
            wall_time_budget_sec=args.wall_time_budget_sec,
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


# --- live mode helpers ------------------------------------------------------


def resolve_godot_exe(arg_value: str | None, env: Mapping[str, str]) -> Path:
    """Resolve Godot exe from arg, env, or known local fallback. Existence required."""
    candidates: list[Path] = []
    if arg_value:
        candidates.append(Path(arg_value))
    env_val = env.get(GODOT_EXE_ENV_VAR)
    if env_val:
        candidates.append(Path(env_val))
    candidates.append(Path(DEFAULT_GODOT_EXE_FALLBACK))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "no Godot executable found; tried "
        + ", ".join(str(c) for c in candidates)
        + f". Set --godot-exe or ${GODOT_EXE_ENV_VAR}."
    )


def resolve_project_dir(arg_value: str | None) -> Path:
    if arg_value:
        return Path(arg_value)
    return _repo_root() / DEFAULT_PROJECT_REL


def episode_id_for_index(idx: int) -> str:
    """ep000001, ep000002, ..."""
    return f"ep{idx:06d}"


def build_live_child_env(
    base_env: Mapping[str, str],
    *,
    port: int,
    godot_log_path: Path,
    actions_budget: int,
    episode_id: str,
) -> dict[str, str]:
    """Construct subprocess env for Godot.

    Spec invariants:
      - Strips SIGHT_TCP_IGNORE_DEATH from the inherited env. P3 live runs must
        never tolerate the ignore-death flag.
      - Sets the GDScript surface vars logger.gd / tcp_controller.gd / main.gd
        rely on for log path, TCP mode, action budget, and episode id.
    """
    env = dict(base_env)
    env.pop(IGNORE_DEATH_ENV_VAR, None)
    env["SIGHT_TCP_MODE"] = "1"
    env["SIGHT_TCP_PORT"] = str(port)
    env["SIGHT_GODOT_LOG_PATH"] = str(godot_log_path)
    env["SIGHT_P3_ACTIONS_BUDGET"] = str(actions_budget)
    env["SIGHT_EPISODE_ID"] = episode_id
    return env


@dataclass
class EpisodeMeta:
    batch_run_id: str
    episode_id: str
    wire_run_id: str
    mode: str
    actions_budget: int
    wall_time_budget_sec: float
    port: int
    godot_path: str
    python_path: str
    diagnostics_dir: str
    godot_exit_code: Optional[int]
    client_exit_code: Optional[int]
    harness_status: str
    started_at: str
    ended_at: str


def _wait_for_port(
    host: str,
    port: int,
    godot_proc: "subprocess.Popen[bytes]",
    timeout_sec: float,
) -> bool:
    """Poll until socket connect succeeds, godot exits, or deadline hit."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if godot_proc.poll() is not None:
            return False
        try:
            s = socket.create_connection((host, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(PORT_BIND_POLL_SEC)
    return False


def launch_godot(
    *,
    godot_exe: Path,
    project_dir: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> "subprocess.Popen[bytes]":
    """Launch Godot headless with stdout/stderr redirected to diagnostics files."""
    stdout_f = stdout_path.open("wb")
    stderr_f = stderr_path.open("wb")
    proc = subprocess.Popen(
        [str(godot_exe), "--headless", "--path", str(project_dir)],
        env=dict(env),
        stdout=stdout_f,
        stderr=stderr_f,
    )
    # Stash file handles on the proc so the orchestrator can close them after exit.
    proc._stdout_file = stdout_f  # type: ignore[attr-defined]
    proc._stderr_file = stderr_f  # type: ignore[attr-defined]
    return proc


def run_python_client(
    *,
    actions_budget: int,
    port: int,
    out_path: Path,
    wire_run_id: str,
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: float,
    cwd: Path,
) -> subprocess.CompletedProcess:
    """Run scripts/run_phase_b.py as a subprocess; capture stdout/stderr to disk."""
    cmd = [
        sys.executable,
        str(cwd / "scripts" / "run_phase_b.py"),
        "--actions", str(actions_budget),
        "--port", str(port),
        "--out", str(out_path),
        "--run-id", wire_run_id,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        # subprocess.run carries whatever stdout/stderr was captured up to the
        # timeout on the exception itself. Without this branch we would lose
        # all client diagnostics on the exact failure mode they're most useful
        # for. text=True normally implies str, but be defensive about bytes.
        so = exc.stdout if exc.stdout is not None else ""
        se = exc.stderr if exc.stderr is not None else ""
        if isinstance(so, (bytes, bytearray)):
            so = so.decode("utf-8", errors="replace")
        if isinstance(se, (bytes, bytearray)):
            se = se.decode("utf-8", errors="replace")
        stdout_path.write_text(so, encoding="utf-8")
        stderr_path.write_text(se, encoding="utf-8")
        raise
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return proc


def _ndjson_has_terminal(godot_path: Path) -> bool:
    """Cheap scan for an authoritative Godot terminal event."""
    if not godot_path.exists():
        return False
    with godot_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") in _GODOT_TERMINAL_TYPES:
                return True
    return False


def _stop_godot(
    proc: "subprocess.Popen[bytes]",
    kill_after_sec: float = 5.0,
) -> int | None:
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
        try:
            proc.wait(timeout=kill_after_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=kill_after_sec)
    except Exception:
        pass
    return proc.returncode


def _close_godot_log_files(proc: "subprocess.Popen[bytes]") -> None:
    for attr in ("_stdout_file", "_stderr_file"):
        f = getattr(proc, attr, None)
        if f is None:
            continue
        try:
            f.close()
        except OSError:
            pass


# --- live episode orchestration --------------------------------------------


def run_one_live_episode(
    *,
    batch_run_id: str,
    episode_idx: int,
    eval_root: Path,
    diagnostics_root: Path,
    godot_exe: Path,
    project_dir: Path,
    port: int,
    actions_budget: int,
    wall_time_budget_sec: float,
    apply_grace_sec: float,
    base_env: Mapping[str, str],
) -> EpisodeMeta:
    """Run one live episode end-to-end. Writes godot.ndjson, python.ndjson, meta.json."""
    episode_id = episode_id_for_index(episode_idx)
    wire_run_id = f"{batch_run_id}-{episode_id}"

    eval_episode_dir = eval_root / "episodes" / episode_id
    diag_episode_dir = diagnostics_root / episode_id
    eval_episode_dir.mkdir(parents=True, exist_ok=True)
    diag_episode_dir.mkdir(parents=True, exist_ok=True)

    godot_log_path = eval_episode_dir / "godot.ndjson"
    python_log_path = eval_episode_dir / "python.ndjson"

    godot_stdout = diag_episode_dir / "godot_stdout.log"
    godot_stderr = diag_episode_dir / "godot_stderr.log"
    python_stdout = diag_episode_dir / "python_stdout.log"
    python_stderr = diag_episode_dir / "python_stderr.log"

    child_env = build_live_child_env(
        base_env,
        port=port,
        godot_log_path=godot_log_path,
        actions_budget=actions_budget,
        episode_id=episode_id,
    )

    started_at = _now_iso()
    harness_status = "ok"
    godot_exit_code: Optional[int] = None
    client_exit_code: Optional[int] = None

    godot_proc: Optional["subprocess.Popen[bytes]"] = None
    try:
        godot_proc = launch_godot(
            godot_exe=godot_exe,
            project_dir=project_dir,
            env=child_env,
            stdout_path=godot_stdout,
            stderr_path=godot_stderr,
        )

        bound = _wait_for_port("127.0.0.1", port, godot_proc, PORT_BIND_TIMEOUT_SEC)
        if not bound:
            harness_status = "harness_abort"
            godot_exit_code = _stop_godot(godot_proc)
        else:
            try:
                client = run_python_client(
                    actions_budget=actions_budget,
                    port=port,
                    out_path=python_log_path,
                    wire_run_id=wire_run_id,
                    stdout_path=python_stdout,
                    stderr_path=python_stderr,
                    timeout_sec=wall_time_budget_sec + apply_grace_sec,
                    cwd=_repo_root(),
                )
                client_exit_code = client.returncode
            except subprocess.TimeoutExpired:
                client_exit_code = None
                harness_status = "timeout"

            # Wait for Godot to exit naturally; bounded by apply_grace_sec.
            wait_deadline = time.monotonic() + apply_grace_sec
            while (
                time.monotonic() < wait_deadline and godot_proc.poll() is None
            ):
                time.sleep(0.1)

            if godot_proc.poll() is None:
                # Godot did not exit on its own; force stop. If we haven't
                # already classified the run, this is a wall-time timeout.
                if harness_status == "ok":
                    harness_status = "timeout"
                godot_exit_code = _stop_godot(godot_proc)
            else:
                godot_exit_code = godot_proc.returncode

            # Godot terminal evidence wins. Only escalate to harness_abort if
            # godot.ndjson lacks an authoritative terminal AND the client failed.
            if harness_status == "ok":
                terminal_present = _ndjson_has_terminal(godot_log_path)
                client_failed = (
                    client_exit_code is None or client_exit_code != 0
                )
                if not terminal_present and client_failed:
                    harness_status = "harness_abort"
    except FileNotFoundError:
        # Godot exe disappeared between resolve and launch.
        harness_status = "harness_abort"
    except Exception as exc:  # pragma: no cover - defensive
        print(f"unexpected episode error: {exc}", file=sys.stderr)
        harness_status = "harness_abort"
        if godot_proc is not None:
            godot_exit_code = _stop_godot(godot_proc)
    finally:
        if godot_proc is not None:
            _close_godot_log_files(godot_proc)

    ended_at = _now_iso()

    meta = EpisodeMeta(
        batch_run_id=batch_run_id,
        episode_id=episode_id,
        wire_run_id=wire_run_id,
        mode="live",
        actions_budget=actions_budget,
        wall_time_budget_sec=wall_time_budget_sec,
        port=port,
        godot_path=str(godot_log_path),
        python_path=str(python_log_path),
        diagnostics_dir=str(diag_episode_dir),
        godot_exit_code=godot_exit_code,
        client_exit_code=client_exit_code,
        harness_status=harness_status,
        started_at=started_at,
        ended_at=ended_at,
    )

    (eval_episode_dir / "meta.json").write_text(
        json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def aggregate_live_run(
    *,
    eval_root: Path,
    metas: list[EpisodeMeta],
    actions_budget: int,
    wall_time_budget_sec: float,
    batch_run_id: str,
) -> Path:
    """Load each episode's NDJSON via the loader, aggregate, write summary.json."""
    from sight_agent.evaluator.episodes import load_episode
    from sight_agent.evaluator.metrics import aggregate

    episodes = []
    for m in metas:
        ep = load_episode(
            godot_path=Path(m.godot_path),
            python_path=Path(m.python_path),
            episode_id=m.episode_id,
            actions_budget=actions_budget,
            wall_time_budget_sec=wall_time_budget_sec,
            harness_status=m.harness_status,
        )
        episodes.append(ep)

    summary = aggregate(episodes)
    summary["run_id"] = batch_run_id
    summary["mode"] = "live"
    summary["episode_ids"] = [m.episode_id for m in metas]
    summary["actions_budget"] = actions_budget
    summary["wall_time_budget_sec"] = wall_time_budget_sec

    out_path = eval_root / "summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def live_mode(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    batch_run_id = args.run_id or _ts_run_id()

    try:
        godot_exe = resolve_godot_exe(args.godot_exe, os.environ)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 6

    project_dir = resolve_project_dir(args.project_dir)
    if not project_dir.exists():
        print(f"project-dir not found: {project_dir}", file=sys.stderr)
        return 7

    eval_root = (
        Path(args.out_dir)
        if args.out_dir
        else (repo_root / "runs" / "eval" / batch_run_id)
    )
    diagnostics_root = repo_root / "runs" / "diagnostics" / f"p3-live-{batch_run_id}"
    eval_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    metas: list[EpisodeMeta] = []
    for idx in range(1, max(1, int(args.episodes)) + 1):
        meta = run_one_live_episode(
            batch_run_id=batch_run_id,
            episode_idx=idx,
            eval_root=eval_root,
            diagnostics_root=diagnostics_root,
            godot_exe=godot_exe,
            project_dir=project_dir,
            port=args.port,
            actions_budget=args.actions_budget,
            wall_time_budget_sec=args.wall_time_budget_sec,
            apply_grace_sec=args.apply_grace_sec,
            base_env=os.environ,
        )
        metas.append(meta)

    summary_path = aggregate_live_run(
        eval_root=eval_root,
        metas=metas,
        actions_budget=args.actions_budget,
        wall_time_budget_sec=args.wall_time_budget_sec,
        batch_run_id=batch_run_id,
    )
    print(json.dumps({"summary": str(summary_path), "episodes": len(metas)}))
    return 0


# --- entrypoint ------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    refuse_if_ignore_death()
    args = parse_args(argv)

    if args.from_artifacts:
        return from_artifacts_mode(args)
    if args.live:
        return live_mode(args)
    # parse_args enforces mutual exclusivity with required=True; unreachable.
    print("internal: no mode selected", file=sys.stderr)
    return 5


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IgnoreDeathRefusal as exc:
        sys.exit(exc.EXIT_CODE)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
