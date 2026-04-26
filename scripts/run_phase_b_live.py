import os, sys, json, time, glob, shutil, subprocess, datetime, pathlib, socket
from collections import Counter

REPO = r"C:\Projects\Sight"
PROJECT = r"C:\Projects\Sight\games\signal-dodge"
GODOT_EXE = r"C:\Users\maste\AppData\Local\Microsoft\WinGet\Packages\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\Godot_v4.6.2-stable_win64_console.exe"
GODOT_USER_RUNS = r"C:\Users\maste\AppData\Roaming\Godot\app_userdata\Signal Dodge\runs"
PORT = 8765
ACTIONS = 300

TS = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
diag = pathlib.Path(REPO) / "runs" / "diagnostics" / f"phase_b_live_{TS}"
diag.mkdir(parents=True, exist_ok=True)
print(f"=== TS={TS} ===")
print(f"diag_dir={diag}")

snap_before = set(glob.glob(os.path.join(GODOT_USER_RUNS, "*.ndjson")))
print(f"godot_user_runs_before_count={len(snap_before)}")

env = os.environ.copy()
env["SIGHT_TCP_MODE"] = "1"
env["SIGHT_TCP_PORT"] = str(PORT)
# 300-action transport endurance: suppress in-Godot death termination so the run
# stays alive past the t~8s collision and the harness can drive the full ACTIONS
# budget over TCP. Requires SIGHT_TCP_MODE=1; the flag is otherwise inert.
env["SIGHT_TCP_IGNORE_DEATH"] = "1"

godot_stdout_path = diag / "godot_stdout.log"
godot_stderr_path = diag / "godot_stderr.log"
godot_stdout_f = godot_stdout_path.open("wb")
godot_stderr_f = godot_stderr_path.open("wb")

t0 = time.monotonic()
godot = subprocess.Popen(
    [GODOT_EXE, "--headless", "--path", PROJECT],
    env=env,
    stdout=godot_stdout_f,
    stderr=godot_stderr_f,
)
print(f"godot_pid={godot.pid}")

# Wait for Godot to bind the TCP port
deadline = time.monotonic() + 25.0
listening = False
while time.monotonic() < deadline:
    if godot.poll() is not None:
        print(f"FATAL godot exited early code={godot.returncode}")
        godot_stdout_f.close(); godot_stderr_f.close()
        print("--- godot stderr ---")
        try: print(godot_stderr_path.read_text(encoding="utf-8", errors="replace")[:2000])
        except Exception as e: print(f"(read err: {e})")
        sys.exit(2)
    try:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=0.5)
        s.close()
        listening = True
        break
    except OSError:
        time.sleep(0.25)

print(f"godot_listening={listening} elapsed={time.monotonic()-t0:.2f}s")
if not listening:
    godot.kill(); godot.wait(timeout=5.0)
    godot_stdout_f.close(); godot_stderr_f.close()
    print("FATAL godot did not bind port in time")
    sys.exit(3)

# Run Python client
py_out = pathlib.Path(REPO) / "runs" / f"phase_b_python_{TS}.ndjson"
client_stdout_path = diag / "python_stdout.log"
client_stderr_path = diag / "python_stderr.log"
client = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts", "run_phase_b.py"),
     "--actions", str(ACTIONS),
     "--port", str(PORT),
     "--out", str(py_out),
     "--run-id", f"phase-b-{TS}"],
    cwd=REPO,
    capture_output=True,
    text=True,
    timeout=60.0,
)
client_stdout_path.write_text(client.stdout, encoding="utf-8")
client_stderr_path.write_text(client.stderr, encoding="utf-8")
print(f"python_client_exit={client.returncode}")
print(f"python_client_stdout={client.stdout.strip()[:300]}")
if client.stderr.strip():
    print(f"python_client_stderr_head={client.stderr.strip()[:300]}")

# Grace period for Godot to apply final actions
time.sleep(2.5)

# Stop Godot. terminate() = TerminateProcess on Windows = no _exit_tree.
# Per-event flush in logger.gd guards against truncation; only the run_end event is at risk.
# In SIGHT_TCP_IGNORE_DEATH=1 runs there is intentionally no run_end (no death path executes).
try:
    godot.terminate()
    try:
        godot.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        godot.kill()
        godot.wait(timeout=5.0)
except Exception as e:
    print(f"godot_stop_err: {e}")
finally:
    godot_stdout_f.close()
    godot_stderr_f.close()
print(f"godot_exit={godot.returncode} elapsed={time.monotonic()-t0:.2f}s")

# Let filesystem settle
time.sleep(0.3)

# Find new Godot ndjson
snap_after = set(glob.glob(os.path.join(GODOT_USER_RUNS, "*.ndjson")))
new_files = sorted(snap_after - snap_before, key=os.path.getmtime, reverse=True)
print(f"godot_user_runs_after_count={len(snap_after)}")
print(f"new_ndjson_count={len(new_files)}")
if not new_files:
    print("FATAL no new godot ndjson")
    sys.exit(4)
godot_ndjson = new_files[0]
print(f"godot_ndjson={godot_ndjson}")

shutil.copy2(godot_ndjson, diag / "godot.ndjson")
shutil.copy2(py_out, diag / "python.ndjson")

# Truncation check
raw = pathlib.Path(godot_ndjson).read_bytes()
print(f"godot_ndjson_size={len(raw)}")
print(f"godot_ndjson_ends_with_lf={raw.endswith(b'\n')}")
print(f"py_ndjson_size={py_out.stat().st_size}")

# Run evaluator
sys.path.insert(0, os.path.join(REPO, "src"))
from sight_agent.evaluator.reconcile import load_ndjson, evaluate

try:
    py_events = load_ndjson(py_out)
    g_events = load_ndjson(godot_ndjson)
except ValueError as e:
    print(f"FATAL ndjson load failed: {e}")
    sys.exit(5)

print(f"py_events_count={len(py_events)}")
print(f"godot_events_count={len(g_events)}")

gt = Counter(e.get("type") for e in g_events)
pt = Counter(e.get("type") for e in py_events)
print(f"godot_event_types={dict(gt)}")
print(f"python_event_types={dict(pt)}")

metrics = evaluate(g_events, py_events)
applied_seqs = sorted(set(e["seq"] for e in g_events if e.get("type") == "controller_cmd_applied" and "seq" in e))
py_seqs = sorted(set(e["seq"] for e in py_events if e.get("type") == "decision" and "seq" in e))

gate = {
    "joined_count": metrics["joined_count"],
    "unmatched_python_count": metrics["unmatched_python_count"],
    "unmatched_godot_count": metrics["unmatched_godot_count"],
    "duplicate_applied_seq_count": metrics["duplicate_applied_seq_count"],
    "run_id_mismatch": metrics["run_id_mismatch"],
    "applied_seq_min": applied_seqs[0] if applied_seqs else None,
    "applied_seq_max": applied_seqs[-1] if applied_seqs else None,
    "applied_seq_count_unique": len(applied_seqs),
    "seq_zero_applied": 0 in applied_seqs,
    "py_seq_min": py_seqs[0] if py_seqs else None,
    "py_seq_max": py_seqs[-1] if py_seqs else None,
    "godot_run_id": metrics["godot_run_id"],
    "python_run_id": metrics["python_run_id"],
    "tcp_death_ignored_count": gt.get("tcp_death_ignored", 0),
}
print("=== GATE ===")
print(json.dumps(gate, indent=2, default=str))

clean = (
    gate["joined_count"] == ACTIONS and
    gate["unmatched_python_count"] == 0 and
    gate["unmatched_godot_count"] == 0 and
    gate["duplicate_applied_seq_count"] == 0 and
    gate["run_id_mismatch"] is False and
    gate["seq_zero_applied"] is True
)
print(f"GATE_CLEAN={clean}")
print("=== END ===")
