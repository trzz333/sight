"""Phase B TCP client runner for Sight.

Connects to the Godot TCP server on 127.0.0.1:8765, sends a hello frame
and paced action commands, and writes Python-side decision NDJSON.
No perception, no screen capture. Standard library only.

Wire contract is newline-delimited UTF-8 JSON.
"""
import argparse
import datetime as _dt
import json
import os
import socket
import sys
import time
from pathlib import Path

PROTOCOL = 1


def action_for_seq(seq):
    """Deterministic stub policy. Returns (action, move_x)."""
    m = seq % 30
    if m < 10:
        return "stay", 0
    if m < 20:
        return "left", -1
    return "right", 1


def build_hello(run_id, agent):
    return {
        "type": "hello",
        "protocol": PROTOCOL,
        "run_id": run_id,
        "agent": agent,
    }


def build_action(seq, ts_unix_ns):
    action, move_x = action_for_seq(seq)
    return {
        "type": "action",
        "seq": seq,
        "ts_unix_ns": ts_unix_ns,
        "action": action,
        "move_x": move_x,
    }


def build_decision(run_id, seq, capture_ts_unix_ns, sent_ts_unix_ns):
    action, move_x = action_for_seq(seq)
    return {
        "type": "decision",
        "run_id": run_id,
        "seq": seq,
        "action": action,
        "move_x": move_x,
        "capture_ts_unix_ns": capture_ts_unix_ns,
        "decision_ts_unix_ns": capture_ts_unix_ns,
        "sent_ts_unix_ns": sent_ts_unix_ns,
    }


def connect_with_retry(host, port, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    last_err = None
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            s.settimeout(2.0)
            return s
        except OSError as e:
            last_err = e
            time.sleep(0.25)
    raise ConnectionError(
        "could not connect to {}:{} within {}s: {}".format(
            host, port, timeout_sec, last_err
        )
    )


def _send_json_line(sock, payload):
    line = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    sock.sendall(line)
    return time.time_ns()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phase B TCP client runner")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--actions", type=int, default=90)
    p.add_argument("--interval-sec", type=float, default=0.033)
    p.add_argument("--connect-timeout-sec", type=float, default=15.0)
    p.add_argument("--agent", default="phase_b_stub")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    now = _dt.datetime.now()
    ts_str = now.strftime("%Y%m%dT%H%M%S")
    run_id = args.run_id or "phase-b-" + ts_str
    out_path = args.out or os.path.join("runs", "phase_b_python_" + ts_str + ".ndjson")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    sock = connect_with_retry(args.host, args.port, args.connect_timeout_sec)

    actions_sent = 0
    try:
        hello = build_hello(run_id, args.agent)
        _send_json_line(sock, hello)

        with open(out_path, "w", encoding="utf-8") as f:
            for seq in range(args.actions):
                ts_ns = time.time_ns()
                msg = build_action(seq, ts_ns)
                sent_ts = _send_json_line(sock, msg)
                decision = build_decision(run_id, seq, ts_ns, sent_ts)
                f.write(json.dumps(decision, separators=(",", ":")) + "\n")
                f.flush()
                actions_sent += 1
                if args.interval_sec > 0:
                    time.sleep(args.interval_sec)
    finally:
        try:
            sock.close()
        except OSError:
            pass

    summary = {
        "run_id": run_id,
        "out": out_path,
        "actions_sent": actions_sent,
        "host": args.host,
        "port": args.port,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)