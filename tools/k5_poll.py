"""Lean poll helper for the detached QR-DQN 200k run.

Prints sentinel state and the latest train_metrics line, retrying the
read because the training callback rewrites the whole ndjson each rollout
(truncate-then-write), so a naive read can land on an empty file.
"""
import json
import os
import sys
import time

RUN = r"C:\Projects\Sight\runs\phase_k\k5_7_qrdqn"
SENTINEL = r"C:\Projects\Sight\runs\phase_k\k5_7_qrdqn_200k.sentinel"
NDJSON = os.path.join(RUN, "train_metrics.ndjson")


def latest():
    for _ in range(8):
        try:
            lines = [l for l in open(NDJSON, encoding="utf-8") if l.strip()]
            if lines:
                return json.loads(lines[-1]), os.path.getmtime(NDJSON)
        except Exception:
            pass
        time.sleep(0.5)
    return None, None


def main():
    done = os.path.exists(SENTINEL)
    print("SENTINEL", "DONE" if done else "absent")
    if done:
        print("SENTINEL_BODY", open(SENTINEL, encoding="utf-8").read().strip())
    rec, mt = latest()
    if rec:
        age = round(time.time() - mt, 1)
        print(
            "steps", rec.get("num_timesteps"),
            "loss", round(rec.get("loss", float("nan")), 1),
            "ep_len_mean", round(rec.get("ep_len_mean", float("nan")), 1),
            "expl", round(rec.get("exploration_rate", float("nan")), 3),
            "age_s", age,
        )
    else:
        print("no_read")


if __name__ == "__main__":
    sys.exit(main())
