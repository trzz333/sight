#!/usr/bin/env python3
"""Atomic Sight handoff update.

Edits docs\\sight-handoff.md per-field via re.subn, commits everything dirty
under a substantive message, pushes, refreshes the **Last commit:** line with
the new short hash, commits the chore, pushes again. Returns JSON with both
hashes plus the substantive subject.

Why this exists:
- Whole-file rewrites of sight-handoff.md cause silent transcription drift
  on unchanged fields. Per-field re.subn anchored on the bold **Field:**
  markers eliminates that drift.
- Desktop Commander's start_process wrapping breaks `git commit -m "..."`
  (cmd.exe sees args word-split into pathspecs). Writing the message to a
  tempfile and using `git commit -F <file>` is the only reliable path.
- The two-commit dance (substantive then chore-refresh-hash) is the project
  pattern. Doing it inside one script keeps the tree clean and the bootstrap
  hash truthful.

Input JSON (stdin or --input <path>). All fields optional except
commit_subject; missing schema fields are not touched.

  {
    "phase":          "P3 in progress. ...",
    "current_task":   "...",
    "next_action":    "...",
    "blockers":       "None on main. ...",
    "notes":          ["note 1", "note 2", "note 3"],
    "commit_subject": "handoff: <72-char subject>",
    "commit_body":    "<optional 1-3 sentence body>"
  }

Output JSON to stdout:

  {"substantive_hash": "787b9f0", "refresh_hash": "67fa151",
   "subject": "handoff: ..."}

Exit codes:
  0 success
  2 input JSON missing or invalid
  3 doc not found at canonical path
  5 git push failed
  6 field substitution failed (anchor not found)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(r"C:\Projects\Sight")
DOC = REPO / "docs" / "sight-handoff.md"
GIT = r"C:\Program Files\Git\cmd\git.exe"

# Anchored on the bold field markers. Each pattern matches the entire line
# including the marker; the replacement carries the marker plus the new value.
_FIELD_PATTERNS: dict[str, bytes] = {
    "phase":        rb"\*\*Phase:\*\*[^\r\n]*",
    "last_commit":  rb"\*\*Last commit:\*\*[^\r\n]*",
    "current_task": rb"\*\*Current task:\*\*[^\r\n]*",
    "next_action":  rb"\*\*Next action:\*\*[^\r\n]*",
    "blockers":     rb"\*\*Blockers:\*\*[^\r\n]*",
}
_FIELD_PREFIX: dict[str, bytes] = {
    "phase":        b"**Phase:** ",
    "last_commit":  b"**Last commit:** ",
    "current_task": b"**Current task:** ",
    "next_action":  b"**Next action:** ",
    "blockers":     b"**Blockers:** ",
}


class HandoffError(RuntimeError):
    pass


def _git(*args: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        [GIT, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if check and cp.returncode != 0:
        raise HandoffError(
            f"git {' '.join(args)} failed (exit {cp.returncode}): "
            f"{cp.stderr.strip() or cp.stdout.strip()}"
        )
    return cp


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.close(fd)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _update_field(data: bytes, field: str, value: str) -> bytes:
    pat = _FIELD_PATTERNS[field]
    prefix = _FIELD_PREFIX[field]
    new, n = re.subn(pat, prefix + value.encode("utf-8"), data, count=1)
    if n != 1:
        raise HandoffError(f"field {field!r}: expected 1 anchor match, got {n}")
    return new


def _update_notes(data: bytes, notes: list[str]) -> bytes:
    if not isinstance(notes, list):
        raise HandoffError(f"notes must be a list, got {type(notes).__name__}")
    if len(notes) > 5:
        raise HandoffError(f"max 5 notes; got {len(notes)}")
    if any(not isinstance(n, str) or not n.strip() for n in notes):
        raise HandoffError("each note must be a non-empty string")
    block = (
        b"**Notes:**\n\n"
        + b"\n".join(b"- " + n.encode("utf-8") for n in notes)
        + b"\n"
    )
    # Match the existing Notes block from the marker through the trailing
    # bullet line. The doc ends after the bullets so we anchor on the marker
    # plus all bullets that follow.
    new, n = re.subn(
        rb"\*\*Notes:\*\*\r?\n\r?\n(?:- [^\r\n]*\r?\n)+",
        block,
        data,
        count=1,
    )
    if n != 1:
        raise HandoffError(f"notes block: expected 1 match, got {n}")
    return new


def _commit(subject: str, body: str = "") -> str:
    """Write commit message to a tempfile and `git commit -F` it.

    Avoids `git commit -m "..."` because Desktop Commander's start_process
    word-splits the quoted subject into pathspecs.
    """
    msg = subject.rstrip() + (("\n\n" + body.rstrip()) if body.strip() else "") + "\n"
    fd, tmp = tempfile.mkstemp(dir=REPO, prefix="COMMIT_MSG_", suffix=".txt")
    try:
        os.write(fd, msg.encode("utf-8"))
        os.close(fd)
        _git("add", "-A")
        _git("commit", "-F", tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return _git("rev-parse", "--short", "HEAD", capture=True).stdout.strip()


_PLACEHOLDER = b"PLACEHOLDER_HASH PLACEHOLDER_SUBJECT"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="handoff_update")
    ap.add_argument("--input", type=Path, default=None,
                    help="JSON input path. If omitted, read from stdin.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write doc updates but do not commit or push.")
    args = ap.parse_args(argv)

    if not DOC.exists():
        print(f"doc not found: {DOC}", file=sys.stderr)
        return 3

    try:
        if args.input:
            payload = json.loads(args.input.read_text(encoding="utf-8"))
        else:
            payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        print(f"invalid input json: {e}", file=sys.stderr)
        return 2

    if not isinstance(payload, dict):
        print("input must be a JSON object", file=sys.stderr)
        return 2

    data = DOC.read_bytes()

    # Update simple schema fields. Last commit becomes the placeholder so the
    # post-push hash refresh has something stable to substitute on.
    for field in ("phase", "current_task", "next_action", "blockers"):
        if field in payload:
            data = _update_field(data, field, payload[field])
    data = _update_field(data, "last_commit",
                         _PLACEHOLDER.decode("utf-8"))
    if "notes" in payload:
        data = _update_notes(data, payload["notes"])

    _atomic_write(DOC, data)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "doc": str(DOC)}))
        return 0

    subject = payload.get("commit_subject", "handoff: update")
    body = payload.get("commit_body", "")

    sub_hash = _commit(subject, body)

    # Substitute the real hash + subject into the **Last commit:** line.
    data = DOC.read_bytes()
    real_line = f"{sub_hash} {subject}".encode("utf-8")
    data, n = re.subn(_PLACEHOLDER, real_line, data, count=1)
    if n != 1:
        print("hash refresh: PLACEHOLDER not found in doc", file=sys.stderr)
        return 6
    _atomic_write(DOC, data)

    refresh_hash = _commit("chore: refresh handoff hash")

    try:
        _git("push", "origin", "main")
    except HandoffError as e:
        print(f"push failed: {e}", file=sys.stderr)
        return 5

    print(json.dumps({
        "substantive_hash": sub_hash,
        "refresh_hash": refresh_hash,
        "subject": subject,
    }))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HandoffError as e:
        print(f"handoff error: {e}", file=sys.stderr)
        sys.exit(1)
