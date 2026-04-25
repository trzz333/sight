"""Per-run NDJSON logger.

One file per side per run:
    runs/<run_id>/python.ndjson
    runs/<run_id>/godot.ndjson    (written by Godot, not by this module)
    runs/<run_id>/manifest.json

Godot currently writes to `%APPDATA%\\Godot\\app_userdata\\Signal Dodge\\runs\\run_<ts>.ndjson`.
That path is preserved for backward compatibility. Shared run_id/run_dir coordination happens
later once live Godot runs confirm the schema. See docs/sight-handoff.md.
"""

from .ndjson import NDJSONLogger, new_run_id

__all__ = ["NDJSONLogger", "new_run_id"]
