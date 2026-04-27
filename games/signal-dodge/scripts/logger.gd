extends Node

# Sight - minimal NDJSON logger.
# Logs to user://runs/run_<timestamp>.ndjson. One event per line.
# Schema: {"t": seconds_since_run_start, "type": <event>, ...fields}

var _file: FileAccess = null
var _run_start_ms: int = 0

func start_run(meta: Dictionary = {}) -> void:
	# SIGHT_GODOT_LOG_PATH overrides the default user://runs target. When set, the logger
	# writes only to that absolute path; user://runs is not touched. The harness uses this
	# to land godot.ndjson directly under runs/eval/<run_id>/episodes/<episode_id>/ without
	# a user://-to-runs copy step. Empty/unset preserves legacy behavior.
	var override_path := OS.get_environment("SIGHT_GODOT_LOG_PATH")
	var path: String
	var resolved_path: String
	if override_path != "":
		path = override_path
		var parent := path.get_base_dir()
		if parent != "":
			DirAccess.make_dir_recursive_absolute(parent)
		resolved_path = path
	else:
		DirAccess.make_dir_recursive_absolute("user://runs")
		var ts := Time.get_datetime_string_from_system().replace(":", "-")
		path = "user://runs/run_%s.ndjson" % ts
		resolved_path = ProjectSettings.globalize_path(path)
	_file = FileAccess.open(path, FileAccess.WRITE)
	_run_start_ms = Time.get_ticks_msec()
	var evt := {"path": resolved_path}
	for k in meta.keys():
		evt[k] = meta[k]
	log_event("run_start", evt)

func log_event(event_type: String, data: Dictionary) -> void:
	if _file == null:
		return
	var record := {
		"t": (Time.get_ticks_msec() - _run_start_ms) / 1000.0,
		"type": event_type,
	}
	for k in data.keys():
		record[k] = data[k]
	_file.store_line(JSON.stringify(record))
	# Per-event flush guards against mid-write truncation when the process is killed
	# externally before _exit_tree fires (TerminateProcess, harness kill, crash).
	_file.flush()

func end_run() -> void:
	# Idempotent: nulling _file at the end makes a second call a no-op.
	if _file == null:
		return
	log_event("run_end", {})
	_file.flush()
	_file.close()
	_file = null

func _exit_tree() -> void:
	# Autoload _exit_tree fires on SceneTree.quit() and on engine shutdown, covering
	# harness-driven and player_died exits. end_run() is idempotent so this acts as
	# a safety net when the death path did not run.
	end_run()