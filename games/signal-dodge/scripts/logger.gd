extends Node

# Sight - minimal NDJSON logger.
# Logs to user://runs/run_<timestamp>.ndjson. One event per line.
# Schema: {"t": seconds_since_run_start, "type": <event>, ...fields}

var _file: FileAccess = null
var _run_start_ms: int = 0

func start_run() -> void:
	DirAccess.make_dir_recursive_absolute("user://runs")
	var ts := Time.get_datetime_string_from_system().replace(":", "-")
	var path := "user://runs/run_%s.ndjson" % ts
	_file = FileAccess.open(path, FileAccess.WRITE)
	_run_start_ms = Time.get_ticks_msec()
	log_event("run_start", {"path": ProjectSettings.globalize_path(path)})

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

func end_run() -> void:
	if _file == null:
		return
	log_event("run_end", {})
	_file.flush()
	_file.close()
	_file = null
