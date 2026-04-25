extends Node

# Sight - Godot-side TCP controller source (server).
# Enabled by env var SIGHT_TCP_MODE=1. Default main loop remains the in-Godot rule agent.
#
# Wire contract:
#   Python (client) sends newline-delimited JSON to 127.0.0.1:SIGHT_TCP_PORT.
#   hello   {"type":"hello","protocol":1,"run_id":"<id>","agent":"<n>"}
#   action  {"type":"action","seq":<int>,"ts_unix_ns":<int>,"action":"left|right|stay","move_x":-1|0|1}
#
# Godot never sends game state back. Frame/apply acknowledgments are reserved for reconciliation,
# not perception. See docs/sight-charter.md ethics armor.
#
# Apply semantics:
#   - poll() returns the latest move_x every frame; held action carries across frames.
#   - log_applied() emits controller_cmd_applied at most ONCE per new seq (first-applied frame).
#     The reconciler is also defensive against duplicates, but the source of truth is here.
#   - run_id captured from hello is stamped onto every controller_* event after the hello so
#     the evaluator can detect Python/Godot run_id mismatches without legacy log changes.

const DEFAULT_HOST := "127.0.0.1"
const DEFAULT_PORT := 8765

var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
var _recv_buf := PackedByteArray()

# Latest action state. Held across frames per spec: if no new command, hold previous action.
var _last_action := "stay"
var _last_move_x := 0
var _last_seq := 0
var _last_ts_unix_ns := 0
# First-applied-frame guard. log_applied() compares _last_seq against _last_logged_seq.
var _last_logged_seq := 0

# run_id captured from hello; stamped onto every controller_* event afterwards.
var _run_id := ""

var _connected_once := false
var _disconnected_logged := false
var _hello: Dictionary = {}
var _active := false
var _host := DEFAULT_HOST
var _port := DEFAULT_PORT

func is_active() -> bool:
	return _active

func run_id() -> String:
	return _run_id

# Call once from Main._ready if SIGHT_TCP_MODE is enabled.
func start(host: String = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Error:
	_host = host
	_port = port
	var err := _server.listen(port, host)
	if err != OK:
		push_error("tcp_controller: listen(%s:%d) failed with %d" % [host, port, err])
		return err
	_active = true
	return OK

func stop() -> void:
	if _peer != null:
		_peer.disconnect_from_host()
		_peer = null
	if _server.is_listening():
		_server.stop()
	_active = false

# Called every physics tick by Main BEFORE player movement. Returns the action to apply as int
# in {-1, 0, +1}. Holds previous action if no new command arrived.
func poll(_frame: int) -> int:
	if not _active:
		return _last_move_x

	# Accept a pending connection.
	if _peer == null and _server.is_connection_available():
		_peer = _server.take_connection()
		_connected_once = true
		_disconnected_logged = false
		SightLog.log_event("controller_connected", _decorate({"host": _host, "port": _port}))

	if _peer == null:
		return _last_move_x

	# Drain any bytes; parse complete JSON lines.
	_peer.poll()
	var status := _peer.get_status()
	if status != StreamPeerTCP.STATUS_CONNECTED:
		if _connected_once and not _disconnected_logged:
			SightLog.log_event("controller_disconnect", _decorate({"status": status}))
			_disconnected_logged = true
			# Stay neutral on disconnect.
			_last_action = "stay"
			_last_move_x = 0
		_peer = null
		return _last_move_x

	var available: int = _peer.get_available_bytes()
	if available > 0:
		var data := _peer.get_data(available)
		if typeof(data) == TYPE_ARRAY and data.size() >= 2 and int(data[0]) == OK:
			_recv_buf.append_array(data[1])

	# Split on newlines.
	while true:
		var nl := _recv_buf.find(0x0A)  # '\n'
		if nl < 0:
			break
		var line_bytes := _recv_buf.slice(0, nl)
		_recv_buf = _recv_buf.slice(nl + 1)
		var line := line_bytes.get_string_from_utf8().strip_edges()
		if line == "":
			continue
		_handle_line(line)

	return _last_move_x

func _handle_line(line: String) -> void:
	var parse: Variant = JSON.parse_string(line)
	if typeof(parse) != TYPE_DICTIONARY:
		SightLog.log_event("controller_bad_line", _decorate({"line": line}))
		return
	var msg: Dictionary = parse
	var mtype := str(msg.get("type", ""))
	if mtype == "hello":
		_hello = msg.duplicate()
		_run_id = str(msg.get("run_id", ""))
		# controller_hello stamps run_id explicitly even if _decorate would omit empty string.
		SightLog.log_event("controller_hello", {
			"protocol": msg.get("protocol"),
			"run_id": _run_id,
			"agent": msg.get("agent"),
		})
		return
	if mtype == "action":
		var action := str(msg.get("action", "stay"))
		var move_x: int = int(msg.get("move_x", 0))
		var seq: int = int(msg.get("seq", 0))
		var ts: int = int(msg.get("ts_unix_ns", 0))
		_last_action = action
		_last_move_x = clamp(move_x, -1, 1)
		_last_seq = seq
		_last_ts_unix_ns = ts
		return
	SightLog.log_event("controller_unknown_type", _decorate({"type": mtype}))

# Main calls this after move_action to log that the command was applied on this frame.
# First-applied-frame semantics: at most one controller_cmd_applied per new seq.
func log_applied(frame: int) -> void:
	if _last_seq <= 0:
		return
	if _last_seq == _last_logged_seq:
		return  # already logged this seq on its first applied frame; held action carries silently
	_last_logged_seq = _last_seq
	SightLog.log_event("controller_cmd_applied", _decorate({
		"seq": _last_seq,
		"frame": frame,
		"action": _last_action,
		"move_x": _last_move_x,
		"ts_unix_ns": _last_ts_unix_ns,
	}))

func latest_move_x() -> int:
	return _last_move_x

# Inject run_id into outbound Logger events when known. Empty run_id is omitted so legacy
# evaluator behavior on logs that pre-date the hello stays unchanged.
func _decorate(data: Dictionary) -> Dictionary:
	if _run_id != "" and not data.has("run_id"):
		data["run_id"] = _run_id
	return data
