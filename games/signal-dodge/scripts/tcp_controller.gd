extends Node

# Sight - Godot-side TCP controller source (server).
# Enabled by env var SIGHT_TCP_MODE=1. Default main loop remains the in-Godot rule agent.
#
# Two protocol modes share this listener:
#
# Legacy v1 (unidirectional, pre-H3): Python sends newline-delimited JSON with field
# "protocol":1. Godot logs and applies but never sends responses. Preserved for
# backwards compatibility with existing harness clients.
#
# H3 v2 (bidirectional): Python sends newline-delimited JSON with field "protocol_version":2.
# Mode dispatch keys on the first message that carries a class-identifying field:
#   - protocol_version present -> H3 mode
#   - protocol present          -> legacy mode (existing unidirectional path only)
# After a mode is selected, opposite-mode messages are errors. In H3 mode, any message
# carrying the legacy "protocol" field produces an "error" response with code
# "protocol_version_mismatch". In legacy mode, H3-shaped messages are logged and dropped
# because the legacy channel cannot send structured responses.
#
# H3 wire contract (newline-delimited UTF-8 JSON, one object per line, loopback only):
#   hello   {"type":"hello","protocol_version":2,"run_id":"<id>"}
#   reset   {"type":"reset","protocol_version":2,"run_id":"<id>","episode_id":"<id>",
#            "seed":<int>,"max_steps":<int>}
#   step    {"type":"step","protocol_version":2,"run_id":"<id>","episode_id":"<id>",
#            "seq":<int>,"action":0|1|2}
#   reset_ok    {"type":"reset_ok","protocol_version":2,"run_id":"<id>","episode_id":"<id>",
#                "frame":<int>,"obs":[...],"terminated":false,"truncated":false,"info":{...}}
#   step_result {"type":"step_result","protocol_version":2,"run_id":"<id>","episode_id":"<id>",
#                "seq":<int>,"frame":<int>,"obs":[...],"reward":<float>,"terminated":<bool>,
#                "truncated":<bool>,"terminal_reason":""|"collision"|"timeout","info":{...}}
#   error       {"type":"error","protocol_version":2,"code":"<str>","message":"<str>"}
#
# H3 step semantics: tcp_controller parses the request and parks it in a pending slot.
# main.gd consumes the slot at the deterministic physics point, applies the action,
# advances physics, builds the observation, and calls the matching send helper. The
# controller never builds responses on its own except for protocol-level errors and
# pipeline-overrun errors.
#
# Legacy apply semantics (unchanged):
#   - poll() returns the latest move_x every frame; held action carries across frames.
#   - log_applied() emits controller_cmd_applied at most ONCE per new seq (first-applied frame).
#     The reconciler is also defensive against duplicates, but the source of truth is here.
#   - run_id captured from hello is stamped onto every controller_* event after the hello so
#     the evaluator can detect Python/Godot run_id mismatches without legacy log changes.

const DEFAULT_HOST := "127.0.0.1"
const DEFAULT_PORT := 8765

# H3 protocol literals. Mirror src/sight_agent/protocol.py. Duplicated rather than imported
# because GDScript has no Python-import seam; the Python module is the authoritative
# contract and any drift here is a bug.
const H3_PROTOCOL_VERSION := 2
const FIELD_PROTOCOL_VERSION := "protocol_version"
const FIELD_LEGACY_PROTOCOL := "protocol"

# Mode tripwire. Locked by the first message that carries a class-identifying field.
const MODE_UNSET := 0
const MODE_LEGACY := 1
const MODE_H3 := 2

const MSG_HELLO := "hello"
const MSG_RESET := "reset"
const MSG_STEP := "step"
const MSG_RESET_OK := "reset_ok"
const MSG_STEP_RESULT := "step_result"
const MSG_ERROR := "error"

# H3 error codes. Mirror src/sight_agent/protocol.py.
const ERROR_PROTOCOL_VERSION_MISMATCH := "protocol_version_mismatch"
const ERROR_RUN_ID_MISMATCH := "run_id_mismatch"
const ERROR_EPISODE_ID_MISMATCH := "episode_id_mismatch"
const ERROR_BAD_REQUEST := "bad_request"
const ERROR_INTERNAL := "internal"

# H3 discrete action wire values. Mirror sight_agent.protocol.VALID_DISCRETE_ACTIONS.
const ACTION_DISCRETE_LEFT := 0
const ACTION_DISCRETE_STAY := 1
const ACTION_DISCRETE_RIGHT := 2

# H4 observation-mode literals. Mirror sight_agent.protocol.OBS_MODE_*.
# State mode is the H3-default wire shape (length-10 numeric list).
# Pixel and both modes are H4-introduced and require the structured
# obs payload defined in docs/sight-h4-plan.md Decision 4. The pixel
# capture path is not yet implemented in Godot; H4 Step 2 (this code)
# accepts the protocol fields, validates them, and rejects pixel/both
# modes at reset time with bad_request so the caller cannot reach a
# half-implemented path. Step 4 (per the H4 plan implementation
# sequence) is where pixel emission lands.
const OBS_MODE_STATE := "state"
const OBS_MODE_PIXEL := "pixel"
const OBS_MODE_BOTH := "both"

const FIELD_OBSERVATION_MODE := "observation_mode"
const FIELD_PIXEL_WIDTH := "pixel_width"
const FIELD_PIXEL_HEIGHT := "pixel_height"
const FIELD_PIXEL_CHANNELS := "pixel_channels"

# Start-state curriculum injection (Godot port of the replica CurriculumSDF,
# tools/sd_fast_ppo_curriculum.py). Optional non-negative int on the reset
# request. Default/absent -> 0, which preserves byte-identical clean-start
# behavior. When > 0, main.gd pre-spawns N hazards above the player at reset.
# The held-out eval never sends this field, so the eval-of-record is untouched.
const FIELD_CURRICULUM_N_INIT := "curriculum_n_init"

# Default pixel dims when observation_mode is set without explicit dims
# (the env layer always sends explicit dims, but the Godot side defaults
# safely so a malformed payload from a future caller is not silently
# accepted). 84/84/1 mirrors the env constructor defaults.
const DEFAULT_PIXEL_WIDTH := 84
const DEFAULT_PIXEL_HEIGHT := 84
const DEFAULT_PIXEL_CHANNELS := 1

var _server := TCPServer.new()
var _peer: StreamPeerTCP = null
var _recv_buf := PackedByteArray()

# Latest legacy action state. Held across frames per spec: if no new command, hold previous.
var _last_action := "stay"
var _last_move_x := 0
# Sentinels for "no command received yet". Python's first action carries seq=0, so the
# pre-command sentinel must be a value seq cannot take. -1 is reserved for that role.
var _last_seq := -1
var _last_ts_unix_ns := 0
# First-applied-frame guard. log_applied() compares _last_seq against _last_logged_seq.
var _last_logged_seq := -1
# Distinct-seq apply counter. Increments only when log_applied() actually emits a
# controller_cmd_applied event for a new seq. Held actions across frames do not count.
# Used by main.gd to enforce SIGHT_P3_ACTIONS_BUDGET in legacy mode.
var _applied_count := 0

# run_id captured from hello; stamped onto controller_* events afterwards. Shared across
# legacy and H3 modes (only one mode locks per controller lifetime).
var _run_id := ""

# Locked mode after first class-identifying message. See dispatch comments above.
var _mode: int = MODE_UNSET

# Active H3 episode_id, set by the most recent valid `reset` request. Empty string until
# the first reset. Step 2 only field-validates; episode_id mismatch detection lands later.
var _h3_episode_id := ""

# Active H4 observation mode for the current episode, locked at the most recent valid
# reset. Defaults to OBS_MODE_STATE so a reset that omits the H4 fields preserves H3
# byte-compatible behavior. Pixel/both modes are protocol-accepted but rejected at
# reset (see _h3_handle_reset) until the pixel source lands in H4 Step 4.
var _h3_observation_mode := OBS_MODE_STATE
var _h3_pixel_width := DEFAULT_PIXEL_WIDTH
var _h3_pixel_height := DEFAULT_PIXEL_HEIGHT
var _h3_pixel_channels := DEFAULT_PIXEL_CHANNELS

# Single-slot pending H3 request. Empty dict means "no pending request". Populated when a
# valid `reset` or `step` arrives in H3 mode; consumed by main.gd via take_pending_h3_request().
# A second pending request before the slot is consumed is a protocol violation -> error
# response with code bad_request.
var _pending_request: Dictionary = {}

var _connected_once := false
var _disconnected_logged := false
var _hello: Dictionary = {}
var _active := false
var _host := DEFAULT_HOST
var _port := DEFAULT_PORT

# --- Public accessors -----------------------------------------------------

func is_active() -> bool:
	return _active

func mode() -> int:
	return _mode

func run_id() -> String:
	return _run_id

func h3_episode_id() -> String:
	return _h3_episode_id

# H4 accessors: active observation mode and pixel dims for the current
# episode. Defaults preserve H3 state behavior. Set by _h3_handle_reset.
func h3_observation_mode() -> String:
	return _h3_observation_mode

func h3_pixel_width() -> int:
	return _h3_pixel_width

func h3_pixel_height() -> int:
	return _h3_pixel_height

func h3_pixel_channels() -> int:
	return _h3_pixel_channels

# True iff a valid H3 reset/step request is parked and waiting for main.gd to consume it.
func has_pending_h3_request() -> bool:
	return not _pending_request.is_empty()

# Returns the parked H3 request and clears the slot. Caller (main.gd in step 3+) inspects
# the "type" field and calls send_reset_ok / send_step_result accordingly. Returns an
# empty dict if nothing is pending.
func take_pending_h3_request() -> Dictionary:
	var r := _pending_request
	_pending_request = {}
	return r

# Distinct count of seq values that have been first-applied (legacy mode only). Held
# actions across frames do not count.
func applied_count() -> int:
	return _applied_count

# --- Lifecycle ------------------------------------------------------------

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

# --- Main poll loop -------------------------------------------------------

# Called every physics tick by Main BEFORE player movement. Returns the legacy action to
# apply as int in {-1, 0, +1}. In H3 mode actions arrive via parked step requests rather
# than this return value; main.gd in step 3+ branches on mode() and consumes
# take_pending_h3_request() instead of using this return. The return is kept stable so
# legacy callers continue to work unchanged.
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

# --- Mode dispatch --------------------------------------------------------

func _handle_line(line: String) -> void:
	var parse: Variant = JSON.parse_string(line)
	if typeof(parse) != TYPE_DICTIONARY:
		SightLog.log_event("controller_bad_line", _decorate({"line": line}))
		return
	var msg: Dictionary = parse
	var has_h3_field := msg.has(FIELD_PROTOCOL_VERSION)
	var has_legacy_field := msg.has(FIELD_LEGACY_PROTOCOL)

	# Cross-mode rejection. Once a mode is locked, opposite-mode messages are errors
	# (in H3 mode) or logged-and-dropped (in legacy mode, which has no response channel).
	if _mode == MODE_H3 and has_legacy_field:
		send_error(ERROR_PROTOCOL_VERSION_MISMATCH,
			"received legacy 'protocol' field in H3 mode")
		return
	if _mode == MODE_LEGACY and has_h3_field:
		SightLog.log_event("controller_unknown_type",
			_decorate({"type": str(msg.get("type", "")), "reason": "h3_in_legacy_mode"}))
		return

	# Field-class dispatch. UNSET locks here on first class-identifying field.
	if has_h3_field:
		if _mode == MODE_UNSET:
			_mode = MODE_H3
		_h3_dispatch(msg)
		return
	if has_legacy_field:
		if _mode == MODE_UNSET:
			_mode = MODE_LEGACY
		_legacy_dispatch(msg)
		return

	# Neither field. Legacy actions don't carry a protocol field; preserve that path
	# by routing through legacy_dispatch in UNSET or LEGACY mode without locking.
	if _mode == MODE_H3:
		send_error(ERROR_BAD_REQUEST, "missing required field protocol_version")
		return
	_legacy_dispatch(msg)

# --- Legacy v1 dispatch (unchanged behavior) -----------------------------

func _legacy_dispatch(msg: Dictionary) -> void:
	var mtype := str(msg.get("type", ""))
	if mtype == MSG_HELLO:
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

# --- H3 v2 dispatch -------------------------------------------------------

func _h3_dispatch(msg: Dictionary) -> void:
	# Protocol-version check applies to every H3-class message.
	# JSON has no int/float distinction at the wire level. Godot 4.6.2's
	# JSON.parse_string widens integer numbers to TYPE_FLOAT, so a Python
	# json.dumps({"protocol_version": 2}) arrives here as a float. Accept
	# both TYPE_INT and TYPE_FLOAT and compare via int() coercion; the
	# values we exchange are always small whole numbers so int() round-trip
	# is exact.
	var pv = msg.get(FIELD_PROTOCOL_VERSION)
	var pv_type := typeof(pv)
	if (pv_type != TYPE_INT and pv_type != TYPE_FLOAT) or int(pv) != H3_PROTOCOL_VERSION:
		send_error(ERROR_PROTOCOL_VERSION_MISMATCH,
			"expected protocol_version=%d, got %s" % [H3_PROTOCOL_VERSION, str(pv)])
		return

	var mtype := str(msg.get("type", ""))
	match mtype:
		MSG_HELLO:
			_h3_handle_hello(msg)
		MSG_RESET:
			_h3_handle_reset(msg)
		MSG_STEP:
			_h3_handle_step(msg)
		_:
			send_error(ERROR_BAD_REQUEST, "unknown message type: %s" % mtype)

func _h3_handle_hello(msg: Dictionary) -> void:
	if not _h3_validate_required_fields(msg, ["type", FIELD_PROTOCOL_VERSION, "run_id"]):
		return
	_hello = msg.duplicate()
	_run_id = str(msg.get("run_id", ""))
	SightLog.log_event("controller_hello", {
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"run_id": _run_id,
	})

func _h3_handle_reset(msg: Dictionary) -> void:
	if not _h3_validate_required_fields(msg,
			["type", FIELD_PROTOCOL_VERSION, "run_id", "episode_id", "seed", "max_steps"]):
		return
	if not _pending_request.is_empty():
		send_error(ERROR_BAD_REQUEST, "pipeline overrun: previous request unhandled")
		return

	# H4 optional fields. Defaults preserve H3 state-mode behavior. Per
	# docs/sight-h4-plan.md sec 7, observation_mode is optional on the
	# reset request; pixel dims are honored only when mode is set. The
	# Python env always sends all four together when mode is set, so a
	# partial set on the wire is treated as a protocol error to keep
	# misuse loud. Validation rules:
	#   - observation_mode (if present) must be a String and one of
	#     {state, pixel, both}.
	#   - pixel_width / pixel_height / pixel_channels (if present) must
	#     be positive ints. JSON.parse_string widens integers to
	#     TYPE_FLOAT in Godot 4.6.2; accept both TYPE_INT and TYPE_FLOAT
	#     and round-trip through int() the way protocol_version does.
	#   - If observation_mode is absent, pixel dims must also be absent.
	#   - If observation_mode is present, all three pixel dims must be
	#     present.
	#   - Pixel and both modes are accepted at the protocol level but
	#     rejected here with bad_request because the pixel source is
	#     not yet implemented (H4 plan implementation step 4). Failing
	#     at reset is correct: it prevents the engine from advancing
	#     into a half-implemented mode where step responses cannot
	#     fulfill the pixel-obs contract.
	var has_mode := msg.has(FIELD_OBSERVATION_MODE)
	var has_w := msg.has(FIELD_PIXEL_WIDTH)
	var has_h := msg.has(FIELD_PIXEL_HEIGHT)
	var has_c := msg.has(FIELD_PIXEL_CHANNELS)
	var any_pixel_dim := has_w or has_h or has_c

	if not has_mode and any_pixel_dim:
		send_error(ERROR_BAD_REQUEST,
			"pixel dims provided without observation_mode; either send all four together or none")
		return

	var requested_mode := OBS_MODE_STATE
	var requested_w := DEFAULT_PIXEL_WIDTH
	var requested_h := DEFAULT_PIXEL_HEIGHT
	var requested_c := DEFAULT_PIXEL_CHANNELS

	if has_mode:
		var mode_value: Variant = msg.get(FIELD_OBSERVATION_MODE)
		if typeof(mode_value) != TYPE_STRING:
			send_error(ERROR_BAD_REQUEST,
				"observation_mode must be string, got type %d" % typeof(mode_value))
			return
		var mode_str := str(mode_value)
		if mode_str != OBS_MODE_STATE and mode_str != OBS_MODE_PIXEL and mode_str != OBS_MODE_BOTH:
			send_error(ERROR_BAD_REQUEST,
				"unknown observation_mode: %s; must be one of state, pixel, both" % mode_str)
			return
		if not (has_w and has_h and has_c):
			send_error(ERROR_BAD_REQUEST,
				"observation_mode requires all three pixel dims (pixel_width, pixel_height, pixel_channels)")
			return
		var w_ok := _h3_parse_positive_int(msg.get(FIELD_PIXEL_WIDTH))
		var h_ok := _h3_parse_positive_int(msg.get(FIELD_PIXEL_HEIGHT))
		var c_ok := _h3_parse_positive_int(msg.get(FIELD_PIXEL_CHANNELS))
		if w_ok < 0:
			send_error(ERROR_BAD_REQUEST,
				"pixel_width must be positive int, got %s" % str(msg.get(FIELD_PIXEL_WIDTH)))
			return
		if h_ok < 0:
			send_error(ERROR_BAD_REQUEST,
				"pixel_height must be positive int, got %s" % str(msg.get(FIELD_PIXEL_HEIGHT)))
			return
		if c_ok < 0:
			send_error(ERROR_BAD_REQUEST,
				"pixel_channels must be positive int, got %s" % str(msg.get(FIELD_PIXEL_CHANNELS)))
			return
		requested_mode = mode_str
		requested_w = w_ok
		requested_h = h_ok
		requested_c = c_ok

	# H4 step 3 unblocks pixel mode via the windowed Godot viewport
	# capture path (docs/sight-h4-spike.md option 2). main.gd performs
	# the actual viewport capture inside the consumed reset/step; this
	# controller's responsibility is to lock the active mode and dims
	# so the dispatcher can branch deterministically. Both mode remains
	# deferred per docs/sight-h4-plan.md section 1; channels != 1 is
	# also deferred for this slice (only L8 grayscale is implemented).
	# Loud rejection here prevents main.gd from being asked to emit a
	# half-built payload.
	if requested_mode == OBS_MODE_BOTH:
		send_error(ERROR_BAD_REQUEST,
			"observation_mode=both not yet implemented in this build; use state or pixel")
		return
	if requested_mode == OBS_MODE_PIXEL and requested_c != 1:
		send_error(ERROR_BAD_REQUEST,
			"pixel_channels=%d not supported in this build; only channels=1 (grayscale L8) is implemented" % requested_c)
		return

	# Lock active mode and dims for this episode. Mid-episode mode change
	# is structurally impossible because step requests do not carry mode
	# fields and the next mode change can only occur on the next reset.
	_h3_observation_mode = requested_mode
	_h3_pixel_width = requested_w
	_h3_pixel_height = requested_h
	_h3_pixel_channels = requested_c

	# Curriculum injection count (optional). Validated to a non-negative int and
	# stamped back onto the parked request as a clean int so main.gd reads a
	# coerced value regardless of JSON float widening. Absent -> 0 (clean start),
	# which leaves the parked request and the downstream reset path byte-identical
	# to the pre-curriculum behavior.
	var curriculum_n := 0
	if msg.has(FIELD_CURRICULUM_N_INIT):
		var cn := _h3_parse_nonneg_int(msg.get(FIELD_CURRICULUM_N_INIT))
		if cn < 0:
			send_error(ERROR_BAD_REQUEST,
				"curriculum_n_init must be a non-negative int, got %s"
					% str(msg.get(FIELD_CURRICULUM_N_INIT)))
			return
		curriculum_n = cn

	_h3_episode_id = str(msg.get("episode_id", ""))
	_pending_request = msg.duplicate()
	_pending_request[FIELD_CURRICULUM_N_INIT] = curriculum_n
	SightLog.log_event("controller_reset_received", _decorate({
		"episode_id": _h3_episode_id,
		"seed": int(msg.get("seed", 0)),
		"max_steps": int(msg.get("max_steps", 0)),
		FIELD_OBSERVATION_MODE: _h3_observation_mode,
		FIELD_PIXEL_WIDTH: _h3_pixel_width,
		FIELD_PIXEL_HEIGHT: _h3_pixel_height,
		FIELD_PIXEL_CHANNELS: _h3_pixel_channels,
		FIELD_CURRICULUM_N_INIT: curriculum_n,
	}))

# Returns the parsed positive int on success, -1 on type mismatch or
# non-positive value. Accepts TYPE_INT and TYPE_FLOAT (widened by Godot
# 4.6.2 JSON.parse_string), rejecting TYPE_BOOL and other types. Float
# values must be whole numbers; partial floats indicate a non-int wire
# payload and are rejected.
func _h3_parse_positive_int(value: Variant) -> int:
	var t := typeof(value)
	if t == TYPE_BOOL:
		return -1
	if t != TYPE_INT and t != TYPE_FLOAT:
		return -1
	if t == TYPE_FLOAT:
		var f := float(value)
		if f != floor(f):
			return -1
	var n := int(value)
	if n <= 0:
		return -1
	return n

# Returns the parsed non-negative int on success, -1 on type mismatch or a
# negative value. Mirrors _h3_parse_positive_int but admits 0 (the clean-start
# and anneal-terminal curriculum value). Accepts TYPE_INT and TYPE_FLOAT
# (widened by Godot 4.6.2 JSON.parse_string); rejects TYPE_BOOL and non-whole
# floats so a malformed curriculum count fails loud rather than truncating.
func _h3_parse_nonneg_int(value: Variant) -> int:
	var t := typeof(value)
	if t == TYPE_BOOL:
		return -1
	if t != TYPE_INT and t != TYPE_FLOAT:
		return -1
	if t == TYPE_FLOAT:
		var f := float(value)
		if f != floor(f):
			return -1
	var n := int(value)
	if n < 0:
		return -1
	return n

func _h3_handle_step(msg: Dictionary) -> void:
	if not _h3_validate_required_fields(msg,
			["type", FIELD_PROTOCOL_VERSION, "run_id", "episode_id", "seq", "action"]):
		return
	var action := int(msg.get("action", -1))
	if action != ACTION_DISCRETE_LEFT and action != ACTION_DISCRETE_STAY \
			and action != ACTION_DISCRETE_RIGHT:
		send_error(ERROR_BAD_REQUEST, "invalid discrete action: %s" % str(action))
		return
	if not _pending_request.is_empty():
		send_error(ERROR_BAD_REQUEST, "pipeline overrun: previous request unhandled")
		return
	_pending_request = msg.duplicate()

func _h3_validate_required_fields(msg: Dictionary, required: Array) -> bool:
	for f in required:
		if not msg.has(f):
			send_error(ERROR_BAD_REQUEST, "missing required field: %s" % str(f))
			return false
	return true

# --- H3 response helpers --------------------------------------------------
#
# Public API for main.gd. Caller is expected to have just consumed the matching pending
# request via take_pending_h3_request() and to have applied the action / advanced physics
# / built the observation. The controller stamps run_id, episode_id, and protocol_version
# from its own state so callers cannot accidentally desynchronize the wire-level keys.

func send_reset_ok(frame: int, obs: Array, terminated: bool, truncated: bool,
		info: Dictionary) -> void:
	var payload := {
		"type": MSG_RESET_OK,
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"run_id": _run_id,
		"episode_id": _h3_episode_id,
		"frame": frame,
		"obs": obs,
		"terminated": terminated,
		"truncated": truncated,
		"info": info,
	}
	_send_json_line(payload)

func send_step_result(seq: int, frame: int, obs: Array, reward: float,
		terminated: bool, truncated: bool, terminal_reason: String,
		info: Dictionary) -> void:
	var payload := {
		"type": MSG_STEP_RESULT,
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"run_id": _run_id,
		"episode_id": _h3_episode_id,
		"seq": seq,
		"frame": frame,
		"obs": obs,
		"reward": reward,
		"terminated": terminated,
		"truncated": truncated,
		"terminal_reason": terminal_reason,
		"info": info,
	}
	_send_json_line(payload)

# H4 step 3 send helpers. Identical envelope to the state-mode helpers
# above except the obs field carries the H4 structured pixel-obs
# Dictionary (mode/shape/dtype/encoding/data/pixel_source/capture_point/
# headless_allowed/viewport_width/viewport_height) instead of the
# length-10 numeric Array. Kept distinct so the GDScript type signatures
# stay tight and so a future regression cannot silently emit an Array
# where a Dict is required (or vice versa). Caller (main.gd) is
# responsible for awaiting RenderingServer.frame_post_draw and building
# the dict; the controller only stamps run_id / episode_id / protocol.

func send_reset_ok_pixel(frame: int, obs: Dictionary, terminated: bool,
		truncated: bool, info: Dictionary) -> void:
	var payload := {
		"type": MSG_RESET_OK,
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"run_id": _run_id,
		"episode_id": _h3_episode_id,
		"frame": frame,
		"obs": obs,
		"terminated": terminated,
		"truncated": truncated,
		"info": info,
	}
	_send_json_line(payload)

func send_step_result_pixel(seq: int, frame: int, obs: Dictionary, reward: float,
		terminated: bool, truncated: bool, terminal_reason: String,
		info: Dictionary) -> void:
	var payload := {
		"type": MSG_STEP_RESULT,
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"run_id": _run_id,
		"episode_id": _h3_episode_id,
		"seq": seq,
		"frame": frame,
		"obs": obs,
		"reward": reward,
		"terminated": terminated,
		"truncated": truncated,
		"terminal_reason": terminal_reason,
		"info": info,
	}
	_send_json_line(payload)

func send_error(code: String, message: String) -> void:
	var payload := {
		"type": MSG_ERROR,
		FIELD_PROTOCOL_VERSION: H3_PROTOCOL_VERSION,
		"code": code,
		"message": message,
	}
	_send_json_line(payload)

func _send_json_line(payload: Dictionary) -> void:
	if _peer == null:
		SightLog.log_event("controller_send_error",
			_decorate({"err": "no_peer", "type": str(payload.get("type", ""))}))
		return
	var line := JSON.stringify(payload) + "\n"
	var bytes := line.to_utf8_buffer()
	var err := _peer.put_data(bytes)
	if err != OK:
		SightLog.log_event("controller_send_error",
			_decorate({"err": err, "type": str(payload.get("type", ""))}))

# --- Legacy log_applied (unchanged) ---------------------------------------

# Main calls this after move_action to log that the legacy command was applied on this
# frame. First-applied-frame semantics: at most one controller_cmd_applied per new seq.
# H3 mode never sets _last_seq away from its sentinel, so this no-ops in H3 mode.
func log_applied(frame: int) -> void:
	if _last_seq < 0:
		return
	if _last_seq == _last_logged_seq:
		return  # already logged this seq on its first applied frame; held action carries silently
	_last_logged_seq = _last_seq
	_applied_count += 1
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
