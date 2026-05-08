extends Node2D

# Sight - Main loop. Owns the deterministic per-tick pipeline:
#   world step (hazards) -> agent.capture -> agent.perceive -> agent.decide
#   -> player.move_action -> log -> spawn -> UI
# All logic runs in _physics_process at 60 Hz.
#
# TCP mode (opt-in): set env var SIGHT_TCP_MODE=1 (optionally SIGHT_TCP_PORT=<int>).
# In TCP mode the in-Godot agent pipeline is bypassed and player actions come from a
# loopback TCP client (see scripts/tcp_controller.gd and docs/sight-handoff.md).
#
# H3 mode (locked once tcp_controller sees a `protocol_version` field in any incoming
# message): pause-world cadence per docs/sight-h3-plan.md Decision 2/3 and the GPT
# step-4 directive. The H3 fast path replaces the autonomous loop entirely. Idle
# physics ticks advance nothing; one consumed `step` request advances exactly one
# Godot physics frame and produces exactly one `step_result`.

const HAZARD_SCENE := preload("res://scenes/hazard.tscn")
const TCP_CONTROLLER := preload("res://scripts/tcp_controller.gd")
const SPAWN_INTERVAL_FRAMES := 30
const SCREEN_WIDTH := 720
const SCREEN_HEIGHT := 540
const HAZARD_SIZE := 24
const RANDOM_SEED := 42

# H3 step 4 observation contract. Mirrors docs/sight-h3-plan.md section 2.
const H3_OBS_DIM := 10

var _frame_counter := 0
var _run_start_ms: int = 0
var _alive := true
var _hazards: Array = []
var _hazard_id_counter := 0
var _player_start_pos := Vector2.ZERO  # H3 step 3: deterministic player respawn position captured at scene init.
var _tcp: TCP_CONTROLLER = null  # H3 step 3: typed via preloaded TCP_CONTROLLER script const so Godot 4.6.2 type inference resolves applied_count()/run_id() and the H3 API surface (mode/has_pending_h3_request/take_pending_h3_request/send_reset_ok/send_step_result/send_error). Same nullable runtime semantics as the prior untyped Variant.
var _tcp_mode := false
var _tcp_ignore_death := false  # SIGHT_TCP_IGNORE_DEATH=1 + TCP mode: keep the run alive past collision for transport endurance tests. Default false; gameplay unchanged in non-TCP mode.
# SIGHT_P3_ACTIONS_BUDGET is read once at _ready in TCP mode. Empty/missing/non-positive
# values disable the success-budget terminal entirely. Gameplay and non-TCP behavior unchanged.
var _actions_budget: int = 0

# H3 step 4 episode-level state. All cleared by _h3_perform_soft_reset.
var _h3_max_steps: int = 0
var _h3_last_move_x: int = 0
var _h3_episode_done: bool = false
var _h3_step_terminated: bool = false
var _h3_terminal_reason: String = ""
var _h3_collision_info: Dictionary = {}

@onready var _player: Area2D = $Player
@onready var _agent: Node = $Agent
@onready var _survival_label: Label = $SurvivalLabel

func _ready() -> void:
	seed(RANDOM_SEED)
	_player_start_pos = _player.position
	_run_start_ms = Time.get_ticks_msec()
	_tcp_mode = OS.get_environment("SIGHT_TCP_MODE") == "1"
	_tcp_ignore_death = _tcp_mode and (OS.get_environment("SIGHT_TCP_IGNORE_DEATH") == "1")
	# SIGHT_P3_ACTIONS_BUDGET only consulted in TCP mode. int("") and int("abc") both
	# return 0 in GDScript; zero or negative values keep _actions_budget at 0, which the
	# physics_process check treats as "no success-budget terminal".
	if _tcp_mode:
		var parsed_budget := int(OS.get_environment("SIGHT_P3_ACTIONS_BUDGET"))
		if parsed_budget > 0:
			_actions_budget = parsed_budget
	var mode := "tcp" if _tcp_mode else "in_godot"
	var run_meta := {
		"seed": RANDOM_SEED,
		"mode": mode,
		"screen_width": SCREEN_WIDTH,
		"screen_height": SCREEN_HEIGHT,
		"spawn_interval_frames": SPAWN_INTERVAL_FRAMES,
		"hazard_size": HAZARD_SIZE,
		"physics_hz": 60,
	}
	if _tcp_mode:
		run_meta["tcp_ignore_death"] = _tcp_ignore_death
		if _actions_budget > 0:
			run_meta["actions_budget"] = _actions_budget
	SightLog.start_run(run_meta)
	_player.died.connect(_on_player_died)
	if _tcp_mode:
		_tcp = TCP_CONTROLLER.new()
		add_child(_tcp)
		var port_env := OS.get_environment("SIGHT_TCP_PORT")
		var port := 8765 if port_env == "" else int(port_env)
		var err: int = _tcp.start("127.0.0.1", port)
		if err != OK:
			push_error("tcp controller failed to start; falling back to in-Godot agent")
			_tcp_mode = false

func _physics_process(delta: float) -> void:
	# H3 mode (locked) takes the pause-world fast path; bypass the legacy autonomous
	# loop and the _alive gate. _h3_physics_tick polls the socket every tick and only
	# advances state when it consumes a valid reset/step request.
	if _tcp_mode and _tcp != null and _tcp.mode() == TCP_CONTROLLER.MODE_H3:
		_h3_physics_tick(delta)
		return

	if not _alive:
		return
	_frame_counter += 1

	# 1. World step: advance hazards, cull offscreen.
	var survivors: Array = []
	for h in _hazards:
		if not is_instance_valid(h):
			continue
		if h.step(delta):
			h.queue_free()
		else:
			survivors.append(h)
	_hazards = survivors

	# 2. Resolve action: TCP mode polls the loopback socket, otherwise run the in-Godot pipeline.
	var action := 0
	var state := {}
	var perceived := {"threat": false}
	if _tcp_mode:
		action = _tcp.poll(_frame_counter)
		# H3 mode-locking transition tick. poll() may have just locked H3 mode and
		# parked a reset/step. Reset gets dispatched here because its state-clobber
		# semantics wipe the legacy world advance that already happened above. Step
		# before reset is a Python protocol error (step requires an episode_id
		# established by a prior reset); reply with bad_request so Python can
		# self-correct. Subsequent ticks take the H3 fast path at the top of this
		# function.
		if _tcp.mode() == TCP_CONTROLLER.MODE_H3 and _tcp.has_pending_h3_request():
			var req: Dictionary = _tcp.take_pending_h3_request()
			var rtype := str(req.get("type", ""))
			match rtype:
				"reset":
					_h3_perform_soft_reset(req)
				"step":
					_tcp.send_error(TCP_CONTROLLER.ERROR_BAD_REQUEST,
						"step received before reset on the H3 mode-locking tick")
				_:
					push_warning("h3: unexpected pending request type: %s" % rtype)
			return
	else:
		state = _agent.capture(_player, _hazards)
		perceived = _agent.perceive(state)
		action = _agent.decide(perceived, state["player_x"], float(SCREEN_WIDTH))

	# 3. Controller: apply action.
	_player.move_action(action, delta)
	if _tcp_mode:
		_tcp.log_applied(_frame_counter)
		# Success-budget terminal. Distinct from _on_player_died: runs only while alive and
		# only when the configured number of distinct seqs have been first-applied. Logs the
		# terminal event, ends the run cleanly, stops TCP, and quits. SIGHT_TCP_IGNORE_DEATH
		# is not consulted here; this branch is gated by _alive (which the ignore-death path
		# keeps true after collision) and the budget alone.
		if _actions_budget > 0 and _alive:
			var ac := _tcp.applied_count()
			if ac >= _actions_budget:
				_alive = false
				var evt := {
					"frame": _frame_counter,
					"applied_count": ac,
					"actions_budget": _actions_budget,
				}
				var rid := _tcp.run_id()
				if rid != "":
					evt["run_id"] = rid
				var ep := OS.get_environment("SIGHT_EPISODE_ID")
				if ep != "":
					evt["episode_id"] = ep
				SightLog.log_event("success_budget_reached", evt)
				SightLog.end_run()
				_tcp.stop()
				_survival_label.text = "BUDGET MET  applied=%d" % ac
				get_tree().quit()
				return

	# 4. Log player/agent tick.
	var rec := {
		"frame": _frame_counter,
		"player_x": _player.position.x,
		"player_y": _player.position.y,
		"action": action,
	}
	if _tcp_mode:
		SightLog.log_event("player_tick", rec)
	else:
		rec["threat"] = perceived.get("threat", false)
		if perceived.get("threat", false):
			rec["threat_x"] = perceived["x"]
			rec["threat_y"] = perceived["y"]
			rec["threat_dist"] = perceived["dist"]
		SightLog.log_event("agent_tick", rec)

	# 5. Spawn.
	if _frame_counter % SPAWN_INTERVAL_FRAMES == 0:
		_spawn_hazard()

	# 6. UI.
	var t := (Time.get_ticks_msec() - _run_start_ms) / 1000.0
	_survival_label.text = "Survival: %.2fs  action=%d" % [t, action]

func _spawn_hazard() -> void:
	_hazard_id_counter += 1
	var hz := HAZARD_SCENE.instantiate()
	var x := randf_range(HAZARD_SIZE / 2.0, SCREEN_WIDTH - HAZARD_SIZE / 2.0)
	var y := float(-HAZARD_SIZE)
	hz.position = Vector2(x, y)
	add_child(hz)
	# H3 step 4: stamp deterministic spawn id on each hazard for the threat-priority
	# sort tie-break (_h3_sort_hazards_by_threat).
	hz.set_meta("h3_spawn_id", _hazard_id_counter)
	_hazards.append(hz)
	SightLog.log_event("spawn", {
		"hazard_id": _hazard_id_counter,
		"frame": _frame_counter,
		"x": x,
		"y": y,
	})

func _on_player_died(survival_time: float, hazard_pos: Vector2, player_pos: Vector2) -> void:
	# H3 mode (locked): record collision state for the in-flight step and let
	# _h3_perform_step send a terminated step_result. Do NOT stop TCP, do NOT call
	# get_tree().quit(); reset must be able to re-arm the episode without relaunching
	# Godot. The collision/death NDJSON events still fire so logs reflect the terminal.
	if _tcp_mode and _tcp != null and _tcp.mode() == TCP_CONTROLLER.MODE_H3:
		_h3_step_terminated = true
		_h3_terminal_reason = "collision"
		_h3_collision_info = {
			"frame": _frame_counter,
			"player_x": player_pos.x,
			"player_y": player_pos.y,
			"hazard_x": hazard_pos.x,
			"hazard_y": hazard_pos.y,
			"survival_time": survival_time,
		}
		SightLog.log_event("collision", {
			"frame": _frame_counter,
			"player_x": player_pos.x,
			"player_y": player_pos.y,
			"hazard_x": hazard_pos.x,
			"hazard_y": hazard_pos.y,
		})
		SightLog.log_event("death", {
			"survival_time": survival_time,
			"frame": _frame_counter,
		})
		return

	# TCP test mode: when SIGHT_TCP_MODE=1 and SIGHT_TCP_IGNORE_DEATH=1, log a non-terminal
	# event and keep the run alive so the harness can verify TCP transport endurance over
	# the full ACTIONS budget. Default and non-TCP gameplay behavior is unchanged. The flag
	# is opt-in and only consulted when _tcp_mode is true.
	if _tcp_mode and _tcp_ignore_death:
		SightLog.log_event("tcp_death_ignored", {
			"survival_time": survival_time,
			"player_x": player_pos.x,
			"player_y": player_pos.y,
			"hazard_x": hazard_pos.x,
			"hazard_y": hazard_pos.y,
			"frame": _frame_counter,
		})
		return
	_alive = false
	SightLog.log_event("collision", {
		"player_x": player_pos.x,
		"player_y": player_pos.y,
		"hazard_x": hazard_pos.x,
		"hazard_y": hazard_pos.y,
	})
	SightLog.log_event("death", {"survival_time": survival_time})
	SightLog.end_run()
	if _tcp != null:
		_tcp.stop()
	_survival_label.text = "DEAD  t=%.2fs" % survival_time
	get_tree().quit()

# --- H3 mode pause-world cadence ---------------------------------------------
#
# Step 4 boundary. _h3_physics_tick is the H3-locked replacement for the autonomous
# loop in _physics_process. Per docs/sight-h3-plan.md Decision 3 and the step-4 GPT
# directive: the engine still calls _physics_process at 60 Hz, but in H3 mode no
# frame-counter increment, no hazard movement, no spawn, no player movement, no
# reward, and no player_tick log occur on idle ticks. Exactly one consumed H3 step
# request advances exactly one physics frame and produces one step_result.

func _h3_physics_tick(delta: float) -> void:
	# Always poll the socket so we accept new connections and drain bytes even when
	# paused. _tcp.poll() also handles peer-disconnect logging. The _frame_counter
	# argument is for legacy log_applied bookkeeping; H3 mode does not use it.
	_tcp.poll(_frame_counter)
	if not _tcp.has_pending_h3_request():
		return  # idle tick: no world advance.
	var req: Dictionary = _tcp.take_pending_h3_request()
	var rtype := str(req.get("type", ""))
	match rtype:
		"reset":
			_h3_perform_soft_reset(req)
		"step":
			_h3_perform_step(req, delta)
		_:
			push_warning("h3: unexpected pending request type: %s" % rtype)

func _h3_map_action(action_wire: int) -> int:
	# Wire-protocol discrete action -> player.move_action argument.
	# 0 -> -1 (left), 1 -> 0 (stay), 2 -> +1 (right). tcp_controller validates the wire
	# value before parking the request, so this re-clamp is a defensive belt-and-
	# suspenders against future controller drift.
	match action_wire:
		0:
			return -1
		2:
			return 1
		_:
			return 0  # stay covers wire=1 plus any unexpected value.

func _h3_perform_soft_reset(req: Dictionary) -> void:
	# Reset ordering follows docs/sight-h3-plan.md Decision 2 plus the GPT step-4
	# directive: clear hazards, reset frame/death/spawn counters, reset run timing,
	# reposition player to the deterministic start, reseed the global RNG from the
	# request seed, clear all H3 episode-level state, log episode_start, build the
	# real observation, and send reset_ok. The caller in _physics_process /
	# _h3_physics_tick returns immediately after this so no normal game step runs on
	# the reset frame. tcp_controller has already validated required fields and
	# stamped the active episode_id; the send helper stamps run_id / episode_id /
	# protocol so wire keys cannot drift here.
	for h in _hazards:
		if is_instance_valid(h):
			h.queue_free()
	_hazards.clear()
	_frame_counter = 0
	_alive = true
	_hazard_id_counter = 0
	_run_start_ms = Time.get_ticks_msec()
	_player.position = _player_start_pos
	# H3 episode-level state.
	_h3_episode_done = false
	_h3_step_terminated = false
	_h3_terminal_reason = ""
	_h3_collision_info = {}
	_h3_last_move_x = 0
	var seed_value: int = int(req.get("seed", RANDOM_SEED))
	seed(seed_value)
	_h3_max_steps = int(req.get("max_steps", 0))
	var episode_id: String = str(req.get("episode_id", ""))
	SightLog.log_event("episode_start", {
		"episode_id": episode_id,
		"seed": seed_value,
		"max_steps": _h3_max_steps,
		"frame": _frame_counter,
	})
	var info := {
		"seed": seed_value,
		"max_steps": _h3_max_steps,
		"frame": _frame_counter,
	}
	_tcp.send_reset_ok(_frame_counter, _h3_build_observation(), false, false, info)
	_survival_label.text = "RESET seed=%d ep=%s" % [seed_value, episode_id]

func _h3_perform_step(req: Dictionary, delta: float) -> void:
	# One consumed H3 step request advances exactly one physics frame and produces
	# exactly one step_result. Ordering matches the step-4 directive section 5:
	# increment frame, advance hazards, apply player action, observe collision flag
	# set by _on_player_died, spawn (only if not already terminated), check timeout,
	# build real observation, send step_result. _h3_episode_done gates further steps
	# until a new reset arrives.
	var seq: int = int(req.get("seq", -1))
	var action_wire: int = int(req.get("action", 1))  # default stay; real validation lives in tcp_controller.
	if _h3_episode_done:
		# Python misuse: step on a terminal episode. Reset is required before further
		# stepping. Reply bad_request so the Python harness raises rather than silently
		# accumulating reward on a corpse.
		_tcp.send_error(TCP_CONTROLLER.ERROR_BAD_REQUEST,
			"step received on a done episode; reset required before continuing")
		return
	var mapped: int = _h3_map_action(action_wire)
	_h3_last_move_x = mapped
	# Reset per-step terminal flags. _on_player_died (in the H3 branch above) sets
	# these synchronously inside the move_action() call below if a collision occurs.
	_h3_step_terminated = false
	_h3_terminal_reason = ""
	_h3_collision_info = {}
	# 1. Advance frame counter.
	_frame_counter += 1
	# 2. World step: advance hazards, cull offscreen.
	var survivors: Array = []
	for h in _hazards:
		if not is_instance_valid(h):
			continue
		if h.step(delta):
			h.queue_free()
		else:
			survivors.append(h)
	_hazards = survivors
	# 3. Apply player action. _on_player_died fires synchronously on collision.
	_player.move_action(mapped, delta)
	# 4. Read terminal state set by _on_player_died (if fired).
	var terminated: bool = _h3_step_terminated
	var truncated: bool = false
	var terminal_reason: String = _h3_terminal_reason
	# 5. Spawn check, but only if this step did not already terminate by collision.
	if not terminated and _frame_counter % SPAWN_INTERVAL_FRAMES == 0:
		_spawn_hazard()
	# 6. Timeout check. Only fires if not already collision-terminated.
	if not terminated and _h3_max_steps > 0 and _frame_counter >= _h3_max_steps:
		truncated = true
		terminal_reason = "timeout"
	if terminated or truncated:
		_h3_episode_done = true
	# 7. Reward: sparse survival per plan section 4.
	var reward: float = 0.0 if terminated else 1.0
	# 8. Build the real observation AFTER the state update.
	var obs: Array = _h3_build_observation()
	# 9. Local NDJSON evidence event. TCP response remains the source of Gym semantics.
	SightLog.log_event("h3_step", {
		"frame": _frame_counter,
		"seq": seq,
		"action_wire": action_wire,
		"action": mapped,
		"player_x": _player.position.x,
		"player_y": _player.position.y,
		"reward": reward,
		"terminated": terminated,
		"truncated": truncated,
		"terminal_reason": terminal_reason,
	})
	# 10. UI.
	var status_suffix := ""
	if terminated:
		status_suffix = " TERM"
	elif truncated:
		status_suffix = " TRUNC"
	_survival_label.text = "H3 step %d  action=%d  r=%.0f%s" % [
		_frame_counter, mapped, reward, status_suffix,
	]
	# 11. Send step_result. tcp_controller stamps run_id/episode_id/protocol_version.
	var info := {
		"frame": _frame_counter,
		"action": mapped,
	}
	if terminated and not _h3_collision_info.is_empty():
		info["collision"] = _h3_collision_info.duplicate()
	_tcp.send_step_result(seq, _frame_counter, obs, reward, terminated, truncated,
		terminal_reason, info)

# --- H3 observation builder --------------------------------------------------
#
# Plan section 2: 10-element float Array. All values clamped to [-1, 1].
#   0 player_x_norm
#   1 player_vx_norm  (-1, 0, +1 from last applied wire-mapped action)
#   2 nearest_hazard_dx_norm
#   3 nearest_hazard_dy_norm
#   4 nearest_hazard_present
#   5 second_hazard_dx_norm
#   6 second_hazard_dy_norm
#   7 second_hazard_present
#   8 third_hazard_dx_norm
#   9 third_hazard_dy_norm
# Note: third hazard has no explicit present flag per spec; absence is signaled by
# both dx and dy being 0.0.

func _h3_build_observation() -> Array:
	var obs: Array = []
	obs.resize(H3_OBS_DIM)
	for i in range(H3_OBS_DIM):
		obs[i] = 0.0
	var px: float = _player.position.x
	var py: float = _player.position.y
	obs[0] = clampf((px / float(SCREEN_WIDTH)) * 2.0 - 1.0, -1.0, 1.0)
	obs[1] = clampf(float(_h3_last_move_x), -1.0, 1.0)
	var sorted_hazards: Array = _h3_sort_hazards_by_threat()
	if sorted_hazards.size() >= 1:
		var h0: Node2D = sorted_hazards[0]
		obs[2] = clampf((h0.position.x - px) / float(SCREEN_WIDTH), -1.0, 1.0)
		obs[3] = clampf((h0.position.y - py) / float(SCREEN_HEIGHT), -1.0, 1.0)
		obs[4] = 1.0
	if sorted_hazards.size() >= 2:
		var h1: Node2D = sorted_hazards[1]
		obs[5] = clampf((h1.position.x - px) / float(SCREEN_WIDTH), -1.0, 1.0)
		obs[6] = clampf((h1.position.y - py) / float(SCREEN_HEIGHT), -1.0, 1.0)
		obs[7] = 1.0
	if sorted_hazards.size() >= 3:
		var h2: Node2D = sorted_hazards[2]
		obs[8] = clampf((h2.position.x - px) / float(SCREEN_WIDTH), -1.0, 1.0)
		obs[9] = clampf((h2.position.y - py) / float(SCREEN_HEIGHT), -1.0, 1.0)
	return obs

func _h3_sort_hazards_by_threat() -> Array:
	# Filter to hazards at or above the player (still threats), then sort by:
	#   primary  : smallest positive (player.y - hazard.y), i.e. closest above player
	#   secondary: smallest abs(hazard.x - player.x)
	#   tertiary : smallest h3_spawn_id (stable, deterministic across runs because
	#              _hazard_id_counter resets to 0 on each soft reset and increments
	#              once per spawn)
	# Hazards strictly below the player are filtered out; they have already passed and
	# are no longer threats per docs/sight-h3-plan.md section 2.
	var px: float = _player.position.x
	var py: float = _player.position.y
	var candidates: Array = []
	for h in _hazards:
		if not is_instance_valid(h):
			continue
		if h.position.y <= py:
			candidates.append(h)
	candidates.sort_custom(func(a: Node2D, b: Node2D) -> bool:
		var dy_a: float = py - a.position.y
		var dy_b: float = py - b.position.y
		if dy_a != dy_b:
			return dy_a < dy_b
		var adx_a: float = absf(a.position.x - px)
		var adx_b: float = absf(b.position.x - px)
		if adx_a != adx_b:
			return adx_a < adx_b
		var sid_a: int = int(a.get_meta("h3_spawn_id", 0))
		var sid_b: int = int(b.get_meta("h3_spawn_id", 0))
		return sid_a < sid_b
	)
	return candidates
