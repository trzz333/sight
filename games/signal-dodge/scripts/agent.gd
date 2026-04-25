extends Node

# Sight - minimal deterministic rule agent.
# Pipeline: capture -> perceive -> decide
# All three are pure functions of their inputs. No hidden state beyond constants.

# Alignment threshold = player_half (16) + hazard_half (12) = 28.
# Any hazard whose x is within this distance of player.x will hit if it keeps falling.
const ALIGN_THRESHOLD := 28.0

# capture: read current world state into a plain dict (no node references).
func capture(player: Node2D, hazards: Array) -> Dictionary:
	var hz: Array = []
	for h in hazards:
		if is_instance_valid(h):
			hz.append({"x": h.position.x, "y": h.position.y})
	return {
		"player_x": player.position.x,
		"player_y": player.position.y,
		"hazards": hz,
	}

# perceive: find nearest threat (aligned + above player). Returns threat dict or {"threat": false}.
func perceive(state: Dictionary) -> Dictionary:
	var px: float = state["player_x"]
	var py: float = state["player_y"]
	var best_dist: float = INF
	var best_x: float = 0.0
	var best_y: float = 0.0
	var found := false
	for h in state["hazards"]:
		if h["y"] > py:
			continue  # already past player
		if abs(h["x"] - px) > ALIGN_THRESHOLD:
			continue  # not in player's column
		var d: float = py - h["y"]
		if d < best_dist:
			best_dist = d
			best_x = h["x"]
			best_y = h["y"]
			found = true
	if found:
		return {"threat": true, "x": best_x, "y": best_y, "dist": best_dist}
	return {"threat": false}

# decide: action in {-1, 0, +1}. Dodge away from aligned threat. Stay if safe.
func decide(perceived: Dictionary, player_x: float, screen_width: float) -> int:
	if not perceived.get("threat", false):
		return 0
	var hx: float = perceived["x"]
	if hx < player_x:
		return 1
	elif hx > player_x:
		return -1
	else:
		# Centered on hazard. Dodge toward the side with more room.
		if player_x < screen_width / 2.0:
			return 1
		else:
			return -1
