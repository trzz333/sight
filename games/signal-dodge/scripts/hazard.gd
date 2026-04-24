extends Area2D

# Hazard - Area2D. Falls straight down at constant speed. Frees when offscreen.

const SPEED := 200.0
const SIZE := 24
const SCREEN_HEIGHT := 540

func _ready() -> void:
	add_to_group("hazard")
	queue_redraw()

func _physics_process(delta: float) -> void:
	position.y += SPEED * delta
	if position.y > SCREEN_HEIGHT + SIZE:
		queue_free()

func _draw() -> void:
	draw_rect(Rect2(-SIZE / 2.0, -SIZE / 2.0, SIZE, SIZE), Color.RED)
