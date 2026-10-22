extends RefCounted
## Camera modes, all in float64 GCRS:
##
## - ORBIT: around the spacecraft, angles measured in a reference frame:
##   "lvlh" (up = radial, side = orbit normal), "inertial" (GCRS axes) or
##   "body" (spacecraft axes, so the craft looks fixed).
## - EARTH: around Earth's centre in GCRS.
##
## Distance changes logarithmically (mouse wheel) from just outside the
## spacecraft to beyond the Moon.

const D := preload("res://src/core/dmath.gd")
const R_EARTH := 6378137.0
const MAX_DIST := 5.0e8

enum Mode { ORBIT, EARTH }

var mode := Mode.ORBIT
var frame_kind := "lvlh"
var yaw := deg_to_rad(-140.0)
var pitch := deg_to_rad(18.0)
var dist := 3.0
var earth_dist := 4.0 * R_EARTH
var min_dist := 0.1
var _dragging := false


func fit_to_craft(radius: float) -> void:
	min_dist = 1.05 * radius
	dist = maxf(4.0 * radius, min_dist)


## Reference frame for ORBIT mode as three GCRS unit vectors [e1, e2, e3=up].
func reference_frame(frame: Dictionary) -> Array:
	match frame_kind:
		"inertial":
			return [D.v3(1, 0, 0), D.v3(0, 1, 0), D.v3(0, 0, 1)]
		"body":
			var m := D.rot_mat(frame["q_gcrs_body"])
			return [PackedFloat64Array([m[0], m[3], m[6]]), PackedFloat64Array([m[1], m[4], m[7]]),
				PackedFloat64Array([m[2], m[5], m[8]])]
		_:
			var up := D.unit(frame["r_sc"])
			var normal := D.unit(D.cross(frame["r_sc"], frame["v_sc"]))
			return [D.cross(normal, up), normal, up]


## Camera position (GCRS m) and rotation (camera local -> GCRS, row-major).
func compute(frame: Dictionary) -> Array:
	var center: PackedFloat64Array
	var axes: Array
	var d: float
	if mode == Mode.EARTH:
		center = D.v3(0, 0, 0)
		axes = [D.v3(1, 0, 0), D.v3(0, 1, 0), D.v3(0, 0, 1)]
		d = earth_dist
	else:
		center = frame["r_sc"]
		axes = reference_frame(frame)
		d = dist
	var cp := cos(pitch)
	var offset := D.add(D.add(D.scale(axes[0], cp * cos(yaw)), D.scale(axes[1], cp * sin(yaw))),
		D.scale(axes[2], sin(pitch)))
	var cam := D.add(center, D.scale(offset, d))
	var up: PackedFloat64Array = axes[2] if absf(pitch) < deg_to_rad(85.0) else axes[0]
	return [cam, D.look_rot(D.scale(offset, -1.0), up)]


func zoom(steps: float) -> void:
	var f := pow(1.15, steps)
	if mode == Mode.EARTH:
		earth_dist = clampf(earth_dist * f, 1.02 * R_EARTH, MAX_DIST)
	else:
		dist = clampf(dist * f, min_dist, MAX_DIST)


## Mouse handling. Returns true if the event was used.
func handle_input(event: InputEvent) -> bool:
	if event is InputEventMouseButton:
		var mb := event as InputEventMouseButton
		if mb.button_index == MOUSE_BUTTON_WHEEL_UP and mb.pressed:
			zoom(-1.0)
			return true
		if mb.button_index == MOUSE_BUTTON_WHEEL_DOWN and mb.pressed:
			zoom(1.0)
			return true
		if mb.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT]:
			_dragging = mb.pressed
			return true
	elif event is InputEventMouseMotion and _dragging:
		var mm := event as InputEventMouseMotion
		yaw -= mm.relative.x * 0.006
		pitch = clampf(pitch + mm.relative.y * 0.006, deg_to_rad(-89.0), deg_to_rad(89.0))
		return true
	return false


func describe() -> String:
	if mode == Mode.EARTH:
		return "Earth-centred, %s from centre" % _fmt_dist(earth_dist)
	return "Spacecraft (%s), %s" % [frame_kind.to_upper() if frame_kind == "lvlh" else frame_kind, _fmt_dist(dist)]


static func _fmt_dist(m: float) -> String:
	if m < 1000.0:
		return "%.2f m" % m
	if m < 1.0e6:
		return "%.1f km" % (m / 1000.0)
	return "%.0f km" % (m / 1000.0)
