extends RefCounted
## Camera-relative placement and scaled space (see tools/viz_validation/PHASE0.md).
##
## Godot builds projections in float32: one frustum stops working beyond
## far/near ~ 1e7. So the frustum follows the nearest true-scale object
## (far = near x FRUSTUM_RATIO) and distant bodies are drawn scaled about the
## camera at a log-compressed distance. Scaling about the camera leaves their
## image unchanged; the compression is monotonic, so depth order is kept.
##
## All positions are float64 (PackedFloat64Array, GCRS metres) until the camera
## has been subtracted.

const D := preload("res://src/core/dmath.gd")

const FRUSTUM_RATIO := 1.0e6
const NEAR_MIN := 0.01
const NEAR_MAX := 100.0
## Beyond this distance (m) even the spacecraft is drawn in scaled space.
const TRUE_SCALE_LIMIT := 1.0e4
const FAR_BODY_MAX_DISTANCE := 1.0e10

var near := NEAR_MIN
var far := NEAR_MIN * FRUSTUM_RATIO
var cam_pos := PackedFloat64Array([0.0, 0.0, 0.0])
## Debug/control: feed absolute float32 coordinates (reproduces jitter).
var absolute := false


## Set the frustum for this frame from the distance and radius of the one
## object drawn at true scale (the spacecraft). Returns true if it is near.
func begin_frame(camera_pos: PackedFloat64Array, near_obj_dist: float, near_obj_radius: float) -> bool:
	cam_pos = camera_pos
	var surface := near_obj_dist - near_obj_radius
	var is_near := surface < TRUE_SCALE_LIMIT
	near = clampf(0.2 * surface, NEAR_MIN, NEAR_MAX) if is_near else NEAR_MAX
	far = near * FRUSTUM_RATIO
	return is_near


## Scale for a far object at distance d: its drawn distance is compressed into
## [0.2, 0.9] x far, strictly increasing with d from the near plane outwards, so
## no two far objects at different distances share a depth.
func far_scale(d: float) -> float:
	var t := clampf(log(maxf(d, near) / near) / log(FAR_BODY_MAX_DISTANCE / near), 0.0, 1.0)
	return far * (0.2 + 0.7 * t) / d


## Place `node` (local axes -> GCRS by `rot`, row-major) at GCRS position `p`,
## scaled about the camera by k.
func place(node: Node3D, p: PackedFloat64Array, rot: PackedFloat64Array, k: float) -> void:
	var rel := D.sub(p, cam_pos)
	var b := D.basis_from(rot).scaled(Vector3.ONE * k)
	if absolute:
		node.transform = Transform3D(b, D.to_godot(cam_pos) + D.to_godot(D.scale(rel, k)))
	else:
		node.transform = Transform3D(b, D.to_godot(D.scale(rel, k)))


## Camera node transform for orientation `rot` (camera local -> GCRS).
func place_camera(camera: Camera3D, rot: PackedFloat64Array) -> void:
	camera.basis = D.basis_from(rot)
	camera.position = D.to_godot(cam_pos) if absolute else Vector3.ZERO
	camera.near = near
	camera.far = far
