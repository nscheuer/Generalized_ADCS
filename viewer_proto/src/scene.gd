extends RefCounted
## Prototype scene. All nodes live in one SubViewport whose camera always sits
## at the Godot origin: every frame, object positions are set to
## (object - camera) computed in float64, then converted to float32.
## `absolute = true` instead feeds raw ECI metres to float32 (the control case).

const D := preload("res://src/dmath.gd")
const R_EARTH := 6378137.0
const R_MOON := 1737400.0

var viewport: SubViewport
var camera: Camera3D
var env: Environment
var earth: MeshInstance3D
var earth_mat: ShaderMaterial
var moon: MeshInstance3D
var moon_mat: ShaderMaterial
var craft: Node3D
var craft_mat: ShaderMaterial
var fiducial: MeshInstance3D
var stars: MeshInstance3D
var model: MeshInstance3D  # optional heavy mesh for the performance test
var absolute := false


func _init(parent: Node, size: Vector2i) -> void:
	viewport = SubViewport.new()
	viewport.size = size
	viewport.msaa_3d = Viewport.MSAA_4X
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	parent.add_child(viewport)

	var view := TextureRect.new()
	view.texture = viewport.get_texture()
	view.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	view.set_anchors_preset(Control.PRESET_FULL_RECT)
	parent.add_child(view)

	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color.BLACK
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR

	camera = Camera3D.new()
	# update() sets near/far every frame; this initial range stays inside the
	# measured single-frustum limit (far/near <= ~1e7).
	camera.near = 0.01
	camera.far = 1.0e4
	camera.fov = 40.0
	camera.environment = env
	viewport.add_child(camera)
	camera.current = true

	earth_mat = _material("res://shaders/earth.gdshader")
	earth = _sphere(R_EARTH, 256, 128, earth_mat)
	moon_mat = _material("res://shaders/lambert.gdshader")
	moon_mat.set_shader_parameter("albedo", Vector3(0.12, 0.12, 0.12))
	moon = _sphere(R_MOON, 96, 48, moon_mat)

	craft = Node3D.new()
	viewport.add_child(craft)
	craft_mat = _material("res://shaders/lambert.gdshader")
	var box := MeshInstance3D.new()
	var box_mesh := BoxMesh.new()
	box_mesh.size = Vector3.ONE
	box.mesh = box_mesh
	box.material_override = craft_mat
	craft.add_child(box)
	# 10 cm fiducial on the body +X face (Godot local +X = body +X).
	fiducial = MeshInstance3D.new()
	var quad := QuadMesh.new()
	quad.size = Vector2(0.1, 0.1)
	quad.orientation = PlaneMesh.FACE_X
	fiducial.mesh = quad
	var fid_mat := _material("res://shaders/flat.gdshader")
	fid_mat.set_shader_parameter("color", Vector3(0.0, 1.0, 0.0))
	fiducial.material_override = fid_mat
	fiducial.position = Vector3(0.5005, 0.0, 0.0)
	craft.add_child(fiducial)

	stars = _stars(9000, 1.0e10)
	viewport.add_child(stars)


func _material(path: String) -> ShaderMaterial:
	var m := ShaderMaterial.new()
	m.shader = load(path)
	return m


func _sphere(radius: float, segments: int, rings: int, mat: ShaderMaterial) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var s := SphereMesh.new()
	s.radius = radius
	s.height = 2.0 * radius
	s.radial_segments = segments
	s.rings = rings
	mi.mesh = s
	mi.material_override = mat
	viewport.add_child(mi)
	return mi


func _stars(count: int, radius: float) -> MeshInstance3D:
	var rng := RandomNumberGenerator.new()
	rng.seed = 12345
	var verts := PackedVector3Array()
	var colors := PackedColorArray()
	for i in count:
		var v := Vector3(rng.randfn(), rng.randfn(), rng.randfn()).normalized() * radius
		verts.append(v)
		var b := pow(10.0, -0.4 * rng.randf_range(-1.0, 6.5)) * 4.0
		colors.append(Color(b, b, b))
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = _material("res://shaders/stars.gdshader")
	mi.custom_aabb = AABB(Vector3.ONE * -2.0 * radius, Vector3.ONE * 4.0 * radius)
	return mi


func add_model(triangles_target: int) -> void:
	# Stand-in for an imported spacecraft model: a dense sphere on the body.
	var rings := int(sqrt(triangles_target / 2.0))
	model = MeshInstance3D.new()
	var s := SphereMesh.new()
	s.radius = 0.35
	s.height = 0.7
	s.radial_segments = rings * 2
	s.rings = rings / 2
	model.mesh = s
	model.material_override = craft_mat
	model.position = Vector3(0.0, 0.0, 0.9)
	craft.add_child(model)


## Depth range of one frustum. Measured on Godot 4.7: geometry disappears once
## far/near exceeds ~1e7 (float32 projection), so stay an order of magnitude below.
const FRUSTUM_RATIO := 1.0e6
const CRAFT_RADIUS := 1.3
## Far bodies are drawn at a log-compressed distance in [0.2, 0.9] x far.
const FAR_BODY_MAX_DISTANCE := 1.0e10

var last_near := 0.0
var last_far := 0.0
var last_scaled := {}


## Place `node` (pose `rot`, local -> GCRS) at GCRS position `p`, seen from `cam`.
## Near objects are placed exactly; far ones are scaled about the camera by k
## (same image, compressed depth).
func _place(node: Node3D, p: PackedFloat64Array, rot: PackedFloat64Array, cam: PackedFloat64Array,
		k: float) -> void:
	var rel := D.sub(p, cam)
	var b := D.basis_from(rot).scaled(Vector3.ONE * k)
	if absolute:
		node.transform = Transform3D(b, D.to_godot(cam) + D.to_godot(D.scale(rel, k)))
	else:
		node.transform = Transform3D(b, D.to_godot(D.scale(rel, k)))


## Distance compression for far bodies: monotonic, so their mutual depth order
## (and their order behind every near object) is preserved.
func _far_scale(d: float, far: float) -> float:
	var lo := 0.1 * far
	var t := clampf(log(maxf(d, lo) / lo) / log(FAR_BODY_MAX_DISTANCE / lo), 0.0, 1.0)
	return far * (0.2 + 0.7 * t) / d


## Pose everything for one frame.
## frame: interpolated sample (metres). cam_pos: GCRS metres. cam_rot: camera local -> GCRS.
func update(frame: Dictionary, cam_pos: PackedFloat64Array, cam_rot: PackedFloat64Array) -> void:
	var origin := D.v3(0.0, 0.0, 0.0)
	var ident := PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
	var r_itrs_gcrs := D.rot_mat(frame["q_itrs_gcrs"])  # v_itrs = R v_gcrs
	var sun_dir := D.unit(D.sub(frame["r_sun"], frame["r_sc"]))

	camera.basis = D.basis_from(cam_rot)
	camera.position = D.to_godot(cam_pos) if absolute else Vector3.ZERO

	# The spacecraft is the only object drawn at true scale; it sets the frustum.
	var d_craft := D.norm(D.sub(frame["r_sc"], cam_pos))
	var craft_near := craft.visible and d_craft - CRAFT_RADIUS < 1.0e4
	var near := clampf(0.2 * (d_craft - CRAFT_RADIUS), 0.01, 100.0) if craft_near else 100.0
	var far := near * FRUSTUM_RATIO
	camera.near = near
	camera.far = far
	last_near = near
	last_far = far

	var k_craft := 1.0 if craft_near else _far_scale(d_craft, far)
	_place(craft, frame["r_sc"], D.rot_mat(frame["q_gcrs_body"]), cam_pos, k_craft)
	craft_mat.set_shader_parameter("sun_dir_world", D.to_godot(sun_dir))
	craft_mat.set_shader_parameter("sun_scale", frame["shadow_factor"])

	var k_earth := _far_scale(D.norm(cam_pos), far)
	_place(earth, origin, D.transpose(r_itrs_gcrs), cam_pos, k_earth)  # local ITRS -> GCRS
	# Shader works in raw ITRS coordinates (the mesh's local space), not Godot axes.
	earth_mat.set_shader_parameter("sun_dir_itrs", _vec3(D.unit(D.mat_vec(r_itrs_gcrs, frame["r_sun"]))))

	var k_moon := 0.0
	if frame.has("r_moon"):
		k_moon = _far_scale(D.norm(D.sub(frame["r_moon"], cam_pos)), far)
		_place(moon, frame["r_moon"], ident, cam_pos, k_moon)
		moon_mat.set_shader_parameter("sun_dir_world", D.to_godot(D.unit(D.sub(frame["r_sun"], frame["r_moon"]))))

	# Stars: infinitely far, just inside the far plane, never translating.
	var k_stars := 0.95 * far / 1.0e10
	stars.transform = Transform3D(D.basis_from(ident).scaled(Vector3.ONE * k_stars), camera.position)
	last_scaled = {"craft": k_craft, "earth": k_earth, "moon": k_moon}


## Raw (unmapped) float64 triple as Vector3: for vectors used in local ITRS shader space.
static func _vec3(v: PackedFloat64Array) -> Vector3:
	return Vector3(v[0], v[1], v[2])
