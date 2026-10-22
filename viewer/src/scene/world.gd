extends RefCounted
## Everything that is drawn: Earth, Moon, Sun, stars and the spacecraft, in one
## SubViewport whose camera sits at the Godot origin (see core/origin.gd).

const D := preload("res://src/core/dmath.gd")
const Origin := preload("res://src/core/origin.gd")
const Assets := preload("res://src/scene/assets.gd")
const Spacecraft := preload("res://src/scene/spacecraft.gd")

const R_EARTH := 6378137.0
const R_MOON := 1737400.0
const SUN_ANGULAR_RADIUS := 0.004652  # rad, mean
const OBLIQUITY := 0.40909  # rad, J2000; Moon frame approximation

var viewport: SubViewport
var view_rect: TextureRect
var camera: Camera3D
var env: Environment
var light: DirectionalLight3D
var origin := Origin.new()
var earth: MeshInstance3D
var earth_mat: ShaderMaterial
var moon: MeshInstance3D
var moon_mat: ShaderMaterial
var stars: MeshInstance3D
var sun: MeshInstance3D
var craft: Spacecraft
var fixed_size := Vector2i.ZERO
var info: Dictionary = {}  ## per-frame diagnostics (near/far, scales, LOD)
## Checks that measure geometry only switch off time-varying light and animation.
var dynamic_lighting := true


func _init(parent: Node, size: Vector2i = Vector2i.ZERO) -> void:
	fixed_size = size
	viewport = SubViewport.new()
	viewport.size = size if size != Vector2i.ZERO else parent.get_viewport().get_visible_rect().size
	viewport.msaa_3d = Viewport.MSAA_4X
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	parent.add_child(viewport)
	view_rect = TextureRect.new()
	view_rect.texture = viewport.get_texture()
	view_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	view_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	view_rect.set_anchors_preset(Control.PRESET_FULL_RECT)
	view_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(view_rect)

	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color.BLACK
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.glow_enabled = true
	env.glow_hdr_threshold = 1.5
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.03, 0.035, 0.045)

	camera = Camera3D.new()
	camera.fov = 45.0
	camera.environment = env
	viewport.add_child(camera)
	camera.current = true

	light = DirectionalLight3D.new()
	light.shadow_enabled = true
	light.light_energy = 1.0
	viewport.add_child(light)

	earth_mat = _material("res://shaders/earth.gdshader")
	earth = _sphere(R_EARTH, 384, 192, earth_mat)
	moon_mat = _material("res://shaders/moon.gdshader")
	moon = _sphere(R_MOON, 128, 64, moon_mat)
	sun = MeshInstance3D.new()
	sun.mesh = QuadMesh.new()
	sun.material_override = _material("res://shaders/sun.gdshader")
	sun.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	viewport.add_child(sun)
	craft = Spacecraft.new()
	viewport.add_child(craft)


func resize_to(size: Vector2i) -> void:
	if fixed_size == Vector2i.ZERO and size.x > 0 and size.y > 0:
		viewport.size = size


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
	mi.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	viewport.add_child(mi)
	return mi


## Textures and stars from the asset cache; anything missing falls back to flat colours.
func load_assets(assets: Assets) -> Dictionary:
	var loaded := {}
	for pair: Array in [["earth_day", "day_tex", "has_day"], ["earth_night", "night_tex", "has_night"],
			["earth_clouds", "cloud_tex", "has_clouds"], ["earth_normals", "normal_tex", "has_normals"],
			["earth_ocean_mask", "ocean_tex", "has_ocean"]]:
		var cube := assets.cubemap(pair[0])
		if cube != null:
			earth_mat.set_shader_parameter(pair[1], cube)
			earth_mat.set_shader_parameter(pair[2], true)
			loaded[pair[0]] = true
	for pair: Array in [["moon_color", "color_tex", "has_color"], ["moon_normals", "normal_tex", "has_normals"]]:
		var cube := assets.cubemap(pair[0])
		if cube != null:
			moon_mat.set_shader_parameter(pair[1], cube)
			moon_mat.set_shader_parameter(pair[2], true)
			loaded[pair[0]] = true
	_build_stars(assets.stars())
	loaded["stars"] = stars.get_meta("count", 0)
	return loaded


func _build_stars(table: Dictionary) -> void:
	var verts := PackedVector3Array()
	var colors := PackedColorArray()
	var radius := 1.0e10
	if table.is_empty():
		var rng := RandomNumberGenerator.new()
		rng.seed = 12345
		for i in 4000:
			verts.append(Vector3(rng.randfn(), rng.randfn(), rng.randfn()).normalized() * radius)
			var b := pow(10.0, -0.16 * rng.randf_range(-1.0, 6.5))
			colors.append(Color(b, b, b, 1.5))
	else:
		var cols: Array = table["columns"]
		var w := cols.size()
		var data: PackedFloat32Array = table["data"]
		var ix := cols.find("x")
		var il := cols.find("illuminance_lux")
		var ir := cols.find("r")
		for i in int(table["rows"]):
			var o := i * w
			var dir := PackedFloat64Array([data[o + ix], data[o + ix + 1], data[o + ix + 2]])
			verts.append(D.to_godot(dir) * radius)
			# Display brightness (artistic until physical exposure, Phase 6): V=0 -> 1.
			var b := clampf(pow(data[o + il] / 2.54e-6, 0.4), 0.03, 4.0)
			var size := clampf(1.0 + b, 1.0, 4.0)
			colors.append(Color(data[o + ir] * b, data[o + ir + 1] * b, data[o + ir + 2] * b, size))
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_POINTS, arrays)
	if stars != null:
		stars.queue_free()
	stars = MeshInstance3D.new()
	stars.mesh = mesh
	stars.material_override = _material("res://shaders/stars.gdshader")
	stars.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	stars.custom_aabb = AABB(Vector3.ONE * -2.0 * radius, Vector3.ONE * 4.0 * radius)
	stars.set_meta("count", verts.size())
	stars.set_meta("radius", radius)
	viewport.add_child(stars)


## Approximate Moon body frame (local -> GCRS, row-major): +X toward Earth
## (tidal locking), +Z toward the ecliptic pole. Libration (+-7 deg) ignored.
static func moon_rotation(r_moon: PackedFloat64Array) -> PackedFloat64Array:
	var x := D.unit(D.scale(r_moon, -1.0))
	var pole := D.v3(0.0, -sin(OBLIQUITY), cos(OBLIQUITY))
	var z := D.unit(D.sub(pole, D.scale(x, D.dot(pole, x))))
	var y := D.cross(z, x)
	return PackedFloat64Array([x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]])


## Pose everything for one frame. frame: FrameStore.sample() (metres).
func update(frame: Dictionary, cam_pos: PackedFloat64Array, cam_rot: PackedFloat64Array) -> void:
	var r_sc: PackedFloat64Array = frame["r_sc"]
	var r_sun: PackedFloat64Array = frame["r_sun"]
	var d_craft := D.norm(D.sub(r_sc, cam_pos))
	var craft_near := origin.begin_frame(cam_pos, d_craft, craft.radius) and craft.visible
	origin.place_camera(camera, cam_rot)

	# Spacecraft: true scale when near, otherwise in scaled space like the bodies.
	var k_craft := 1.0 if craft_near else origin.far_scale(d_craft)
	origin.place(craft, r_sc, D.rot_mat(frame["q_gcrs_body"]), k_craft)
	var focal := (viewport.size.y / 2.0) / tan(deg_to_rad(camera.fov) / 2.0)
	craft.choose_lod(craft.radius * focal / maxf(d_craft, 1e-6))
	if dynamic_lighting:
		craft.update_articulation(frame)

	# Sunlight on the spacecraft (standard PBR materials), dimmed in eclipse.
	var sun_dir := D.unit(D.sub(r_sun, r_sc))
	light.basis = D.basis_from(D.look_rot(D.scale(sun_dir, -1.0), _any_perpendicular(sun_dir)))
	# Energy 1 = the Earth shader's full sunlight (Godot's Lambert has no 1/pi).
	light.light_energy = 1.0 * float(frame["shadow_factor"])
	light.visible = dynamic_lighting and light.light_energy > 1e-4
	light.directional_shadow_max_distance = clampf(4.0 * d_craft, 2.0, 1.0e4) if craft_near else 0.0
	light.shadow_enabled = craft_near

	var r_itrs_gcrs := D.rot_mat(frame["q_itrs_gcrs"])  # v_itrs = R v_gcrs
	var k_earth := origin.far_scale(D.norm(cam_pos))
	origin.place(earth, D.v3(0.0, 0.0, 0.0), D.transpose(r_itrs_gcrs), k_earth)
	var s_itrs := D.unit(D.mat_vec(r_itrs_gcrs, r_sun))
	earth_mat.set_shader_parameter("sun_dir_itrs", Vector3(s_itrs[0], s_itrs[1], s_itrs[2]))

	var k_moon := 0.0
	if frame.has("r_moon"):
		var r_moon: PackedFloat64Array = frame["r_moon"]
		var rot := moon_rotation(r_moon)
		k_moon = origin.far_scale(D.norm(D.sub(r_moon, cam_pos)))
		origin.place(moon, r_moon, rot, k_moon)
		var s_local := D.unit(D.mat_t_vec(rot, D.sub(r_sun, r_moon)))
		moon_mat.set_shader_parameter("sun_dir_local", Vector3(s_local[0], s_local[1], s_local[2]))
		moon.visible = true
	else:
		moon.visible = false

	# Stars and Sun: at infinity, just inside the far plane, never translating.
	var k_stars := 0.95 * origin.far / float(stars.get_meta("radius", 1.0e10))
	stars.transform = Transform3D(D.basis_from(PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])).scaled(Vector3.ONE * k_stars),
		camera.position)
	var sun_from_cam := D.unit(D.sub(r_sun, cam_pos))
	var dist := 0.9 * origin.far
	var half := maxf(dist * tan(SUN_ANGULAR_RADIUS), 3.0 * dist / focal) / 0.35  # disk >= 3 px radius
	sun.transform = Transform3D(camera.basis.scaled(Vector3.ONE * 2.0 * half),
		camera.position + D.to_godot(D.scale(sun_from_cam, dist)))

	info = {"near": origin.near, "far": origin.far, "k_craft": k_craft, "k_earth": k_earth, "k_moon": k_moon,
		"craft_near": craft_near, "lod": craft.lod, "d_craft": d_craft}


static func _any_perpendicular(v: PackedFloat64Array) -> PackedFloat64Array:
	return D.v3(0, 0, 1) if absf(v[2]) < 0.9 else D.v3(1, 0, 0)
