extends RefCounted
## Everything that is drawn: Earth, Moon, Sun, stars and the spacecraft, in one
## SubViewport whose camera sits at the Godot origin (see core/origin.gd).

const D := preload("res://src/core/dmath.gd")
const Origin := preload("res://src/core/origin.gd")
const Assets := preload("res://src/scene/assets.gd")
const Spacecraft := preload("res://src/scene/spacecraft.gd")
const Atmosphere := preload("res://src/scene/atmosphere.gd")
const OverlayManager := preload("res://src/overlays/manager.gd")

const R_EARTH := 6378137.0
const R_MOON := 1737400.0
const SUN_ANGULAR_RADIUS := 0.004652  # rad, mean
const OBLIQUITY := 0.40909  # rad, J2000; Moon frame approximation
const SUN_SOLID_ANGLE := 6.8e-5  # sr
## Illuminance of the full Sun at 1 au [lux]: the viewer's display unit is a
## white Lambertian surface under this illuminance (value 1).
const SUN_LUX := 1.27e5
const EARTH_ALBEDO := 0.3

var viewport: SubViewport
var view_rect: TextureRect
var camera: Camera3D
var env: Environment
var light: DirectionalLight3D
var earthshine: DirectionalLight3D
var atmosphere: Atmosphere
var shell: MeshInstance3D
var shell_mat: ShaderMaterial
## "artistic": stars and the Sun compressed so they show next to a lit Earth.
## "physical": true relative brightness (stars vanish beside a sunlit Earth).
var exposure_mode := "artistic"
var exposure_ev := 0.0
## Vertical field of view [deg]. Below 1 deg (Godot's limit for `fov`), e.g.
## DSCOVR/EPIC's 0.62 deg, the camera uses an equivalent frustum projection.
var fov_deg := 45.0
var quality := "balanced"
var overlays := OverlayManager.new()
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
	# Fewer, coarser blur levels: the glow is for the Sun and specular highlights.
	for level in 7:
		env.set_glow_level(level, 1.0 if level in [2, 3, 4] else 0.0)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.03, 0.035, 0.045)

	camera = Camera3D.new()
	camera.fov = fov_deg
	camera.environment = env
	viewport.add_child(camera)
	camera.current = true

	light = DirectionalLight3D.new()
	light.shadow_enabled = true
	light.light_energy = 1.0
	viewport.add_child(light)
	earthshine = DirectionalLight3D.new()
	earthshine.light_color = Color(0.85, 0.92, 1.0)
	earthshine.shadow_enabled = false
	viewport.add_child(earthshine)

	earth_mat = _material("res://shaders/earth.gdshader")
	earth = _sphere(R_EARTH, 384, 192, earth_mat)
	shell_mat = _material("res://shaders/atmosphere_shell.gdshader")
	shell = MeshInstance3D.new()
	var shell_mesh := SphereMesh.new()
	shell_mesh.radius = R_EARTH + 100.5e3
	shell_mesh.height = 2.0 * shell_mesh.radius
	shell_mesh.radial_segments = 192
	shell_mesh.rings = 96
	shell.mesh = shell_mesh
	shell.material_override = shell_mat
	shell.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	shell.visible = false
	earth.add_child(shell)  # same transform (and scaled-space scale) as the Earth
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
	atmosphere = assets.atmosphere()
	if atmosphere != null:
		atmosphere.apply(earth_mat)
		atmosphere.apply(shell_mat)
		shell.visible = true
		loaded["atmosphere"] = true
	return loaded


func _build_stars(table: Dictionary) -> void:
	var verts := PackedVector3Array()
	var colors := PackedColorArray()
	var uvs := PackedVector2Array()
	var radius := 1.0e10
	if table.is_empty():
		var rng := RandomNumberGenerator.new()
		rng.seed = 12345
		for i in 4000:
			verts.append(Vector3(rng.randfn(), rng.randfn(), rng.randfn()).normalized() * radius)
			colors.append(Color(1, 1, 1))
			uvs.append(Vector2(pow(10.0, -0.4 * rng.randf_range(-1.0, 6.5)), 0.0))
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
			colors.append(Color(data[o + ir], data[o + ir + 1], data[o + ir + 2]))
			uvs.append(Vector2(data[o + il] / 2.54e-6, 0.0))  # illuminance relative to V = 0
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_COLOR] = colors
	arrays[Mesh.ARRAY_TEX_UV] = uvs
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
	# A hidden spacecraft must not size the frustum (it may sit at the camera).
	var craft_near := origin.begin_frame(cam_pos, d_craft if craft.visible else 1.0e30, craft.radius)
	origin.place_camera(camera, cam_rot, fov_deg)

	# Spacecraft: true scale when near, otherwise in scaled space like the bodies.
	var k_craft := 1.0 if craft_near else origin.far_scale(d_craft)
	origin.place(craft, r_sc, D.rot_mat(frame["q_gcrs_body"]), k_craft)
	var focal := focal_px()
	craft.choose_lod(craft.radius * focal / maxf(d_craft, 1e-6))
	if dynamic_lighting:
		craft.update_articulation(frame)

	# Sunlight on the spacecraft (standard PBR materials), dimmed in eclipse.
	var sun_dir := D.unit(D.sub(r_sun, r_sc))
	light.basis = D.basis_from(D.look_rot(D.scale(sun_dir, -1.0), _any_perpendicular(sun_dir)))
	# Energy 1 = the Earth shader's full sunlight (Godot's Lambert has no 1/pi);
	# coloured by the atmosphere when the Sun is seen through the limb.
	var sun_color := atmosphere.sunlight(r_sc, sun_dir) if atmosphere != null else Vector3.ONE
	var peak := maxf(sun_color.x, maxf(sun_color.y, sun_color.z))
	light.light_color = Color(sun_color.x / maxf(peak, 1e-6), sun_color.y / maxf(peak, 1e-6), sun_color.z / maxf(peak, 1e-6))
	light.light_energy = float(frame["shadow_factor"]) * peak
	light.visible = dynamic_lighting and light.light_energy > 1e-4
	light.directional_shadow_max_distance = clampf(4.0 * d_craft, 2.0, 1.0e4) if craft_near else 0.0
	light.shadow_enabled = craft_near

	# Earthshine: sunlight reflected by the visible Earth, from the nadir direction.
	# Flat plate facing Earth: view factor (R/r)^2, times the lit fraction of the disk.
	var r_norm := D.norm(r_sc)
	var up := D.scale(r_sc, 1.0 / r_norm)
	var lit_fraction := 0.5 * (1.0 + D.dot(up, D.unit(r_sun)))
	earthshine.basis = D.basis_from(D.look_rot(up, _any_perpendicular(up)))  # shines outward from Earth
	earthshine.light_energy = EARTH_ALBEDO * pow(R_EARTH / r_norm, 2.0) * lit_fraction if dynamic_lighting else 0.0
	earthshine.visible = earthshine.light_energy > 1e-4

	var r_itrs_gcrs := D.rot_mat(frame["q_itrs_gcrs"])  # v_itrs = R v_gcrs
	var k_earth := origin.far_scale(D.norm(cam_pos))
	origin.place(earth, D.v3(0.0, 0.0, 0.0), D.transpose(r_itrs_gcrs), k_earth)
	var s_itrs := D.unit(D.mat_vec(r_itrs_gcrs, r_sun))
	var c_itrs := D.mat_vec(r_itrs_gcrs, cam_pos)  # camera in ITRS, float64 until here
	for m: ShaderMaterial in [earth_mat, shell_mat]:
		m.set_shader_parameter("sun_dir_itrs", Vector3(s_itrs[0], s_itrs[1], s_itrs[2]))
		m.set_shader_parameter("cam_local", Vector3(c_itrs[0], c_itrs[1], c_itrs[2]))

	var k_moon := 0.0
	if frame.has("r_moon"):
		var r_moon: PackedFloat64Array = frame["r_moon"]
		var rot := moon_rotation(r_moon)
		k_moon = origin.far_scale(D.norm(D.sub(r_moon, cam_pos)))
		origin.place(moon, r_moon, rot, k_moon)
		var s_local := D.unit(D.mat_t_vec(rot, D.sub(r_sun, r_moon)))
		moon_mat.set_shader_parameter("sun_dir_local", Vector3(s_local[0], s_local[1], s_local[2]))
		var c_local := D.mat_t_vec(rot, D.sub(cam_pos, r_moon))
		moon_mat.set_shader_parameter("cam_local", Vector3(c_local[0], c_local[1], c_local[2]))
		moon.visible = true
	else:
		moon.visible = false

	# Stars and Sun: at infinity, just inside the far plane, never translating.
	var k_stars := 0.95 * origin.far / float(stars.get_meta("radius", 1.0e10))
	stars.transform = Transform3D(D.basis_from(PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])).scaled(Vector3.ONE * k_stars),
		camera.position)
	var sun_from_cam := D.unit(D.sub(r_sun, cam_pos))
	var dist := 0.92 * origin.far  # behind the far bodies (<= 0.9 x far), in front of the stars (0.95)
	var true_half := dist * tan(SUN_ANGULAR_RADIUS)
	var half := maxf(true_half, 3.0 * dist / focal) / 0.35  # disk >= 3 px radius
	_update_exposure(focal, (true_half / 0.35) / half)
	sun.transform = Transform3D(camera.basis.scaled(Vector3.ONE * 2.0 * half),
		camera.position + D.to_godot(D.scale(sun_from_cam, dist)))

	info = {"near": origin.near, "far": origin.far, "k_craft": k_craft, "k_earth": k_earth, "k_moon": k_moon,
		"craft_near": craft_near, "lod": craft.lod, "d_craft": d_craft}
	overlays.update(frame, self, cam_pos, cam_rot)


## Quality presets. Measured on an Intel Arc 140V at 1080p (GPU p50/p99):
## high (MSAA 4x) ~13/29 ms, balanced ~6/11 ms, fast lower still.
const QUALITY := {
	"high": {"msaa": Viewport.MSAA_4X, "fxaa": false, "glow": true, "shell_steps": 24, "pixel_steps": 16, "vertex_steps": 24},
	"balanced": {"msaa": Viewport.MSAA_DISABLED, "fxaa": true, "glow": true, "shell_steps": 12, "pixel_steps": 12, "vertex_steps": 24},
	"fast": {"msaa": Viewport.MSAA_DISABLED, "fxaa": true, "glow": false, "shell_steps": 8, "pixel_steps": 8, "vertex_steps": 16},
}


func set_quality(name: String) -> void:
	if not QUALITY.has(name):
		name = "balanced"
	quality = name
	var q: Dictionary = QUALITY[name]
	viewport.msaa_3d = q["msaa"]
	viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA if q["fxaa"] else Viewport.SCREEN_SPACE_AA_DISABLED
	env.glow_enabled = q["glow"]
	shell_mat.set_shader_parameter("atm_steps", q["shell_steps"])
	earth_mat.set_shader_parameter("pixel_steps", q["pixel_steps"])
	earth_mat.set_shader_parameter("atm_steps", q["vertex_steps"])


## Focal length in pixels for the current field of view and viewport height.
func focal_px() -> float:
	return (viewport.size.y / 2.0) / tan(deg_to_rad(fov_deg) / 2.0)


## Exposure: one EV control for everything; stars and Sun either compressed
## (artistic) or at their true brightness relative to a sunlit surface (physical).
func set_exposure(mode: String, ev: float) -> void:
	exposure_mode = mode
	exposure_ev = ev
	env.tonemap_exposure = pow(2.0, ev)


func _update_exposure(focal_px: float, sun_size_ratio: float) -> void:
	var physical := exposure_mode == "physical"
	var star_mat := stars.material_override as ShaderMaterial
	star_mat.set_shader_parameter("physical", physical)
	# One pixel's solid angle; a star's flux lands in one pixel.
	var omega := 1.0 / (focal_px * focal_px)
	star_mat.set_shader_parameter("phys_gain", PI * 2.54e-6 / (omega * SUN_LUX))
	var sun_mat := sun.material_override as ShaderMaterial
	# Physical: disk radiance pi / Omega_sun in display units, spread over the drawn size.
	sun_mat.set_shader_parameter("intensity", PI / SUN_SOLID_ANGLE * sun_size_ratio * sun_size_ratio if physical else 40.0)


static func _any_perpendicular(v: PackedFloat64Array) -> PackedFloat64Array:
	return D.v3(0, 0, 1) if absf(v[2]) < 0.9 else D.v3(1, 0, 0)
