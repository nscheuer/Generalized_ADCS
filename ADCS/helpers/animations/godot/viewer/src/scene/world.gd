extends RefCounted
## Everything that is drawn: Earth, Moon, Sun, stars and the spacecraft, in one
## SubViewport whose camera sits at the Godot origin (see core/origin.gd).

const D := preload("res://src/core/dmath.gd")
const Origin := preload("res://src/core/origin.gd")
const Assets := preload("res://src/scene/assets.gd")
const Spacecraft := preload("res://src/scene/spacecraft.gd")
const SolarSystem := preload("res://src/scene/solar_system.gd")
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
var rays_mat: ShaderMaterial
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
var quality := "fast"
var overlays := OverlayManager.new()
var origin := Origin.new()
var earth: MeshInstance3D
var earth_mat: ShaderMaterial
var moon: MeshInstance3D
var moon_mat: ShaderMaterial
var stars: MeshInstance3D
var sun: MeshInstance3D
var craft: Spacecraft
var craft_pixel: MeshInstance3D
var earth_pixel: MeshInstance3D
var moon_pixel: MeshInstance3D
var craft_label: Label
var satellite_label_enabled := true
var solar: RefCounted
var fixed_size := Vector2i.ZERO
var info: Dictionary = {}  ## per-frame diagnostics (near/far, scales, LOD)
var last_camera_position := PackedFloat64Array()
## Checks that measure geometry only switch off time-varying light and animation.
var dynamic_lighting := true
var rays_strength := 0.0
var _rays_start_strength := 0.0
var _rays_target_strength := 0.0
var _rays_transition_start_ms := 0


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
	rays_mat = ShaderMaterial.new()
	rays_mat.shader = load("res://shaders/volumetric_rays.gdshader")
	view_rect.material = rays_mat
	parent.add_child(view_rect)

	env = Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color.BLACK
	env.tonemap_mode = Environment.TONE_MAPPER_AGX
	env.glow_enabled = true
	env.glow_hdr_threshold = 1.5
	for level in 7:
		env.set_glow_level(level, 1.0 if level in [2, 3, 4] else 0.0)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.03, 0.035, 0.045)

	camera = Camera3D.new()
	camera.fov = fov_deg
	camera.environment = env
	viewport.add_child(camera)
	camera.current = true
	solar = SolarSystem.new(parent, viewport, camera, origin)

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
	shell_mesh.radius = R_EARTH + 80.5e3
	shell_mesh.height = 2.0 * shell_mesh.radius
	# Match Earth's angular tessellation so its depth silhouette never crosses
	# the atmosphere shell as a dotted arc while the camera moves.
	shell_mesh.radial_segments = 384
	shell_mesh.rings = 192
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
	earth_pixel = _pixel_marker(Color(0.18, 0.31, 0.43))
	moon_pixel = _pixel_marker(Color(0.48, 0.46, 0.43))
	craft_pixel = _pixel_marker(craft.average_color)
	viewport.add_child(earth_pixel)
	viewport.add_child(moon_pixel)
	viewport.add_child(craft_pixel)
	craft_label = Label.new()
	craft_label.add_theme_color_override("font_color", Color(0.82, 0.88, 1.0))
	craft_label.add_theme_color_override("font_shadow_color", Color.BLACK)
	craft_label.mouse_filter = Control.MOUSE_FILTER_IGNORE
	craft_label.visible = false
	parent.add_child(craft_label)


static func _pixel_marker(color: Color) -> MeshInstance3D:
	var marker := MeshInstance3D.new()
	var sphere := SphereMesh.new()
	sphere.radius = 1.0
	sphere.height = 2.0
	sphere.radial_segments = 8
	sphere.rings = 4
	marker.mesh = sphere
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.albedo_color = color
	marker.material_override = material
	marker.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
	marker.visible = false
	return marker


func resize_to(size: Vector2i) -> void:
	if fixed_size == Vector2i.ZERO and size.x > 0 and size.y > 0:
		viewport.size = size


func set_label_visibility(satellite: bool, planets: bool) -> void:
	satellite_label_enabled = satellite
	solar.set_labels_enabled(planets)


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


func _body_lod(node: MeshInstance3D, radius: float, pixels: float) -> void:
	var level := 0 if pixels < 2.0 else (1 if pixels < 40.0 or quality == "fast" else 2)
	if node.get_meta("lod", -1) == level:
		return
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = 2.0 * radius
	# Use matching tessellation for Earth and its atmosphere shell so their
	# silhouettes share the same vertices. The shell is intentionally denser
	# than before to avoid a faceted ground sphere showing through its smooth rim.
	var detail := 32 if level == 0 else (128 if level == 1 else (256 if quality == "high" else 192))
	mesh.radial_segments = detail
	mesh.rings = detail / 2
	node.mesh = mesh
	node.set_meta("lod", level)


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
	var using_lut := atmosphere != null
	if atmosphere == null:
		atmosphere = Atmosphere.new()
	atmosphere.apply(earth_mat)
	atmosphere.apply(shell_mat)
	shell.visible = true
	loaded["atmosphere"] = true
	loaded["atmosphere_mode"] = "lut" if using_lut else "procedural"
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
	last_camera_position = cam_pos
	craft.set_after_simulation(bool(frame.get("after_simulation", false)))
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
	var craft_px := craft.radius * focal / maxf(d_craft, 1.0)
	craft.choose_lod(craft_px)
	var craft_marker_radius := maxf(craft.radius, 0.75 * d_craft / maxf(focal, 1.0))
	origin.place(craft_pixel, r_sc, PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
		k_craft * craft_marker_radius)
	craft_pixel.visible = craft_px < 1.0
	var craft_pixel_material := craft_pixel.material_override as StandardMaterial3D
	var after_simulation := bool(frame.get("after_simulation", false))
	craft_pixel_material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA if after_simulation else BaseMaterial3D.TRANSPARENCY_DISABLED
	craft_pixel_material.albedo_color = Color(0.65, 0.65, 0.65, 0.8) if after_simulation else craft.average_color
	var craft_screen_ok := not camera.is_position_behind(craft_pixel.global_position)
	var craft_screen := camera.unproject_position(craft_pixel.global_position) if craft_screen_ok else Vector2(-1, -1)
	craft_label.visible = satellite_label_enabled and craft_px < 1.0 and craft_screen_ok and craft_screen.x >= 0.0 and craft_screen.y >= 0.0 \
		and craft_screen.x < viewport.size.x and craft_screen.y < viewport.size.y
	if craft_label.visible:
		craft_label.text = "< SATELLITE"
		craft_label.position = craft_screen + Vector2(8, -8)
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
	var earth_px := R_EARTH * focal / maxf(D.norm(cam_pos), 1.0)
	_body_lod(earth, R_EARTH, earth_px)
	_body_lod(shell, R_EARTH + 80500.0, earth_px)
	var k_earth := origin.far_scale(D.norm(cam_pos))
	origin.place(earth, D.v3(0.0, 0.0, 0.0), D.transpose(r_itrs_gcrs), k_earth)
	var earth_marker_radius := maxf(R_EARTH, 0.75 * D.norm(cam_pos) / maxf(focal, 1.0))
	origin.place(earth_pixel, D.v3(0, 0, 0), PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
		k_earth * earth_marker_radius)
	earth.visible = earth_px >= 1.0
	earth_pixel.visible = earth_px < 1.0
	var s_itrs := D.unit(D.mat_vec(r_itrs_gcrs, r_sun))
	var c_itrs := D.mat_vec(r_itrs_gcrs, cam_pos)  # camera in ITRS, float64 until here
	for m: ShaderMaterial in [earth_mat, shell_mat]:
		m.set_shader_parameter("sun_dir_itrs", Vector3(s_itrs[0], s_itrs[1], s_itrs[2]))
		m.set_shader_parameter("cam_local", Vector3(c_itrs[0], c_itrs[1], c_itrs[2]))

	var k_moon := 0.0
	if frame.has("r_moon"):
		var r_moon: PackedFloat64Array = frame["r_moon"]
		var d_moon := D.norm(D.sub(r_moon, cam_pos))
		var moon_px := R_MOON * focal / maxf(d_moon, 1.0)
		_body_lod(moon, R_MOON, moon_px)
		var rot := D.rot_mat(frame["q_moon_gcrs"]) if frame.has("q_moon_gcrs") else moon_rotation(r_moon)
		k_moon = origin.far_scale(d_moon)
		origin.place(moon, r_moon, rot, k_moon)
		var moon_marker_radius := maxf(R_MOON, 0.75 * d_moon / maxf(focal, 1.0))
		origin.place(moon_pixel, r_moon, PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
			k_moon * moon_marker_radius)
		moon.visible = moon_px >= 1.0
		moon_pixel.visible = moon_px < 1.0
		var s_local := D.unit(D.mat_t_vec(rot, D.sub(r_sun, r_moon)))
		moon_mat.set_shader_parameter("sun_dir_local", Vector3(s_local[0], s_local[1], s_local[2]))
		var c_local := D.mat_t_vec(rot, D.sub(cam_pos, r_moon))
		moon_mat.set_shader_parameter("cam_local", Vector3(c_local[0], c_local[1], c_local[2]))
	else:
		moon.visible = false
		moon_pixel.visible = false
	solar.update(frame, cam_pos, focal, quality)

	# Stars and Sun: at infinity, just inside the far plane, never translating.
	var k_stars := 0.95 * origin.far / float(stars.get_meta("radius", 1.0e10))
	stars.transform = Transform3D(D.basis_from(PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1])).scaled(Vector3.ONE * k_stars),
		camera.position)
	var sun_from_cam := D.unit(D.sub(r_sun, cam_pos))
	if frame.has("q_sun_gcrs"):
		var sun_rot := D.rot_mat(frame["q_sun_gcrs"])
		var camera_right := D.mat_t_vec(sun_rot, PackedFloat64Array([cam_rot[0], cam_rot[3], cam_rot[6]]))
		var camera_up := D.mat_t_vec(sun_rot, PackedFloat64Array([cam_rot[1], cam_rot[4], cam_rot[7]]))
		var camera_back := D.mat_t_vec(sun_rot, PackedFloat64Array([cam_rot[2], cam_rot[5], cam_rot[8]]))
		(sun.material_override as ShaderMaterial).set_shader_parameter("camera_to_sun", Basis(
			Vector3(camera_right[0], camera_right[1], camera_right[2]),
			Vector3(camera_up[0], camera_up[1], camera_up[2]),
			Vector3(camera_back[0], camera_back[1], camera_back[2])))
	var dist := 0.92 * origin.far  # behind the far bodies (<= 0.9 x far), in front of the stars (0.95)
	var true_half := dist * tan(SUN_ANGULAR_RADIUS)
	var half := maxf(true_half, 3.0 * dist / focal) / 0.35  # disk >= 3 px radius
	_update_exposure(focal, (true_half / 0.35) / half)
	var sun_position := camera.position + D.to_godot(D.scale(sun_from_cam, dist))
	sun.transform = Transform3D(camera.basis.scaled(Vector3.ONE * 2.0 * half), sun_position)
	var sun_screen := camera.unproject_position(sun_position)
	var sun_uv := Vector2(sun_screen.x / float(viewport.size.x), sun_screen.y / float(viewport.size.y))
	rays_mat.set_shader_parameter("sun_uv", sun_uv)
	var sun_direction := Vector3(sun_from_cam[0], sun_from_cam[1], sun_from_cam[2])
	rays_mat.set_shader_parameter("rays_visibility", smoothstep(0.0, 0.12, sun_direction.dot(-camera.basis.z)))
	var ramp := clampf(float(Time.get_ticks_msec() - _rays_transition_start_ms) / 1200.0, 0.0, 1.0)
	ramp = ramp * ramp * (3.0 - 2.0 * ramp)
	rays_strength = lerpf(_rays_start_strength, _rays_target_strength, ramp)
	rays_mat.set_shader_parameter("rays_strength", rays_strength)
	info = {"near": origin.near, "far": origin.far, "k_craft": k_craft, "k_earth": k_earth, "k_moon": k_moon,
		"craft_near": craft_near, "lod": craft.lod, "d_craft": d_craft}
	overlays.update(frame, self, cam_pos, cam_rot)


func pick_body(screen: Vector2, frame: Dictionary) -> String:
	var planet: String = solar.pick(screen)
	if planet != "":
		return planet
	for item: Array in [["spacecraft", craft, frame["r_sc"], craft.radius],
		["earth", earth, D.v3(0, 0, 0), R_EARTH],
		["moon", moon, frame.get("r_moon", D.v3(0, 0, 0)), R_MOON],
		["sun", sun, frame["r_sun"], 695700000.0]]:
		var node: Node3D = item[1]
		if not node.visible or camera.is_position_behind(node.global_position):
			continue
		var pos := camera.unproject_position(node.global_position)
		var px := float(item[3]) * focal_px() / maxf(D.norm(D.sub(item[2], last_camera_position)), 1.0)
		if screen.distance_to(pos) <= maxf(14.0, px):
			return item[0]
	return ""


## High adds denser atmospheric integration and gently faded screen-space light shafts;
## it is intended for a discrete GPU. Balanced and fast keep the lighter path.
const QUALITY := {
	"high": {"msaa": Viewport.MSAA_4X, "fxaa": false, "glow": true, "shell_steps": 64, "pixel_steps": 40, "vertex_steps": 48, "scattering_gain": 1.6},
	"balanced": {"msaa": Viewport.MSAA_DISABLED, "fxaa": true, "glow": true, "shell_steps": 12, "pixel_steps": 12, "vertex_steps": 24},
	"fast": {"msaa": Viewport.MSAA_DISABLED, "fxaa": true, "glow": false, "shell_steps": 8, "pixel_steps": 8, "vertex_steps": 16},
}


func set_quality(name: String) -> void:
	if not QUALITY.has(name):
		name = "balanced"
	quality = name
	var q: Dictionary = QUALITY[name]
	# The AMD Vulkan driver loses opaque Earth pixels with 4x MSAA plus HDR glow.
	# 2x retains hardware edge smoothing and the full glow/reflection path.
	viewport.msaa_3d = Viewport.MSAA_2X if name == "high" and RenderingServer.get_video_adapter_name().to_lower().contains("amd") else q["msaa"]
	viewport.screen_space_aa = Viewport.SCREEN_SPACE_AA_FXAA if q["fxaa"] else Viewport.SCREEN_SPACE_AA_DISABLED
	env.glow_enabled = q["glow"]
	shell_mat.set_shader_parameter("atm_steps", q["shell_steps"])
	earth_mat.set_shader_parameter("pixel_steps", q["pixel_steps"])
	earth_mat.set_shader_parameter("atm_steps", q["vertex_steps"])
	for material: ShaderMaterial in [earth_mat, shell_mat]:
		material.set_shader_parameter("scattering_gain", q.get("scattering_gain", 1.0))
	_rays_start_strength = rays_strength
	_rays_target_strength = 0.04 if name == "high" else 0.0
	_rays_transition_start_ms = Time.get_ticks_msec()


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
