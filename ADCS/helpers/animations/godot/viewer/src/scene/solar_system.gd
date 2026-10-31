extends RefCounted
## Planetary bodies at ephemeris positions; all lengths are metres.

const D := preload("res://src/core/dmath.gd")
const SPECS := {
	"mercury": [2439500.0, Color(0.73, 0.69, 0.65)],
	"venus": [6052000.0, Color(0.92, 0.78, 0.58)],
	"mars": [3396000.0, Color(0.94, 0.45, 0.32)],
	"jupiter": [71492000.0, Color(0.92, 0.74, 0.57)],
	"saturn": [60268000.0, Color(0.91, 0.81, 0.58)],
	"uranus": [25559000.0, Color(0.62, 0.84, 0.91)],
	"neptune": [24764000.0, Color(0.39, 0.57, 0.99)],
}
## Approximate visible haze layers. The shared atmosphere shader uses these
## radii and scattering coefficients in metres, as it does for Earth.
const ATMOSPHERES := {
	"venus": {"height": 180000.0, "scale": 16000.0, "rayleigh": Vector3(1.3e-6, 1.0e-6, 0.6e-6), "mie": 7.0e-6, "gain": 2.8, "haze": Color(0.98, 0.80, 0.54)},
	"mars": {"height": 110000.0, "scale": 18000.0, "rayleigh": Vector3(0.7e-6, 0.4e-6, 0.25e-6), "mie": 1.5e-6, "gain": 3.4, "haze": Color(0.95, 0.51, 0.32)},
	"jupiter": {"height": 3000000.0, "scale": 650000.0, "rayleigh": Vector3(0.8e-6, 0.6e-6, 0.4e-6), "mie": 0.8e-6, "haze": Color(0.94, 0.80, 0.65)},
	"saturn": {"height": 2600000.0, "scale": 600000.0, "rayleigh": Vector3(0.8e-6, 0.7e-6, 0.5e-6), "mie": 0.7e-6, "haze": Color(0.95, 0.85, 0.67)},
	"uranus": {"height": 1200000.0, "scale": 350000.0, "rayleigh": Vector3(0.4e-6, 0.9e-6, 1.4e-6), "mie": 0.4e-6, "haze": Color(0.66, 0.88, 0.94)},
	"neptune": {"height": 1200000.0, "scale": 350000.0, "rayleigh": Vector3(0.3e-6, 0.6e-6, 1.6e-6), "mie": 0.3e-6, "haze": Color(0.38, 0.61, 1.0)},
}
const GAS_BANDS := {
	"jupiter": [Color(0.77, 0.52, 0.39), 0.72],
	"saturn": [Color(0.73, 0.65, 0.50), 0.38],
	"uranus": [Color(0.48, 0.72, 0.82), 0.18],
	"neptune": [Color(0.27, 0.42, 0.84), 0.24],
}

var nodes: Dictionary = {}
var meshes: Dictionary = {}
var shells: Dictionary = {}
var ring_mesh: MeshInstance3D
var pixel_markers: Dictionary = {}
var labels: Dictionary = {}
var projected: Dictionary = {}
var focal := 1.0
var camera: Camera3D
var origin: RefCounted
var asset_root := ""
var earth_mat: ShaderMaterial
var moon_mat: ShaderMaterial
var sun_mat: ShaderMaterial
var current_texture_lod: Dictionary = {}
var textures_2k: Dictionary = {}
var labels_enabled := true


func set_labels_enabled(on: bool) -> void:
	labels_enabled = on


func load_textures(root: String, earth_material: ShaderMaterial, moon_material: ShaderMaterial,
		sun_material: ShaderMaterial) -> void:
	asset_root = root.path_join("planet_textures") if root != "" else ""
	current_texture_lod.clear()
	textures_2k.clear()
	earth_mat = earth_material
	moon_mat = moon_material
	sun_mat = sun_material
	for name: String in SPECS:
		_apply_texture(name, "2k")
	_apply_texture("earth", "2k")
	_apply_texture("earth_night", "2k")
	_apply_texture("earth_clouds", "2k")
	_apply_texture("earth_normal", "2k")
	_apply_texture("earth_specular", "2k")
	_apply_texture("moon", "2k")
	_apply_texture("sun", "2k")


func _filename(name: String, lod: String) -> String:
	if name == "venus":
		return "4k_venus_atmosphere.jpg" if lod == "8k" else "2k_venus_atmosphere.jpg"
	if name == "earth":
		return lod + "_earth_daymap.jpg"
	if name == "earth_night":
		return lod + "_earth_nightmap.jpg"
	if name == "earth_clouds":
		return lod + "_earth_clouds.jpg"
	if name == "earth_normal":
		return lod + "_earth_normal_map.png"
	if name == "earth_specular":
		return lod + "_earth_specular_map.png"
	return lod + "_" + name + ".jpg"


func _load_texture(name: String, lod: String) -> Array:
	if asset_root == "":
		return [null, ""]
	var actual := lod
	var path := asset_root.path_join(_filename(name, actual))
	if not FileAccess.file_exists(path) and actual == "8k":
		actual = "2k"
		path = asset_root.path_join(_filename(name, actual))
	if not FileAccess.file_exists(path):
		return [null, actual]
	if actual == "2k" and textures_2k.has(name):
		return [textures_2k[name], actual]
	var image := Image.load_from_file(path)
	if image == null:
		return [null, actual]
	image.generate_mipmaps()
	var texture := ImageTexture.create_from_image(image)
	if actual == "2k":
		textures_2k[name] = texture
	return [texture, actual]


func _apply_texture(name: String, lod: String) -> void:
	if current_texture_lod.get(name, "") == lod:
		return
	var loaded: Array = _load_texture(name, lod)
	var texture: Texture2D = loaded[0]
	var actual_lod: String = loaded[1]
	current_texture_lod[name] = lod
	match name:
		"earth":
			if texture != null:
				earth_mat.set_shader_parameter("day_map", texture)
			earth_mat.set_shader_parameter("has_day_map", texture != null)
		"earth_night":
			if texture != null:
				earth_mat.set_shader_parameter("night_map", texture)
			earth_mat.set_shader_parameter("has_night_map", texture != null)
		"earth_clouds":
			if texture != null:
				earth_mat.set_shader_parameter("cloud_map", texture)
			earth_mat.set_shader_parameter("has_cloud_map", texture != null)
		"earth_normal":
			if texture != null:
				earth_mat.set_shader_parameter("normal_map", texture)
			earth_mat.set_shader_parameter("has_normal_map", texture != null)
		"earth_specular":
			if texture != null:
				earth_mat.set_shader_parameter("specular_map", texture)
			earth_mat.set_shader_parameter("has_specular_map", texture != null)
		"moon":
			if texture != null:
				moon_mat.set_shader_parameter("color_map", texture)
			moon_mat.set_shader_parameter("has_color_map", texture != null)
		"sun":
			if texture != null:
				sun_mat.set_shader_parameter("surface_map", texture)
			sun_mat.set_shader_parameter("has_surface_map", texture != null)
		_:
			var mat: ShaderMaterial = meshes[name].material_override
			if texture != null:
				mat.set_shader_parameter("albedo_texture", texture)
			mat.set_shader_parameter("has_texture", texture != null)


func _init(ui_parent: Node, viewport: SubViewport, cam: Camera3D, scene_origin: RefCounted) -> void:
	camera = cam
	origin = scene_origin
	for name: String in SPECS:
		var root := Node3D.new()
		viewport.add_child(root)
		nodes[name] = root
		var body := MeshInstance3D.new()
		body.mesh = _mesh(float(SPECS[name][0]), 32, 16)
		body.rotation_degrees = Vector3(90, 0, 0)  # Godot sphere +Y pole -> IAU body +Z.
		var mat := ShaderMaterial.new()
		mat.shader = load("res://shaders/planet.gdshader")
		mat.set_shader_parameter("base_color", SPECS[name][1])
		if ATMOSPHERES.has(name):
			var haze: Color = ATMOSPHERES[name]["haze"]
			mat.set_shader_parameter("haze_color", Vector3(haze.r, haze.g, haze.b))
			mat.set_shader_parameter("haze_strength", 0.88 if name == "venus" else (0.72 if name == "mars" else 0.65))
		if GAS_BANDS.has(name):
			var band: Color = GAS_BANDS[name][0]
			mat.set_shader_parameter("gas_giant", true)
			mat.set_shader_parameter("band_color", Vector3(band.r, band.g, band.b))
			mat.set_shader_parameter("band_strength", GAS_BANDS[name][1])
		body.material_override = mat
		body.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		body.visible = false
		root.add_child(body)
		meshes[name] = body
		if ATMOSPHERES.has(name):
			var atmosphere: Dictionary = ATMOSPHERES[name]
			var radius: float = SPECS[name][0]
			var shell := MeshInstance3D.new()
			shell.mesh = _mesh(radius + atmosphere["height"], 96, 48)
			var shell_mat := ShaderMaterial.new()
			shell_mat.shader = load("res://shaders/atmosphere_shell.gdshader")
			shell_mat.set_shader_parameter("has_atmosphere", true)
			shell_mat.set_shader_parameter("atm_bottom", radius)
			shell_mat.set_shader_parameter("atm_top", radius + atmosphere["height"])
			shell_mat.set_shader_parameter("rayleigh_scattering", atmosphere["rayleigh"])
			shell_mat.set_shader_parameter("rayleigh_h", atmosphere["scale"])
			shell_mat.set_shader_parameter("mie_scattering", Vector3.ONE * atmosphere["mie"])
			shell_mat.set_shader_parameter("mie_extinction", Vector3.ONE * atmosphere["mie"] * 1.12)
			shell_mat.set_shader_parameter("mie_h", atmosphere["scale"] * 0.7)
			shell_mat.set_shader_parameter("ozone_absorption", Vector3.ZERO)
			shell_mat.set_shader_parameter("scattering_gain", atmosphere.get("gain", 1.15))
			shell.material_override = shell_mat
			shell.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			shell.visible = false
			root.add_child(shell)
			shells[name] = shell
		var marker := _pixel_marker(SPECS[name][1])
		viewport.add_child(marker)
		pixel_markers[name] = marker
		if name == "saturn":
			ring_mesh = MeshInstance3D.new()
			ring_mesh.mesh = _saturn_rings()
			var ring_mat := ShaderMaterial.new()
			ring_mat.shader = load("res://shaders/rings.gdshader")
			ring_mesh.material_override = ring_mat
			ring_mesh.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			root.add_child(ring_mesh)
		var label := Label.new()
		label.text = name.capitalize()
		label.add_theme_color_override("font_color", Color(0.82, 0.88, 1.0))
		label.add_theme_color_override("font_shadow_color", Color.BLACK)
		label.visible = false
		label.mouse_filter = Control.MOUSE_FILTER_IGNORE
		ui_parent.add_child(label)
		labels[name] = label
	for name: String in ["earth", "moon", "sun"]:
		var label := Label.new()
		label.text = name.capitalize()
		label.add_theme_color_override("font_color", Color(0.82, 0.88, 1.0))
		label.visible = false
		label.mouse_filter = Control.MOUSE_FILTER_IGNORE
		ui_parent.add_child(label)
		labels[name] = label


static func _mesh(radius: float, segments: int, rings: int) -> SphereMesh:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = 2.0 * radius
	mesh.radial_segments = segments
	mesh.rings = rings
	return mesh


## Main C/B/A rings and the narrow F ring, in Saturn's equatorial (local XY)
## plane. Radial UV is measured in thousands of kilometres for the shader.
static func _saturn_rings() -> ArrayMesh:
	var vertices := PackedVector3Array()
	var uvs := PackedVector2Array()
	var indices := PackedInt32Array()
	var radii := PackedFloat64Array([74510000.0, 92000000.0, 117580000.0,
		122170000.0, 136780000.0, 140180000.0, 140230000.0])
	var sides := 256
	for r in radii:
		for i in sides + 1:
			var angle := TAU * float(i) / float(sides)
			vertices.append(Vector3(r * cos(angle), r * sin(angle), 0.0))
			uvs.append(Vector2(r / 1.0e6, float(i) / float(sides)))
	for band in radii.size() - 1:
		for i in sides:
			var a := band * (sides + 1) + i
			var b := a + sides + 1
			indices.append_array(PackedInt32Array([a, b, a + 1, a + 1, b, b + 1]))
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = indices
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh


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


func update(frame: Dictionary, cam_pos: PackedFloat64Array, focal_px: float, quality: String) -> void:
	focal = focal_px
	projected.clear()
	for name: String in SPECS:
		var body: MeshInstance3D = meshes[name]
		var root: Node3D = nodes[name]
		var label: Label = labels[name]
		if not frame.has("r_" + name):
			body.visible = false
			if shells.has(name):
				shells[name].visible = false
			if name == "saturn":
				ring_mesh.visible = false
			pixel_markers[name].visible = false
			label.visible = false
			continue
		var p: PackedFloat64Array = frame["r_" + name]
		var d := D.norm(D.sub(p, cam_pos))
		var radius: float = SPECS[name][0]
		var px := radius * focal_px / maxf(d, 1.0)
		var marker: MeshInstance3D = pixel_markers[name]
		var marker_radius := maxf(radius, 0.75 * d / maxf(focal_px, 1.0))
		origin.place(marker, p, PackedFloat64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
			origin.far_scale(d) * marker_radius)
		marker.visible = px < 1.0
		var lod := 0 if px < 2.0 else (1 if px < 40.0 or quality == "fast" else 2)
		_apply_texture(name, "8k" if quality == "high" and px >= 128.0 else "2k")
		var detail := 12 if lod == 0 else (48 if lod == 1 else (192 if quality == "high" and px >= 128.0 else 96))
		if body.get_meta("segments", -1) != detail:
			body.mesh = _mesh(radius, detail, detail / 2)
			body.set_meta("segments", detail)
		var body_rot: PackedFloat64Array = D.rot_mat(frame["q_" + name + "_gcrs"])
		origin.place(root, p, body_rot, origin.far_scale(d))
		var sun_dir := D.unit(D.sub(frame["r_sun"], p))
		var mat: ShaderMaterial = body.material_override
		mat.set_shader_parameter("sun_dir_world", D.to_godot(sun_dir))
		if shells.has(name):
			var shell: MeshInstance3D = shells[name]
			shell.visible = px >= 1.0 and quality != "fast"
			if shell.visible:
				var shell_detail := 192 if quality == "high" and px >= 128.0 else 96
				if shell.get_meta("segments", -1) != shell_detail:
					shell.mesh = _mesh(radius + ATMOSPHERES[name]["height"], shell_detail, shell_detail / 2)
					shell.set_meta("segments", shell_detail)
				var shell_mat: ShaderMaterial = shell.material_override
				var local_sun := D.mat_t_vec(body_rot, sun_dir)
				var local_cam := D.mat_t_vec(body_rot, D.sub(cam_pos, p))
				shell_mat.set_shader_parameter("sun_dir_itrs", Vector3(local_sun[0], local_sun[1], local_sun[2]))
				shell_mat.set_shader_parameter("cam_local", Vector3(local_cam[0], local_cam[1], local_cam[2]))
				shell_mat.set_shader_parameter("atm_steps", 24 if quality == "high" else 12)
		if name == "saturn":
			ring_mesh.visible = px >= 1.0
			var ring_mat: ShaderMaterial = ring_mesh.material_override
			var ring_sun := D.mat_t_vec(body_rot, sun_dir)
			ring_mat.set_shader_parameter("sun_dir_local", Vector3(ring_sun[0], ring_sun[1], ring_sun[2]))
		body.visible = px >= 1.0
		var visible := not camera.is_position_behind(root.global_position)
		var screen := camera.unproject_position(root.global_position) if visible else Vector2(-1, -1)
		projected[name] = [screen, px] if visible else []
		label.visible = labels_enabled and visible and px < 1.0 and screen.x >= 0.0 and screen.y >= 0.0
		if label.visible:
			label.text = "< " + name.to_upper()
			label.position = screen + Vector2(8, -8)
	var earth_px := 6378137.0 * focal_px / maxf(D.norm(cam_pos), 1.0)
	var moon_px := 1737400.0 * focal_px / maxf(D.norm(D.sub(frame.get("r_moon", D.v3(0, 0, 0)), cam_pos)), 1.0)
	var detailed := quality == "high" and earth_px >= 128.0
	_apply_texture("earth", "8k" if detailed else "2k")
	_apply_texture("earth_night", "8k" if detailed else "2k")
	_apply_texture("earth_clouds", "8k" if detailed else "2k")
	_apply_texture("earth_normal", "8k" if detailed else "2k")
	_apply_texture("earth_specular", "8k" if detailed else "2k")
	_apply_texture("moon", "8k" if quality == "high" and moon_px >= 128.0 else "2k")
	_apply_texture("sun", "2k")
	for name: String in ["earth", "moon", "sun"]:
		var label: Label = labels[name]
		var key := "r_" + name
		if name == "earth":
			key = ""
		if key != "" and not frame.has(key):
			label.visible = false
			continue
		var p: PackedFloat64Array = D.v3(0, 0, 0) if name == "earth" else frame[key]
		var radius := 6378137.0 if name == "earth" else (1737400.0 if name == "moon" else 695700000.0)
		var d := D.norm(D.sub(p, cam_pos))
		var px := radius * focal_px / maxf(d, 1.0)
		var pos := D.to_godot(D.scale(D.sub(p, cam_pos), origin.far_scale(d)))
		var visible := not camera.is_position_behind(pos)
		var screen := camera.unproject_position(pos) if visible else Vector2(-1, -1)
		var label_enabled := labels_enabled if name == "earth" else true
		label.visible = label_enabled and visible and px < 1.0 and screen.x >= 0.0 and screen.y >= 0.0
		if label.visible:
			label.text = "< " + name.to_upper()
			label.position = screen + Vector2(8, -8)


func pick(screen: Vector2) -> String:
	for name: String in projected:
		var v: Array = projected[name]
		if not v.is_empty() and screen.distance_to(v[0]) <= maxf(14.0, v[1]):
			return name
	return ""
