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

var nodes: Dictionary = {}
var meshes: Dictionary = {}
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
		body.material_override = mat
		body.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
		body.visible = false
		root.add_child(body)
		meshes[name] = body
		if name == "saturn":
			var rings := MeshInstance3D.new()
			var quad := QuadMesh.new()
			quad.size = Vector2.ONE * 2.0 * 140180000.0
			rings.mesh = quad
			var ring_mat := ShaderMaterial.new()
			ring_mat.shader = load("res://shaders/rings.gdshader")
			rings.material_override = ring_mat
			rings.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			root.add_child(rings)
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


func update(frame: Dictionary, cam_pos: PackedFloat64Array, focal_px: float, quality: String) -> void:
	focal = focal_px
	projected.clear()
	for name: String in SPECS:
		var body: MeshInstance3D = meshes[name]
		var root: Node3D = nodes[name]
		var label: Label = labels[name]
		if not frame.has("r_" + name):
			body.visible = false
			label.visible = false
			continue
		var p: PackedFloat64Array = frame["r_" + name]
		var d := D.norm(D.sub(p, cam_pos))
		var radius: float = SPECS[name][0]
		var px := radius * focal_px / maxf(d, 1.0)
		var lod := 0 if px < 2.0 else (1 if px < 40.0 or quality == "fast" else 2)
		_apply_texture(name, "8k" if quality == "high" and px >= 128.0 else "2k")
		if body.get_meta("lod", -1) != lod:
			body.mesh = _mesh(radius, 8 if lod == 0 else (24 if lod == 1 else 64),
				4 if lod == 0 else (12 if lod == 1 else 32))
			body.set_meta("lod", lod)
		origin.place(root, p, D.rot_mat(frame["q_" + name + "_gcrs"]), origin.far_scale(d))
		var sun_dir := D.unit(D.sub(frame["r_sun"], p))
		var mat: ShaderMaterial = body.material_override
		mat.set_shader_parameter("sun_dir_world", Vector3(sun_dir[0], sun_dir[1], sun_dir[2]))
		if name == "saturn":
			var ring_mat: ShaderMaterial = (root.get_child(1) as MeshInstance3D).material_override
			ring_mat.set_shader_parameter("sun_dir_world", Vector3(sun_dir[0], sun_dir[1], sun_dir[2]))
		body.visible = px >= 1.0
		var visible := not camera.is_position_behind(root.global_position)
		var screen := camera.unproject_position(root.global_position) if visible else Vector2(-1, -1)
		projected[name] = [screen, px] if visible else []
		label.visible = visible and px < 1.0 and screen.x >= 0.0 and screen.y >= 0.0
		if label.visible:
			label.position = screen + Vector2(6, -8)
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
		label.visible = visible and px < 1.0 and screen.x >= 0.0 and screen.y >= 0.0
		if label.visible:
			label.position = screen + Vector2(6, -8)


func pick(screen: Vector2) -> String:
	for name: String in projected:
		var v: Array = projected[name]
		if not v.is_empty() and screen.distance_to(v[0]) <= maxf(14.0, v[1]):
			return name
	return ""
