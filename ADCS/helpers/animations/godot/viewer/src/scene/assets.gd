extends RefCounted
## Loads products built by `python -m ADCS.viz.assets` (docs/source/viz/assets.rst).
##
## Cube faces are stored in OpenGL/Vulkan order and orientation (+X, -X, +Y,
## -Y, +Z, -Z), which is also Godot's Cubemap layer order, so shaders sample
## them with the raw body-fixed direction.

var root := ""
var current: Dictionary = {}
var max_face := 0  ## downscale faces larger than this (0 = keep)
var load_ms: Dictionary = {}


func open(asset_root: String) -> bool:
	root = asset_root
	var path := root.path_join("processed").path_join("current.json")
	if asset_root == "" or not FileAccess.file_exists(path):
		return false
	var c: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	current = c if c is Dictionary else {}
	return not current.is_empty()


func has(product: String) -> bool:
	return current.has(product)


func _manifest(product: String) -> Dictionary:
	var dir := root.path_join("processed").path_join(current[product])
	var m: Variant = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("asset_manifest.json")))
	return m if m is Dictionary else {}


func cubemap(product: String) -> Cubemap:
	if not has(product):
		return null
	var t0 := Time.get_ticks_msec()
	var dir := root.path_join("processed").path_join(current[product])
	var images: Array[Image] = []
	for f: Dictionary in _manifest(product).get("files", []):
		var img := Image.load_from_file(dir.path_join(f["name"]))
		if img == null:
			push_error("cannot load %s" % f["name"])
			return null
		if max_face > 0 and img.get_width() > max_face:
			img.resize(max_face, max_face, Image.INTERPOLATE_BILINEAR)
		if img.get_format() == Image.FORMAT_RGB8:
			img.convert(Image.FORMAT_RGBA8)  # GPUs rarely support 3-channel textures
		img.generate_mipmaps()
		images.append(img)
	if images.size() != 6:
		push_error("%s: expected 6 cube faces, got %d" % [product, images.size()])
		return null
	var cube := Cubemap.new()
	var err := cube.create_from_images(images)
	load_ms[product] = Time.get_ticks_msec() - t0
	return cube if err == OK else null


## Atmosphere tables, or null if not built.
func atmosphere() -> RefCounted:
	if not has("atmosphere"):
		return null
	var atm: RefCounted = load("res://src/scene/atmosphere.gd").new()
	var dir := root.path_join("processed").path_join(current["atmosphere"])
	return atm if atm.load_from(dir, _manifest("atmosphere")) else null


## Star table rows: x, y, z, vmag, b_v, r, g, b, illuminance_lux, pm_ra_mas, pm_dec_mas.
func stars() -> Dictionary:
	if not has("stars"):
		return {}
	var m := _manifest("stars")
	var f: Dictionary = m["files"][0]
	var dir := root.path_join("processed").path_join(current["stars"])
	var data := FileAccess.get_file_as_bytes(dir.path_join(f["name"])).to_float32_array()
	return {"columns": f["columns"], "rows": int(f["rows"]), "data": data}
