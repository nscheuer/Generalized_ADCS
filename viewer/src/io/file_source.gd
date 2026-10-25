extends RefCounted
## Data source: an adcs-vizlog directory read from disk (docs/source/viz/vizlog_v1.rst).

const FrameStore := preload("res://src/core/frame_store.gd")

var manifest: Dictionary = {}
var scene: Dictionary = {}
var assets: Dictionary = {}
var store: FrameStore
var run_id := ""
var live := false
var finished := true
var error_message := ""


func open(path: String, wanted_run: String = "") -> bool:
	var m: Variant = JSON.parse_string(FileAccess.get_file_as_string(path.path_join("manifest.json")))
	if not m is Dictionary or m.get("format", "") != "adcs-vizlog":
		error_message = "%s is not an adcs-vizlog directory" % path
		return false
	manifest = m
	if int(manifest.get("schema_version", 0)) != 1:
		error_message = "unsupported vizlog schema version %s" % manifest.get("schema_version")
		return false
	var sc: Variant = JSON.parse_string(FileAccess.get_file_as_string(path.path_join("scene.json")))
	scene = sc if sc is Dictionary else {}
	var runs: Array = manifest.get("runs", [])
	if runs.is_empty():
		error_message = "vizlog has no runs"
		return false
	run_id = wanted_run if wanted_run != "" else str(runs[0]["id"])
	var run_dir := path.path_join("runs").path_join(run_id)
	var names := PackedStringArray()
	var widths := PackedInt32Array()
	for c: Dictionary in manifest["channels"]:
		names.append(c["name"])
		widths.append(int(c["width"]))
	# Usable rows: rows.json if complete, else whole rows present in every file.
	var rows: Variant = JSON.parse_string(FileAccess.get_file_as_string(run_dir.path_join("rows.json")))
	var n := int(rows["n_rows"]) if rows is Dictionary else 0
	var complete := rows is Dictionary and bool(rows.get("complete", false))
	var data := {}
	var on_disk := 1 << 62
	for i in names.size():
		var bytes := FileAccess.get_file_as_bytes(run_dir.path_join(names[i] + ".f64"))
		data[names[i]] = bytes
		on_disk = mini(on_disk, bytes.size() / (8 * widths[i]))
	if not complete:
		n = on_disk
	store = FrameStore.new(names, widths)
	var cols := {}
	for i in names.size():
		cols[names[i]] = (data[names[i]] as PackedByteArray).slice(0, 8 * n * widths[i]).to_float64_array()
	store.append_block(cols, n)
	# Model files referenced by the scene, if readable from here.
	var mesh: Variant = scene.get("mesh")
	if mesh is Dictionary:
		for lod: Dictionary in mesh.get("lods", []):
			var p := host_path(String(lod.get("path", "")))
			if FileAccess.file_exists(p):
				assets[p.get_file()] = {"meta": {"kind": "model_lod", "sha256": lod.get("sha256")},
					"data": FileAccess.get_file_as_bytes(p)}
	return n >= 2


## A path written on the Python side, as this process can open it: a viewer on
## Windows reading a vizlog written inside WSL sees /mnt/c/... as C:/...
static func host_path(p: String) -> String:
	if OS.get_name() == "Windows" and p.begins_with("/mnt/") and p.length() > 6 and p[6] == "/":
		return "%s:%s" % [p[5].to_upper(), p.substr(6)]
	return p


func poll() -> bool:
	return false
