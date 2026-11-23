extends RefCounted
## Bookmarks: named moments on the timeline. Those from the log (scene.json
## "timeline") are locked; the user's own are saved per log in
## user://bookmarks/<key>.json, where the key identifies the run (epoch, run id,
## initial state), so they come back when the same run is opened again.

signal changed

const DIR := "user://bookmarks"
const COLOR := "#ffd166"

var items: Array = []  ## [{id, t_s, label, note, color, locked}], sorted by time
var key := ""
var path := ""
var _next := 1


func setup(manifest: Dictionary, scene: Dictionary, run_id: String, first_row: PackedFloat64Array,
		dir: String = DIR) -> void:
	items.clear()
	for b: Dictionary in scene.get("timeline", {}).get("bookmarks", []):
		items.append({"id": String(b.get("id", "log%d" % items.size())), "t_s": float(b["t_s"]),
			"label": String(b.get("label", "")), "note": String(b.get("note", "")),
			"color": String(b.get("color", COLOR)), "locked": true})
	key = ("%s|%s|%s" % [JSON.stringify(manifest.get("epoch", {})), run_id, first_row]).md5_text()
	DirAccess.make_dir_recursive_absolute(dir)
	path = dir.path_join(key + ".json")
	_load()
	_sort()


func _load() -> void:
	if not FileAccess.file_exists(path):
		return
	var data: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not data is Dictionary:
		push_warning("bookmarks: unreadable %s" % path)
		return
	for b: Dictionary in (data as Dictionary).get("bookmarks", []):
		var id := String(b.get("id", ""))
		items.append({"id": id, "t_s": float(b.get("t_s", 0.0)), "label": String(b.get("label", "")),
			"note": String(b.get("note", "")), "color": String(b.get("color", COLOR)), "locked": false})
		if id.begins_with("u") and id.substr(1).is_valid_int():
			_next = maxi(_next, int(id.substr(1)) + 1)


func save() -> void:
	var own := items.filter(func(b: Dictionary) -> bool: return not b["locked"]).map(
		func(b: Dictionary) -> Dictionary: return {"id": b["id"], "t_s": b["t_s"], "label": b["label"],
			"note": b["note"], "color": b["color"]})
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_warning("bookmarks: cannot write %s" % path)
		return
	f.store_string(JSON.stringify({"format": "adcs-viewer-bookmarks", "version": 1, "bookmarks": own}, " "))


func _sort() -> void:
	items.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return a["t_s"] < b["t_s"])


func add(t: float, label: String = "") -> Dictionary:
	var b := {"id": "u%d" % _next, "t_s": t, "label": label if label != "" else "Bookmark %d" % _next,
		"note": "", "color": COLOR, "locked": false}
	_next += 1
	items.append(b)
	_sort()
	save()
	changed.emit()
	return b


func find(id: String) -> Dictionary:
	for b: Dictionary in items:
		if b["id"] == id:
			return b
	return {}


func rename(id: String, label: String) -> void:
	var b := find(id)
	if b.is_empty() or b["locked"]:
		return
	b["label"] = label
	save()
	changed.emit()


func remove(id: String) -> void:
	var b := find(id)
	if b.is_empty() or b["locked"]:
		return
	items.erase(b)
	save()
	changed.emit()


## All bookmarks as JSON (for the clipboard), each with the state at its time
## from `state_at(t) -> Dictionary`.
func export_json(state_at: Callable) -> String:
	var out := []
	for b: Dictionary in items:
		out.append({"t_s": b["t_s"], "label": b["label"], "note": b["note"], "from_log": b["locked"],
			"state": state_at.call(b["t_s"])})
	return JSON.stringify({"bookmarks": out}, " ")
