extends RefCounted
## Creates the overlays listed in scene.json and updates them every frame
## after the world has been posed (called from World.update).

const Registry := preload("res://src/overlays/registry.gd")
const Overlay := preload("res://src/overlays/overlay.gd")

var nodes: Array[Node3D] = []
var store: RefCounted
var frames: Dictionary = {}  ## channel -> frame (BODY/GCRS), from the manifest
var enabled := true
var labels: Array[Label3D] = []
var show_labels := true
var _label_w := {}  ## Label3D -> text width [font px] (texts are fixed)
## Display groups the user toggles from the layer chips and menus.
const GROUP := {"arrow": "vectors", "cone": "vectors", "target_line": "vectors", "label": "vectors",
	"triad": "axes", "polyline": "tracks"}
const GROUPS := ["vectors", "labels", "axes", "tracks"]
var shown := {}  ## group -> bool, for the groups present
var label_visibility := {"satellite": true, "planets": true}
var label_setter: Callable
var cost_us := 0  ## CPU time of the last update (all overlays + label layout)
var layout_us := 0  ## of which label layout


func build(scene: Dictionary, manifest: Dictionary, data_store: RefCounted, craft: Node3D, root: Node) -> void:
	store = data_store
	for c: Dictionary in manifest.get("channels", []):
		frames[c["name"]] = String(c.get("frame", "-"))
	for d: Dictionary in scene.get("overlays", []):
		var n := Registry.create(d, scene)
		if n == null:
			continue
		if Registry.ATTACHED.has(String(d.get("type", ""))):
			craft.add_child(n)
		else:
			root.add_child(n)
		n.set_meta("enabled", bool(d.get("visible", true)))
		nodes.append(n)
	for n in nodes:
		for l in n.find_children("*", "Label3D", true, false):
			labels.append(l)
	_apply()


static func group_of(n: Node3D) -> String:
	if n.desc.get("type") == "arrow" and String(n.desc.get("source", "")).begins_with("user_inertia_axis_"):
		return "axes"
	return GROUP.get(String(n.desc.get("type", "")), "vectors")


## Groups present in this scene, in display order.
func groups() -> Array:
	return GROUPS.filter(func(g: String) -> bool: return shown.has(g))


func set_label_controller(setter: Callable) -> void:
	label_setter = setter
	_apply()


static func item_key(n: Node3D) -> String:
	var d: Dictionary = n.desc
	match String(d.get("type", "")):
		"triad": return "axes_" + String(d.get("frame", "body"))
		"arrow":
			var src := String(d.get("source", ""))
			return "axes_principal" if src.begins_with("user_inertia_axis_") else "vector_" + src
		"polyline":
			if d.get("track") == "ground":
				return "ground_track"
			return "orbit_track"
	return "overlay_" + String(d.get("id", n.name))


static func item_label(n: Node3D) -> String:
	var key := item_key(n)
	if n.desc.get("type") == "target_line":
		return "goal direction"
	var labels := {"vector_@nadir": "Nadir", "vector_@sun": "Sun", "vector_@velocity": "Velocity",
		"axes_body": "Satellite body axes", "axes_gcrs": "Inertial space axes",
		"axes_lvlh": "Orbital frame axes", "axes_principal": "Satellite principal axes",
		"orbit_track": "Orbit track", "ground_track": "Ground track"}
	return String(labels.get(key, n.desc.get("label", n.desc.get("id", n.name))))


func menu_items(group: String) -> Array:
	if group == "labels":
		return [
			{"key": "satellite_label", "label": "Satellite label", "checked": label_visibility["satellite"]},
			{"key": "planet_labels", "label": "Planet labels", "checked": label_visibility["planets"]},
		]
	var items := []
	var keys := {}
	for n in nodes:
		if group_of(n) != group:
			continue
		var key := item_key(n)
		if keys.has(key):
			continue
		keys[key] = true
		items.append({"key": key, "label": item_label(n), "checked": bool(n.get_meta("enabled", true))})
	var order := ["vector_@nadir", "vector_@sun", "vector_@velocity",
		"axes_body", "axes_gcrs", "axes_principal", "axes_lvlh",
		"orbit_track", "ground_track"]
	items.sort_custom(func(a: Dictionary, b: Dictionary) -> bool:
		var ai := order.find(a["key"])
		var bi := order.find(b["key"])
		return (ai if ai >= 0 else 1000) < (bi if bi >= 0 else 1000))
	return items


func toggle_item(group: String, key: String) -> void:
	if group == "labels":
		if key == "satellite_label":
			label_visibility["satellite"] = not bool(label_visibility["satellite"])
		elif key == "planet_labels":
			label_visibility["planets"] = not bool(label_visibility["planets"])
		else:
			return
		_apply()
		return
	var selected: Array = nodes.filter(func(n: Node3D) -> bool: return group_of(n) == group and item_key(n) == key)
	if selected.is_empty():
		return
	var on := not bool(selected[0].get_meta("enabled", true))
	for n in selected:
		n.set_meta("enabled", on)
	_apply()


func set_group(group: String, on: bool) -> void:
	if group == "labels":
		label_visibility["satellite"] = on
		label_visibility["planets"] = on
		_apply()
	elif shown.has(group):
		for n in nodes:
			if group_of(n) == group:
				n.set_meta("enabled", on)
		_apply()


func toggle_group(group: String) -> void:
	set_group(group, not bool(shown.get(group, false)))


## O: everything off if anything is shown, else everything on.
func toggle_all() -> void:
	var any: bool = shown.values().any(func(v: bool) -> bool: return v)
	for n in nodes:
		n.set_meta("enabled", not any)
	label_visibility["satellite"] = not any
	label_visibility["planets"] = not any
	_apply()


func set_enabled(on: bool) -> void:
	enabled = on
	_apply()


func _apply() -> void:
	shown.clear()
	for n in nodes:
		var group := group_of(n)
		var item_on := bool(n.get_meta("enabled", true))
		shown[group] = bool(shown.get(group, false)) or item_on
		n.visible = enabled and item_on
	shown["labels"] = bool(label_visibility["satellite"]) or bool(label_visibility["planets"])
	if label_setter.is_valid():
		label_setter.call(bool(label_visibility["satellite"]), bool(label_visibility["planets"]))


func update(frame: Dictionary, world: RefCounted, cam_pos: PackedFloat64Array, cam_rot: PackedFloat64Array) -> void:
	cost_us = 0
	if not enabled or nodes.is_empty():
		return
	var t0 := Time.get_ticks_usec()
	var ctx := {"frame": frame, "store": store, "world": world, "d_craft": world.info.get("d_craft", 1.0), "k_craft": world.info.get("k_craft", 1.0),
		"focal": world.focal_px(), "rot": preload("res://src/core/dmath.gd").rot_mat(frame["q_gcrs_body"]),
		"cam_pos": cam_pos, "cam_rot": cam_rot, "t": frame["t"], "frames": frames}
	for n in nodes:
		if n.visible:
			n.update(ctx)
	var t1 := Time.get_ticks_usec()
	declutter(world.camera)
	cost_us = Time.get_ticks_usec() - t0
	layout_us = Time.get_ticks_usec() - t1


## Push overlapping labels apart vertically on screen (labels are billboards
## of known pixel height, so this is a 2-D box layout, top to bottom).
func declutter(camera: Camera3D) -> void:
	var shown: Array[Label3D] = []
	var x0 := PackedFloat32Array()
	var x1 := PackedFloat32Array()
	var y := PackedFloat32Array()
	var order: Array[Vector2] = []  # (screen y, index): sorted natively
	for l in labels:
		l.offset = Vector2.ZERO
		if not show_labels:
			l.visible = false
		if not l.is_visible_in_tree() or camera.is_position_behind(l.global_position):
			continue
		if not _label_w.has(l):
			var font: Font = l.font if l.font else ThemeDB.fallback_font
			_label_w[l] = font.get_string_size(l.text, HORIZONTAL_ALIGNMENT_LEFT, -1, l.font_size).x \
				* Overlay.LABEL_PX / float(l.font_size)  # screen px
		var p := camera.unproject_position(l.global_position)
		var w: float = _label_w[l]
		var left := p.x if l.horizontal_alignment == HORIZONTAL_ALIGNMENT_LEFT else p.x - 0.5 * w
		order.append(Vector2(p.y, shown.size()))
		shown.append(l)
		x0.append(left)
		x1.append(left + w)
		y.append(p.y)
	order.sort()
	var h := Overlay.LABEL_PX * 1.25
	var placed := PackedInt32Array()
	for o in order:
		var i := int(o.y)
		var yi := y[i]
		for j in placed:
			if x0[i] < x1[j] and x0[j] < x1[i] and yi - y[j] < h and y[j] - yi < h:
				yi = y[j] + h
		y[i] = yi
		placed.append(i)
		if yi != o.x:  # Label3D offset is in font px, +y up
			shown[i].offset = Vector2(0.0, -(yi - o.x) * float(shown[i].font_size) / Overlay.LABEL_PX)


func summary() -> Array:
	var out := []
	for n in nodes:
		out.append({"id": n.name, "type": n.desc.get("type"), "visible": n.visible, "info": n.diagnostics()})
	return out
