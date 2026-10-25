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
## Display groups the user toggles (on-screen chips, keys V / A / G). A group starts
## shown if any of its overlays is declared visible; toggling it shows or hides all.
const GROUP := {"arrow": "vectors", "cone": "vectors", "target_line": "vectors", "label": "vectors",
	"triad": "axes", "polyline": "tracks"}
const GROUPS := ["vectors", "axes", "tracks"]
var shown := {}  ## group -> bool, for the groups present
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
		nodes.append(n)
	for n in nodes:
		for l in n.find_children("*", "Label3D", true, false):
			labels.append(l)
		var g := group_of(n)
		shown[g] = bool(shown.get(g, false)) or bool(n.desc.get("visible", true))
	_apply()


static func group_of(n: Node3D) -> String:
	return GROUP.get(String(n.desc.get("type", "")), "vectors")


## Groups present in this scene, in display order.
func groups() -> Array:
	return GROUPS.filter(func(g: String) -> bool: return shown.has(g))


func set_group(group: String, on: bool) -> void:
	if shown.has(group):
		shown[group] = on
		_apply()


func toggle_group(group: String) -> void:
	set_group(group, not bool(shown.get(group, false)))


## O: everything off if anything is shown, else everything on.
func toggle_all() -> void:
	var any: bool = shown.values().any(func(v: bool) -> bool: return v)
	for g: String in shown:
		shown[g] = not any
	_apply()


func set_enabled(on: bool) -> void:
	enabled = on
	_apply()


func _apply() -> void:
	for n in nodes:
		n.visible = enabled and bool(shown.get(group_of(n), true))


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
